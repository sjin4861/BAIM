from __future__ import annotations

import argparse
import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from transformers import StoppingCriteria, StoppingCriteriaList, Qwen3VLForConditionalGeneration, Qwen3VLProcessor


_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z0-9_]+)\}")
_IMAGE_MARKER_RE = re.compile(r"\bquestion_\d+-image_\d+\b")


@dataclass
class PromptTemplate:
    name: str = ""
    version: str = ""
    notes: str = ""
    system_prompt: str = ""
    instruction: str = ""

    @staticmethod
    def from_yaml(path: str | Path) -> "PromptTemplate":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Prompt YAML not found: {p}")

        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        meta = data.get("meta", {}) or {}

        return PromptTemplate(
            name=str(meta.get("name", "")),
            version=str(meta.get("version", "")),
            notes=str(meta.get("notes", "")),
            system_prompt=(data.get("system_prompt") or "").strip(),
            instruction=(data.get("instruction") or data.get("instruction_C") or "").strip(),
        )

def render_placeholders(template: str, ctx: Dict[str, Any]) -> str:
    def repl(m: re.Match) -> str:
        key = m.group(1)
        return str(ctx.get(key, m.group(0)))

    return _PLACEHOLDER_RE.sub(repl, template)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML not found: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_options(options: Any) -> List[str]:
    if options is None:
        return []

    if isinstance(options, dict):
        out: List[str] = []
        for key in sorted(options.keys()):
            out.append(f"{key} : {str(options[key])}")
        return out

    if isinstance(options, list):
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        out = []
        for i, opt in enumerate(options):
            if i >= len(labels):
                break
            out.append(f"{labels[i]} : {str(opt)}")
        return out

    return []


def build_question_text(qid: str, item: Dict[str, Any]) -> str:
    q_type = str(item.get("type", "unknown"))
    content = str(item.get("content", "")).strip()

    lines = [f"Q#{qid} [{q_type}]", f"Question: {content}"]
    for opt_line in normalize_options(item.get("options")):
        lines.append(opt_line)

    return "\n".join(lines)


def extract_image_markers(item: Dict[str, Any], qid: str) -> List[str]:
    markers: List[str] = []

    content = str(item.get("content", ""))
    markers.extend(_IMAGE_MARKER_RE.findall(content))

    for field in ["images", "image_paths"]:
        value = item.get(field)
        if isinstance(value, list):
            for v in value:
                if v is None:
                    continue
                stem = Path(str(v)).stem
                if stem:
                    markers.append(stem)
        elif isinstance(value, str):
            stem = Path(value).stem
            if stem:
                markers.append(stem)

    if not markers:
        markers.append(f"question_{qid}-image_0")

    unique = []
    seen = set()
    for m in markers:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    return unique


def resolve_image_paths(item: Dict[str, Any], qid: str, images_root: Path) -> List[Path]:
    if not images_root.exists():
        return []

    exts = [".jpg", ".jpeg", ".png", ".webp"]
    stems = extract_image_markers(item, qid)

    found: List[Path] = []
    for stem in stems:
        direct = images_root / stem
        if direct.exists():
            found.append(direct)
            continue

        matched = None
        for ext in exts:
            p = images_root / f"{stem}{ext}"
            if p.exists():
                matched = p
                break
        if matched is not None:
            found.append(matched)

    unique = []
    seen = set()
    for p in found:
        s = str(p.resolve())
        if s not in seen:
            seen.add(s)
            unique.append(p)

    return unique


def build_messages(
    template: PromptTemplate,
    question_text: str,
    analysis: str,
    image_paths: List[Path],
) -> List[Dict[str, Any]]:
    ctx = {
        "question": (question_text or "").strip(),
        "analysis": (analysis or "").strip(),
    }

    user_text = render_placeholders(template.instruction, ctx)

    if not image_paths:
        return [
            {"role": "system", "content": template.system_prompt},
            {"role": "user", "content": user_text},
        ]

    user_parts: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
    for p in image_paths:
        user_parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"file://{str(p.resolve())}"},
            }
        )

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": template.system_prompt}],
        },
        {"role": "user", "content": user_parts},
    ]


def sanitize_json_string(json_str: str) -> str:
    def repl(m: re.Match) -> str:
        return "\\\\" + m.group(1)

    return re.sub(r"\\([^\"\\/bfnrtu])", repl, json_str)


def parse_polya_json(text: str) -> Tuple[bool, Optional[Dict[str, str]]]:
    search_region = text.split("</think>", 1)[1] if "</think>" in text else text
    start_idx = search_region.find("{")
    if start_idx < 0:
        return False, None

    raw = search_region[start_idx:].strip()
    raw = sanitize_json_string(raw)

    try:
        parsed = json.loads(raw)
    except Exception:
        return False, None

    required_keys = ["understand", "plan", "carry_out", "look_back"]
    for key in required_keys:
        if key not in parsed or not isinstance(parsed[key], str):
            return False, None

    out = {k: parsed[k] for k in required_keys}
    return True, out


class JsonBalanceStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length: int):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        generated_ids = input_ids[0][self.prompt_length:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        if "</think>" not in text:
            return False

        json_part = text.split("</think>", 1)[1]
        open_count = json_part.count("{")
        close_count = json_part.count("}")

        return open_count > 0 and open_count == close_count


class SimpleTrajectoryExtractor:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.config = load_yaml(self.config_path)

        self._setup_logging()

        self.model_name = self.config.get("model_name")
        paths_cfg = self.config.get("paths", {}) or {}
        gen_cfg = self.config.get("generation", {}) or {}
        rt_cfg = self.config.get("runtime", {}) or {}

        self.seed = int(rt_cfg.get("seed", 42))
        set_global_seed(self.seed)
        logging.info("Global seed set to %d", self.seed)

        self.dataset = str(self.config.get("dataset", "NIPS34")).strip()
        self.prompt_path = Path(paths_cfg.get("prompt_path", "src/prompts/reasoning_template.yaml"))
        self.question_path, self.images_root = self._resolve_dataset_paths(paths_cfg)

        save_dir_raw = str(paths_cfg.get("save_dir", "data/{dataset}/solutions"))
        self.save_dir = Path(save_dir_raw.replace("{dataset}", self.dataset))
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype_str = str(rt_cfg.get("torch_dtype", "bfloat16")).lower()
        self.torch_dtype = dtype_map.get(torch_dtype_str, torch.bfloat16)

        self.generation_params = {
            "temperature": float(gen_cfg.get("temperature", 0.7)),
            "top_p": float(gen_cfg.get("top_p", 0.9)),
            "repetition_penalty": float(gen_cfg.get("repetition_penalty", 1.1)),
            "max_new_tokens": int(gen_cfg.get("max_new_tokens", 12000)),
            "max_retry": int(gen_cfg.get("max_retry", 10)),
            "json_mode": bool(gen_cfg.get("json_mode", True)),
        }

        self.template = PromptTemplate.from_yaml(self.prompt_path)

        logging.info("Loading model: %s", self.model_name)
        self.processor = Qwen3VLProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=bool(rt_cfg.get("trust_remote_code", True)),
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            trust_remote_code=bool(rt_cfg.get("trust_remote_code", True)),
        )
        self.model.eval()

        self.save_dir.mkdir(parents=True, exist_ok=True)
        logging.info(
            "Extractor ready. dataset=%s, question_path=%s, images_root=%s, save_dir=%s",
            self.dataset,
            self.question_path,
            self.images_root,
            self.save_dir,
        )

    def _resolve_dataset_paths(self, paths_cfg: Dict[str, Any]) -> Tuple[Path, Path]:
        dataset_root = Path(self.dataset)

        question_override = str(paths_cfg.get("question_path", "")).strip()
        if question_override:
            question_path = Path(question_override.replace("{dataset}", self.dataset))
        else:
            candidates = [
                dataset_root / "metadata" / "question_en.json",
                dataset_root / "metadata" / "questions_en.json",
            ]
            question_path = next((p for p in candidates if p.exists()), None)
            if question_path is None:
                attempted = ", ".join(str(p) for p in candidates)
                raise FileNotFoundError(
                    f"No question JSON found for dataset={self.dataset}. Tried: {attempted}"
                )

        images_override = str(paths_cfg.get("images_root", "")).strip()
        if images_override:
            images_root = Path(images_override.replace("{dataset}", self.dataset))
        else:
            images_root = dataset_root / "metadata" / "images"

        if not images_root.exists():
            logging.warning(
                "images_root does not exist yet: %s (continuing without images)",
                images_root,
            )

        return question_path, images_root

    def _setup_logging(self) -> None:
        logging_cfg = self.config.get("logging", {}) or {}
        level = str(logging_cfg.get("level", "INFO")).upper()
        runtime_log = logging_cfg.get("runtime_log")

        handlers: List[logging.Handler] = [logging.StreamHandler()]
        if runtime_log:
            handlers.append(logging.FileHandler(runtime_log, encoding="utf-8"))

        logging.basicConfig(
            level=getattr(logging, level, logging.INFO),
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=handlers,
        )

    def _prepare_inputs(
        self,
        question_text: str,
        analysis_text: str,
        image_paths: List[Path],
    ) -> Dict[str, Any]:
        messages = build_messages(
            template=self.template,
            question_text=question_text,
            analysis=analysis_text,
            image_paths=image_paths,
        )

        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            return_tensors=None,
        )

        images = [Image.open(str(p)).convert("RGB") for p in image_paths] if image_paths else None

        inputs = self.processor(
            text=[prompt],
            images=images,
            return_tensors="pt",
            padding=True,
        )

        device_inputs = {
            k: (v.to(self.model.device) if torch.is_tensor(v) else v)
            for k, v in inputs.items()
        }
        return device_inputs

    def extract_one(
        self,
        qid: str,
        item: Dict[str, Any],
        include_trajectory: bool = True,
    ) -> Dict[str, Any]:
        question_text = build_question_text(qid, item)
        analysis_text = str(item.get("analysis", "") or "")
        image_paths = resolve_image_paths(item, qid, self.images_root)

        inputs = self._prepare_inputs(question_text, analysis_text, image_paths)
        prompt_length = int(inputs["input_ids"].shape[1])

        stopping_criteria = None
        if self.generation_params["json_mode"]:
            stopping_criteria = StoppingCriteriaList(
                [JsonBalanceStoppingCriteria(self.processor.tokenizer, prompt_length)]
            )

        outputs = None
        parsed_json = None
        generated_text = ""
        generated_token_ids: List[int] = []

        max_retry = self.generation_params["max_retry"]
        for attempt in range(1, max_retry + 1):
            logging.info("Q%s attempt %d/%d", qid, attempt, max_retry)
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=self.generation_params["max_new_tokens"],
                    do_sample=True,
                    temperature=self.generation_params["temperature"],
                    top_p=self.generation_params["top_p"],
                    repetition_penalty=self.generation_params["repetition_penalty"],
                    output_hidden_states=include_trajectory,
                    return_dict_in_generate=True,
                    use_cache=True,
                    stopping_criteria=stopping_criteria,
                )

            all_token_ids = outputs.sequences[0].detach().cpu().tolist()
            generated_token_ids = all_token_ids[prompt_length:]
            generated_text = self.processor.tokenizer.decode(
                generated_token_ids,
                skip_special_tokens=True,
            )

            is_valid, parsed = parse_polya_json(generated_text)
            if is_valid:
                parsed_json = parsed
                break

            logging.warning("Q%s invalid JSON on attempt %d", qid, attempt)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if parsed_json is None:
            raise RuntimeError(f"Q{qid}: failed to generate valid JSON")

        result: Dict[str, Any] = {
            "qid": str(qid),
            "type": item.get("type"),
            "raw_text": generated_text,
            "output": parsed_json,
            "answer": item.get("answer"),
            "used_images": [str(p) for p in image_paths],
            "meta": {
                "model_name": self.model_name,
                "dataset": self.dataset,
                "prompt_path": str(self.prompt_path),
                "question_path": str(self.question_path),
                "num_generated_tokens": len(generated_token_ids),
                "generation": self.generation_params,
            },
        }

        if include_trajectory and outputs is not None and outputs.hidden_states is not None:
            num_layers = len(outputs.hidden_states[0])
            num_tokens = len(outputs.hidden_states)
            hidden_dim = int(outputs.hidden_states[0][0].shape[-1])

            traj = torch.zeros(num_layers, num_tokens, hidden_dim)
            for t, step_states in enumerate(outputs.hidden_states):
                for l, layer_state in enumerate(step_states):
                    traj[l, t, :] = layer_state[0, 0, :]

            result["trajectory"] = traj.detach().cpu().to(torch.float32)
            result["token_ids"] = generated_token_ids
            result["meta"]["trajectory_shape"] = [num_layers, num_tokens, hidden_dim]

        return result

    def run_dataset(
        self,
        question_path: str | Path | None,
        include_trajectory: bool,
        limit: Optional[int] = None,
        start_index: int = 0,
    ) -> List[Dict[str, Any]]:
        p = Path(question_path) if question_path else self.question_path
        if not p.exists():
            raise FileNotFoundError(f"Question JSON not found: {p}")

        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Question JSON must be a dict keyed by index strings")

        keys = sorted(data.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
        if start_index > 0:
            keys = keys[start_index:]
        if limit is not None:
            keys = keys[:limit]

        results: List[Dict[str, Any]] = []
        for qid in keys:
            item = data[qid]
            if not isinstance(item, dict):
                logging.warning("Skipping qid=%s because item is not a dict", qid)
                continue

            result = self.extract_one(
                qid=qid,
                item=item,
                include_trajectory=include_trajectory,
            )
            results.append(result)

        return results

    def save_results(self, results: List[Dict[str, Any]], output_name: str = "solutions.jsonl") -> Path:
        out_path = self.save_dir / output_name

        with out_path.open("w", encoding="utf-8") as f:
            for row in results:
                payload = dict(row)
                traj = payload.pop("trajectory", None)
                token_ids = payload.get("token_ids")
                if isinstance(token_ids, list) and len(token_ids) > 0:
                    payload["token_ids"] = token_ids
                if traj is not None:
                    payload["trajectory_file"] = f"{payload['qid']}.pt"
                    torch.save(traj, self.save_dir / payload["trajectory_file"])
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        logging.info("Saved %d results to %s", len(results), out_path)
        return out_path

    def cleanup(self) -> None:
        if hasattr(self, "model"):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple Qwen trajectory extractor")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/extract.yaml",
        help="Path to extract YAML",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    extractor = SimpleTrajectoryExtractor(config_path=args.config)
    run_cfg = extractor.config.get("run", {}) or {}

    question_path = run_cfg.get("questions")
    if question_path is None:
        question_path = run_cfg.get("question_path")

    limit = run_cfg.get("limit")
    start_index = int(run_cfg.get("start_index", 0))
    include_trajectory = bool(run_cfg.get("include_trajectory", True))
    output_name = str(run_cfg.get("output_name", "solutions.jsonl"))

    try:
        results = extractor.run_dataset(
            question_path=question_path,
            include_trajectory=include_trajectory,
            limit=limit,
            start_index=start_index,
        )
        extractor.save_results(results, output_name=output_name)
    finally:
        extractor.cleanup()


if __name__ == "__main__":
    main()
