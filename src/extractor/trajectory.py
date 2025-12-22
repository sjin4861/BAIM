"""
Trajectory Extractor for BAIM - Solution Trajectory Analysis.

This module extends IExtractor to extract the FULL reasoning trajectory
(all token hidden states) instead of just the final state, enabling analysis of
the problem-solving path in embedding space.

Based on Problem Solving Theory (Newell & Simon, 1972):
- Problem solving = search through state space
- Each token generation = state transition
- Full trajectory = reasoning path from problem to solution
"""

from __future__ import annotations

import logging
from PIL import Image
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

import torch
import json
import re
import numpy as np
from transformers import StoppingCriteria, StoppingCriteriaList
from src.entities.question import Question
from src.entities.solution import Solution
from src.utils.message_router import build_messages_auto

logger = logging.getLogger(__name__)

MAX_RETRY = 5


class StopOnSequence(StoppingCriteria):
    def __init__(self, stop_sequence_ids):
        super().__init__()
        self.stop_sequence_ids = torch.tensor(stop_sequence_ids).to(torch.long)

    def __call__(self, input_ids, scores, **kwargs):
        # input_ids shape: [batch, seq_len]
        seq = input_ids[0][-len(self.stop_sequence_ids) :]
        return torch.equal(seq.cpu(), self.stop_sequence_ids)


class JsonBalanceStoppingCriteria(StoppingCriteria):
    """
    Stops generation when JSON brace balance is matched after </think> tag.
    Ignores braces inside <think> tag (e.g., math set notation).
    """

    def __init__(self, tokenizer, prompt_length):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs):
        # Decode only generated portion
        generated_ids = input_ids[0][self.prompt_length :]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Find </think> tag
        think_close_tag = "</think>"

        # If no </think> tag yet, still in thinking region
        if think_close_tag not in text:
            return False

        # Extract part after </think>
        parts = text.split(think_close_tag, 1)
        if len(parts) < 2:
            return False

        json_part = parts[1]

        # Check brace balance only in JSON part
        open_count = json_part.count("{")
        close_count = json_part.count("}")

        # 1. Must have at least one opening brace
        # 2. If open and close counts match, JSON is complete -> stop (True)
        if open_count > 0 and open_count == close_count:
            return True

        return False


class TrajectoryExtractor:
    """
    Extract full solution trajectories (all token hidden states).

    Extracts complete reasoning path from question-solution pairs.

    Output format (via extract()):
        {
            'trajectory': torch.Tensor([num_tokens, hidden_dim]),
            'metadata': dict,
            'token_ids': List[int]
        }
    """

    def __init__(
        self,
        model_name: str = None,  # Optional - will use config if not provided
        device: str = None,
        torch_dtype: str = None,
        layer_idx: int = -1,
        extract_all_layers: bool = True,
        mode: str = "generate",  # Default to generate mode
        extract_granularity: str = "token_wise",  # "token_wise", "layer_mean", or "both"
    ):
        """
        Args:
            model_name: HuggingFace model identifier (defaults to config)
            device: "cuda" or "cpu" (defaults to config)
            torch_dtype: "float16", "bfloat16", or "float32" (defaults to config)
            layer_idx: Which layer to extract (-1 = last layer)
            extract_all_layers: If True, extract all layers [num_layers, T, D]
            mode: "generate" (extract during generation) - replay mode removed
            extract_granularity: Extraction granularity mode:
                - "token_wise": Full token trajectory [T, D] or [L, T, D]
                - "layer_mean": Layer-averaged representation [L, D]
                - "both": Returns both token_wise and layer_mean as dict
        """
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
        from src.utils.config_loader import get_config

        # Load config
        config = get_config()

        # Use config values if not provided
        self.model_name = model_name or config.llm.model
        self.device = device or config.llm.transformers.device
        self.layer_idx = layer_idx
        self.extract_all_layers = extract_all_layers
        self.mode = mode
        self.extract_granularity = extract_granularity

        # Map string dtype to torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype_str = torch_dtype or config.llm.transformers.torch_dtype
        self.torch_dtype = dtype_map.get(torch_dtype_str, torch.bfloat16)

        # Store generation parameters from config
        self.generation_params = {
            "max_new_tokens": config.llm.max_tokens,
            "temperature": config.llm.temperature,
            "top_p": config.llm.top_p,
            "repetition_penalty": config.llm.repetition_penalty,
            "do_sample": True,  # Always sample for trajectory extraction
        }

        logger.info(f"Loading Qwen3-VL model for trajectory extraction: {model_name}")

        self.processor = Qwen3VLProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        # Load config for images root (reused across extractions)
        config = get_config()
        self.images_root = Path(config.data.images_root)

        # Cache for prompt builders by (prompt_path, solution_type)
        self._prompt_builder_cache = {}

        logger.info(f"Model loaded on {self.device} with dtype {torch_dtype_str}")
        logger.info(f"Extraction mode: {self.mode}")
        logger.info(f"Extraction granularity: {self.extract_granularity}")
        logger.info(f"Generation params: {self.generation_params}")
        logger.info(f"Images root: {self.images_root}")

    def extract(self, question: Question, solution: Solution = None) -> Dict[str, Any]:
        """
        Extract full reasoning trajectory.

        Returns the complete sequence of hidden states for all generated tokens.

        **Generate Mode (only mode supported):**
        - Generates solution with model.generate()
        - Extracts hidden states during generation
        - Only needs question
        - Always uses C (Correct reasoning) mode

        Args:
            question: Question entity (contains text, images, options)
            solution: Not used (kept for interface compatibility)

        Returns:
            Dictionary with:
                - trajectory: torch.Tensor [num_tokens, hidden_dim] or
                              [num_layers, num_tokens, hidden_dim]
                - metadata: dict with num_tokens, etc.
                - token_ids: List of generated token IDs

        Example:
            result = extractor.extract(question)
        """
        # Delegate to extract_trajectory for actual implementation
        return self.extract_trajectory(question, solution)

    def extract_trajectory(
        self, question: Question, solution: Solution = None
    ) -> Dict[str, Any]:
        """
        Extract full reasoning trajectory for a solution.

        **Generate Mode (only mode supported):**
        Generates solution text with model.generate() and extracts hidden states
        during generation. This captures the actual reasoning process.
        Uses parameters from config (temperature, max_tokens, etc.)
        Always uses C (Correct reasoning) mode.

        Args:
            question: Question entity (contains text, images, options)
            solution: Not used (kept for interface compatibility)

        Returns:
            Dictionary containing:
                - trajectory: torch.Tensor [num_tokens, hidden_dim] or
                              [num_layers, num_tokens, hidden_dim]
                - metadata: dict with num_tokens, etc.
                - token_ids: List of generated token IDs

        Example:
            result = extractor.extract_trajectory(question)
        """
        if self.mode == "generate":

            # 1. 기존 설정 백업
            _orig_granularity = self.extract_granularity
            _orig_all_layers = self.extract_all_layers

            # 2. 설정 강제 오버라이드 (Step 분석용)
            self.extract_granularity = "token_wise"  # 토큰 차원 살리기
            self.extract_all_layers = True  # 레이어 차원(65개) 살리기 [핵심!]

            try:
                # 3. 생성 및 추출 (이제 무조건 [L, T, D] 형태가 나옴)
                raw_result = self._extract_generate_mode(question)
            finally:
                # 4. 설정 원상복구 (안전장치)
                self.extract_granularity = _orig_granularity
                self.extract_all_layers = _orig_all_layers

            # 5. Step별 분할 및 평균 (Refining)
            # raw_result는 [65, Total_Tokens, 5120] 형태임이 보장됨.
            step_wise_result = self._refine_to_step_wise(raw_result)

            return step_wise_result
        else:
            raise ValueError(
                f"Unknown mode: {self.mode}. Only 'generate' mode is supported."
            )

    def _extract_section_boundaries_from_text(
        self, generated_text: str, generated_token_ids: list[int]
    ) -> dict:
        """
        JSON value 내용을 parsed 텍스트가 아니라,
        *raw generated_text 안의 JSON substring* 기준으로 정렬하는 버전.

        - thinking: </think> 이후 첫 '{' 전까지
        - phases: JSON 안에서 `"understand": "..."` 등 value 영역을 직접 찾아서
                해당 substring을 토큰 인덱스로 매핑
        """
        boundaries: Dict[str, Tuple[int, int]] = {}
        total_tokens = len(generated_token_ids)

        try:
            think_close = "</think>"
            if think_close in generated_text:
                before, after = generated_text.split(think_close, 1)
                search_region = after
                offset = len(before) + len(think_close)  # full_text 기준 offset
            else:
                search_region = generated_text
                offset = 0

            # JSON 시작 지점: 첫 '{'
            rel_json_start = search_region.find("{")
            if rel_json_start == -1:
                # JSON 자체가 없다면 전부 thinking
                logger.warning(
                    "No JSON block found when extracting boundaries → all thinking."
                )
                return {"thinking": (0, total_tokens)}

            json_region = search_region[rel_json_start:]
            json_start_char = offset + rel_json_start

            # 1) thinking 영역: JSON 시작 전까지
            thinking_text = generated_text[:json_start_char]

            _, thinking_end_tok = self._find_span_by_char_index(
                generated_text,
                generated_token_ids,
                thinking_text,
                start_token_hint=0,
            )
            boundaries["thinking"] = (0, thinking_end_tok)

            # 2) 각 phase value를 raw JSON 문자열에서 직접 찾기
            phases = ["understand", "plan", "carry_out", "look_back"]
            current_search_start = thinking_end_tok

            for phase in phases:
                # "phase": "...."  에서 따옴표 안쪽 value만 캡처
                pattern = r'"%s"\s*:\s*"(?P<val>(?:[^"\\]|\\.)*)"' % phase
                m = re.search(pattern, json_region, re.DOTALL)
                if not m:
                    logger.warning(
                        f"Could not find JSON field for phase '{phase}' in raw JSON."
                    )
                    continue

                raw_val = m.group(
                    "val"
                )  # escape 포함된 raw value (예: '\\n', '\\(' 등)
                # json_region 기준 char index → full_text 기준으로 환산
                rel_val_start = m.start("val")
                abs_val_start_char = (
                    json_start_char + rel_val_start
                )  # 사실 직접 쓰진 않지만 참고용

                # raw_val 은 generated_text 안에 그대로 등장하는 substring 이므로
                # _find_span_by_char_index 로 토큰 범위를 찾을 수 있음
                s_idx, e_idx = self._find_span_by_char_index(
                    generated_text,
                    generated_token_ids,
                    raw_val,
                    start_token_hint=current_search_start,
                )

                # s == e 인 경우는 의미 없는 영역이라 로그 남김
                if s_idx >= e_idx:
                    logger.warning(
                        f"Phase '{phase}' alignment produced empty span "
                        f"(start={s_idx}, end={e_idx})."
                    )
                boundaries[phase] = (s_idx, e_idx)
                current_search_start = e_idx

            return boundaries

        except Exception as e:
            logger.warning(f"Boundary extraction failed with error: {e}")
            return {"thinking": (0, total_tokens)}

    def _extract_generate_mode(self, question: Question) -> Dict[str, Any]:
        """
        Generate mode: Generate solution and extract trajectory.

        This mode generates the solution from scratch using model.generate()
        and extracts hidden states during generation.
        Always uses C (Correct reasoning) mode.

        Args:
            question: Question entity (contains text, images, options)

        Returns:
            Dictionary with trajectory, metadata, token_ids
        """
        from src.utils.prompt_builder import PromptBuilder, PromptTemplate
        from src.utils.config_loader import get_config

        logger.info("Extracting trajectory in GENERATE mode (C - Correct reasoning)...")

        # Load config to get prompt template path
        config = get_config()
        prompt_template_path = config.prompts.template

        # Get or create cached prompt builder
        cache_key = (
            prompt_template_path,
        )  # Simplified cache key without solution_type
        if cache_key not in self._prompt_builder_cache:
            template = PromptTemplate.from_yaml(prompt_template_path)
            prompt_builder = PromptBuilder(template)  # No mode parameter needed
            self._prompt_builder_cache[cache_key] = prompt_builder
        else:
            prompt_builder = self._prompt_builder_cache[cache_key]

        # Reconstruct the original user message (question + images)
        question_lines = question.get_lines(
            include_options=True, options_multiline=True
        )
        question_text = "\n".join(question_lines)
        analysis_text = question.analysis

        # Prepare image paths (all are local paths)
        image_paths = []
        if hasattr(question, "images") and question.images:
            for img_path in question.images:
                abs_path = self.images_root / img_path
                # Only add if file actually exists
                if abs_path.exists():
                    image_paths.append(str(abs_path))
                else:
                    logger.warning(f"Image file not found, skipping: {abs_path}")

        messages = build_messages_auto(
            prompt_builder,
            question_text=question_text,
            images_root=self.images_root,
            image_paths=image_paths,
            analysis=analysis_text,  # <--- 핵심: Analysis 전달
        )

        # 2️⃣ 텍스트만 뽑기 (tokenize=False, return_tensors=None)
        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            return_tensors=None,
        )
        # Load actual images (as PIL objects)
        images = (
            [Image.open(p).convert("RGB") for p in image_paths] if image_paths else None
        )

        # Preprocess text + images with processor (returns dict)
        inputs = self.processor(
            text=[prompt],
            images=images,
            return_tensors="pt",
            padding=True,
        )

        # Move to model device
        inputs = {
            k: (v.to(self.model.device) if torch.is_tensor(v) else v)
            for k, v in inputs.items()
        }

        # Generate with hidden state tracking using config parameters
        logger.info("Generating trajectory (C - Correct reasoning)...")
        logger.info(
            f"Generation params: temp={self.generation_params['temperature']}, "
            f"max_tokens={self.generation_params['max_new_tokens']}, "
            f"top_p={self.generation_params['top_p']}"
        )

        prompt_length = inputs["input_ids"].shape[1]

        stopping_criteria = StoppingCriteriaList(
            [JsonBalanceStoppingCriteria(self.processor.tokenizer, prompt_length)]
        )

        outputs = None
        valid_json = None  # Initialize before loop
        for attempt in range(MAX_RETRY):
            logger.info(f"[Attempt {attempt+1}/{MAX_RETRY}] Generating...")

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=self.generation_params["max_new_tokens"],
                    do_sample=self.generation_params["do_sample"],
                    temperature=self.generation_params["temperature"],
                    top_p=self.generation_params["top_p"],
                    repetition_penalty=self.generation_params["repetition_penalty"],
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    use_cache=True,
                    stopping_criteria=stopping_criteria,
                )

            # Extract only generated tokens
            all_token_ids = outputs.sequences[0].cpu().tolist()
            prompt_length = inputs["input_ids"].shape[1]
            generated_token_ids = all_token_ids[prompt_length:]
            generated_text = self.processor.tokenizer.decode(
                generated_token_ids, skip_special_tokens=True
            )

            # Validate JSON
            is_valid, parsed_json = self._is_valid_json_output(generated_text)

            if is_valid:
                logger.info("Valid JSON detected! Proceeding.")
                valid_json = parsed_json
                break
            else:
                logger.warning("Invalid JSON output detected. Retrying generation...")
                logger.warning(f"Failed Output : {generated_text}")
                # Free GPU memory from failed generation before retry
                # Delete outputs object (contains sequences, hidden_states, past_key_values)
                del outputs
                # Clear unreferenced tensors from GPU cache
                torch.cuda.empty_cache()
                logger.debug(f"Freed memory after failed attempt {attempt+1}")

        if valid_json is None:
            raise RuntimeError(
                "Failed to generate valid JSON after MAX_RETRY attempts."
            )

        # Extract trajectory from generation outputs
        # outputs.hidden_states: tuple of tuples
        # outer tuple: one element per generated token
        # inner tuple: one tensor per layer [batch, 1, hidden_dim]

        # 1️⃣ Always extract ALL layers first: [num_layers, num_tokens, hidden_dim]
        num_layers = len(outputs.hidden_states[0])
        num_tokens = len(outputs.hidden_states)
        hidden_dim = outputs.hidden_states[0][0].shape[-1]

        all_layers_trajectory = torch.zeros(num_layers, num_tokens, hidden_dim)

        for token_idx, step_hidden_states in enumerate(outputs.hidden_states):
            for layer_idx, layer_hidden in enumerate(step_hidden_states):
                # layer_hidden: [batch, 1, hidden_dim]
                all_layers_trajectory[layer_idx, token_idx, :] = layer_hidden[0, 0, :]

        logger.debug(f"Extracted all layers: {all_layers_trajectory.shape}")

        # Convert to float32 and move to CPU
        all_layers_trajectory = all_layers_trajectory.cpu().to(torch.float32)

        # Store all token IDs (prompt + generated) for reference
        all_token_ids = outputs.sequences[0].cpu().tolist()
        # But only use generated portion for metadata
        prompt_length = inputs["input_ids"].shape[1]
        generated_token_ids = all_token_ids[prompt_length:]

        logger.info(
            f"Generated {num_tokens} new tokens (prompt length: {prompt_length})"
        )
        logger.info(
            f"All layers trajectory shape: {all_layers_trajectory.shape} [L, T, D]"
        )

        # Decode ONLY the newly generated tokens (not the prompt)
        generated_text = self.processor.tokenizer.decode(
            generated_token_ids, skip_special_tokens=True
        )

        # Extract section boundaries from generated text
        section_boundaries = self._extract_section_boundaries_from_text(
            generated_text,
            generated_token_ids,
        )

        # Apply granularity transformation
        final_representation = self._apply_granularity(
            all_layers_trajectory, section_boundaries
        )

        # Build metadata
        # Handle shape for both single tensor and dict cases
        if isinstance(final_representation, dict):
            shape_info = {k: list(v.shape) for k, v in final_representation.items()}
        else:
            shape_info = list(final_representation.shape)

        metadata = {
            "extraction_type": "trajectory",
            "mode": "generate",
            "granularity": self.extract_granularity,
            "num_tokens": num_tokens,
            "num_layers": num_layers,
            "question_id": (
                question.question_id if hasattr(question, "question_id") else None
            ),
            "model_name": self.model_name,
            "layer_idx": self.layer_idx,
            "shape": shape_info,
            "all_layers_shape": [
                num_layers,
                num_tokens,
                hidden_dim,
            ],  # Original shape info
            "generation_params": self.generation_params,
            "section_boundaries": section_boundaries,
            "generated_text": generated_text,
        }

        if isinstance(final_representation, dict):
            logger.info(
                f"Trajectory extracted with {self.extract_granularity} granularity: shapes={shape_info}"
            )
        else:
            logger.info(
                f"Trajectory extracted with {self.extract_granularity} granularity: shape={final_representation.shape}"
            )
        logger.info(f"Section boundaries: {section_boundaries}")

        return {
            "trajectory": final_representation,  # Return granular representation
            "metadata": metadata,
            "token_ids": generated_token_ids,
        }

    def _apply_granularity(
        self, all_layers_trajectory: torch.Tensor, section_boundaries: Dict[str, tuple]
    ) -> torch.Tensor:
        """
        Apply granularity transformation to raw all-layers trajectory.

        Args:
            all_layers_trajectory: [num_layers, num_tokens, hidden_dim]
            section_boundaries: Section boundary information (for future use)

        Returns:
            Transformed trajectory based on self.extract_granularity:
            - "token_wise": Returns specific layer(s) trajectory
                * If layer_idx specified: [num_tokens, hidden_dim]
                * If extract_all_layers=True: [num_layers, num_tokens, hidden_dim]
            - "layer_mean": Returns layer-wise token-averaged representation
                * [num_layers, hidden_dim]
        """
        L, T, D = all_layers_trajectory.shape

        if self.extract_granularity == "token_wise":
            # Return token-level trajectory
            if self.extract_all_layers:
                # Return all layers: [L, T, D]
                logger.info(
                    f"Granularity=token_wise (all layers): returning {all_layers_trajectory.shape}"
                )
                return all_layers_trajectory
            else:
                # Return specific layer: [T, D]
                selected_layer = all_layers_trajectory[self.layer_idx]  # [T, D]
                logger.info(
                    f"Granularity=token_wise (layer {self.layer_idx}): {all_layers_trajectory.shape} → {selected_layer.shape}"
                )
                return selected_layer

        elif self.extract_granularity == "layer_mean":
            # Average across tokens for each layer: [L, T, D] → [L, D]
            layer_means = all_layers_trajectory.mean(
                dim=1
            )  # Average across tokens (dim=1)
            logger.info(
                f"Granularity=layer_mean: {all_layers_trajectory.shape} → {layer_means.shape}"
            )
            return layer_means

        elif self.extract_granularity == "both":
            # Return both representations as dict
            token_wise = (
                all_layers_trajectory[self.layer_idx]
                if not self.extract_all_layers
                else all_layers_trajectory
            )
            layer_means = all_layers_trajectory.mean(dim=1)

            result = {"token_wise": token_wise, "layer_mean": layer_means}
            logger.info(
                f"Granularity=both: token_wise={token_wise.shape}, layer_mean={layer_means.shape}"
            )
            return result

        else:
            raise ValueError(
                f"Unknown extract_granularity: {self.extract_granularity}. "
                f"Use 'token_wise', 'layer_mean', or 'both'"
            )

    def cleanup(self):
        """Cleanup resources (IExtractor interface implementation)"""
        if hasattr(self, "model"):
            del self.model
            torch.cuda.empty_cache()
            logger.info("TrajectoryExtractor cleaned up")

    def _sanitize_json_string(self, json_str: str) -> str:
        # 이미 올바른 이스케이프(\n, \t, \", \\ 등)를 제외하고
        # 나머지 `\x` 꼴은 `\\x`로 바꾼다
        def repl(m):
            return "\\\\" + m.group(1)

        return re.sub(r'\\([^"\\/bfnrtu])', repl, json_str)

    def _is_valid_json_output(self, text: str):
        """
        Validate if generated text contains valid JSON with required Polya fields.

        Returns:
            (is_valid, parsed_json): tuple of bool and dict (or None if invalid)
        """

        # Extract JSON only from the part AFTER </think> tag
        # This prevents matching braces inside <think> section (like {1,2} or {red, black})
        if "</think>" in text:
            # Only search for JSON after </think> tag
            json_search_region = text.split("</think>", 1)[1]
        else:
            # Fallback: use entire text if no </think> tag found
            json_search_region = text

        # Find the start of JSON (first '{')
        # Don't use greedy regex - let json.loads handle nested braces
        start_idx = json_search_region.find("{")
        if start_idx == -1:
            return False, None

        # Try to parse JSON starting from the first '{'
        # json.loads will correctly handle nested braces like {red, black} in values
        json_str = json_search_region[start_idx:].strip()
        json_str = self._sanitize_json_string(json_str)

        # JSON parse
        try:
            parsed = json.loads(json_str)
        except Exception:
            return False, None

        # key existence - Polya 4단계 키 확인
        required_keys = ["understand", "plan", "carry_out", "look_back"]
        for k in required_keys:
            if k not in parsed:
                return False, None

        # values type check
        if not all(isinstance(parsed[k], str) for k in required_keys):
            return False, None

        # 모든 조건 만족
        return True, parsed

    def __del__(self):
        """Cleanup model from memory"""
        self.cleanup()

    def _refine_to_step_wise(self, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        기존 통짜 Trajectory를 Polya 4단계(understand/plan/carry_out/look_back)별로
        [L, T, D] -> 각 구간 [L, D] 평균 벡터로 만드는 단계.

        우선적으로 metadata['section_boundaries'] (generate 시 계산된 결과)를 사용하고,
        필요하면 텍스트 정렬 fallback을 사용한다.
        """
        trajectory_tensor = raw_result["trajectory"]
        token_ids = raw_result["token_ids"]
        metadata = raw_result["metadata"]
        full_text = metadata.get("generated_text", "")

        phases = ["understand", "plan", "carry_out", "look_back"]

        # 1) generate 시 계산된 section_boundaries를 우선 사용
        boundaries: Dict[str, Tuple[int, int]] = (
            metadata.get("section_boundaries", {}) or {}
        )

        # 2) JSON 파싱은 'phase별 텍스트' 메타데이터 용도로만 사용
        phase_segments: Dict[str, str] = {p: "" for p in phases}
        try:
            search_text = (
                full_text.split("</think>", 1)[1]
                if "</think>" in full_text
                else full_text
            )
            json_match = re.search(r"(\{.*\})", search_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                json_str = self._sanitize_json_string(json_str)
                parsed_json = json.loads(json_str)
                for p in phases:
                    if isinstance(parsed_json.get(p, ""), str):
                        phase_segments[p] = parsed_json[p]
        except Exception as e:
            logger.warning(f"JSON parsing failed in refine step: {e}")

        # 3) boundaries 중 쓸만한(길이>0) 구간이 하나도 없으면 fallback
        has_valid = any(
            (p in boundaries) and (boundaries[p][0] < boundaries[p][1]) for p in phases
        )
        if not has_valid:
            logger.warning(
                "No usable per-phase boundaries in metadata; "
                "falling back to alignment based on phase text."
            )
            boundaries = self._align_segments_to_tokens(
                full_text, token_ids, phases, phase_segments
            )

        refined_trajectory: Dict[str, torch.Tensor] = {}

        # Helper function to slice and average segments
        def process_segment(tensor_data, start_idx: int, end_idx: int):
            sliced = self._slice_tensor(tensor_data, start_idx, end_idx)
            if isinstance(sliced, torch.Tensor):
                if sliced.ndim == 3:  # [L, T, D] -> [L, D]
                    return sliced.mean(dim=1)
                elif sliced.ndim == 2:  # [T, D] -> [D]
                    return sliced.mean(dim=0)
            elif isinstance(sliced, dict):
                return {
                    k: (
                        v.mean(dim=1)
                        if isinstance(v, torch.Tensor) and v.ndim == 3
                        else (
                            v.mean(dim=0)
                            if isinstance(v, torch.Tensor) and v.ndim == 2
                            else v
                        )
                    )
                    for k, v in sliced.items()
                }
            return sliced

        # For each phase, if boundary exists, slice and average
        for phase in phases:
            if phase not in boundaries:
                continue
            s, e = boundaries[phase]
            if s < e:
                refined_trajectory[phase] = process_segment(trajectory_tensor, s, e)
            else:
                logger.warning(
                    f"Empty span detected for phase '{phase}' during refinement "
                    f"(start={s}, end={e})."
                )

        # Update metadata
        metadata["polya_segments"] = phase_segments
        metadata["step_boundaries"] = boundaries

        if not refined_trajectory:
            logger.warning(
                "No valid phases extracted during refinement. "
                "Returning raw (unrefined) trajectory."
            )
            return raw_result

        logger.info(
            f"Refined trajectory into phases: {list(refined_trajectory.keys())}"
        )
        return {
            "trajectory": refined_trajectory,
            "metadata": metadata,
            "token_ids": token_ids,
        }

    def _align_segments_to_tokens(
        self,
        full_text: str,
        token_ids: List[int],
        ordered_keys: List[str],
        segment_dict: Dict[str, str],
    ) -> Dict[str, Tuple[int, int]]:
        """
        Find token positions for text segments sequentially.
        Uses cumulative decoding to accurately align text segments to tokens.
        """
        boundaries = {}

        # Start search cursor (token index)
        current_search_start_token = 0

        for key in ordered_keys:
            target_text = segment_dict.get(key, "")

            # Skip if content is empty
            if not target_text.strip():
                continue

            # Use existing char index finder with sequential search starting from last found position
            start_idx, end_idx = self._find_span_by_char_index(
                full_text,
                token_ids,
                target_text,
                start_token_hint=current_search_start_token,
            )

            boundaries[key] = (start_idx, end_idx)

            # Next search starts from current end position
            # (JSON structure naturally creates small offset due to key names like "plan": )
            current_search_start_token = end_idx

        return boundaries

    def _find_span_by_char_index(
        self, full_text, token_ids, target_subtext, start_token_hint=0
    ):
        """
        Improved string matching and fast token mapping using binary search.
        Finds character positions in text and maps them to token indices.
        """
        import json

        if not target_subtext:
            return (start_token_hint, start_token_hint)

        # 1. Text 상에서 위치 찾기
        # start_token_hint를 이용하여 검색 시작 위치(char index)를 계산
        # 주의: decode는 느리므로 start_token_hint가 0일 때만 안전하게 0부터 시작,
        # 아니라면 근사치를 써야 하지만 정확도를 위해 start_token_hint까지는 decode를 한 번 수행
        prefix_text = ""
        if start_token_hint > 0:
            prefix_text = self.processor.tokenizer.decode(
                token_ids[:start_token_hint], skip_special_tokens=True
            )

        prefix_len = len(prefix_text)

        # [핵심 수정] json.dumps 대신 원본 텍스트로 먼저 검색 시도
        # full_text와 target_subtext는 모두 decode된 상태이므로 포맷이 일치해야 함
        start_char = full_text.find(target_subtext, prefix_len)

        # 실패 시에만 json.dumps 시도 (이스케이프 문자 차이 대응용 fallback)
        if start_char == -1:
            try:
                escaped_target = json.dumps(target_subtext, ensure_ascii=False)[1:-1]
                start_char = full_text.find(escaped_target, prefix_len)
                if start_char != -1:
                    target_subtext = escaped_target  # Found with escaping
            except:
                pass

        if start_char == -1:
            logger.warning(
                f"Could not find target text via exact match. Snippet: {target_subtext[:50]}..."
            )
            return (start_token_hint, start_token_hint)

        end_char = start_char + len(target_subtext)

        # 2. Find token index (Binary Search optimization)
        # Improved from O(N^2) to O(N log N)

        # Function to return decoded length up to a specific token index
        # (lru_cache would be ideal but complex for class methods, so simple call)
        def get_decoded_len(idx):
            if idx == 0:
                return 0
            # Partial decoding
            txt = self.processor.tokenizer.decode(
                token_ids[:idx], skip_special_tokens=True
            )
            return len(txt)

        # Binary Search Helper
        def binary_search_token_index(low, high, target_char_len):
            # lower_bound: first index where decoded_len >= target_char_len
            ans = high
            while low <= high:
                mid = (low + high) // 2
                length = get_decoded_len(mid)

                if length >= target_char_len:
                    ans = mid
                    high = mid - 1
                else:
                    low = mid + 1
            return ans

        # Search range: from hint to end
        total_tokens = len(token_ids)

        # Find start token
        # First position where length >= start_char includes the start token
        # Search starting from start_token_hint
        start_token_idx = binary_search_token_index(
            start_token_hint, total_tokens, start_char
        )

        # 끝 토큰 찾기 (시작 토큰 이후부터 검색)
        # end_char보다 길이가 같거나 커지는 지점
        end_token_idx = binary_search_token_index(
            start_token_idx, total_tokens, end_char
        )

        # 보정: end_token_idx는 해당 텍스트가 끝나는 지점을 '포함'하는 토큰까지임.
        # slicing을 위해 +1 할 필요가 있는지 확인해야 함.
        # binary_search가 lower_bound를 찾으므로, decoded_len >= end_char가 되는 첫 지점.
        # 즉, end_token_idx까지 포함해야 end_char를 커버함. python slice는 exclusive하므로 +1 필요?
        # 아니요, get_decoded_len(end_token_idx) >= end_char 이므로,
        # token_ids[:end_token_idx] 만으로 텍스트가 잘릴 수 있음.
        # 따라서 end_token_idx 자체를 포함해야 함 -> slice 시 end_token_idx
        # (Qwen 토크나이저는 byte 단위라 토큰 하나가 문자의 일부일 수 있음, 넉넉하게 잡는게 좋음)

        # 약간의 여유를 두지 않으면 마지막 글자가 잘릴 수 있으니 체크
        if get_decoded_len(end_token_idx) < end_char:
            end_token_idx += 1

        return (start_token_idx, end_token_idx)

    def _slice_tensor(self, tensor_data, start, end):
        """Slice tensor or dictionary of tensors"""
        if isinstance(tensor_data, dict):
            return {
                k: self._slice_tensor(v, start, end) for k, v in tensor_data.items()
            }

        # Tensor shapes: [Num_Layers, Total_Tokens, Hidden_Dim] or [Total_Tokens, Hidden_Dim]
        # Shape varies depending on extract_granularity

        # [L, T, D] case
        if tensor_data.ndim == 3:
            return tensor_data[:, start:end, :]
        # [T, D] case
        elif tensor_data.ndim == 2:
            return tensor_data[start:end, :]
        # [L, D] case (Layer Mean) -> Cannot slice (already averaged)
        else:
            return tensor_data
