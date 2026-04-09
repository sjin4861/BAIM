#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA


DEFAULT_STAGE_FILES: List[Tuple[str, int]] = [
    ("understand.pt", 0),
    ("plan.pt", 1),
    ("carry_out.pt", 2),
    ("look_back.pt", 3),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build stage-mean embeddings from raw trajectories, then run PCA."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/process.yaml",
        help="Path to process YAML config.",
    )
    return parser.parse_args()


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


def replace_dataset_placeholder(value: str, dataset: str) -> str:
    return value.replace("{dataset}", dataset)


def resolve_path(raw: str, dataset: str) -> Path:
    return Path(replace_dataset_placeholder(str(raw), dataset))


def resolve_stage_files(cfg: Dict[str, Any]) -> List[Tuple[str, int]]:
    raw = cfg.get("stage_files")
    if raw is None:
        return list(DEFAULT_STAGE_FILES)

    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError("stage_files must be a list of 4 objects with keys: name, stage_id")

    parsed: List[Tuple[str, int]] = []
    seen: set[int] = set()
    for row in raw:
        if not isinstance(row, dict):
            raise ValueError("Each stage_files entry must be a dict")
        name = str(row.get("name", "")).strip()
        stage_id = int(row.get("stage_id", -1))
        if not name:
            raise ValueError("stage_files entry has empty name")
        if stage_id < 0 or stage_id > 3:
            raise ValueError(f"Invalid stage_id={stage_id}; expected 0..3")
        if stage_id in seen:
            raise ValueError(f"Duplicate stage_id found: {stage_id}")
        seen.add(stage_id)
        parsed.append((name, stage_id))

    if seen != {0, 1, 2, 3}:
        raise ValueError("stage_files must contain stage_id 0,1,2,3 exactly once")

    return parsed


def infer_end_index(base: Path, start: int) -> int:
    candidates: List[int] = []
    for child in base.iterdir():
        if not child.is_dir():
            continue
        if child.name.isdigit():
            candidates.append(int(child.name))

    if not candidates:
        raise ValueError(f"No numeric index folders found under: {base}")

    max_idx = max(candidates)
    if max_idx < start:
        raise ValueError(f"No index folder >= start={start} under: {base}")
    return max_idx


def load_stage_tensor(path: Path, expected_d: int, expected_t: int = 0) -> Tuple[torch.Tensor, int]:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"Not a torch.Tensor: type={type(obj)}")
    if obj.ndim != 2:
        raise ValueError(f"Expected 2D tensor [T,D], got shape={tuple(obj.shape)}")

    t, d = int(obj.shape[0]), int(obj.shape[1])
    if d != expected_d:
        raise ValueError(f"Embedding dim mismatch: expected D={expected_d}, got D={d}")
    if expected_t and t != expected_t:
        raise ValueError(f"T mismatch: expected T={expected_t}, got T={t}")

    return obj.to(dtype=torch.float32).contiguous(), t


def build_mean_tensor(
    base: Path,
    start: int,
    end: int,
    expected_d: int,
    expected_t: int,
    stage_files: Sequence[Tuple[str, int]],
    skip_bad: bool,
) -> Tuple[torch.Tensor, List[Tuple[int, str]], int]:
    bad: List[Tuple[int, str]] = []
    mean_rows: List[torch.Tensor] = []
    inferred_t: Optional[int] = None

    for idx in range(start, end + 1):
        folder = base / str(idx)
        try:
            if not folder.exists():
                raise FileNotFoundError("Folder missing")

            stage_vecs: List[Optional[torch.Tensor]] = [None, None, None, None]
            for fname, stage_id in stage_files:
                fpath = folder / fname
                if not fpath.exists():
                    raise FileNotFoundError(f"Missing file: {fname}")

                t2d, t = load_stage_tensor(
                    fpath,
                    expected_d=expected_d,
                    expected_t=expected_t,
                )

                if inferred_t is None:
                    inferred_t = t
                if expected_t == 0 and inferred_t is not None and t != inferred_t:
                    raise ValueError(
                        f"Inconsistent T across files: first T={inferred_t}, now T={t} at {fpath}"
                    )

                stage_vecs[stage_id] = t2d.mean(dim=0)  # [D]

            if any(v is None for v in stage_vecs):
                raise RuntimeError("Some stage vectors were not assigned")

            stacked = torch.stack([v for v in stage_vecs if v is not None], dim=0)  # [4,D]
            mean_rows.append(stacked)

        except Exception as e:
            bad.append((idx, str(e)))
            print(f"[BAD] idx={idx} path={folder} error={e}")
            if not skip_bad:
                print("Exiting due to error (set skip_bad=true in config to continue).")
                raise

    if not mean_rows:
        raise RuntimeError("No valid samples were built. Check base path and index range.")

    if inferred_t is None:
        inferred_t = expected_t if expected_t else 0

    out = torch.stack(mean_rows, dim=0)  # [N_good,4,D]
    return out, bad, inferred_t


def run_pca(
    x_mean: torch.Tensor,
    n_components: int,
    random_state: int,
    svd_solver: str,
) -> Tuple[torch.Tensor, float]:
    if x_mean.ndim != 3:
        raise ValueError(f"Expected [N,4,D], got {tuple(x_mean.shape)}")

    n, s, d = x_mean.shape
    if s != 4:
        raise ValueError(f"Expected stage dim=4, got {s}")

    x_flat = x_mean.reshape(n * s, d).cpu().numpy()
    print(f"[PCA] Fitting PCA: {x_flat.shape} -> {n_components} (svd_solver={svd_solver})")

    pca = PCA(
        n_components=n_components,
        random_state=random_state,
        svd_solver=svd_solver,
    )
    x_reduced = pca.fit_transform(x_flat)
    explained = float(pca.explained_variance_ratio_.sum())

    out = torch.from_numpy(x_reduced).reshape(n, s, n_components)
    return out, explained


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    dataset = str(config.get("dataset", "nips34")).strip()
    seed = int(config.get("seed", 42))
    set_global_seed(seed)

    io_cfg = config.get("io", {}) or {}
    shape_cfg = config.get("shape", {}) or {}
    pca_cfg = config.get("pca", {}) or {}
    run_cfg = config.get("run", {}) or {}

    base = resolve_path(io_cfg.get("base", "data/embedding_trajectories/{dataset}"), dataset)
    mean_out = resolve_path(
        io_cfg.get("mean_out", "embedding/{dataset}/polya_tensor_layer_mean.pt"), dataset
    )
    pca_out = resolve_path(
        io_cfg.get("pca_out", "embedding/{dataset}/polya_tensor_layer_mean_pca768.pt"), dataset
    )

    expected_d = int(shape_cfg.get("expected_d", 5120))
    expected_t = int(shape_cfg.get("expected_t", 0))

    n_components = int(pca_cfg.get("n_components", 768))
    random_state = int(pca_cfg.get("random_state", seed))
    svd_solver = str(pca_cfg.get("svd_solver", "auto"))

    start = int(run_cfg.get("start", 0))
    end_cfg = run_cfg.get("end")
    skip_bad = bool(run_cfg.get("skip_bad", False))
    rebuild_mean = bool(run_cfg.get("rebuild_mean", False))

    stage_files = resolve_stage_files(config)

    if not base.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base}")

    end = int(end_cfg) if end_cfg is not None else infer_end_index(base, start)
    if end < start:
        raise ValueError(f"Invalid range: start={start}, end={end}")

    mean_out.parent.mkdir(parents=True, exist_ok=True)
    pca_out.parent.mkdir(parents=True, exist_ok=True)

    print("========================================")
    print("Dataset:", dataset)
    print("Seed:", seed)
    print("Base:", base)
    print("Range:", f"{start}..{end}")
    print("Expected shape per stage:", f"[T,{expected_d}] (T={expected_t if expected_t else 'infer'})")
    print("Mean output:", mean_out)
    print("PCA output:", pca_out)
    print("========================================")

    if mean_out.exists() and not rebuild_mean:
        print(f"[LOAD MEAN] {mean_out}")
        x_mean = torch.load(mean_out, map_location="cpu")
        if not isinstance(x_mean, torch.Tensor):
            raise TypeError(f"mean_out is not a torch.Tensor: {type(x_mean)}")
        if x_mean.ndim != 3:
            raise ValueError(f"Expected mean tensor [N,4,D], got {tuple(x_mean.shape)}")
        if x_mean.shape[1] != 4 or x_mean.shape[2] != expected_d:
            raise ValueError(
                f"mean tensor mismatch: got {tuple(x_mean.shape)}, expected [N,4,{expected_d}]"
            )
        bad: List[Tuple[int, str]] = []
        inferred_t = expected_t
        print(f"[MEAN] loaded shape={tuple(x_mean.shape)} dtype={x_mean.dtype}")
    else:
        print("[BUILD MEAN] loading raw stage trajectories and applying layer-wise mean pooling")
        x_mean, bad, inferred_t = build_mean_tensor(
            base=base,
            start=start,
            end=end,
            expected_d=expected_d,
            expected_t=expected_t,
            stage_files=stage_files,
            skip_bad=skip_bad,
        )
        print(f"[MEAN] shape={tuple(x_mean.shape)} dtype={x_mean.dtype} inferred_T={inferred_t}")
        print(f"[SAVE MEAN] {mean_out}")
        torch.save(x_mean, mean_out)

    print(f"[RUN PCA] n_components={n_components}")
    x_pca, explained = run_pca(
        x_mean=x_mean,
        n_components=n_components,
        random_state=random_state,
        svd_solver=svd_solver,
    )

    print("========================================")
    print("Mean tensor:", tuple(x_mean.shape), x_mean.dtype)
    print("PCA tensor:", tuple(x_pca.shape), x_pca.dtype)
    print("Explained variance ratio (sum):", f"{explained:.4f}")
    print("Bad count:", len(bad))
    if bad:
        print("Bad indices (first 50):", [b[0] for b in bad][:50], ("..." if len(bad) > 50 else ""))
    print("Saving PCA to:", pca_out)
    print("========================================")

    torch.save(x_pca, pca_out)
    print("[DONE] Saved.")


if __name__ == "__main__":
    main()
