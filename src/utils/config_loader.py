# -*- coding: utf-8 -*-
"""
Configuration loader for BAIM project.
Loads settings from YAML and supports environment variable overrides.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ProjectConfig:
    """Project metadata"""

    name: str
    version: str
    description: str = ""


@dataclass
class DataConfig:
    """Data paths configuration"""

    dataset_root: str
    questions_json: str  # Contains both questions and answers
    images_root: str

    # BAIM output directories
    solutions_dir: str = "data/solutions"
    solution_embeddings_dir: str = "data/solution_embeddings"
    question_embeddings_dir: str = "data/question_embeddings"
    verifications_dir: str = "data/verifications"  # Verification results

    @property
    def questions_path(self) -> str:
        """Alias for questions_json for backward compatibility."""
        return self.questions_json


@dataclass
class OutputConfig:
    """Output settings"""

    base_dir: str
    prefix: str
    save_raw: bool
    save_each: bool


@dataclass
class PromptsConfig:
    """Prompt templates configuration"""

    template: str

    def get_template_path(self, solution_type: Optional[str] = None) -> str:
        """Get template path (same for all solution types in BAIM)."""
        return self.template


@dataclass
class TransformersConfig:
    """Transformers-specific settings"""

    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    extract_hidden: bool = True
    hidden_layer: int = -1


@dataclass
class LLMConfig:
    """LLM configuration"""

    model: str
    base_url: str
    api_key: str
    temperature: float
    top_p: float
    repetition_penalty: float
    max_tokens: int
    timeout_s: int
    max_retries: int
    json_mode: bool
    backend: str
    transformers: TransformersConfig


@dataclass
class SolutionGenerationConfig:
    """Solution generation settings"""

    num_samples_per_type: int
    types: List[str]
    rescue_retry: int
    batch_size: int


@dataclass
class EmbeddingConfig:
    """Embedding settings"""

    hidden_dim: int
    aggregation_method: str  # Renamed from 'aggregation' for clarity
    embedding_format: str = "pt"  # Format for both solution and question embeddings
    save_meta: bool = True
    meta_filename: str = "baim_question_meta.json"


@dataclass
class VerificationConfig:
    """Verification settings"""

    fallback_method: str = "exact_match"


@dataclass
class LoggingConfig:
    """Logging configuration"""

    runtime_log: str
    infer_log: str
    level: str


@dataclass
class KTConfig:
    """Knowledge Tracing configuration"""

    pykt_root: str
    models: Dict[str, str]


@dataclass
class BAIMConfig:
    """Main BAIM configuration"""

    project: ProjectConfig
    data: DataConfig
    prompts: PromptsConfig
    llm: LLMConfig
    solution_generation: SolutionGenerationConfig
    embedding: EmbeddingConfig
    verification: VerificationConfig
    logging: LoggingConfig
    kt: KTConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> BAIMConfig:
        """Load configuration from YAML file"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        # Expand environment variables in string values
        raw = cls._expand_env_vars(raw)

        # Build dataclass instances
        project = ProjectConfig(**raw["project"])
        data = DataConfig(**raw["data"])
        prompts = PromptsConfig(**raw["prompts"])

        transformers_cfg = TransformersConfig(**raw["llm"].get("transformers", {}))
        llm = LLMConfig(**{**raw["llm"], "transformers": transformers_cfg})

        solution_gen = SolutionGenerationConfig(**raw["solution_generation"])
        embedding = EmbeddingConfig(**raw["embedding"])
        verification = VerificationConfig(**raw.get("verification", {}))
        logging_cfg = LoggingConfig(**raw["logging"])
        kt = KTConfig(**raw["kt"])

        return cls(
            project=project,
            data=data,
            prompts=prompts,
            llm=llm,
            solution_generation=solution_gen,
            embedding=embedding,
            verification=verification,
            logging=logging_cfg,
            kt=kt,
        )

    @staticmethod
    def _expand_env_vars(obj: Any) -> Any:
        """Recursively expand environment variables in config"""
        if isinstance(obj, dict):
            return {k: BAIMConfig._expand_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [BAIMConfig._expand_env_vars(v) for v in obj]
        elif isinstance(obj, str):
            # Handle ${VAR:-default} syntax
            if "${" in obj:
                import re

                pattern = r"\$\{([^:}]+)(?::[-]([^}]*))?\}"

                def replace_var(match):
                    var_name = match.group(1)
                    default_val = match.group(2) or ""
                    return os.environ.get(var_name, default_val)

                return re.sub(pattern, replace_var, obj)
            # Handle ~ for home directory
            if obj.startswith("~"):
                return os.path.expanduser(obj)
        return obj

    def get_prompt_template(self, solution_type: str = "default") -> str:
        """Get prompt template path for a solution type"""
        return self.prompts.get_template_path(solution_type)


# Global config instance (loaded on demand)
_config: Optional[BAIMConfig] = None


def load_config(path: Optional[str | Path] = None) -> BAIMConfig:
    """
    Load BAIM configuration.

    Args:
        path: Path to config YAML. If None, uses default location.

    Returns:
        BAIMConfig instance
    """
    global _config

    if path is None:
        # Default config path
        default_path = (
            Path(__file__).parent.parent.parent / "configs" / "baim_config.yaml"
        )
        path = os.environ.get("BAIM_CONFIG", str(default_path))

    if _config is None or path:
        _config = BAIMConfig.from_yaml(path)

    return _config


def get_config() -> BAIMConfig:
    """Get the global config instance (load if not already loaded)"""
    global _config
    if _config is None:
        _config = load_config()
    return _config
