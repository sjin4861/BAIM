# src/utils/prompt_builder.py
"""
Prompt template loader and builder for BAIM project.
Simplified version without knowledge_state dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import re
import yaml
import logging

logger = logging.getLogger("prompt_builder")

_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z0-9_]+)\}")


def render_placeholders(template: str, ctx: Dict[str, str]) -> str:
    """{key} 형태의 플레이스홀더를 ctx로 치환. 미존재 키는 그대로 둠."""

    def repl(m: re.Match) -> str:
        k = m.group(1)
        return str(ctx.get(k, m.group(0)))

    return _PLACEHOLDER_RE.sub(repl, template)


@dataclass
class PromptTemplate:
    """Prompt template with system_prompt and instruction."""

    name: str = ""
    version: str = ""
    notes: str = ""
    system_prompt: str = ""
    instruction_C: str = ""  # Only C (Correct reasoning) is used

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
            instruction_C=(data.get("instruction_C") or "").strip(),
        )

    def get_instruction(self) -> str:
        """Get instruction (always returns C instruction)."""
        return self.instruction_C


class PromptBuilder:
    """Build prompts from templates for BAIM solution generation."""

    @staticmethod
    def load_template(path: str | Path) -> PromptTemplate:
        """
        Load PromptTemplate from YAML file.

        Args:
            path: Path to prompt template YAML file

        Returns:
            PromptTemplate instance
        """
        return PromptTemplate.from_yaml(path)

    def __init__(self, template: PromptTemplate):
        """
        Args:
            template: PromptTemplate instance
        """
        self.tpl = template

    def build_messages(self, *, question: str, **kwargs) -> List[Dict[str, str]]:
        """
        Build text-only messages for OpenAI-compatible API.

        Args:
            question: Question text
            **kwargs: Additional context variables

        Returns:
            List of message dicts with 'role' and 'content'
        """
        ctx = {
            "question": (question or "").strip(),
            "analysis": (kwargs.get("analysis") or "").strip(),  # [옵션] 명시적 추가
        }
        ctx.update(kwargs)

        instruction_template = self.tpl.get_instruction()
        user_msg = render_placeholders(instruction_template, ctx)

        return [
            {"role": "system", "content": self.tpl.system_prompt},
            {"role": "user", "content": user_msg},
        ]

    def build_mm_messages(
        self, *, question: str, image_paths: List[str], **kwargs
    ) -> List[Dict[str, any]]:
        """
        Build multimodal messages with local images.

        Args:
            question: Question text
            image_paths: List of local image file paths
            **kwargs: Additional context variables

        Returns:
            List of message dicts with multimodal content
        """
        ctx = {
            "question": (question or "").strip(),
            "analysis": (kwargs.get("analysis") or "").strip(),  # [옵션] 명시적 추가
        }
        ctx.update(kwargs)

        instruction_template = self.tpl.get_instruction()
        user_text = render_placeholders(instruction_template, ctx)

        # Build multimodal content: text + images
        parts = [{"type": "text", "text": user_text}]
        for img_path in image_paths:
            abs_path = str(Path(img_path).absolute())
            # Use OpenAI API standard format for images
            parts.append(
                {"type": "image_url", "image_url": {"url": f"file://{abs_path}"}}
            )

        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.tpl.system_prompt}],
            },
            {"role": "user", "content": parts},
        ]
