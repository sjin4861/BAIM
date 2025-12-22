# src/utils/message_router.py
"""
Message router for BAIM project.
Automatically routes between text-only and multimodal message building.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Dict, Any
from src.utils.prompt_builder import PromptBuilder


def build_messages_auto(
    prompt_builder: PromptBuilder,
    *,
    question_text: str,
    images_root: Optional[Path] = None,
    image_paths: Optional[List[str]] = None,
    analysis: Optional[str] = None,  # [수정] analysis 파라미터 명시
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Auto-route between text-only and multimodal messages.

    Args:
        prompt_builder: PromptBuilder instance
        question_text: Question text content
        images_root: Root directory for resolving relative image paths
        image_paths: List of image paths (relative or absolute)
        analysis: Detailed solution analysis text (for {analysis} placeholder)
        **kwargs: Additional context variables for template

    Returns:
        List of message dicts for OpenAI-compatible API
    """
    # analysis가 있으면 kwargs에 추가
    if analysis:
        kwargs["analysis"] = analysis

    # If image paths provided, use multimodal
    if image_paths and len(image_paths) > 0:
        return prompt_builder.build_mm_messages(
            question=question_text, image_paths=image_paths, **kwargs
        )

    # Otherwise, text-only
    return prompt_builder.build_messages(question=question_text, **kwargs)
