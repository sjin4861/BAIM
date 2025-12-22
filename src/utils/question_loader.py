# src/utils/question_loader.py
"""
Question loader for BAIM project.
Loads questions from JSON metadata files (e.g., XES3G5M questions.json).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Union

from src.entities.question import Question

log = logging.getLogger(__name__)


class QuestionLoader:
    """
    Load Question entities from JSON files.
    Supports both dict-based ({"qid": {...}}) and list-based ([{"qid": ...}, ...]) formats.
    """

    def __init__(self, path: Union[str, Path], *, encoding: str = "utf-8") -> None:
        self.path = Path(path)
        self.questions_path = self.path  # Alias for compatibility
        self.encoding = encoding
        self._data: Optional[Any] = None  # raw json (dict or list)

    @property
    def metadata(self) -> Any:
        """Get raw loaded metadata (dict or list)."""
        self._ensure_loaded()
        return self._data

    def get(self, qid: str) -> Question:
        """
        Get a single Question by qid.
        """
        self._ensure_loaded()
        raw = self._find_raw_item(qid)
        if raw is None:
            qid_fallback = qid.split("-")[-1]
            if qid_fallback != qid:
                raw = self._find_raw_item(qid_fallback)
        if raw is None:
            raise KeyError(f"Question not found for qid={qid!r}")

        # --- content + image auto-detection ---
        content_raw = str(raw.get("content") or raw.get("question") or "")
        image_pattern = re.compile(r"(question_[\w\-]+(?:\.[a-zA-Z0-9]+)?)")
        images = []
        for match in image_pattern.findall(content_raw):
            img_name = match
            if not img_name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                img_name += ".png"  # 기본 확장자
            images.append(img_name)
            content_raw = content_raw.replace(match, "").strip()

        # --- rest as before ---
        opts_raw = raw.get("options") or {}
        if isinstance(opts_raw, dict) and opts_raw:
            order = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            keys = [k for k in order if k in opts_raw]
            options = [str(opts_raw[k]) for k in keys]
        elif isinstance(opts_raw, list):
            options = [str(o) for o in opts_raw]
        else:
            options = None

        ans_raw = raw.get("answer")
        if isinstance(ans_raw, list):
            answer = (
                None
                if not ans_raw
                else (
                    str(ans_raw[0])
                    if len(ans_raw) == 1
                    else ",".join(map(str, ans_raw))
                )
            )
        else:
            answer = str(ans_raw) if ans_raw is not None else None

        concepts_raw = (
            raw.get("concepts")
            or raw.get("knowledge")
            or raw.get("kc_routes")
            or raw.get("skills")
            or []
        )
        if isinstance(concepts_raw, list):
            concepts = [str(c) for c in concepts_raw]
        elif isinstance(concepts_raw, str):
            concepts = [concepts_raw]
        else:
            concepts = []

        return Question(
            qid=str(qid),
            type=str(raw.get("type", "Unknown")),
            concepts=concepts,
            content=content_raw,
            options=options,
            analysis=raw.get("analysis") or raw.get("explanation"),
            answer=answer,
            images=images or None,  # ✅ 자동추출
        )

    def get_all_qids(self) -> List[str]:
        """Get list of all question IDs in the file."""
        self._ensure_loaded()

        if isinstance(self._data, dict):
            return list(self._data.keys())
        elif isinstance(self._data, list):
            qids = []
            for item in self._data:
                if isinstance(item, dict):
                    qid = item.get("qid") or item.get("id") or item.get("question_id")
                    if qid is not None:
                        qids.append(str(qid))
            return qids
        return []

    # ------------- Internals -------------

    def _ensure_loaded(self) -> None:
        """Load JSON file if not already loaded."""
        if self._data is not None:
            return
        if not self.path.exists():
            raise FileNotFoundError(f"Question file not found: {self.path}")
        with self.path.open("r", encoding=self.encoding) as f:
            self._data = json.load(f)

    def _find_raw_item(self, qid: str) -> Optional[Mapping[str, Any]]:
        """Find raw item dict by qid."""
        data = self._data

        if isinstance(data, dict):
            # Direct lookup
            item = data.get(qid)
            if isinstance(item, dict):
                return item

            # Search through values
            for v in data.values():
                if isinstance(v, dict):
                    item_qid = v.get("qid") or v.get("id") or v.get("question_id")
                    if str(item_qid) == str(qid):
                        return v
            return None

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    item_qid = (
                        item.get("qid") or item.get("id") or item.get("question_id")
                    )
                    if str(item_qid) == str(qid):
                        return item
            return None

        raise ValueError(f"Unsupported JSON structure: {type(data)}")
