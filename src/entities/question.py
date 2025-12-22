# src/entities/question.py
"""Question entity for BAIM project."""
from typing import List, Optional
from pydantic import BaseModel, Field


class Question(BaseModel):
    """Question entity for knowledge tracing and math tutoring."""

    qid: str = Field(..., description="Question ID (string)")
    type: str = Field(..., description="Question type label to render in bracket")
    concepts: List[str] = Field(
        default_factory=list, description="Human-readable concept labels"
    )
    content: str = Field(..., description="Question text (already shortened if needed)")
    options: Optional[List[str]] = Field(
        default=None, description="Choices if multiple choice"
    )
    analysis: Optional[str] = Field(
        default=None, description="Detailed solution/analysis text"
    )
    answer: Optional[str] = Field(
        default=None, description="Correct answer (string for fill-in, key for MCQ)"
    )
    images: Optional[List[str]] = Field(
        default=None, description="Relative or absolute paths to image files"
    )

    def get_header(self) -> str:
        """e.g., Q#4123 [Fill-in-the-blank]"""
        return f"Q#{self.qid} [{self.type}]"

    def get_header_with_kc(self) -> str:
        """e.g., Q#4123 [Fill-in-the-blank] | Concepts: [A, B]"""
        c_list = ", ".join(self.concepts) if self.concepts else ""
        c_str = f" | Concepts: [{c_list}]" if c_list else ""
        return f"Q#{self.qid} [{self.type}]{c_str}"

    def get_lines(
        self, include_options: bool = True, *, options_multiline: bool = False
    ) -> List[str]:
        """
        Renders lines WITHOUT correctness marker.
        Order:
          Q#... [Type]
          Question: <content>
          (Options...)  # only if present & include_options
        """
        lines = [self.get_header(), f"Question: {self.content}"]
        if include_options and self.options:
            abcd = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            if options_multiline:
                # One option per line, e.g., "A : ..."
                for i, o in enumerate(self.options):
                    if i >= len(abcd):
                        break
                    lines.append(f"{abcd[i]} : {str(o)}")
            else:
                # Inline rendering
                opt = " | ".join(
                    f"({abcd[i]}) {str(o)}" for i, o in enumerate(self.options)
                )
                lines.append(f"Options: {opt}")
        return lines

    def get_lines_with_kc(
        self, include_options: bool = True, *, options_multiline: bool = False
    ) -> List[str]:
        """
        Renders lines WITH knowledge concepts.
        Order:
          Q#... [Type] | Concepts: [...]
          Question: <content>
          (Options...)  # only if present & include_options
        """
        lines = [self.get_header_with_kc(), f"Question: {self.content}"]
        if include_options and self.options:
            abcd = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            if options_multiline:
                for i, o in enumerate(self.options):
                    if i >= len(abcd):
                        break
                    lines.append(f"{abcd[i]} : {str(o)}")
            else:
                opt = " | ".join(
                    f"({abcd[i]}) {str(o)}" for i, o in enumerate(self.options)
                )
                lines.append(f"Options: {opt}")
        return lines

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict representation."""
        return {
            "qid": self.qid,
            "type": self.type,
            "concepts": self.concepts,
            "content": self.content,
            "options": self.options,
            "analysis": self.analysis,
            "answer": self.answer,
        }

    def __str__(self) -> str:
        """Default string representation without options."""
        return "\n".join(self.get_lines(include_options=False))
