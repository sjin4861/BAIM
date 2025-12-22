"""
BAIM Solution Entity

Represents a single reasoning solution with structured output and metadata.
Each solution corresponds to one generation attempt for a specific solution type (C/M/I/H).
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import uuid as uuid_module


class SolutionOutput(BaseModel):
    """
    Structured output from LLM reasoning.

    This represents the parsed JSON response from the model, following the
    standard BAIM output schema with problem understanding, step-by-step
    solution, and final answer.
    """

    problem_understanding: str = Field(
        ..., description="Model's interpretation of the problem"
    )
    solution: str = Field(..., description="Step-by-step solution process")
    final_answer: str = Field(..., description="The final answer")

    class Config:
        json_schema_extra = {
            "example": {
                "problem_understanding": "Find x in the equation 3x+2=11",
                "solution": "Step 1: Subtract 2 from both sides: 3x = 9. Step 2: Divide both sides by 3: x = 3.",
                "final_answer": "3",
            }
        }


class SolutionMetadata(BaseModel):
    """
    Metadata about solution generation process.

    Contains information about the model, generation parameters, timing,
    verification results, and other process-level details.
    """

    # Identifiers
    uuid: str = Field(
        default_factory=lambda: str(uuid_module.uuid4()),
        description="Unique solution ID",
    )
    question_id: str = Field(..., description="Question ID")
    solution_type: str = Field(..., description="Solution type: C/M/I/H")

    # Model configuration
    model: str = Field(..., description="LLM model name")
    prompt_path: str = Field(..., description="Path to prompt template")
    temperature: float = Field(..., description="Sampling temperature")
    repetition_penalty: float = Field(
        1.0, description="Repetition penalty applied during generation"
    )

    # Generation metrics
    latency_s: Optional[float] = Field(
        None, description="Generation latency in seconds"
    )
    tokens: Optional[int] = Field(None, description="Number of tokens generated")

    # JSON parsing status
    valid_json: Optional[bool] = Field(
        None, description="Whether output was valid JSON"
    )
    parse_error: Optional[str] = Field(
        None, description="JSON parsing error message if any"
    )

    # Verification metadata (optional, for retry flow)
    attempt_idx: Optional[int] = Field(
        None, description="Retry attempt number (1-indexed)"
    )
    expected_correct: Optional[bool] = Field(
        None, description="Expected correctness for this type"
    )
    actual_correct: Optional[bool] = Field(
        None, description="Actual correctness vs ground truth"
    )
    verification_passed: Optional[bool] = Field(
        None, description="Whether verification passed"
    )
    verification_method: Optional[str] = Field(
        None, description="Verification method used"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "uuid": "b3b7b302-0a72-4ef7-b92e-5a47cdb8e502",
                "question_id": "001",
                "solution_type": "M",
                "model": "Qwen/Qwen3-VL-4B-Thinking",
                "prompt_path": "src/prompts/reasoning_template.yaml",
                "temperature": 0.7,
                "repetition_penalty": 1.15,
                "latency_s": 3.48,
                "attempt_idx": 1,
                "expected_correct": False,
                "actual_correct": False,
                "verification_passed": True,
            }
        }


class Solution(BaseModel):
    """
    Complete solution wrapper containing output and metadata.

    This is the main entity used throughout the BAIM pipeline, combining
    the structured LLM output with generation metadata and raw text.

    Attributes:
        metadata: Generation metadata (model, timing, verification, etc.)
        output: Structured output (parsed JSON)
        raw_text: Full raw LLM response including <think> tags
    """

    metadata: SolutionMetadata = Field(..., description="Solution generation metadata")
    output: SolutionOutput = Field(..., description="Parsed structured output")
    raw_text: str = Field(..., description="Full raw LLM response")

    # Convenience properties for backward compatibility

    @property
    def uuid(self) -> str:
        """Get solution UUID"""
        return self.metadata.uuid

    @property
    def question_id(self) -> str:
        """Get question ID"""
        return self.metadata.question_id

    @property
    def solution_type(self) -> str:
        """Get solution type"""
        return self.metadata.solution_type

    @property
    def model(self) -> str:
        """Get model name"""
        return self.metadata.model

    @property
    def prompt_path(self) -> str:
        """Get prompt path"""
        return self.metadata.prompt_path

    @property
    def temperature(self) -> float:
        """Get temperature"""
        return self.metadata.temperature

    @property
    def final_answer(self) -> str:
        """Get final answer"""
        return self.output.final_answer

    @property
    def reasoning_steps(self) -> list[str]:
        """Extract reasoning steps from solution field"""
        solution_text = self.output.solution
        if not solution_text:
            return []
        # Split by newlines or periods
        steps = [s.strip() for s in solution_text.split("\n") if s.strip()]
        return steps if steps else [solution_text]

    class Config:
        json_schema_extra = {
            "example": {
                "metadata": {
                    "uuid": "b3b7b302-0a72-4ef7-b92e-5a47cdb8e502",
                    "question_id": "001",
                    "solution_type": "M",
                    "model": "Qwen/Qwen3-VL-4B-Thinking",
                    "prompt_path": "src/prompts/reasoning_template.yaml",
                    "temperature": 0.7,
                    "repetition_penalty": 1.15,
                    "latency_s": 3.48,
                    "attempt_idx": 1,
                    "expected_correct": False,
                    "actual_correct": False,
                    "verification_passed": True,
                },
                "output": {
                    "problem_understanding": "Find x in equation 3x+2=11",
                    "solution": "Step 1: Subtract 2 from both sides: 3x = 9. Step 2: Divide by 3: x = 3.",
                    "final_answer": "3",
                },
                "raw_text": "<think>Let me solve 3x+2=11...</think>\n{...}",
            }
        }
