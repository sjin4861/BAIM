"""
Embedding Entities for BAIM

- SolutionEmbeddingMetadata: Metadata about a solution embedding
- SolutionEmbedding: Complete solution embedding (metadata + vector)
- QuestionEmbeddingMetadata: Metadata about a question embedding
- QuestionEmbedding: Complete question embedding (metadata + vector)
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Any
import torch


class SolutionEmbeddingMetadata(BaseModel):
    """
    Metadata about a solution embedding.

    Contains information about the source solution and embedding properties
    without the actual vector data.
    """

    solution_uuid: str = Field(..., description="UUID of the source solution")
    question_id: str = Field(..., description="Question ID")
    solution_type: str = Field(..., description="Solution type (C/M/I/H)")

    # Vector metadata
    embedding_path: str = Field(..., description="Path to .pt or .npz file")
    dim: int = Field(default=4096, description="Embedding dimension")
    dtype: str = Field(
        default="float32", description="Data type (always float32 for KT compatibility)"
    )

    # Model information
    model: str = Field(..., description="Source LLM model")
    layer: int = Field(default=-1, description="Layer index (-1 for last layer)")

    class Config:
        json_schema_extra = {
            "example": {
                "solution_uuid": "b3b7b302-0a72-4ef7-b92e-5a47cdb8e502",
                "question_id": "001",
                "solution_type": "C",
                "embedding_path": "data/solution_embeddings/q001/C_b3b7b302.pt",
                "dim": 4096,
                "dtype": "float32",
                "model": "Qwen/Qwen2.5-VL-8B-Thinking",
                "layer": -1,
            }
        }


class SolutionEmbedding(BaseModel):
    """
    Complete solution embedding with metadata and vector.

    Represents the full embedding including both the metadata (where it came from,
    dimensions, etc.) and the actual vector tensor.
    """

    metadata: SolutionEmbeddingMetadata = Field(..., description="Embedding metadata")
    vector: Any = Field(..., description="Embedding vector (torch.Tensor)")

    class Config:
        arbitrary_types_allowed = True  # Allow torch.Tensor
        json_schema_extra = {
            "example": {
                "metadata": {
                    "solution_uuid": "b3b7b302-0a72-4ef7-b92e-5a47cdb8e502",
                    "question_id": "001",
                    "solution_type": "C",
                    "embedding_path": "data/solution_embeddings/q001/C_b3b7b302.pt",
                    "dim": 4096,
                    "dtype": "float32",
                    "model": "Qwen/Qwen2.5-VL-8B-Thinking",
                    "layer": -1,
                },
                "vector": "torch.Tensor([4096])",
            }
        }

    @property
    def solution_uuid(self) -> str:
        """Get solution UUID"""
        return self.metadata.solution_uuid

    @property
    def question_id(self) -> str:
        """Get question ID"""
        return self.metadata.question_id

    @property
    def solution_type(self) -> str:
        """Get solution type"""
        return self.metadata.solution_type

    @property
    def dim(self) -> int:
        """Get embedding dimension"""
        return self.metadata.dim


class QuestionEmbeddingMetadata(BaseModel):
    """
    Metadata about a question embedding.

    Contains information about how the embedding was created (aggregation method,
    source solutions, etc.) without the actual vector data.
    """

    question_id: str = Field(..., description="Question ID")
    num_solutions: int = Field(
        ..., description="Number of solutions used for aggregation"
    )
    solution_types: List[str] = Field(
        ..., description="Solution types used (e.g., ['C', 'M', 'I', 'H'])"
    )
    solution_embedding_files: List[str] = Field(
        ..., description="Paths to source solution embedding files"
    )

    # Embedding metadata
    embedding_path: str = Field(
        ..., description="Path to aggregated embedding .pt file"
    )
    aggregation_method: str = Field(
        default="mean", description="Aggregation method (mean/max/weighted)"
    )
    hidden_dim: int = Field(default=4096, description="Embedding dimension")
    dtype: str = Field(default="float32", description="Data type")

    class Config:
        json_schema_extra = {
            "example": {
                "question_id": "001",
                "num_solutions": 4,
                "solution_types": ["C", "M", "I", "H"],
                "solution_embedding_files": [
                    "data/solution_embeddings/q001/C_a1b2c3d4.pt",
                    "data/solution_embeddings/q001/M_e5f6g7h8.pt",
                    "data/solution_embeddings/q001/I_i9j0k1l2.pt",
                    "data/solution_embeddings/q001/H_m3n4o5p6.pt",
                ],
                "embedding_path": "data/question_embeddings/q001.pt",
                "aggregation_method": "mean",
                "hidden_dim": 4096,
                "dtype": "float32",
            }
        }


class QuestionEmbedding(BaseModel):
    """
    Complete question embedding with metadata and vector.

    Represents the aggregated embedding for a question, including both the
    metadata (how it was created) and the actual vector tensor.
    """

    metadata: QuestionEmbeddingMetadata = Field(..., description="Embedding metadata")
    vector: Any = Field(..., description="Aggregated embedding vector (torch.Tensor)")

    class Config:
        arbitrary_types_allowed = True  # Allow torch.Tensor
        json_schema_extra = {
            "example": {
                "metadata": {
                    "question_id": "001",
                    "num_solutions": 4,
                    "solution_types": ["C", "M", "I", "H"],
                    "solution_embedding_files": ["..."],
                    "embedding_path": "data/question_embeddings/q001.pt",
                    "aggregation_method": "mean",
                    "hidden_dim": 4096,
                    "dtype": "float32",
                },
                "vector": "torch.Tensor([4096])",
            }
        }

    @property
    def question_id(self) -> str:
        """Get question ID"""
        return self.metadata.question_id

    @property
    def dim(self) -> int:
        """Get embedding dimension"""
        return self.metadata.hidden_dim
