"""
Storage utilities for BAIM pipeline

Handles saving/loading of:
- Solution text (JSONL)
- Solution embeddings (.pt or .npz) - individual solution's hidden state vectors
- Question embeddings (.pt) - aggregated from multiple solution embeddings
- Metadata (JSON)

All vectors are stored in float32 for KT model compatibility.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from src.entities.solution import Solution
from src.entities.embedding import (
    SolutionEmbedding,
    SolutionEmbeddingMetadata,
    QuestionEmbedding,
    QuestionEmbeddingMetadata,
)

logger = logging.getLogger(__name__)


class SolutionStorage:
    """
    Manages storage for solution text (JSONL) ONLY.
    Does NOT handle embeddings - use SolutionEmbeddingStorage for that.

    Directory structure:
    - solutions/: JSONL files with solution text and metadata
    """

    def __init__(
        self,
        solutions_dir: str | Path,
    ):
        self.solutions_dir = Path(solutions_dir)

        # Create directories
        self.solutions_dir.mkdir(parents=True, exist_ok=True)

    def save(self, solution_data: Dict[str, Any]):
        """
        Save solution data (dict) to JSONL.
        Simple method for pipeline use.
        """
        qid = solution_data.get("question_id")
        if not qid:
            raise ValueError("solution_data must have 'question_id' field")

        jsonl_path = self.solutions_dir / f"{qid}_solutions.jsonl"
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(solution_data, ensure_ascii=False) + "\n")

        logger.debug(f"Saved solution: {qid} -> {jsonl_path}")

    def load_by_uuid(self, uuid: str) -> Optional[Dict[str, Any]]:
        """Load solution by UUID (searches all JSONL files)"""
        for jsonl_file in self.solutions_dir.glob("*_solutions.jsonl"):
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line.strip())
                    if data.get("uuid") == uuid:
                        return data
        return None

    def save_solution(
        self,
        solution: Solution,
    ) -> None:
        """
        Save solution text to JSONL.

        Args:
            solution: Solution object to save
        """
        qid = solution.question_id

        # Save solution text to JSONL
        jsonl_path = self.solutions_dir / f"{qid}_solutions.jsonl"
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(solution.model_dump_json(exclude_none=True) + "\n")

        logger.info(f"Saved solution: {qid} ({solution.solution_type}) -> {jsonl_path}")

    def load_solutions(self, question_id: str) -> List[Dict[str, Any]]:
        """Load all solutions for a question from JSONL (as dicts)"""
        jsonl_path = self.solutions_dir / f"{question_id}_solutions.jsonl"

        if not jsonl_path.exists():
            return []

        solutions = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    # Try parsing as Solution object first
                    from src.entities.solution import Solution

                    sol = Solution.model_validate_json(line.strip())
                    solutions.append(sol.model_dump())
                except Exception:
                    # Fallback: parse as dict
                    solutions.append(json.loads(line.strip()))

        return solutions


class SolutionEmbeddingStorage:
    """
    Manages storage for solution embeddings (hidden state vectors) ONLY.

    Directory structure:
    - solution_embeddings/
      - <question_id>/
        - <type>_<uuid>.pt  (or .npz)
    """

    def __init__(self, solution_embeddings_dir: Path, embedding_format: str = "pt"):
        self.solution_embeddings_dir = Path(solution_embeddings_dir)
        self.embedding_format = embedding_format
        self.solution_embeddings_dir.mkdir(parents=True, exist_ok=True)

    def save_solution_embedding(
        self,
        qid: str,
        solution_type: str,
        uuid: str,
        vector: torch.Tensor,
        meta: Dict[str, Any],
    ) -> SolutionEmbeddingMetadata:
        """
        Save solution embedding (hidden state vector) to disk.

        Note: Vectors are always saved in float32 for KT model compatibility.

        Returns:
            SolutionEmbeddingMetadata entity with saved embedding information
        """
        # Create question-specific directory
        embedding_dir = self.solution_embeddings_dir / qid
        embedding_dir.mkdir(parents=True, exist_ok=True)

        # Ensure float32 dtype
        vector = vector.to(torch.float32)

        # Filename: <type>_<uuid>.<format>
        filename = f"{solution_type}_{uuid[:8]}.{self.embedding_format}"
        embedding_path = embedding_dir / filename

        if self.embedding_format == "pt":
            torch.save(
                {
                    "vector": vector.cpu(),
                    "meta": meta,
                    "dim": vector.shape[-1],
                    "dtype": "float32",
                },
                embedding_path,
            )
        elif self.embedding_format == "npz":
            np.savez_compressed(
                embedding_path,
                vector=vector.cpu().numpy().astype(np.float32),
                meta=meta,
            )
        else:
            raise ValueError(f"Unsupported embedding format: {self.embedding_format}")

        logger.debug(f"Saved solution embedding: {embedding_path}")

        # Return metadata entity
        return SolutionEmbeddingMetadata(
            solution_uuid=uuid,
            question_id=qid,
            solution_type=solution_type,
            embedding_path=str(embedding_path),
            dim=vector.shape[-1],
            dtype="float32",
            model=meta.get("model", "unknown"),
            layer=-1,
        )

    def load_solution_embedding(self, embedding_path: str) -> SolutionEmbedding:
        """
        Load solution embedding from disk.

        Args:
            embedding_path: Path to the embedding file

        Returns:
            SolutionEmbedding entity with metadata and vector
        """
        path = Path(embedding_path)

        if not path.exists():
            raise FileNotFoundError(f"Solution embedding not found: {embedding_path}")

        if path.suffix == ".pt":
            data = torch.load(path, map_location="cpu")
            vector = data["vector"].to(torch.float32)
            meta = data.get("meta", {})
        elif path.suffix == ".npz":
            data = np.load(path)
            vector = torch.from_numpy(data["vector"].astype(np.float32))
            meta = data.get("meta", {}).item() if "meta" in data else {}
        else:
            raise ValueError(f"Unsupported embedding format: {path.suffix}")

        # Create metadata from stored info
        metadata = SolutionEmbeddingMetadata(
            solution_uuid=meta.get("solution_uuid", "unknown"),
            question_id=meta.get("question_id", "unknown"),
            solution_type=meta.get("solution_type", "unknown"),
            embedding_path=str(path),
            dim=vector.shape[-1],
            dtype="float32",
            model=meta.get("model", "unknown"),
            layer=-1,
        )

        return SolutionEmbedding(metadata=metadata, vector=vector)


class QuestionEmbeddingStorage:
    """Manages storage for question-level embeddings (aggregated from solution embeddings)"""

    def __init__(
        self,
        question_embeddings_dir: Path,
        meta_filename: str = "question_embeddings_meta.json",
    ):
        self.question_embeddings_dir = Path(question_embeddings_dir)
        self.meta_path = self.question_embeddings_dir / meta_filename

        # Create directory
        self.question_embeddings_dir.mkdir(parents=True, exist_ok=True)

        # Load existing metadata
        self.metadata: Dict[str, QuestionEmbeddingMetadata] = {}
        if self.meta_path.exists():
            self._load_metadata()

    def save_question_embedding(
        self,
        question_id: str,
        embedding: torch.Tensor,
        solution_types: List[str],
        solution_embedding_files: List[str],
        aggregation_method: str = "mean",
    ) -> QuestionEmbeddingMetadata:
        """
        Save question-level embedding.

        Note: Embeddings are saved in float32 for KT model compatibility.

        Returns:
            QuestionEmbeddingMetadata entity with saved embedding information
        """

        # Ensure float32 dtype
        embedding = embedding.to(torch.float32)

        # Save embedding tensor
        emb_path = self.question_embeddings_dir / f"{question_id}.pt"
        torch.save(
            {
                "question_id": question_id,
                "embedding": embedding.cpu(),
                "dim": embedding.shape[-1],
                "aggregation": aggregation_method,
                "dtype": "float32",
            },
            emb_path,
        )

        # Create metadata entity
        emb_meta = QuestionEmbeddingMetadata(
            question_id=question_id,
            num_solutions=len(solution_embedding_files),
            solution_types=solution_types,
            solution_embedding_files=solution_embedding_files,
            embedding_path=str(emb_path),
            aggregation_method=aggregation_method,
            hidden_dim=embedding.shape[-1],
            dtype="float32",
        )

        self.metadata[question_id] = emb_meta
        self._save_metadata()

        logger.info(
            f"Saved question embedding: {question_id} -> {emb_path} (dtype: float32)"
        )

        return emb_meta

    def load_question_embedding(self, question_id: str) -> QuestionEmbedding:
        """
        Load question embedding with metadata and vector.

        Args:
            question_id: Question ID to load

        Returns:
            QuestionEmbedding entity with metadata and vector
        """
        emb_path = self.question_embeddings_dir / f"{question_id}.pt"

        if not emb_path.exists():
            raise FileNotFoundError(f"Question embedding not found: {emb_path}")

        # Load vector
        data = torch.load(emb_path, map_location="cpu")
        embedding = data["embedding"].to(torch.float32)

        # Get or create metadata
        if question_id in self.metadata:
            metadata = self.metadata[question_id]
        else:
            # Fallback: create minimal metadata
            metadata = QuestionEmbeddingMetadata(
                question_id=question_id,
                num_solutions=0,
                solution_types=[],
                solution_embedding_files=[],
                embedding_path=str(emb_path),
                aggregation_method=data.get("aggregation", "mean"),
                hidden_dim=embedding.shape[-1],
                dtype="float32",
            )

        return QuestionEmbedding(metadata=metadata, vector=embedding)

    def _save_metadata(self):
        """Save metadata to JSON"""
        meta_dict = {
            qid: emb.model_dump(exclude_none=True) for qid, emb in self.metadata.items()
        }

        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2, ensure_ascii=False)

    def _load_metadata(self):
        """Load metadata from JSON"""
        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)

        self.metadata = {
            qid: QuestionEmbeddingMetadata(**emb_data)
            for qid, emb_data in meta_dict.items()
        }


# Backward compatibility aliases
LatentStorage = SolutionEmbeddingStorage
EmbeddingStorage = QuestionEmbeddingStorage
