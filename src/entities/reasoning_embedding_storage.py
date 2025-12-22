#!/usr/bin/env python3
"""
Reasoning Embedding Storage

Manages storage and retrieval of trajectory-based reasoning embeddings.
Separate from solution_trajectories to avoid confusion.

Storage structure:
    data/embedding_trajectories/
    ├── <question_id>/
    │   ├── token_wise.pt          # [T, D] fine-grained
    │   ├── layer_mean.pt          # [L, D] coarse global
    │   ├── section_mean.pt        # [S, D] semantic local
    │   └── metadata.json          # extraction config & stats
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

import torch

logger = logging.getLogger(__name__)


class ReasoningEmbeddingStorage:
    """
    Manage storage and retrieval of trajectory-based reasoning embeddings.
    
    This class handles the intermediate representation extraction results,
    which are different from:
    - solution_trajectories: Raw LLM hidden states
    - question_embeddings: Final aggregated question-level embeddings
    
    Purpose: Store multiple granularities of reasoning representations
    for experimental comparison (token-wise, layer-wise, section-wise).
    """
    
    def __init__(
        self, 
        base_dir: str | Path = "data/embedding_trajectories",
        run_subfolder: Optional[str] = None,
    ):
        """
        Args:
            base_dir: Root directory for embedding trajectory storage
            run_subfolder: Optional subfolder for multi-run experiments (e.g., "run_1", "run_2")
                          If provided, structure becomes: base_dir/{question_id}/{run_subfolder}/
        """
        self.base_dir = Path(base_dir)
        self.run_subfolder = run_subfolder
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        if run_subfolder:
            logger.info(f"ReasoningEmbeddingStorage initialized at: {self.base_dir} (run_subfolder={run_subfolder})")
        else:
            logger.info(f"ReasoningEmbeddingStorage initialized at: {self.base_dir}")
    
    def save(
        self,
        question_id: str,
        representations: Dict[str, torch.Tensor],
        metadata: Dict[str, Any],
        compress: bool = True,
    ) -> Path:
        """
        Save all representation types for a question.
        
        Args:
            question_id: Question identifier
            representations: Dict mapping representation type to tensor
                            e.g., {'token_wise': [T, D], 'layer_mean': [L, D], ...}
            metadata: Extraction configuration and stats
            compress: If True, save as float16 (50% space reduction)
        
        Returns:
            Path to the saved directory
        """
        # Create directory structure
        question_dir = self.base_dir / question_id
        
        # Add run subfolder if specified
        if self.run_subfolder:
            question_dir = question_dir / self.run_subfolder
        
        question_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        total_size_mb = 0.0
        
        # Save each representation type
        for rep_type, tensor in representations.items():
            # Convert to float16 for compression if requested
            if compress:
                tensor_to_save = tensor.to(torch.float16)
                logger.debug(f"Compressed {rep_type} to float16: {tensor.shape}")
            else:
                tensor_to_save = tensor
            
            # Save tensor
            tensor_path = question_dir / f"{rep_type}.pt"
            torch.save(tensor_to_save, tensor_path)
            
            file_size_mb = tensor_path.stat().st_size / (1024 * 1024)
            total_size_mb += file_size_mb
            
            saved_files[rep_type] = {
                'shape': list(tensor.shape),
                'dtype': str(tensor_to_save.dtype),
                'size_mb': file_size_mb
            }
            
            logger.debug(f"Saved {rep_type}: {tensor_path} ({file_size_mb:.2f} MB)")
        
        # Enhance metadata
        metadata_enhanced = {
            **metadata,
            'question_id': question_id,
            'compressed': compress,
            'saved_at': datetime.now().isoformat(),
            'saved_files': saved_files,
            'total_size_mb': total_size_mb,
        }
        
        # Add run info if applicable
        if self.run_subfolder:
            metadata_enhanced['run_subfolder'] = self.run_subfolder
        
        # Save metadata
        metadata_path = question_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata_enhanced, f, indent=2)
        
        logger.info(
            f"Saved reasoning embeddings for Q{question_id}"
            f"{f' ({self.run_subfolder})' if self.run_subfolder else ''}: "
            f"{len(representations)} types, {total_size_mb:.2f} MB"
        )
        
        return question_dir
    
    def load(
        self,
        question_id: str,
        representation_types: Optional[List[str]] = None,
        to_float32: bool = True,
        run_subfolder: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load reasoning embeddings for a question.
        
        Args:
            question_id: Question identifier
            representation_types: List of types to load. If None, load all available.
                                 Options: ['token_wise', 'layer_mean', 'section_mean']
            to_float32: If True, convert loaded tensors to float32
            run_subfolder: Override the instance's run_subfolder for this load
        
        Returns:
            Dictionary with:
                - representations: Dict[str, torch.Tensor]
                - metadata: dict
        """
        question_dir = self.base_dir / question_id
        
        # Use provided run_subfolder or fall back to instance default
        subfolder = run_subfolder if run_subfolder is not None else self.run_subfolder
        if subfolder:
            question_dir = question_dir / subfolder
        
        if not question_dir.exists():
            raise FileNotFoundError(f"No embeddings found for question: {question_id}")
        
        # Load metadata
        metadata_path = question_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Determine which representations to load
        if representation_types is None:
            # Load all available
            available_files = metadata.get('saved_files', {}).keys()
            representation_types = list(available_files)
        
        # Load representations
        representations = {}
        for rep_type in representation_types:
            tensor_path = question_dir / f"{rep_type}.pt"
            
            if not tensor_path.exists():
                logger.warning(f"Representation not found: {tensor_path}")
                continue
            
            tensor = torch.load(tensor_path)
            
            if to_float32:
                tensor = tensor.to(torch.float32)
            
            representations[rep_type] = tensor
            logger.debug(f"Loaded {rep_type}: {tensor.shape}")
        
        logger.info(
            f"Loaded {len(representations)} representation(s) for Q{question_id}"
        )
        
        return {
            'representations': representations,
            'metadata': metadata,
        }
    
    def exists(
        self,
        question_id: str,
        representation_type: Optional[str] = None,
        run_subfolder: Optional[str] = None,
    ) -> bool:
        """
        Check if embeddings exist for a question.
        
        Args:
            question_id: Question identifier
            representation_type: Specific type to check. If None, check if directory exists.
            run_subfolder: Override the instance's run_subfolder for this check
        
        Returns:
            True if exists, False otherwise
        """
        question_dir = self.base_dir / question_id
        
        # Use provided run_subfolder or fall back to instance default
        subfolder = run_subfolder if run_subfolder is not None else self.run_subfolder
        if subfolder:
            question_dir = question_dir / subfolder
        
        if not question_dir.exists():
            return False
        
        if representation_type is None:
            # Check if any representation exists
            metadata_path = question_dir / "metadata.json"
            return metadata_path.exists()
        else:
            # Check specific type
            tensor_path = question_dir / f"{representation_type}.pt"
            return tensor_path.exists()
    
    def list_questions(self) -> List[str]:
        """
        List all questions with stored embeddings.
        
        Returns:
            List of question IDs
        """
        question_ids = []
        
        for item in self.base_dir.iterdir():
            if item.is_dir() and (item / "metadata.json").exists():
                question_ids.append(item.name)
        
        return sorted(question_ids)
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored embeddings.
        
        Returns:
            Dictionary with storage statistics
        """
        total_size = 0
        num_questions = 0
        type_counts = {}
        
        for question_dir in self.base_dir.iterdir():
            if not question_dir.is_dir():
                continue
            
            metadata_path = question_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            
            num_questions += 1
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Count types
            for rep_type in metadata.get('saved_files', {}).keys():
                type_counts[rep_type] = type_counts.get(rep_type, 0) + 1
                
                # Sum file sizes
                tensor_path = question_dir / f"{rep_type}.pt"
                if tensor_path.exists():
                    total_size += tensor_path.stat().st_size
        
        return {
            'base_dir': str(self.base_dir),
            'total_size_gb': total_size / (1024 ** 3),
            'num_questions': num_questions,
            'representation_types': type_counts,
            'avg_size_mb': (total_size / num_questions / (1024 ** 2)) if num_questions > 0 else 0,
        }
    
    def delete(self, question_id: str) -> bool:
        """
        Delete all embeddings for a question.
        
        Args:
            question_id: Question identifier
        
        Returns:
            True if deleted, False if not found
        """
        question_dir = self.base_dir / question_id
        
        if not question_dir.exists():
            logger.warning(f"Question directory not found: {question_dir}")
            return False
        
        # Delete all files in directory
        for file in question_dir.iterdir():
            file.unlink()
        
        # Delete directory
        question_dir.rmdir()
        
        logger.info(f"Deleted embeddings for Q{question_id}")
        return True


if __name__ == "__main__":
    """Quick test of ReasoningEmbeddingStorage"""
    
    logging.basicConfig(level=logging.INFO)
    
    # Create test storage
    storage = ReasoningEmbeddingStorage(base_dir="data/embedding_trajectories_test")
    
    # Create synthetic data
    representations = {
        'token_wise': torch.randn(100, 4096),
        'layer_mean': torch.randn(32, 4096),
        'section_mean': torch.randn(3, 4096),
    }
    
    metadata = {
        'solution_type': 'C',
        'extraction_method': 'test',
    }
    
    # Test save
    print("\n=== Testing save ===")
    storage.save('test_001', representations, metadata)
    
    # Test load
    print("\n=== Testing load ===")
    data = storage.load('test_001')
    for rep_type, tensor in data['representations'].items():
        print(f"{rep_type}: {tensor.shape}")
    
    # Test exists
    print("\n=== Testing exists ===")
    print(f"test_001 exists: {storage.exists('test_001')}")
    print(f"test_999 exists: {storage.exists('test_999')}")
    
    # Test stats
    print("\n=== Testing stats ===")
    stats = storage.get_storage_stats()
    print(json.dumps(stats, indent=2))
    
    # Cleanup
    storage.delete('test_001')
    storage.base_dir.rmdir()
    
    print("\n✓ All tests passed!")
