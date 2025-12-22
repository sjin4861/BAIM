"""
Trajectory Storage - Efficient storage for solution trajectories.

Provides utilities to save and load trajectory data with compression
to manage large file sizes (potentially 500GB+ for full dataset).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json

import torch
import numpy as np

logger = logging.getLogger(__name__)


class TrajectoryStorage:
    """
    Manage storage and retrieval of solution trajectories.
    
    Storage format:
        data/solution_trajectories/
        ├── <question_id>/
        │   ├── C_<uuid>/
        │   │   ├── trajectory.pt      # [T, 4096] tensor (float16)
        │   │   └── metadata.json      # {num_tokens, solution_type, ...}
        │   ├── M_<uuid>/
        │   ├── I_<uuid>/
        │   └── H_<uuid>/
    """
    
    def __init__(self, base_dir: str | Path = "data/solution_trajectories"):
        """
        Args:
            base_dir: Root directory for trajectory storage
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"TrajectoryStorage initialized at: {self.base_dir}")
    
    def save(
        self,
        question_id: str,
        solution_id: str,
        solution_type: str,
        trajectory: torch.Tensor,
        metadata: Dict[str, Any],
        compress: bool = True,
    ) -> Path:
        """
        Save a trajectory to disk.
        
        Args:
            question_id: Question identifier
            solution_id: Solution UUID
            solution_type: "C", "M", "I", or "H"
            trajectory: Tensor of shape [T, D] or [L, T, D]
            metadata: Additional metadata to store
            compress: If True, save as float16 (50% space reduction)
        
        Returns:
            Path to the saved directory
        """
        # Create directory structure
        solution_dir = self.base_dir / question_id / f"{solution_type}_{solution_id}"
        solution_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to float16 for compression if requested
        if compress:
            trajectory_to_save = trajectory.to(torch.float16)
            logger.debug(f"Compressed trajectory to float16: {trajectory.shape} -> {trajectory_to_save.dtype}")
        else:
            trajectory_to_save = trajectory
        
        # Save trajectory
        trajectory_path = solution_dir / "trajectory.pt"
        torch.save(trajectory_to_save, trajectory_path)
        
        # Enhance metadata
        metadata_enhanced = {
            **metadata,
            'question_id': question_id,
            'solution_id': solution_id,
            'solution_type': solution_type,
            'compressed': compress,
            'saved_at': datetime.now().isoformat(),
            'file_size_mb': trajectory_path.stat().st_size / (1024 * 1024),
        }
        
        # Save metadata
        metadata_path = solution_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata_enhanced, f, indent=2)
        
        logger.info(f"Saved trajectory: {solution_dir} ({metadata_enhanced['file_size_mb']:.2f} MB)")
        
        return solution_dir
    
    def load(
        self,
        question_id: str,
        solution_id: str,
        solution_type: str,
        to_float32: bool = True,
    ) -> Dict[str, Any]:
        """
        Load a trajectory from disk.
        
        Args:
            question_id: Question identifier
            solution_id: Solution UUID
            solution_type: "C", "M", "I", or "H"
            to_float32: If True, convert loaded tensor to float32
        
        Returns:
            Dictionary with:
                - trajectory: torch.Tensor
                - metadata: dict
        """
        solution_dir = self.base_dir / question_id / f"{solution_type}_{solution_id}"
        
        if not solution_dir.exists():
            raise FileNotFoundError(f"Trajectory not found: {solution_dir}")
        
        # Load trajectory
        trajectory_path = solution_dir / "trajectory.pt"
        trajectory = torch.load(trajectory_path)
        
        if to_float32:
            trajectory = trajectory.to(torch.float32)
        
        # Load metadata
        metadata_path = solution_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.debug(f"Loaded trajectory: {solution_dir} (shape={trajectory.shape})")
        
        return {
            'trajectory': trajectory,
            'metadata': metadata,
        }
    
    def load_question_trajectories(
        self,
        question_id: str,
        to_float32: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Load all trajectories for a single question.
        
        Args:
            question_id: Question identifier
            to_float32: If True, convert loaded tensors to float32
        
        Returns:
            Dictionary mapping solution_type to trajectory data:
            {
                'C': {'trajectory': tensor, 'metadata': dict},
                'M': {'trajectory': tensor, 'metadata': dict},
                ...
            }
        """
        question_dir = self.base_dir / question_id
        
        if not question_dir.exists():
            raise FileNotFoundError(f"No trajectories found for question: {question_id}")
        
        result = {}
        
        # Iterate over solution directories
        for solution_dir in question_dir.iterdir():
            if not solution_dir.is_dir():
                continue
            
            # Parse directory name: <type>_<uuid>
            parts = solution_dir.name.split('_', 1)
            if len(parts) != 2:
                continue
            
            solution_type, solution_id = parts
            
            # Load trajectory
            try:
                data = self.load(question_id, solution_id, solution_type, to_float32)
                result[solution_type] = data
            except Exception as e:
                logger.warning(f"Failed to load {solution_dir}: {e}")
        
        logger.info(f"Loaded {len(result)} trajectories for question {question_id}")
        
        return result
    
    def exists(
        self,
        question_id: str,
        solution_id: str,
        solution_type: str,
    ) -> bool:
        """
        Check if a trajectory exists on disk.
        
        Args:
            question_id: Question identifier
            solution_id: Solution UUID
            solution_type: "C", "M", "I", or "H"
        
        Returns:
            True if trajectory exists, False otherwise
        """
        solution_dir = self.base_dir / question_id / f"{solution_type}_{solution_id}"
        trajectory_path = solution_dir / "trajectory.pt"
        return trajectory_path.exists()
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored trajectories.
        
        Returns:
            Dictionary with storage statistics
        """
        total_size = 0
        num_questions = 0
        num_trajectories = 0
        type_counts = {'C': 0, 'M': 0, 'I': 0, 'H': 0}
        
        for question_dir in self.base_dir.iterdir():
            if not question_dir.is_dir():
                continue
            
            num_questions += 1
            
            for solution_dir in question_dir.iterdir():
                if not solution_dir.is_dir():
                    continue
                
                num_trajectories += 1
                
                # Count by type
                solution_type = solution_dir.name.split('_')[0]
                if solution_type in type_counts:
                    type_counts[solution_type] += 1
                
                # Sum file sizes
                trajectory_path = solution_dir / "trajectory.pt"
                if trajectory_path.exists():
                    total_size += trajectory_path.stat().st_size
        
        return {
            'total_size_gb': total_size / (1024 ** 3),
            'num_questions': num_questions,
            'num_trajectories': num_trajectories,
            'type_counts': type_counts,
            'avg_size_mb': (total_size / num_trajectories / (1024 ** 2)) if num_trajectories > 0 else 0,
        }
