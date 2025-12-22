"""
Extractor Module - Trajectory Extraction for BAIM.

Provides trajectory extraction functionality for reasoning analysis.

Usage:
    from src.extractor import TrajectoryExtractor

    # Create extractor
    extractor = TrajectoryExtractor(
        model_name='Qwen/Qwen3-VL-32B-Thinking',
        device='cuda'
    )

    # Extract trajectory
    result = extractor.extract(question)
"""

from src.extractor.trajectory import TrajectoryExtractor

__all__ = ["TrajectoryExtractor"]
