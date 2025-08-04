"""
NEXUS TMC Engine - Container Format

This module provides the TMC v8.0 container format for storing compressed data.
"""

from typing import Dict, List, Any
from dataclasses import dataclass

from ..core.data_types import ChunkInfo

__all__ = ['TMCv8Container']


@dataclass
class TMCv8Container:
    """TMC v8.0 コンテナフォーマット"""
    header: Dict[str, Any]
    data_chunks: List[bytes]
    metadata: Dict[str, Any]
    compression_info: Dict[str, Any]
    magic: bytes
    version: str
    chunk_count: int
    chunk_infos: List[ChunkInfo]
    compressed_chunks: List[bytes]
