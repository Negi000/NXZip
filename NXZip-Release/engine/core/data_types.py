"""
NEXUS TMC Engine - Data Types Module

This module contains all data type definitions and structures used
throughout the NEXUS TMC compression engine.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

__all__ = ['DataType', 'ChunkInfo', 'PipelineStage', 'AsyncTask', 'TMCv8Container']


class DataType(Enum):
    """データタイプ分類"""
    TEXT_REPETITIVE = "text_repetitive"
    TEXT_NATURAL = "text_natural"
    SEQUENTIAL_INT = "sequential_int"
    FLOAT_ARRAY = "float_array"
    GENERIC_BINARY = "generic_binary"


@dataclass
class ChunkInfo:
    """チャンク情報格納クラス"""
    chunk_id: int
    original_size: int
    compressed_size: int
    data_type: str
    compression_ratio: float
    processing_time: float


@dataclass
class PipelineStage:
    """パイプライン処理ステージ情報"""
    stage_id: int
    stage_name: str
    input_data: bytes
    output_data: bytes
    processing_time: float
    thread_id: int


@dataclass
class AsyncTask:
    """非同期タスク情報"""
    task_id: int
    task_type: str
    data: bytes
    priority: int = 0
    completed: bool = False


@dataclass 
class TMCv8Container:
    """TMC v8.0 コンテナフォーマット"""
    version: str
    timestamp: str
    original_size: int
    compressed_size: int
    compression_ratio: float
    data_type: str
    chunk_count: int
    metadata: dict
    integrity_hash: str
    processing_time: float
    memory_usage: int
    cpu_usage: float
    compression_level: int
    algorithm_version: str
    parallel_workers: int
    optimization_flags: list
