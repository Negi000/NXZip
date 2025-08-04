# NEXUS TMC Core Module
"""
NEXUS TMC Engine - Core System Components

This module contains the fundamental system components for NEXUS TMC:
- Memory management
- Data type definitions  
- Pipeline base structures
- Core compression functionality
"""

__version__ = "9.1.0"
__author__ = "NEXUS TMC Development Team"

# Core exports
from .memory_manager import MemoryManager, MEMORY_MANAGER
from .data_types import DataType, ChunkInfo, PipelineStage, AsyncTask, TMCv8Container
from .core_compressor import CoreCompressor

__all__ = [
    'MemoryManager', 
    'MEMORY_MANAGER',
    'DataType', 
    'ChunkInfo',
    'PipelineStage',
    'AsyncTask',
    'TMCv8Container',
    'CoreCompressor'
]
