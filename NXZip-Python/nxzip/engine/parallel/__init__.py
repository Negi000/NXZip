# NEXUS TMC Parallel Module
"""
NEXUS TMC Engine - Parallel Processing Components

This module contains parallel processing and pipeline management:
- Parallel pipeline processor
- Worker management
- Async task handling
"""

__version__ = "9.1.0"

from .pipeline_processor import ParallelPipelineProcessor

__all__ = ['ParallelPipelineProcessor']
