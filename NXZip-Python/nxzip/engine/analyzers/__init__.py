# NEXUS TMC Analyzers Module
"""
NEXUS TMC Engine - Analysis Components

This module contains analysis and metadata processing components:
- Meta analysis
- Entropy calculation
- Statistical preprocessing
"""

__version__ = "9.1.0"

# Analyzer exports
from .entropy_calculator import (
    calculate_entropy,
    calculate_entropy_vectorized,
    estimate_temporal_similarity,
    estimate_repetition_density,
    estimate_context_predictability,
    calculate_theoretical_compression_gain,
    generate_sample_key
)
from .meta_analyzer import MetaAnalyzer

__all__ = [
    'MetaAnalyzer',
    'calculate_entropy',
    'calculate_entropy_vectorized', 
    'estimate_temporal_similarity',
    'estimate_repetition_density',
    'estimate_context_predictability',
    'calculate_theoretical_compression_gain',
    'generate_sample_key'
]
