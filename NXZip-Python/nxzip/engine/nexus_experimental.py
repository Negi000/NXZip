#!/usr/bin/env python3
"""
🚀 NXZip NEXUS Experimental Engine - 実験版エンジン
実験的な高性能圧縮アルゴリズムのテスト用エンジン

This module provides access to the experimental NEXUS compression engine
for testing and development purposes.
"""

# 実験版エンジンを nexus.py から import
from .nexus import NEXUSExperimentalEngine

# 互換性のための追加エイリアス
NEXUSEngine = NEXUSExperimentalEngine
ExperimentalEngine = NEXUSExperimentalEngine

__all__ = ['NEXUSExperimentalEngine', 'NEXUSEngine', 'ExperimentalEngine']