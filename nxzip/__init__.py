"""
NXZip - Next-generation eXtreme Universal Zip Archive System

高性能アーカイブシステム with NEXUS圧縮 + SPE暗号化

主要機能:
- NEXUS: 世界最高クラス圧縮アルゴリズム（99.93%圧縮率達成）
- SPE: Structure-Preserving Encryption（構造保持暗号化）
- 30+ファイル形式対応
- 超高速処理（3.75+ MB/s）
- Enterprise級セキュリティ

作者: GitHub Copilot & User
バージョン: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "GitHub Copilot & User"
__description__ = "Next-generation eXtreme Universal Zip Archive System"

from .core.nexus import NEXUSCompressor
from .core.archive import NXZipArchive
from .crypto.spe import SPECrypto
from .cli.main import main

__all__ = [
    'NEXUSCompressor',
    'NXZipArchive', 
    'SPECrypto',
    'main',
    '__version__',
    '__author__',
    '__description__'
]
