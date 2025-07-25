#!/usr/bin/env python3
"""
NXZip - 統合圧縮システム
AV1/SRLA/AVIF制約除去技術による次世代圧縮

Features:
- 制約除去技術 (AV1/SRLA/AVIF constraint removal)
- NEXUS統合エンジン (Unified compression engine)
- SPE暗号化技術 (Structure-Preserving Encryption)
- 適応的圧縮戦略 (Adaptive compression strategies)
- 完全可逆圧縮 (100% lossless compression)
"""

__version__ = "3.0.0"
__author__ = "NXZip Development Team"
__email__ = "team@nxzip.org"
__license__ = "MIT"

# Core classes
from .engine.nexus_unified import NEXUSUnified
from .engine.spe_core_jit import SPECoreJIT
from .crypto.encrypt import SuperCrypto
from .utils.constants import (
    CompressionAlgorithm,
    EncryptionAlgorithm, 
    KDFAlgorithm,
    FileFormat,
    SecurityConstants
)

# Exception classes
from .crypto.encrypt import NXZipError

__all__ = [
    # Core classes
    'NXZipFinal',
    'SPECoreJIT', 
    'SuperCompressor',
    'SuperCrypto',
    
    # Constants
    'CompressionAlgorithm',
    'EncryptionAlgorithm',
    'KDFAlgorithm',
    'FileFormat',
    'SecurityConstants',
    
    # Exceptions
    'NXZipError',
    
    # Metadata
    '__version__',
    '__author__',
    '__email__',
    '__license__',
]


def get_version() -> str:
    """バージョン情報を取得"""
    return __version__


def verify_installation() -> bool:
    """インストールの整合性を検証"""
    try:
        # SPEコアの整合性検証
        from .engine.spe_core_jit import SPECoreJIT
        spe = SPECoreJIT()
        
        # 基本的な圧縮・展開テスト
        test_data = b"NXZip Installation Test"
        nexus = NEXUSUnified()
        
        archive = nexus.compress(test_data)
        restored = nexus.decompress(archive)
        
        return restored == test_data
    
    except Exception:
        return False


# パッケージロード時の整合性検証
if not verify_installation():
    import warnings
    warnings.warn(
        "NXZip package integrity check failed. "
        "Please reinstall the package.",
        RuntimeWarning
    )
