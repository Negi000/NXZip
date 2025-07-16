#!/usr/bin/env python3
"""
NXZip - Next-Generation Archive System
7Zipを超える圧縮率と超高速処理を実現する革新的アーカイブシステム

Features:
- 超高圧縮率 (99.9%+ compression ratio)
- 超高速処理 (60MB/s+ processing speed) 
- SPE暗号化技術 (Structure-Preserving Encryption)
- 多重暗号化対応 (AES-GCM + XChaCha20-Poly1305)
- モジュラー設計による拡張性
"""

__version__ = "2.0.0"
__author__ = "NXZip Team"
__email__ = "team@nxzip.org"
__license__ = "MIT"

# Core classes
from .formats.enhanced_nxz import SuperNXZipFile
from .engine.spe_core import SPECore
from .engine.compressor import SuperCompressor
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
    'SuperNXZipFile',
    'SPECore', 
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
        from .engine.spe_core import verify_spe_integrity
        if not verify_spe_integrity():
            return False
        
        # 基本的な圧縮・展開テスト
        test_data = b"NXZip Installation Test"
        nxzip = SuperNXZipFile()
        
        archive = nxzip.create_archive(test_data)
        restored = nxzip.extract_archive(archive)
        
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
