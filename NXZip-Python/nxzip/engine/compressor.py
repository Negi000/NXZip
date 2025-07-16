#!/usr/bin/env python3
"""
NXZip Compression Engine
高性能圧縮アルゴリズムの統合インターフェース
"""

import zlib
import lzma
from typing import Tuple, Optional
from ..utils.progress import ProgressBar
from ..utils.constants import CompressionAlgorithm

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False


class SuperCompressor:
    """7Zipを超える高圧縮率と超高速処理を実現する圧縮器"""
    
    def __init__(self, algorithm: str = CompressionAlgorithm.AUTO, level: int = 6):
        self.algorithm = algorithm
        self.level = level
    
    def compress(self, data: bytes, show_progress: bool = False) -> Tuple[bytes, str]:
        """データを圧縮し、使用したアルゴリズムも返す"""
        if not data:
            return data, CompressionAlgorithm.ZLIB
        
        if self.algorithm == CompressionAlgorithm.AUTO:
            return self._auto_compress(data, show_progress)
        elif self.algorithm == CompressionAlgorithm.ZSTD and ZSTD_AVAILABLE:
            return self._compress_zstd(data, show_progress), CompressionAlgorithm.ZSTD
        elif self.algorithm == CompressionAlgorithm.LZMA2:
            return self._compress_lzma2(data, show_progress), CompressionAlgorithm.LZMA2
        else:
            return self._compress_zlib(data, show_progress), CompressionAlgorithm.ZLIB
    
    def decompress(self, data: bytes, algorithm: str, show_progress: bool = False) -> bytes:
        """指定されたアルゴリズムでデータを展開"""
        if not data:
            return data
        
        if algorithm == CompressionAlgorithm.ZSTD and ZSTD_AVAILABLE:
            return self._decompress_zstd(data, show_progress)
        elif algorithm == CompressionAlgorithm.LZMA2:
            return self._decompress_lzma2(data, show_progress)
        else:
            return self._decompress_zlib(data, show_progress)
    
    def _auto_compress(self, data: bytes, show_progress: bool) -> Tuple[bytes, str]:
        """最適な圧縮アルゴリズムを自動選択"""
        data_size = len(data)
        
        # 小さなファイルはZLIBが高速
        if data_size < 1024:
            return self._compress_zlib(data, show_progress), CompressionAlgorithm.ZLIB
        
        # 大容量ファイルでZstdが利用可能ならZstd、そうでなければLZMA2
        if ZSTD_AVAILABLE and data_size > 1024 * 1024:
            return self._compress_zstd(data, show_progress), CompressionAlgorithm.ZSTD
        elif data_size > 10 * 1024:
            return self._compress_lzma2(data, show_progress), CompressionAlgorithm.LZMA2
        else:
            return self._compress_zlib(data, show_progress), CompressionAlgorithm.ZLIB
    
    def _compress_zlib(self, data: bytes, show_progress: bool) -> bytes:
        """ZLIB圧縮（高速・軽量）"""
        level = min(9, max(1, self.level))
        if show_progress:
            pb = ProgressBar(len(data), "ZLIB圧縮")
            result = zlib.compress(data, level)
            pb.update(len(data))
            pb.close()
            return result
        return zlib.compress(data, level)
    
    def _compress_lzma2(self, data: bytes, show_progress: bool) -> bytes:
        """LZMA2圧縮（高圧縮率）"""
        preset = min(9, max(0, self.level))
        if show_progress:
            pb = ProgressBar(len(data), "LZMA2圧縮")
            result = lzma.compress(data, format=lzma.FORMAT_XZ, preset=preset)
            pb.update(len(data))
            pb.close()
            return result
        return lzma.compress(data, format=lzma.FORMAT_XZ, preset=preset)
    
    def _compress_zstd(self, data: bytes, show_progress: bool) -> bytes:
        """Zstandard圧縮（高速・高圧縮）"""
        if not ZSTD_AVAILABLE:
            raise RuntimeError("Zstandard not available")
        
        level = min(22, max(1, self.level))
        compressor = zstd.ZstdCompressor(level=level)
        
        if show_progress:
            pb = ProgressBar(len(data), "Zstd圧縮")
            result = compressor.compress(data)
            pb.update(len(data))
            pb.close()
            return result
        return compressor.compress(data)
    
    def _decompress_zlib(self, data: bytes, show_progress: bool) -> bytes:
        """ZLIB展開"""
        if show_progress:
            pb = ProgressBar(len(data), "ZLIB展開")
            result = zlib.decompress(data)
            pb.update(len(data))
            pb.close()
            return result
        return zlib.decompress(data)
    
    def _decompress_lzma2(self, data: bytes, show_progress: bool) -> bytes:
        """LZMA2展開"""
        if show_progress:
            pb = ProgressBar(len(data), "LZMA2展開")
            result = lzma.decompress(data, format=lzma.FORMAT_XZ)
            pb.update(len(data))
            pb.close()
            return result
        return lzma.decompress(data, format=lzma.FORMAT_XZ)
    
    def _decompress_zstd(self, data: bytes, show_progress: bool) -> bytes:
        """Zstandard展開"""
        if not ZSTD_AVAILABLE:
            raise RuntimeError("Zstandard not available")
        
        decompressor = zstd.ZstdDecompressor()
        if show_progress:
            pb = ProgressBar(len(data), "Zstd展開")
            result = decompressor.decompress(data)
            pb.update(len(data))
            pb.close()
            return result
        return decompressor.decompress(data)
