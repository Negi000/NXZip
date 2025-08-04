"""
NEXUS TMC Engine - Core Compressor Module

This module provides the fundamental compression/decompression functionality
that serves as the base layer for all TMC operations.
"""

import zlib
import lzma
import bz2
from typing import Tuple, Dict, Any


class CoreCompressor:
    """
    NEXUS TMC コア圧縮器 - 基盤圧縮機能
    統括モジュールから独立した専門圧縮器
    """
    
    def __init__(self, lightweight_mode: bool = False):
        self.lightweight_mode = lightweight_mode
        
        if self.lightweight_mode:
            self.default_method = 'zlib'
            self.compression_level = 6
            print("⚡ CoreCompressor軽量モード: 高速zlib圧縮")
        else:
            self.default_method = 'lzma'
            self.compression_level = 9
            print("🎯 CoreCompressor通常モード: 最高圧縮率追求")
    
    def compress_core(self, data: bytes, method: str = None) -> Tuple[bytes, Dict[str, Any]]:
        """基本圧縮機能"""
        if not method:
            method = self.default_method
        
        original_size = len(data)
        
        try:
            if method == 'zlib':
                compressed = zlib.compress(data, level=self.compression_level)
                compression_method = 'zlib'
            elif method == 'lzma':
                if self.lightweight_mode:
                    compressed = lzma.compress(data, preset=3)
                else:
                    compressed = lzma.compress(data, preset=9)
                compression_method = 'lzma'
            elif method == 'bz2':
                compressed = bz2.compress(data, compresslevel=self.compression_level)
                compression_method = 'bz2'
            else:
                # デフォルトフォールバック
                compressed = zlib.compress(data, level=self.compression_level)
                compression_method = 'zlib_fallback'
            
            compression_ratio = (1 - len(compressed) / original_size) * 100 if original_size > 0 else 0
            
            info = {
                'method': compression_method,
                'original_size': original_size,
                'compressed_size': len(compressed),
                'compression_ratio': compression_ratio,
                'compression_level': self.compression_level
            }
            
            return compressed, info
            
        except Exception as e:
            print(f"[CoreCompressor] 圧縮エラー {method}: {e}")
            # フォールバック: zlib低レベル
            compressed = zlib.compress(data, level=1)
            info = {
                'method': 'zlib_emergency_fallback',
                'original_size': original_size,
                'compressed_size': len(compressed),
                'compression_ratio': (1 - len(compressed) / original_size) * 100,
                'compression_level': 1,
                'error': str(e)
            }
            return compressed, info
    
    def decompress_core(self, compressed_data: bytes, method: str = 'auto') -> bytes:
        """基本解凍機能"""
        try:
            print(f"[CoreCompressor] 解凍開始: method={method}, サイズ={len(compressed_data):,}B")
            
            # 自動判定または指定された方式で解凍
            if method == 'auto':
                # 複数の方式を順次試行
                
                # zlib試行
                try:
                    result = zlib.decompress(compressed_data)
                    print(f"[CoreCompressor] zlib解凍成功: {len(result):,}B")
                    return result
                except:
                    pass
                
                # lzma試行
                try:
                    result = lzma.decompress(compressed_data)
                    print(f"[CoreCompressor] lzma解凍成功: {len(result):,}B")
                    return result
                except:
                    pass
                
                # bz2試行
                try:
                    result = bz2.decompress(compressed_data)
                    print(f"[CoreCompressor] bz2解凍成功: {len(result):,}B")
                    return result
                except:
                    pass
                
                # gzip試行
                try:
                    import gzip
                    result = gzip.decompress(compressed_data)
                    print(f"[CoreCompressor] gzip解凍成功: {len(result):,}B")
                    return result
                except:
                    pass
                
                print(f"[CoreCompressor] 全方式失敗 - 元データ返却")
                return compressed_data
            
            elif method == 'zlib' or method.startswith('zlib'):
                result = zlib.decompress(compressed_data)
                print(f"[CoreCompressor] zlib指定解凍成功: {len(result):,}B")
                return result
            elif method == 'lzma' or method.startswith('lzma'):
                result = lzma.decompress(compressed_data)
                print(f"[CoreCompressor] lzma指定解凍成功: {len(result):,}B")
                return result
            elif method == 'bz2' or method.startswith('bz2'):
                result = bz2.decompress(compressed_data)
                print(f"[CoreCompressor] bz2指定解凍成功: {len(result):,}B")
                return result
            else:
                # 不明な方式の場合は自動検出
                print(f"[CoreCompressor] 不明方式 '{method}' - 自動検出実行")
                return self.decompress_core(compressed_data, 'auto')
                
        except Exception as e:
            print(f"[CoreCompressor] 解凍エラー: {e}")
            if self.lightweight_mode:
                # 軽量モードはエラー耐性を重視
                print(f"[CoreCompressor] 軽量モード - 元データ返却")
                return compressed_data
            else:
                raise Exception(f"CoreCompressor解凍失敗: {e}")
