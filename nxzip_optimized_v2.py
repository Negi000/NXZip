#!/usr/bin/env python3
"""
NXZip TMC v9.1 最適化改良版
TMC変換パイプライン効果向上 + 適応型アルゴリズム選択
"""

import os
import sys
import time
import random
import json
import zlib
import lzma
import math
from typing import Dict, Any, List, Tuple, Optional

class OptimizedNXZipEngine:
    """NXZip TMC v9.1 最適化改良版"""
    
    def __init__(self, lightweight_mode: bool = False):
        self.lightweight_mode = lightweight_mode
        self.chunk_size = 512 * 1024 if lightweight_mode else 1024 * 1024
        
        if lightweight_mode:
            self.strategy = "ultra_fast"
            self.compression_level = 2
            print("⚡ NXZip最適化軽量: 超高速モード")
        else:
            self.strategy = "smart_adaptive"
            self.compression_level = 6
            print("🎯 NXZip最適化通常: スマート適応モード")
        
        # 改良統計
        self.stats = {
            'files_processed': 0,
            'total_input_size': 0,
            'total_compressed_size': 0,
            'strategy': self.strategy,
            'algorithm_selections': {}
        }
    
    def smart_analyze(self, data: bytes) -> Dict[str, Any]:
        """改良メタアナライザー - 最適アルゴリズム選択"""
        if len(data) == 0:
            return {'recommended_algorithm': 'store', 'confidence': 1.0}
        
        # 高速エントロピー計算
        sample_size = min(2048, len(data))
        sample = data[:sample_size]
        
        byte_counts = [0] * 256
        for byte in sample:
            byte_counts[byte] += 1
        
        entropy = 0.0
        for count in byte_counts:
            if count > 0:
                p = count / sample_size
                entropy -= p * math.log2(p)
        
        # パターン検出
        unique_bytes = sum(1 for count in byte_counts if count > 0)
        max_run_length = self._detect_max_run_length(sample)
        repetition_ratio = self._detect_repetition_ratio(sample)
        
        # アルゴリズム選択ロジック
        if self.lightweight_mode:
            # 軽量モード: 速度優先
            if entropy < 2.0 or max_run_length > 50:
                algorithm = 'run_length_zlib'  # ランレングス + zlib
                confidence = 0.9
            elif repetition_ratio > 0.7:
                algorithm = 'lz77_fast'  # 高速LZ77
                confidence = 0.8
            else:
                algorithm = 'zlib_fast'  # 高速zlib
                confidence = 0.7
        else:
            # 通常モード: 適応選択
            if entropy < 3.0 and max_run_length > 20:
                algorithm = 'bwt_lzma'  # BWT + LZMA
                confidence = 0.95
            elif repetition_ratio > 0.5 and unique_bytes < 128:
                algorithm = 'context_lzma'  # コンテキスト + LZMA
                confidence = 0.9
            elif entropy > 7.0:
                algorithm = 'hybrid_fast'  # ハイブリッド高速
                confidence = 0.6
            else:
                algorithm = 'adaptive_zlib'  # 適応zlib
                confidence = 0.8
        
        return {
            'entropy': entropy,
            'unique_bytes': unique_bytes,
            'max_run_length': max_run_length,
            'repetition_ratio': repetition_ratio,
            'recommended_algorithm': algorithm,
            'confidence': confidence,
            'data_type': self._classify_data_type(entropy, unique_bytes, repetition_ratio)
        }
    
    def _detect_max_run_length(self, data: bytes) -> int:
        """最大ランレングス検出"""
        if len(data) <= 1:
            return 1
        
        max_run = 1
        current_run = 1
        
        for i in range(1, min(len(data), 1000)):  # 高速化のため制限
            if data[i] == data[i-1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        
        return max_run
    
    def _detect_repetition_ratio(self, data: bytes) -> float:
        """繰り返し比率検出"""
        if len(data) < 20:
            return 0.0
        
        # 短いパターンの繰り返し検出
        pattern_sizes = [2, 3, 4, 8, 16]
        max_ratio = 0.0
        
        for pattern_size in pattern_sizes:
            if pattern_size * 2 > len(data):
                continue
            
            pattern = data[:pattern_size]
            matches = 0
            
            for i in range(0, min(len(data) - pattern_size, 500), pattern_size):
                if data[i:i+pattern_size] == pattern:
                    matches += 1
            
            ratio = matches / (min(len(data), 500) // pattern_size)
            max_ratio = max(max_ratio, ratio)
        
        return max_ratio
    
    def _classify_data_type(self, entropy: float, unique_bytes: int, repetition_ratio: float) -> str:
        """データタイプ分類"""
        if entropy < 2.0 or repetition_ratio > 0.8:
            return 'highly_compressible'
        elif entropy < 5.0 and unique_bytes < 128:
            return 'text_like'
        elif entropy > 7.5:
            return 'random_like'
        else:
            return 'mixed'
    
    def optimized_compress(self, data: bytes, algorithm: str, confidence: float) -> Tuple[bytes, Dict[str, Any]]:
        """最適化圧縮実行"""
        start_time = time.time()
        
        try:
            if algorithm == 'run_length_zlib':
                compressed, info = self._run_length_compress(data)
            elif algorithm == 'lz77_fast':
                compressed, info = self._lz77_fast_compress(data)
            elif algorithm == 'zlib_fast':
                compressed, info = self._zlib_fast_compress(data)
            elif algorithm == 'bwt_lzma':
                compressed, info = self._bwt_lzma_compress(data)
            elif algorithm == 'context_lzma':
                compressed, info = self._context_lzma_compress(data)
            elif algorithm == 'hybrid_fast':
                compressed, info = self._hybrid_fast_compress(data)
            elif algorithm == 'adaptive_zlib':
                compressed, info = self._adaptive_zlib_compress(data)
            else:
                compressed, info = self._fallback_compress(data)
            
            compress_time = time.time() - start_time
            
            info.update({
                'algorithm_used': algorithm,
                'confidence': confidence,
                'compression_time': compress_time,
                'throughput_mbps': (len(data) / (1024 * 1024) / compress_time) if compress_time > 0 else 0
            })
            
            # アルゴリズム選択統計
            if algorithm not in self.stats['algorithm_selections']:
                self.stats['algorithm_selections'][algorithm] = 0
            self.stats['algorithm_selections'][algorithm] += 1
            
            return compressed, info
            
        except Exception as e:
            return self._fallback_compress(data)
    
    def _run_length_compress(self, data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """ランレングス + Zlib圧縮"""
        try:
            # 簡易ランレングスエンコーディング
            encoded = bytearray()
            i = 0
            while i < len(data):
                byte_val = data[i]
                run_length = 1
                
                # 同じバイトの連続をカウント
                while i + run_length < len(data) and data[i + run_length] == byte_val and run_length < 255:
                    run_length += 1
                
                if run_length >= 3:
                    # ランレングス記録: 0xFF + byte + length
                    encoded.extend([0xFF, byte_val, run_length])
                    i += run_length
                else:
                    # 通常記録
                    if byte_val == 0xFF:
                        encoded.extend([0xFF, 0xFF])  # エスケープ
                    else:
                        encoded.append(byte_val)
                    i += 1
            
            # Zlib圧縮
            compressed = zlib.compress(bytes(encoded), level=self.compression_level)
            
            return compressed, {
                'method': 'run_length_zlib',
                'original_size': len(data),
                'encoded_size': len(encoded),
                'compressed_size': len(compressed),
                'compression_ratio': (1 - len(compressed) / len(data)) * 100 if len(data) > 0 else 0
            }
        except:
            return self._fallback_compress(data)
    
    def _lz77_fast_compress(self, data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """高速LZ77圧縮"""
        try:
            # Zlib（LZ77ベース）高速設定
            compressed = zlib.compress(data, level=1)
            
            return compressed, {
                'method': 'lz77_fast',
                'original_size': len(data),
                'compressed_size': len(compressed),
                'compression_ratio': (1 - len(compressed) / len(data)) * 100 if len(data) > 0 else 0
            }
        except:
            return self._fallback_compress(data)
    
    def _zlib_fast_compress(self, data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """高速Zlib圧縮"""
        try:
            compressed = zlib.compress(data, level=1)
            
            return compressed, {
                'method': 'zlib_fast',
                'original_size': len(data),
                'compressed_size': len(compressed),
                'compression_ratio': (1 - len(compressed) / len(data)) * 100 if len(data) > 0 else 0
            }
        except:
            return self._fallback_compress(data)
    
    def _bwt_lzma_compress(self, data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """BWT + LZMA圧縮"""
        try:
            # 簡易BWT（小さなデータのみ）
            if len(data) <= 8192:
                transformed = self._simple_bwt(data)
            else:
                transformed = data
            
            # LZMA圧縮
            compressed = lzma.compress(transformed, preset=self.compression_level | lzma.PRESET_EXTREME)
            
            return compressed, {
                'method': 'bwt_lzma',
                'original_size': len(data),
                'transformed_size': len(transformed),
                'compressed_size': len(compressed),
                'compression_ratio': (1 - len(compressed) / len(data)) * 100 if len(data) > 0 else 0
            }
        except:
            return self._fallback_compress(data)
    
    def _context_lzma_compress(self, data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """コンテキスト + LZMA圧縮"""
        try:
            # 簡易コンテキスト変換
            transformed = self._simple_context_transform(data)
            
            # LZMA圧縮
            compressed = lzma.compress(transformed, preset=self.compression_level)
            
            return compressed, {
                'method': 'context_lzma',
                'original_size': len(data),
                'transformed_size': len(transformed),
                'compressed_size': len(compressed),
                'compression_ratio': (1 - len(compressed) / len(data)) * 100 if len(data) > 0 else 0
            }
        except:
            return self._fallback_compress(data)
    
    def _hybrid_fast_compress(self, data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """ハイブリッド高速圧縮"""
        try:
            # 複数アルゴリズムで試行し、最良を選択（小さなデータのみ）
            if len(data) <= 4096:
                zlib_compressed = zlib.compress(data, level=3)
                lzma_compressed = lzma.compress(data, preset=1)
                
                if len(zlib_compressed) <= len(lzma_compressed):
                    return zlib_compressed, {
                        'method': 'hybrid_fast_zlib',
                        'original_size': len(data),
                        'compressed_size': len(zlib_compressed),
                        'compression_ratio': (1 - len(zlib_compressed) / len(data)) * 100 if len(data) > 0 else 0
                    }
                else:
                    return lzma_compressed, {
                        'method': 'hybrid_fast_lzma',
                        'original_size': len(data),
                        'compressed_size': len(lzma_compressed),
                        'compression_ratio': (1 - len(lzma_compressed) / len(data)) * 100 if len(data) > 0 else 0
                    }
            else:
                # 大きなデータは高速zlib
                compressed = zlib.compress(data, level=3)
                return compressed, {
                    'method': 'hybrid_fast_fallback',
                    'original_size': len(data),
                    'compressed_size': len(compressed),
                    'compression_ratio': (1 - len(compressed) / len(data)) * 100 if len(data) > 0 else 0
                }
        except:
            return self._fallback_compress(data)
    
    def _adaptive_zlib_compress(self, data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """適応Zlib圧縮"""
        try:
            # データサイズに応じてレベル調整
            if len(data) < 1024:
                level = 6
            elif len(data) < 10240:
                level = 5
            else:
                level = 4
            
            compressed = zlib.compress(data, level=level)
            
            return compressed, {
                'method': f'adaptive_zlib_l{level}',
                'original_size': len(data),
                'compressed_size': len(compressed),
                'compression_ratio': (1 - len(compressed) / len(data)) * 100 if len(data) > 0 else 0
            }
        except:
            return self._fallback_compress(data)
    
    def _simple_bwt(self, data: bytes) -> bytes:
        """簡易BWT実装"""
        try:
            if len(data) <= 1:
                return data
            
            # 回転生成
            rotations = []
            for i in range(len(data)):
                rotation = data[i:] + data[:i]
                rotations.append(rotation)
            
            # ソート
            rotations.sort()
            
            # 最後の文字を取得
            bwt_result = bytes([rotation[-1] for rotation in rotations])
            
            return bwt_result
        except:
            return data
    
    def _simple_context_transform(self, data: bytes) -> bytes:
        """簡易コンテキスト変換"""
        try:
            if len(data) <= 2:
                return data
            
            # バイト頻度計算
            freq = [0] * 256
            for b in data:
                freq[b] += 1
            
            # 頻度順ソート
            sorted_bytes = sorted(range(256), key=lambda x: freq[x], reverse=True)
            
            # 変換テーブル作成
            transform_table = [0] * 256
            for i, byte_val in enumerate(sorted_bytes):
                transform_table[byte_val] = i
            
            # 変換実行
            transformed = bytes([transform_table[b] for b in data])
            
            return transformed
        except:
            return data
    
    def _fallback_compress(self, data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """フォールバック圧縮"""
        try:
            compressed = zlib.compress(data, level=3)
            return compressed, {
                'method': 'fallback_zlib',
                'original_size': len(data),
                'compressed_size': len(compressed),
                'compression_ratio': (1 - len(compressed) / len(data)) * 100 if len(data) > 0 else 0
            }
        except:
            return data, {
                'method': 'store',
                'original_size': len(data),
                'compressed_size': len(data),
                'compression_ratio': 0.0
            }
    
    def compress(self, data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """メイン圧縮インターフェース"""
        if len(data) == 0:
            return b'', {'method': 'nxzip_empty', 'compression_ratio': 0.0}
        
        # スマート分析
        analysis = self.smart_analyze(data)
        algorithm = analysis['recommended_algorithm']
        confidence = analysis['confidence']
        
        print(f"📊 スマート分析: {analysis['data_type']} → {algorithm} (信頼度: {confidence:.2f})")
        
        # 最適化圧縮実行
        compressed, info = self.optimized_compress(data, algorithm, confidence)
        
        # 結果統合
        result_info = {
            'engine_version': 'NXZip TMC v9.1 Optimized',
            'strategy': self.strategy,
            'analysis': analysis,
            **info
        }
        
        # 統計更新
        self.stats['files_processed'] += 1
        self.stats['total_input_size'] += len(data)
        self.stats['total_compressed_size'] += len(compressed)
        
        print(f"✅ 最適化圧縮完了: {info['compression_ratio']:.1f}% ({info['algorithm_used']})")
        
        return compressed, result_info
    
    def get_stats(self) -> Dict[str, Any]:
        """統計取得"""
        stats = self.stats.copy()
        
        if stats['total_input_size'] > 0:
            stats['overall_compression_ratio'] = (
                1 - stats['total_compressed_size'] / stats['total_input_size']
            ) * 100
        
        return stats


def test_optimized_nxzip():
    """最適化NXZipテスト"""
    print("🚀 NXZip TMC v9.1 最適化版テスト\n")
    
    # テストデータ
    test_cases = [
        (b'A' * 1000, "ランレングステスト"),
        (b'Hello World! ' * 100, "繰り返しテキスト"),
        (bytes([random.randint(0, 255) for _ in range(1000)]), "ランダムデータ"),
        (b''.join([f'Line {i}: Test data {i%10}\n'.encode() for i in range(50)]), "構造化テキスト"),
        (b'ABCD' * 500, "短パターン繰り返し")
    ]
    
    for test_data, description in test_cases:
        print(f"📊 テスト: {description} ({len(test_data):,} bytes)")
        
        # 軽量モード
        print("⚡ 軽量モード:")
        engine_light = OptimizedNXZipEngine(lightweight_mode=True)
        compressed_light, info_light = engine_light.compress(test_data)
        
        # 通常モード
        print("🎯 通常モード:")
        engine_normal = OptimizedNXZipEngine(lightweight_mode=False)
        compressed_normal, info_normal = engine_normal.compress(test_data)
        
        print("-" * 50)

if __name__ == "__main__":
    try:
        test_optimized_nxzip()
        print("✅ 最適化テスト完了")
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()
