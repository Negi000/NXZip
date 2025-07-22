#!/usr/bin/env python3
"""
NEXUS SDC Enhanced Compression Algorithms
構造破壊型圧縮の高度化アルゴリズム

理論値に近づけるための最適化実装
"""

import lzma
import zlib
import bz2
import struct
from typing import bytes, Tuple, List
import hashlib
from collections import Counter

class EnhancedCompressionAlgorithms:
    """高度化圧縮アルゴリズム"""
    
    @staticmethod
    def adaptive_compress(data: bytes, compression_potential: float) -> Tuple[bytes, str]:
        """適応的圧縮アルゴリズム"""
        
        # データ特性分析
        entropy = EnhancedCompressionAlgorithms._calculate_entropy(data)
        repetition_ratio = EnhancedCompressionAlgorithms._calculate_repetition_ratio(data)
        pattern_complexity = EnhancedCompressionAlgorithms._calculate_pattern_complexity(data)
        
        # 前処理適用
        preprocessed_data = EnhancedCompressionAlgorithms._apply_preprocessing(data, entropy, repetition_ratio)
        
        # 複数アルゴリズムでテスト
        compression_results = []
        
        # LZMA (最高圧縮)
        try:
            lzma_compressed = lzma.compress(preprocessed_data, preset=9, check=lzma.CHECK_NONE)
            compression_results.append((lzma_compressed, 'lzma_enhanced', len(lzma_compressed)))
        except:
            pass
        
        # Zlib (高速)
        try:
            zlib_compressed = zlib.compress(preprocessed_data, level=9)
            compression_results.append((zlib_compressed, 'zlib_enhanced', len(zlib_compressed)))
        except:
            pass
        
        # BZ2 (中間)
        try:
            bz2_compressed = bz2.compress(preprocessed_data, compresslevel=9)
            compression_results.append((bz2_compressed, 'bz2_enhanced', len(bz2_compressed)))
        except:
            pass
        
        # カスタム高圧縮アルゴリズム
        if compression_potential > 0.7:
            custom_compressed = EnhancedCompressionAlgorithms._custom_high_compression(preprocessed_data)
            if custom_compressed:
                compression_results.append((custom_compressed, 'custom_high', len(custom_compressed)))
        
        # 構造破壊型圧縮
        if compression_potential > 0.8:
            destructive_compressed = EnhancedCompressionAlgorithms._destructive_compression(preprocessed_data)
            if destructive_compressed:
                compression_results.append((destructive_compressed, 'destructive', len(destructive_compressed)))
        
        # 最良の結果を選択
        if compression_results:
            best_result = min(compression_results, key=lambda x: x[2])
            return best_result[0], best_result[1]
        else:
            return data, 'raw'
    
    @staticmethod
    def _calculate_entropy(data: bytes) -> float:
        """エントロピー計算"""
        if len(data) == 0:
            return 0.0
        
        byte_counts = Counter(data)
        entropy = 0.0
        
        import math
        for count in byte_counts.values():
            p = count / len(data)
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    @staticmethod
    def _calculate_repetition_ratio(data: bytes) -> float:
        """繰り返し比率の計算"""
        if len(data) < 4:
            return 0.0
        
        # 2-4バイトのパターンをチェック
        patterns = {}
        total_patterns = 0
        
        for pattern_len in [2, 3, 4]:
            for i in range(len(data) - pattern_len + 1):
                pattern = data[i:i + pattern_len]
                patterns[pattern] = patterns.get(pattern, 0) + 1
                total_patterns += 1
        
        # 最も頻出するパターンの比率
        if total_patterns == 0:
            return 0.0
        
        max_count = max(patterns.values()) if patterns else 0
        return max_count / total_patterns
    
    @staticmethod
    def _calculate_pattern_complexity(data: bytes) -> float:
        """パターン複雑度の計算"""
        if len(data) < 8:
            return 1.0
        
        # 8バイト窓での類似度チェック
        similarities = 0
        comparisons = 0
        
        for i in range(0, len(data) - 16, 8):
            window1 = data[i:i + 8]
            window2 = data[i + 8:i + 16]
            
            # ハミング距離計算
            hamming = sum(b1 != b2 for b1, b2 in zip(window1, window2))
            similarity = 1.0 - (hamming / 8.0)
            
            similarities += similarity
            comparisons += 1
        
        return similarities / comparisons if comparisons > 0 else 1.0
    
    @staticmethod
    def _apply_preprocessing(data: bytes, entropy: float, repetition_ratio: float) -> bytes:
        """前処理の適用"""
        
        # 高繰り返しデータの場合はRLE前処理
        if repetition_ratio > 0.3:
            data = EnhancedCompressionAlgorithms._rle_preprocess(data)
        
        # 低エントロピーデータの場合は差分符号化
        if entropy < 4.0:
            data = EnhancedCompressionAlgorithms._delta_encoding(data)
        
        # バイト順序の最適化
        if len(data) > 1024:
            data = EnhancedCompressionAlgorithms._optimize_byte_order(data)
        
        return data
    
    @staticmethod
    def _rle_preprocess(data: bytes) -> bytes:
        """Run-Length Encoding前処理"""
        result = bytearray()
        i = 0
        
        while i < len(data):
            current_byte = data[i]
            count = 1
            
            # 連続する同じバイトをカウント
            while i + count < len(data) and data[i + count] == current_byte and count < 255:
                count += 1
            
            if count >= 4:  # 4回以上の繰り返しでRLE適用
                result.extend([0xFF, current_byte, count])
                i += count
            else:
                result.append(current_byte)
                i += 1
        
        return bytes(result)
    
    @staticmethod
    def _delta_encoding(data: bytes) -> bytes:
        """差分符号化"""
        if len(data) < 2:
            return data
        
        result = bytearray([data[0]])  # 最初のバイトはそのまま
        
        for i in range(1, len(data)):
            delta = (data[i] - data[i-1]) % 256
            result.append(delta)
        
        return bytes(result)
    
    @staticmethod
    def _optimize_byte_order(data: bytes) -> bytes:
        """バイト順序の最適化"""
        # 頻度順にバイトを並び替え
        byte_freq = Counter(data)
        sorted_bytes = sorted(byte_freq.keys(), key=lambda x: byte_freq[x], reverse=True)
        
        # 変換テーブル作成
        translation_table = {}
        for new_value, original_byte in enumerate(sorted_bytes):
            translation_table[original_byte] = new_value % 256
        
        # データ変換
        result = bytearray()
        for byte in data:
            result.append(translation_table[byte])
        
        # 変換テーブルを先頭に付加
        header = bytearray([len(sorted_bytes)])
        for original_byte in sorted_bytes:
            header.append(original_byte)
        
        return bytes(header + result)
    
    @staticmethod
    def _custom_high_compression(data: bytes) -> bytes:
        """カスタム高圧縮アルゴリズム"""
        try:
            # 多段階圧縮
            stage1 = bz2.compress(data, compresslevel=9)
            stage2 = lzma.compress(stage1, preset=9, check=lzma.CHECK_NONE)
            
            # 元のデータより小さい場合のみ返す
            if len(stage2) < len(data) * 0.8:
                return b'MSTAGE' + stage2
            
        except:
            pass
        
        return None
    
    @staticmethod
    def _destructive_compression(data: bytes) -> bytes:
        """構造破壊型圧縮（実験的）"""
        try:
            # データを複数のパートに分割
            part_size = len(data) // 4
            parts = []
            
            for i in range(0, len(data), part_size):
                part = data[i:i + part_size]
                # 各パートを個別に最適圧縮
                compressed_part = lzma.compress(part, preset=9, check=lzma.CHECK_NONE)
                parts.append(compressed_part)
            
            # パーツを結合
            result = b'DESTRUCT'
            result += struct.pack('<I', len(parts))
            
            for part in parts:
                result += struct.pack('<I', len(part))
                result += part
            
            # 元のデータより小さい場合のみ返す
            if len(result) < len(data) * 0.7:
                return result
            
        except:
            pass
        
        return None
    
    @staticmethod
    def enhanced_decompress(data: bytes, method: str) -> bytes:
        """高度化圧縮の展開"""
        
        if method == 'lzma_enhanced':
            return lzma.decompress(data)
        elif method == 'zlib_enhanced':
            return zlib.decompress(data)
        elif method == 'bz2_enhanced':
            return bz2.decompress(data)
        elif method == 'custom_high':
            if data.startswith(b'MSTAGE'):
                stage2_data = data[6:]  # 'MSTAGE'を除去
                stage1_data = lzma.decompress(stage2_data)
                return bz2.decompress(stage1_data)
        elif method == 'destructive':
            if data.startswith(b'DESTRUCT'):
                pos = 8  # 'DESTRUCT'
                part_count = struct.unpack('<I', data[pos:pos+4])[0]
                pos += 4
                
                parts = []
                for _ in range(part_count):
                    part_size = struct.unpack('<I', data[pos:pos+4])[0]
                    pos += 4
                    compressed_part = data[pos:pos+part_size]
                    pos += part_size
                    
                    decompressed_part = lzma.decompress(compressed_part)
                    parts.append(decompressed_part)
                
                return b''.join(parts)
        elif method == 'raw':
            return data
        else:
            # デフォルトは生データ
            return data

# エクスポート用
__all__ = ['EnhancedCompressionAlgorithms']
