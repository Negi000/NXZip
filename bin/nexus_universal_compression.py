#!/usr/bin/env python3
"""
NEXUS Universal Compression Engine (NUCE)
汎用圧縮アルゴリズム - 完全独自実装

特徴:
1. 汎用性 - あらゆるファイル形式に対応
2. 適応性 - データパターンを自動検出・最適化
3. 段階的圧縮 - 複数アルゴリズムの組み合わせ
4. 高速処理 - 実用的な処理速度
5. 完全独自 - zlib/LZMA等の既存技術不使用

アルゴリズム構成:
- Stage 1: Pattern Analysis (パターン解析)
- Stage 2: Adaptive Preprocessing (適応前処理)
- Stage 3: Multi-tier Compression (多段階圧縮)
- Stage 4: Entropy Optimization (エントロピー最適化)
"""

import os
import sys
import time
import struct
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter, defaultdict
from enum import Enum

class DataPattern(Enum):
    """データパターン分類"""
    BINARY = "binary"
    TEXT = "text"
    REPETITIVE = "repetitive"
    RANDOM = "random"
    STRUCTURED = "structured"
    COMPRESSED = "compressed"

class CompressionStrategy(Enum):
    """圧縮戦略"""
    ULTRA_FAST = "ultra_fast"
    BALANCED = "balanced"
    MAXIMUM = "maximum"
    ADAPTIVE = "adaptive"

@dataclass
class CompressionAnalysis:
    """圧縮解析結果"""
    data_size: int
    pattern_type: DataPattern
    entropy: float
    repetition_ratio: float
    text_ratio: float
    null_ratio: float
    recommended_strategy: CompressionStrategy
    estimated_ratio: float

@dataclass
class CompressionResult:
    """圧縮結果"""
    original_size: int
    compressed_size: int
    compression_ratio: float
    processing_time: float
    algorithm_stages: List[str]
    pattern_analysis: CompressionAnalysis
    checksum: str

class NexusUniversalCompression:
    """汎用圧縮エンジン"""
    
    def __init__(self, strategy: CompressionStrategy = CompressionStrategy.ADAPTIVE):
        self.version = "1.0-Universal"
        self.magic = b'NUCE2025'  # NEXUS Universal Compression Engine
        self.strategy = strategy
        
        # 圧縮設定
        self.enable_pattern_analysis = True
        self.enable_adaptive_preprocessing = True
        self.enable_multitier_compression = True
        self.enable_entropy_optimization = True
        
        # 性能調整
        self.analysis_sample_size = 8192  # 解析サンプルサイズ
        self.max_dict_size = 32768  # 辞書最大サイズ
        self.min_match_length = 3  # 最小マッチ長
        
        print(f"🚀 NEXUS Universal Compression Engine v{self.version}")
        print(f"⚙️  戦略: {strategy.value}")
        print("🔧 汎用圧縮エンジン初期化完了")
    
    def analyze_data_pattern(self, data: bytes) -> CompressionAnalysis:
        """データパターン解析"""
        if len(data) == 0:
            return CompressionAnalysis(
                data_size=0,
                pattern_type=DataPattern.BINARY,
                entropy=0.0,
                repetition_ratio=0.0,
                text_ratio=0.0,
                null_ratio=0.0,
                recommended_strategy=CompressionStrategy.ULTRA_FAST,
                estimated_ratio=0.0
            )
        
        # サンプリング
        sample_size = min(len(data), self.analysis_sample_size)
        sample = data[:sample_size]
        
        # エントロピー計算
        entropy = self._calculate_entropy(sample)
        
        # 反復パターン分析
        repetition_ratio = self._calculate_repetition_ratio(sample)
        
        # テキスト率計算
        text_ratio = self._calculate_text_ratio(sample)
        
        # NULL率計算
        null_ratio = sample.count(0) / len(sample)
        
        # パターン分類
        pattern_type = self._classify_pattern(entropy, repetition_ratio, text_ratio, null_ratio)
        
        # 戦略推奨
        recommended_strategy = self._recommend_strategy(pattern_type, len(data))
        
        # 圧縮率予測
        estimated_ratio = self._estimate_compression_ratio(pattern_type, entropy, repetition_ratio)
        
        return CompressionAnalysis(
            data_size=len(data),
            pattern_type=pattern_type,
            entropy=entropy,
            repetition_ratio=repetition_ratio,
            text_ratio=text_ratio,
            null_ratio=null_ratio,
            recommended_strategy=recommended_strategy,
            estimated_ratio=estimated_ratio
        )
    
    def _calculate_entropy(self, data: bytes) -> float:
        """シャノンエントロピー計算"""
        if len(data) == 0:
            return 0.0
        
        # 頻度計算
        freq = Counter(data)
        total = len(data)
        
        # エントロピー計算
        entropy = 0.0
        for count in freq.values():
            prob = count / total
            if prob > 0:
                import math
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    def _calculate_repetition_ratio(self, data: bytes) -> float:
        """反復パターン率計算"""
        if len(data) < 4:
            return 0.0
        
        repetitive_bytes = 0
        i = 0
        
        while i < len(data) - 1:
            current = data[i]
            count = 1
            
            # 連続する同じバイトをカウント
            while i + count < len(data) and data[i + count] == current:
                count += 1
            
            if count >= 3:  # 3回以上の反復
                repetitive_bytes += count
            
            i += count
        
        return repetitive_bytes / len(data)
    
    def _calculate_text_ratio(self, data: bytes) -> float:
        """テキスト率計算"""
        if len(data) == 0:
            return 0.0
        
        text_bytes = 0
        for byte in data:
            # ASCII印刷可能文字 + 改行・タブ・スペース
            if (32 <= byte <= 126) or byte in [9, 10, 13]:
                text_bytes += 1
        
        return text_bytes / len(data)
    
    def _classify_pattern(self, entropy: float, repetition_ratio: float, 
                         text_ratio: float, null_ratio: float) -> DataPattern:
        """パターン分類"""
        # 高反復率
        if repetition_ratio > 0.3:
            return DataPattern.REPETITIVE
        
        # 高テキスト率
        if text_ratio > 0.8:
            return DataPattern.TEXT
        
        # 高エントロピー（ランダム/既圧縮）
        if entropy > 7.5:
            return DataPattern.RANDOM if null_ratio < 0.1 else DataPattern.COMPRESSED
        
        # 低エントロピー（構造化）
        if entropy < 4.0:
            return DataPattern.STRUCTURED
        
        # デフォルト
        return DataPattern.BINARY
    
    def _recommend_strategy(self, pattern: DataPattern, size: int) -> CompressionStrategy:
        """戦略推奨"""
        if self.strategy != CompressionStrategy.ADAPTIVE:
            return self.strategy
        
        # サイズベース調整
        if size < 1024:  # 1KB未満
            return CompressionStrategy.ULTRA_FAST
        elif size > 10 * 1024 * 1024:  # 10MB以上
            return CompressionStrategy.BALANCED
        
        # パターンベース調整
        if pattern == DataPattern.REPETITIVE:
            return CompressionStrategy.MAXIMUM
        elif pattern == DataPattern.RANDOM:
            return CompressionStrategy.ULTRA_FAST
        elif pattern == DataPattern.TEXT:
            return CompressionStrategy.BALANCED
        else:
            return CompressionStrategy.BALANCED
    
    def _estimate_compression_ratio(self, pattern: DataPattern, entropy: float, 
                                  repetition_ratio: float) -> float:
        """圧縮率予測"""
        base_ratio = {
            DataPattern.REPETITIVE: 0.8,
            DataPattern.TEXT: 0.6,
            DataPattern.STRUCTURED: 0.7,
            DataPattern.BINARY: 0.5,
            DataPattern.RANDOM: 0.1,
            DataPattern.COMPRESSED: 0.05
        }.get(pattern, 0.5)
        
        # エントロピーベース調整
        entropy_factor = max(0.1, min(1.0, (8.0 - entropy) / 8.0))
        
        # 反復率ベース調整
        repetition_factor = 1.0 + repetition_ratio * 2.0
        
        estimated = base_ratio * entropy_factor * repetition_factor
        return max(0.05, min(0.95, estimated))
    
    def compress_universal(self, data: bytes) -> bytes:
        """汎用圧縮メイン処理"""
        if len(data) == 0:
            return self._create_empty_archive()
        
        print(f"📦 汎用圧縮開始: {len(data)} bytes")
        start_time = time.time()
        
        # Stage 1: パターン解析
        analysis = self.analyze_data_pattern(data)
        print(f"🔍 解析: {analysis.pattern_type.value} (エントロピー: {analysis.entropy:.2f})")
        print(f"📊 反復率: {analysis.repetition_ratio:.1%}, テキスト率: {analysis.text_ratio:.1%}")
        print(f"⚡ 推奨戦略: {analysis.recommended_strategy.value}")
        
        # チェックサム計算
        checksum = hashlib.sha256(data).hexdigest()[:16]
        
        # Stage 2-4: 段階的圧縮
        compressed_data = data
        stages = []
        
        # Stage 2: 適応前処理
        if self.enable_adaptive_preprocessing:
            compressed_data = self._adaptive_preprocess(compressed_data, analysis)
            stages.append("adaptive_preprocess")
            print(f"  🔧 適応前処理: {len(data)} → {len(compressed_data)} bytes")
        
        # Stage 3: 多段階圧縮
        if self.enable_multitier_compression:
            compressed_data = self._multitier_compress(compressed_data, analysis)
            stages.append("multitier_compress")
            print(f"  🗜️  多段階圧縮: → {len(compressed_data)} bytes")
        
        # Stage 4: エントロピー最適化
        if self.enable_entropy_optimization:
            compressed_data = self._entropy_optimize(compressed_data, analysis)
            stages.append("entropy_optimize")
            print(f"  ⚗️  エントロピー最適化: → {len(compressed_data)} bytes")
        
        # 結果パッケージング
        processing_time = time.time() - start_time
        result = CompressionResult(
            original_size=len(data),
            compressed_size=len(compressed_data),
            compression_ratio=(1 - len(compressed_data) / len(data)) * 100,
            processing_time=processing_time,
            algorithm_stages=stages,
            pattern_analysis=analysis,
            checksum=checksum
        )
        
        # アーカイブ作成
        archive = self._create_archive(compressed_data, result)
        
        final_ratio = (1 - len(archive) / len(data)) * 100
        print(f"✅ 圧縮完了: {len(data)} → {len(archive)} bytes ({final_ratio:.1f}%, {processing_time:.3f}s)")
        
        return archive
    
    def _adaptive_preprocess(self, data: bytes, analysis: CompressionAnalysis) -> bytes:
        """適応前処理"""
        # パターンに応じた前処理選択
        if analysis.pattern_type == DataPattern.REPETITIVE:
            return self._rle_preprocess(data)
        elif analysis.pattern_type == DataPattern.TEXT:
            return self._text_preprocess(data)
        elif analysis.pattern_type == DataPattern.STRUCTURED:
            return self._delta_preprocess(data)
        else:
            return data
    
    def _rle_preprocess(self, data: bytes) -> bytes:
        """Run-Length前処理"""
        result = bytearray()
        i = 0
        
        while i < len(data):
            current = data[i]
            count = 1
            
            # 連続カウント
            while i + count < len(data) and data[i + count] == current and count < 255:
                count += 1
            
            if count >= 4:  # 4回以上で圧縮
                result.append(0xFE)  # RLEマーカー
                result.append(count)
                result.append(current)
                i += count
            else:
                # エスケープ処理
                if current == 0xFE:
                    result.append(0xFE)
                    result.append(0x00)
                result.append(current)
                i += 1
        
        return bytes(result)
    
    def _text_preprocess(self, data: bytes) -> bytes:
        """テキスト前処理（単語辞書圧縮）"""
        # 簡易単語分割と頻度解析
        text = data.decode('utf-8', errors='ignore')
        words = text.split()
        
        if len(words) < 10:
            return data  # 効果なし
        
        # 頻出単語辞書作成
        word_freq = Counter(words)
        common_words = [word for word, count in word_freq.most_common(128) if len(word) > 3]
        
        if len(common_words) < 5:
            return data
        
        # 辞書置換
        result = text
        dictionary = {}
        
        for i, word in enumerate(common_words):
            marker = f"\xFF{i:02x}\xFF"
            dictionary[marker] = word
            result = result.replace(word, marker)
        
        # 辞書とデータをパッケージ
        dict_data = "|".join([f"{k}:{v}" for k, v in dictionary.items()])
        packaged = f"DICT:{len(dict_data):04x}:{dict_data}|DATA:{result}"
        
        encoded = packaged.encode('utf-8', errors='ignore')
        return encoded if len(encoded) < len(data) else data
    
    def _delta_preprocess(self, data: bytes) -> bytes:
        """Delta前処理"""
        if len(data) < 2:
            return data
        
        result = bytearray([data[0]])  # 最初のバイト
        
        for i in range(1, len(data)):
            delta = (data[i] - data[i-1]) % 256
            result.append(delta)
        
        return bytes(result)
    
    def _multitier_compress(self, data: bytes, analysis: CompressionAnalysis) -> bytes:
        """多段階圧縮"""
        # 戦略に応じた圧縮方式選択
        strategy = analysis.recommended_strategy
        
        if strategy == CompressionStrategy.ULTRA_FAST:
            return self._simple_lz_compress(data)
        elif strategy == CompressionStrategy.MAXIMUM:
            return self._advanced_lz_compress(data)
        else:  # BALANCED or ADAPTIVE
            return self._balanced_lz_compress(data)
    
    def _simple_lz_compress(self, data: bytes) -> bytes:
        """簡易LZ圧縮"""
        if len(data) < 4:
            return data
        
        result = bytearray()
        i = 0
        
        while i < len(data):
            # 後方参照検索（簡易版）
            best_length = 0
            best_distance = 0
            
            # 最大検索範囲
            search_start = max(0, i - 256)
            
            for j in range(search_start, i):
                length = 0
                while (i + length < len(data) and 
                       j + length < i and 
                       data[i + length] == data[j + length] and 
                       length < 255):
                    length += 1
                
                if length >= self.min_match_length and length > best_length:
                    best_length = length
                    best_distance = i - j
            
            if best_length >= self.min_match_length:
                # マッチ符号化: [0xFF][距離][長さ]
                result.append(0xFF)
                result.append(best_distance)
                result.append(best_length)
                i += best_length
            else:
                # リテラル文字
                if data[i] == 0xFF:
                    result.append(0xFF)
                    result.append(0x00)
                result.append(data[i])
                i += 1
        
        return bytes(result)
    
    def _balanced_lz_compress(self, data: bytes) -> bytes:
        """バランスLZ圧縮"""
        # より大きな検索窓での圧縮
        if len(data) < 4:
            return data
        
        result = bytearray()
        i = 0
        
        while i < len(data):
            best_length = 0
            best_distance = 0
            
            # 拡張検索範囲
            search_start = max(0, i - 4096)
            
            for j in range(search_start, i):
                length = 0
                while (i + length < len(data) and 
                       j + length < i and 
                       data[i + length] == data[j + length] and 
                       length < 258):
                    length += 1
                
                if length >= self.min_match_length and length > best_length:
                    best_length = length
                    best_distance = i - j
            
            if best_length >= self.min_match_length:
                # 拡張マッチ符号化
                if best_distance <= 255 and best_length <= 255:
                    result.append(0xFE)
                    result.append(best_distance)
                    result.append(best_length)
                else:
                    result.append(0xFD)
                    result.extend(struct.pack('<HH', best_distance, best_length))
                i += best_length
            else:
                # リテラル
                byte = data[i]
                if byte in [0xFD, 0xFE, 0xFF]:
                    result.append(0xFF)
                    result.append(byte)
                else:
                    result.append(byte)
                i += 1
        
        return bytes(result)
    
    def _advanced_lz_compress(self, data: bytes) -> bytes:
        """高級LZ圧縮（ハッシュテーブル使用）"""
        if len(data) < 4:
            return data
        
        # ハッシュテーブル構築
        hash_table = defaultdict(list)
        
        # 3バイトハッシュでインデックス作成
        for i in range(len(data) - 2):
            hash_val = (data[i] << 16) | (data[i+1] << 8) | data[i+2]
            hash_table[hash_val].append(i)
        
        result = bytearray()
        i = 0
        
        while i < len(data):
            if i + 2 >= len(data):
                result.append(data[i])
                i += 1
                continue
            
            # ハッシュ検索
            hash_val = (data[i] << 16) | (data[i+1] << 8) | data[i+2]
            candidates = hash_table[hash_val]
            
            best_length = 0
            best_distance = 0
            
            for pos in candidates:
                if pos >= i:
                    break
                
                if i - pos > 65535:  # 距離制限
                    continue
                
                length = 0
                while (i + length < len(data) and 
                       pos + length < i and 
                       data[i + length] == data[pos + length] and 
                       length < 258):
                    length += 1
                
                if length > best_length:
                    best_length = length
                    best_distance = i - pos
            
            if best_length >= self.min_match_length:
                # 効率的な符号化
                result.append(0xFC)
                if best_distance <= 255 and best_length <= 255:
                    result.append(0x01)  # 短距離・短長
                    result.append(best_distance)
                    result.append(best_length)
                else:
                    result.append(0x02)  # 長距離・長長
                    result.extend(struct.pack('<HH', best_distance, best_length))
                i += best_length
            else:
                byte = data[i]
                if byte == 0xFC:
                    result.append(0xFC)
                    result.append(0x00)
                result.append(byte)
                i += 1
        
        return bytes(result)
    
    def _entropy_optimize(self, data: bytes, analysis: CompressionAnalysis) -> bytes:
        """エントロピー最適化"""
        if len(data) < 16:
            return data
        
        # 頻度解析
        freq = Counter(data)
        
        if len(freq) <= 1:
            return data
        
        # 頻度順ソート
        sorted_symbols = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        
        # 簡易適応符号化
        if len(sorted_symbols) <= 16:
            return self._nibble_encode(data, sorted_symbols)
        else:
            return self._huffman_like_encode(data, sorted_symbols)
    
    def _nibble_encode(self, data: bytes, sorted_symbols: List[Tuple[int, int]]) -> bytes:
        """4ビット符号化"""
        # 最頻出16シンボルを4ビットで符号化
        encode_table = {}
        decode_table = {}
        
        for i, (symbol, _) in enumerate(sorted_symbols[:16]):
            encode_table[symbol] = i
            decode_table[i] = symbol
        
        # エンコード
        encoded_bits = []
        escaped_symbols = []
        
        for byte in data:
            if byte in encode_table:
                encoded_bits.append(encode_table[byte])
            else:
                encoded_bits.append(15)  # エスケープ
                escaped_symbols.append(byte)
        
        # ビットパッキング
        packed = bytearray()
        
        # テーブル情報
        packed.append(len(decode_table))
        for i in range(len(decode_table)):
            packed.append(decode_table[i])
        
        # エスケープシンボル数
        packed.extend(struct.pack('<I', len(escaped_symbols)))
        packed.extend(escaped_symbols)
        
        # エンコードデータ
        for i in range(0, len(encoded_bits), 2):
            if i + 1 < len(encoded_bits):
                packed_byte = (encoded_bits[i] << 4) | encoded_bits[i + 1]
            else:
                packed_byte = encoded_bits[i] << 4
            packed.append(packed_byte)
        
        return bytes(packed) if len(packed) < len(data) else data
    
    def _huffman_like_encode(self, data: bytes, sorted_symbols: List[Tuple[int, int]]) -> bytes:
        """Huffman風符号化"""
        # 簡易可変長符号作成
        code_table = {}
        
        # 頻度に基づく符号長決定
        total_freq = sum(freq for _, freq in sorted_symbols)
        
        for i, (symbol, freq) in enumerate(sorted_symbols):
            if i < 2:
                code_length = 2
            elif i < 6:
                code_length = 3
            elif i < 14:
                code_length = 4
            elif i < 30:
                code_length = 5
            else:
                code_length = 8
            
            # 符号生成（簡易）
            code = i & ((1 << code_length) - 1)
            code_table[symbol] = (code, code_length)
        
        # エンコード
        bit_stream = []
        
        for byte in data:
            if byte in code_table:
                code, length = code_table[byte]
                for i in range(length):
                    bit_stream.append((code >> (length - 1 - i)) & 1)
            else:
                # エスケープ
                for i in range(8):
                    bit_stream.append((byte >> (7 - i)) & 1)
        
        # パッキング
        packed = bytearray()
        
        # テーブル情報
        packed.append(len(code_table))
        for symbol, (code, length) in code_table.items():
            packed.append(symbol)
            packed.append(length)
            packed.append(code)
        
        # ビットストリーム
        packed.extend(struct.pack('<I', len(bit_stream)))
        
        for i in range(0, len(bit_stream), 8):
            byte = 0
            for j in range(8):
                if i + j < len(bit_stream):
                    byte |= bit_stream[i + j] << (7 - j)
            packed.append(byte)
        
        return bytes(packed) if len(packed) < len(data) else data
    
    def _create_archive(self, compressed_data: bytes, result: CompressionResult) -> bytes:
        """アーカイブ作成"""
        archive = bytearray()
        
        # マジックヘッダー
        archive.extend(self.magic)
        
        # バージョン
        archive.append(1)
        
        # 結果メタデータ
        metadata = self._serialize_result(result)
        archive.extend(struct.pack('<I', len(metadata)))
        archive.extend(metadata)
        
        # 圧縮データ
        archive.extend(struct.pack('<I', len(compressed_data)))
        archive.extend(compressed_data)
        
        return bytes(archive)
    
    def _serialize_result(self, result: CompressionResult) -> bytes:
        """結果シリアライズ"""
        data = bytearray()
        
        # 基本情報
        data.extend(struct.pack('<I', result.original_size))
        data.extend(struct.pack('<I', result.compressed_size))
        data.extend(struct.pack('<f', result.compression_ratio))
        data.extend(struct.pack('<f', result.processing_time))
        
        # チェックサム
        checksum_bytes = result.checksum.encode('utf-8')
        data.append(len(checksum_bytes))
        data.extend(checksum_bytes)
        
        # ステージ情報
        data.append(len(result.algorithm_stages))
        for stage in result.algorithm_stages:
            stage_bytes = stage.encode('utf-8')
            data.append(len(stage_bytes))
            data.extend(stage_bytes)
        
        # パターン解析情報
        analysis = result.pattern_analysis
        data.append(ord(analysis.pattern_type.value[0]))  # 最初の文字
        data.extend(struct.pack('<f', analysis.entropy))
        data.extend(struct.pack('<f', analysis.repetition_ratio))
        data.extend(struct.pack('<f', analysis.text_ratio))
        
        return bytes(data)
    
    def _create_empty_archive(self) -> bytes:
        """空アーカイブ作成"""
        archive = bytearray()
        archive.extend(self.magic)
        archive.append(1)
        archive.extend(struct.pack('<I', 0))  # メタデータサイズ
        archive.extend(struct.pack('<I', 0))  # データサイズ
        return bytes(archive)

def main():
    """メイン関数"""
    if len(sys.argv) < 2:
        print("🚀 NEXUS Universal Compression Engine")
        print("汎用圧縮アルゴリズム - 完全独自実装")
        print()
        print("使用方法:")
        print("  python nexus_universal_compression.py compress <ファイル> [戦略]")
        print("  python nexus_universal_compression.py analyze <ファイル>")
        print("  python nexus_universal_compression.py test")
        print()
        print("戦略オプション:")
        print("  ultra_fast - 超高速圧縮")
        print("  balanced   - バランス型（デフォルト）")
        print("  maximum    - 最大圧縮")
        print("  adaptive   - 適応型")
        print()
        print("特徴:")
        print("  🔍 パターン解析 - データ特性自動検出")
        print("  🔧 適応処理 - 最適アルゴリズム選択")
        print("  🗜️  多段階圧縮 - 複数手法組み合わせ")
        print("  ⚗️  エントロピー最適化 - 理論限界追求")
        return
    
    command = sys.argv[1].lower()
    
    if command == "test":
        print("🧪 Universal Compression テスト実行")
        
        # 各種パターンのテストデータ生成
        test_cases = [
            ("反復データ", b"ABCD" * 1000),
            ("テキストデータ", "Hello World! " * 200),
            ("構造化データ", bytes(range(256)) * 20),
            ("ランダムデータ", os.urandom(2048)),
        ]
        
        for name, test_data in test_cases:
            if isinstance(test_data, str):
                test_data = test_data.encode('utf-8')
            
            print(f"\n📊 {name}テスト: {len(test_data)} bytes")
            
            compressor = NexusUniversalCompression(CompressionStrategy.ADAPTIVE)
            compressed = compressor.compress_universal(test_data)
            
            ratio = (1 - len(compressed) / len(test_data)) * 100
            print(f"結果: {len(test_data)} → {len(compressed)} bytes ({ratio:.1f}%)")
    
    elif command == "analyze" and len(sys.argv) >= 3:
        file_path = sys.argv[2]
        
        if not os.path.exists(file_path):
            print(f"❌ ファイルが見つかりません: {file_path}")
            return
        
        with open(file_path, 'rb') as f:
            data = f.read()
        
        print(f"📁 ファイル解析: {file_path}")
        print(f"📏 ファイルサイズ: {len(data)} bytes")
        
        compressor = NexusUniversalCompression()
        analysis = compressor.analyze_data_pattern(data)
        
        print(f"\n🔍 解析結果:")
        print(f"  パターン: {analysis.pattern_type.value}")
        print(f"  エントロピー: {analysis.entropy:.3f}")
        print(f"  反復率: {analysis.repetition_ratio:.1%}")
        print(f"  テキスト率: {analysis.text_ratio:.1%}")
        print(f"  NULL率: {analysis.null_ratio:.1%}")
        print(f"  推奨戦略: {analysis.recommended_strategy.value}")
        print(f"  予想圧縮率: {analysis.estimated_ratio:.1%}")
    
    elif command == "compress" and len(sys.argv) >= 3:
        file_path = sys.argv[2]
        strategy_name = sys.argv[3] if len(sys.argv) >= 4 else "adaptive"
        
        # 戦略パース
        strategy_map = {
            "ultra_fast": CompressionStrategy.ULTRA_FAST,
            "balanced": CompressionStrategy.BALANCED,
            "maximum": CompressionStrategy.MAXIMUM,
            "adaptive": CompressionStrategy.ADAPTIVE
        }
        
        strategy = strategy_map.get(strategy_name, CompressionStrategy.ADAPTIVE)
        
        if not os.path.exists(file_path):
            print(f"❌ ファイルが見つかりません: {file_path}")
            return
        
        print(f"📁 ファイル圧縮: {file_path}")
        
        with open(file_path, 'rb') as f:
            data = f.read()
        
        compressor = NexusUniversalCompression(strategy)
        compressed = compressor.compress_universal(data)
        
        # 出力ファイル
        base_name = os.path.splitext(file_path)[0]
        output_path = f"{base_name}.nuce"
        
        with open(output_path, 'wb') as f:
            f.write(compressed)
        
        ratio = (1 - len(compressed) / len(data)) * 100
        print(f"✅ 圧縮完了!")
        print(f"📁 出力: {output_path}")
        print(f"📊 圧縮率: {ratio:.1f}%")
    
    else:
        print("❌ 無効なコマンドです。")

if __name__ == "__main__":
    main()
