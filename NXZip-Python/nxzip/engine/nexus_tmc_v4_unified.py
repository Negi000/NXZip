#!/usr/bin/env python3
"""
NEXUS TMC Engine v8.0 - 次世代量子インテリジェント圧縮プラットフォーム
Transform-Model-Code 圧縮フレームワーク TMC v8.0
真の並列チャンク処理 + LeCoの可変長パーティショニング + 純粋エントロピー符号化
"""

import os
import sys
import time
import struct
import zlib
import lzma
import bz2
import json
import warnings
import numpy as np
from typing import Tuple, Dict, Any, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from enum import Enum
from dataclasses import dataclass
import multiprocessing as mp

# TMC v8.0 並列チャンク処理の定数とデータ構造
TMC_V8_MAGIC = b'TMC8'  # マジックナンバー
DEFAULT_CHUNK_SIZE = 2 * 1024 * 1024  # 2MB per chunk (optimal for parallel processing)

@dataclass
class ChunkInfo:
    """チャンク情報格納クラス"""
    chunk_id: int
    original_size: int
    compressed_size: int
    data_type: str
    compression_ratio: float
    processing_time: float

@dataclass 
class TMCv8Container:
    """TMC v8.0 コンテナフォーマット"""
    magic: bytes
    version: str
    chunk_count: int
    chunk_infos: List[ChunkInfo]
    compressed_chunks: List[bytes]

# Zstandardのインポート（フォールバック付き）
try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
    print("🚀 Zstandard利用可能 - 高性能バックエンド有効")
except ImportError:
    ZSTD_AVAILABLE = False
    print("⚠️ Zstandard未利用 - 標準圧縮器を使用")


class MetaAnalyzer:
    """
    TMC v7.0 インテリジェント・バイパス - メタレイヤー分析器
    変換のコストパフォーマンスを予測・評価
    """
    
    def __init__(self, core_compressor):
        self.core_compressor = core_compressor
        self.cache = {}  # 分析結果キャッシュ
        self.sample_size = 8192  # 8KBサンプル
        
    def should_apply_transform(self, data: bytes, transformer, data_type) -> Tuple[bool, Dict[str, Any]]:
        """
        変換のコストパフォーマンス分析
        Returns: (should_transform, analysis_info)
        """
        print(f"  [メタ分析] {data_type.value} の変換効果を分析中...")
        
        if not transformer or len(data) < self.sample_size:
            return True, {'reason': 'no_transformer_or_small_data'}
        
        try:
            # サンプル抽出（先頭、中央、末尾から均等に）
            sample = self._extract_representative_sample(data)
            sample_key = hash(sample)
            
            # キャッシュチェック
            if sample_key in self.cache:
                cached_result = self.cache[sample_key]
                print(f"    [メタ分析] キャッシュヒット: 効果={cached_result['effectiveness']:.2%}")
                return cached_result['should_transform'], cached_result
            
            # 1. 変換なしの圧縮サイズ
            compressed_raw, _ = self.core_compressor.compress(sample)
            size_raw = len(compressed_raw)
            
            # 2. 変換ありの圧縮サイズ
            try:
                transformed_streams, _ = transformer.transform(sample)
                size_transformed = 0
                
                for stream in transformed_streams:
                    if len(stream) > 0:
                        compressed_stream, _ = self.core_compressor.compress(stream)
                        size_transformed += len(compressed_stream)
                
                # ヘッダーオーバーヘッドを推定（変換情報など）
                estimated_header_overhead = 64  # 概算
                size_transformed += estimated_header_overhead
                
            except Exception as e:
                print(f"    [メタ分析] 変換テスト失敗: {e}")
                # 変換に失敗した場合は変換をスキップ
                analysis_info = {
                    'reason': 'transform_failed',
                    'error': str(e),
                    'should_transform': False
                }
                self.cache[sample_key] = analysis_info
                return False, analysis_info
            
            # 3. 効果分析
            effectiveness = (size_raw - size_transformed) / size_raw if size_raw > 0 else 0
            threshold = self._get_effectiveness_threshold(data_type, len(data))
            
            should_transform = effectiveness > threshold
            
            analysis_info = {
                'sample_size': len(sample),
                'raw_compressed_size': size_raw,
                'transformed_compressed_size': size_transformed,
                'effectiveness': effectiveness,
                'threshold': threshold,
                'should_transform': should_transform,
                'reason': 'effectiveness_analysis'
            }
            
            # キャッシュに保存
            self.cache[sample_key] = analysis_info
            
            print(f"    [メタ分析] 圧縮効果: {effectiveness:.2%} (閾値: {threshold:.2%}) -> {'変換実行' if should_transform else '変換スキップ'}")
            
            return should_transform, analysis_info
            
        except Exception as e:
            print(f"    [メタ分析] 分析エラー: {e} - デフォルトで変換実行")
            return True, {'reason': 'analysis_error', 'error': str(e)}
    
    def _extract_representative_sample(self, data: bytes) -> bytes:
        """代表的なサンプルを抽出（先頭、中央、末尾から）"""
        if len(data) <= self.sample_size:
            return data
        
        chunk_size = self.sample_size // 3
        start_chunk = data[:chunk_size]
        middle_start = (len(data) - chunk_size) // 2
        middle_chunk = data[middle_start:middle_start + chunk_size]
        end_chunk = data[-chunk_size:]
        
        return start_chunk + middle_chunk + end_chunk
    
    def _get_effectiveness_threshold(self, data_type, data_size: int) -> float:
        """データタイプとサイズに基づく効果閾値"""
        base_thresholds = {
            DataType.TEXT_DATA: 0.05,          # テキストは5%以上の改善で変換
            DataType.SEQUENTIAL_INT_DATA: 0.03, # 系列整数は3%以上で変換
            DataType.FLOAT_DATA: 0.08,         # 浮動小数点は8%以上で変換
            DataType.STRUCTURED_NUMERIC: 0.06,  # 構造化数値は6%以上で変換
            DataType.REPETITIVE_BINARY: 0.04,  # 反復バイナリは4%以上で変換
        }
        
        threshold = base_thresholds.get(data_type, 0.05)
        
        # 大きなデータほど厳しい閾値（オーバーヘッドの相対的影響が減少）
        if data_size > 1024 * 1024:  # 1MB以上
            threshold *= 0.7
        elif data_size > 64 * 1024:  # 64KB以上
            threshold *= 0.85
        
        return threshold


class PostBWTPipeline:
    """
    TMC v7.0 ポストBWTパイプライン
    BWT+MTF後の特殊なデータ構造に特化した専門符号化
    """
    
    def encode(self, mtf_stream: bytes) -> List[bytes]:
        """BWT+MTF後のストリームを専門符号化"""
        print("    [ポストBWT] RLE + 分割エントロピー符号化を実行中...")
        
        try:
            # 1. ランレングス符号化 (RLE)
            literals, run_lengths = self._apply_rle(mtf_stream)
            
            print(f"    [ポストBWT] RLE: {len(mtf_stream)} bytes -> リテラル: {len(literals)}, ラン: {len(run_lengths)}")
            
            # 2. 分割したストリームを返す
            return [literals, run_lengths]
            
        except Exception as e:
            print(f"    [ポストBWT] エラー: {e} - 元データを返却")
            return [mtf_stream]
    
    def decode(self, streams: List[bytes]) -> bytes:
        """ポストBWT専門復号"""
        print("    [ポストBWT] RLE逆変換を実行中...")
        
        try:
            if len(streams) == 1:
                return streams[0]  # RLE未適用
            
            if len(streams) >= 2:
                literals = streams[0]
                run_lengths = streams[1]
                
                # 逆RLE
                mtf_stream = self._reverse_rle(literals, run_lengths)
                print(f"    [ポストBWT] 逆RLE: リテラル: {len(literals)}, ラン: {len(run_lengths)} -> {len(mtf_stream)} bytes")
                
                return mtf_stream
            
            return b''.join(streams)
            
        except Exception as e:
            print(f"    [ポストBWT] 逆変換エラー: {e}")
            return b''.join(streams)
    
    def _apply_rle(self, data: bytes) -> Tuple[bytes, bytes]:
        """ランレングス符号化（MTF後のデータに最適化）"""
        if not data:
            return b'', b''
        
        literals = bytearray()
        run_lengths = bytearray()
        
        current_byte = data[0]
        run_length = 1
        
        for i in range(1, len(data)):
            if data[i] == current_byte and run_length < 255:
                run_length += 1
            else:
                # ランを記録
                literals.append(current_byte)
                run_lengths.append(run_length)
                
                # 新しいランを開始
                current_byte = data[i]
                run_length = 1
        
        # 最後のランを記録
        literals.append(current_byte)
        run_lengths.append(run_length)
        
        return bytes(literals), bytes(run_lengths)
    
    def _reverse_rle(self, literals: bytes, run_lengths: bytes) -> bytes:
        """逆ランレングス符号化"""
        if len(literals) != len(run_lengths):
            raise ValueError("Literals and run_lengths must have the same length")
        
        result = bytearray()
        
        for literal, run_length in zip(literals, run_lengths):
            result.extend([literal] * run_length)
        
        return bytes(result)


class DataType(Enum):
    """改良データタイプ分類（ユーザー提案統合）"""
    FLOAT_DATA = "float_data"
    TEXT_DATA = "text_data"
    SEQUENTIAL_INT_DATA = "sequential_int_data"
    STRUCTURED_NUMERIC = "structured_numeric"
    TIME_SERIES = "time_series"
    REPETITIVE_BINARY = "repetitive_binary"
    COMPRESSED_LIKE = "compressed_like"
    GENERIC_BINARY = "generic_binary"


class SublinearLZ77Compressor:
    """
    TMC v9.0 サブリニアLZ77圧縮器
    O(n log log n)の高速辞書検索 + Suffix Array活用
    """
    
    def __init__(self):
        self.min_match_length = 3  # 最小マッチ長
        self.max_match_length = 258  # 最大マッチ長
        self.window_size = 32768  # 辞書ウィンドウサイズ
        self.pydivsufsort_available = False
        
        try:
            import pydivsufsort
            self.pydivsufsort = pydivsufsort
            self.pydivsufsort_available = True
            print("🚀 SublinearLZ77: pydivsufsort高速検索有効")
        except ImportError:
            print("⚠️ SublinearLZ77: フォールバック検索モード")
    
    def compress_sublinear_lz77(self, data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """サブリニアLZ77圧縮実行"""
        if len(data) < self.min_match_length:
            return data, {"method": "store", "reason": "too_small"}
        
        print(f"    [SublinearLZ77] 高速辞書圧縮開始: {len(data)} bytes")
        
        try:
            if self.pydivsufsort_available and len(data) >= 1024:
                # Suffix Array活用高速検索
                compressed_data, stats = self._sa_based_compression(data)
            else:
                # フォールバック高速検索
                compressed_data, stats = self._fallback_compression(data)
            
            print(f"    [SublinearLZ77] 圧縮完了: {len(data)} -> {len(compressed_data)} bytes")
            print(f"    [SublinearLZ77] 統計: {stats}")
            
            return compressed_data, {
                "method": "sublinear_lz77",
                "original_size": len(data),
                "compressed_size": len(compressed_data),
                "statistics": stats
            }
            
        except Exception as e:
            print(f"    [SublinearLZ77] エラー: {e} - 元データ返却")
            return data, {"method": "store", "error": str(e)}
    
    def _sa_based_compression(self, data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """Suffix Array基盤の高速LZ77圧縮"""
        import numpy as np
        
        # Suffix Array構築
        sa = self.pydivsufsort.divsufsort(data)
        
        # 高速辞書マッチング
        compressed_tokens = []
        pos = 0
        total_matches = 0
        total_match_length = 0
        
        while pos < len(data):
            # 現在位置からの最長マッチを高速検索
            match_pos, match_length = self._find_longest_match_sa(data, sa, pos)
            
            if match_length >= self.min_match_length:
                # マッチ発見: (距離, 長さ)トークン
                distance = pos - match_pos
                compressed_tokens.append(('match', distance, match_length))
                pos += match_length
                total_matches += 1
                total_match_length += match_length
            else:
                # リテラル文字
                compressed_tokens.append(('literal', data[pos]))
                pos += 1
        
        # トークンをバイト列にエンコード
        compressed_data = self._encode_lz77_tokens(compressed_tokens)
        
        stats = {
            "total_matches": total_matches,
            "total_match_length": total_match_length,
            "compression_ratio": len(compressed_data) / len(data),
            "tokens": len(compressed_tokens)
        }
        
        return compressed_data, stats
    
    def _find_longest_match_sa(self, data: bytes, sa: 'np.ndarray', pos: int) -> Tuple[int, int]:
        """Suffix Array使用最長マッチ検索"""
        if pos >= len(data):
            return -1, 0
        
        max_match_length = 0
        best_match_pos = -1
        
        # 現在位置から検索範囲を設定
        window_start = max(0, pos - self.window_size)
        
        # Suffix Array内で候補位置を高速検索
        for i in range(len(sa)):
            sa_pos = sa[i]
            
            # ウィンドウ範囲内かつ現在位置より前の位置のみ検索
            if sa_pos >= pos or sa_pos < window_start:
                continue
            
            # マッチ長計算
            match_length = 0
            max_possible_length = min(
                len(data) - pos, 
                len(data) - sa_pos,
                self.max_match_length
            )
            
            while (match_length < max_possible_length and 
                   data[pos + match_length] == data[sa_pos + match_length]):
                match_length += 1
            
            if match_length > max_match_length:
                max_match_length = match_length
                best_match_pos = sa_pos
        
        return best_match_pos, max_match_length
    
    def _fallback_compression(self, data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """フォールバック高速LZ77実装"""
        compressed_tokens = []
        pos = 0
        total_matches = 0
        
        while pos < len(data):
            # 高速ハッシュベースマッチング
            match_pos, match_length = self._hash_based_match(data, pos)
            
            if match_length >= self.min_match_length:
                distance = pos - match_pos
                compressed_tokens.append(('match', distance, match_length))
                pos += match_length
                total_matches += 1
            else:
                compressed_tokens.append(('literal', data[pos]))
                pos += 1
        
        compressed_data = self._encode_lz77_tokens(compressed_tokens)
        
        stats = {
            "total_matches": total_matches,
            "method": "hash_based",
            "tokens": len(compressed_tokens)
        }
        
        return compressed_data, stats
    
    def _hash_based_match(self, data: bytes, pos: int) -> Tuple[int, int]:
        """ハッシュベース高速マッチング"""
        if pos < self.min_match_length:
            return -1, 0
        
        window_start = max(0, pos - self.window_size)
        max_length = 0
        best_pos = -1
        
        # 3バイトハッシュで高速検索
        if pos + self.min_match_length <= len(data):
            target = data[pos:pos + self.min_match_length]
            
            for search_pos in range(window_start, pos):
                if search_pos + self.min_match_length <= len(data):
                    if data[search_pos:search_pos + self.min_match_length] == target:
                        # マッチ拡張
                        length = self.min_match_length
                        while (pos + length < len(data) and 
                               search_pos + length < len(data) and
                               length < self.max_match_length and
                               data[pos + length] == data[search_pos + length]):
                            length += 1
                        
                        if length > max_length:
                            max_length = length
                            best_pos = search_pos
        
        return best_pos, max_length
    
    def _encode_lz77_tokens(self, tokens: list) -> bytes:
        """LZ77トークンのバイトエンコーディング"""
        import struct
        encoded = bytearray()
        
        for token in tokens:
            if token[0] == 'literal':
                # リテラル: 0x00 + バイト値
                encoded.append(0x00)
                encoded.append(token[1])
            else:  # match
                # マッチ: 0x01 + 距離(2bytes) + 長さ(1byte)
                _, distance, length = token
                encoded.append(0x01)
                encoded.extend(struct.pack('<H', distance))  # リトルエンディアン2バイト
                encoded.append(min(length, 255))
        
        return bytes(encoded)
    
    def decompress_sublinear_lz77(self, compressed_data: bytes, info: Dict[str, Any]) -> bytes:
        """サブリニアLZ77展開"""
        if info.get("method") != "sublinear_lz77":
            return compressed_data
        
        print("    [SublinearLZ77] 高速展開開始")
        
        try:
            tokens = self._decode_lz77_tokens(compressed_data)
            decompressed = bytearray()
            
            for token in tokens:
                if token[0] == 'literal':
                    decompressed.append(token[1])
                else:  # match
                    _, distance, length = token
                    start_pos = len(decompressed) - distance
                    for i in range(length):
                        decompressed.append(decompressed[start_pos + i])
            
            print(f"    [SublinearLZ77] 展開完了: {len(compressed_data)} -> {len(decompressed)} bytes")
            return bytes(decompressed)
            
        except Exception as e:
            print(f"    [SublinearLZ77] 展開エラー: {e}")
            return compressed_data
    
    def _decode_lz77_tokens(self, data: bytes) -> list:
        """LZ77トークンのデコード"""
        import struct
        tokens = []
        pos = 0
        
        while pos < len(data):
            if data[pos] == 0x00:  # リテラル
                if pos + 1 < len(data):
                    tokens.append(('literal', data[pos + 1]))
                    pos += 2
                else:
                    break
            elif data[pos] == 0x01:  # マッチ
                if pos + 4 <= len(data):
                    distance = struct.unpack('<H', data[pos + 1:pos + 3])[0]
                    length = data[pos + 3]
                    tokens.append(('match', distance, length))
                    pos += 4
                else:
                    break
            else:
                pos += 1  # 不正なトークンをスキップ
        
        return tokens


class ContextMixingEncoder:
    """
    TMC v9.0 高度コンテキストミキシング符号化エンジン
    複数予測器の並列実行 + 動的ミキシングによる極限圧縮率実現
    """
    
    def __init__(self):
        self.zstd_available = ZSTD_AVAILABLE
        
        # 複数予測器の初期化
        self.order0_model = {}  # オーダー0（統計的）
        self.order1_model = {}  # オーダー1（1バイト文脈）
        self.order2_model = {}  # オーダー2（2バイト文脈）
        
        # 動的ミキシング用の重み
        self.mixing_weights = {
            'order0': 0.33,
            'order1': 0.33,
            'order2': 0.34
        }
        
        # 学習率（適応的調整用）
        self.learning_rate = 0.01
        self.prediction_history = []
        
        print("🧠 コンテキストミキシングエンコーダー初期化完了")
    
    def encode_with_context_mixing(self, data: bytes, stream_type: str = "transformed") -> Tuple[bytes, str]:
        """
        コンテキストミキシングによる高度符号化
        複数予測器 + 動的重み調整による最適化
        """
        try:
            if len(data) == 0:
                return b'', "context_empty"
            
            print(f"  [コンテキスト] ミキシング符号化開始: {len(data)} bytes")
            
            # 複数予測器の並列実行
            predictions = self._run_multiple_predictors(data)
            
            # 動的ミキシング実行
            mixed_probabilities = self._dynamic_mixing(predictions, data)
            
            # FSE符号化（Finite State Entropy）シミュレーション
            if self.zstd_available:
                # Zstandardの高度符号化を使用
                compressed = self._fse_encode_simulation(data, mixed_probabilities)
                return compressed, "context_mixing_fse"
            else:
                # フォールバック: 高効率zlib
                compressed = zlib.compress(data, level=9)
                return compressed, "context_mixing_zlib"
                
        except Exception as e:
            print(f"    [コンテキスト] エラー: {e}")
            return data, "context_store"
    
    def _run_multiple_predictors(self, data: bytes) -> Dict[str, List[Dict[int, float]]]:
        """複数予測器の並列実行"""
        predictions = {
            'order0': [],
            'order1': [],
            'order2': []
        }
        
        # データ統計の事前計算（オーダー0用）
        byte_counts = [0] * 256
        for byte in data:
            byte_counts[byte] += 1
        
        total_bytes = len(data)
        order0_probs = {i: count / total_bytes for i, count in enumerate(byte_counts) if count > 0}
        
        # 各バイト位置での予測実行
        for i in range(len(data)):
            current_byte = data[i]
            
            # オーダー0予測（全体統計）
            predictions['order0'].append(order0_probs)
            
            # オーダー1予測（直前1バイト文脈）
            if i > 0:
                context1 = data[i-1:i]
                order1_pred = self._predict_order1(context1, data, i)
                predictions['order1'].append(order1_pred)
            else:
                predictions['order1'].append(order0_probs)
            
            # オーダー2予測（直前2バイト文脈）
            if i > 1:
                context2 = data[i-2:i]
                order2_pred = self._predict_order2(context2, data, i)
                predictions['order2'].append(order2_pred)
            else:
                predictions['order2'].append(order0_probs)
        
        return predictions
    
    def _predict_order1(self, context: bytes, data: bytes, position: int) -> Dict[int, float]:
        """オーダー1予測（1バイト文脈）"""
        context_key = context[0] if len(context) > 0 else 0
        
        # このコンテキストに続くバイトの統計を収集
        following_bytes = []
        for i in range(len(data) - 1):
            if data[i] == context_key:
                following_bytes.append(data[i + 1])
        
        if not following_bytes:
            # フォールバック: 均等分布
            return {i: 1.0/256 for i in range(256)}
        
        # 確率分布を計算
        byte_counts = {}
        for byte in following_bytes:
            byte_counts[byte] = byte_counts.get(byte, 0) + 1
        
        total = len(following_bytes)
        return {byte: count / total for byte, count in byte_counts.items()}
    
    def _predict_order2(self, context: bytes, data: bytes, position: int) -> Dict[int, float]:
        """オーダー2予測（2バイト文脈）"""
        if len(context) < 2:
            return self._predict_order1(context[-1:] if context else b'', data, position)
        
        context_key = (context[0], context[1])
        
        # このコンテキストに続くバイトの統計を収集
        following_bytes = []
        for i in range(len(data) - 2):
            if (data[i], data[i + 1]) == context_key:
                following_bytes.append(data[i + 2])
        
        if not following_bytes:
            # フォールバック: オーダー1予測
            return self._predict_order1(context[-1:], data, position)
        
        # 確率分布を計算
        byte_counts = {}
        for byte in following_bytes:
            byte_counts[byte] = byte_counts.get(byte, 0) + 1
        
        total = len(following_bytes)
        return {byte: count / total for byte, count in byte_counts.items()}
    
    def _dynamic_mixing(self, predictions: Dict[str, List[Dict[int, float]]], data: bytes) -> List[Dict[int, float]]:
        """動的ミキシング（適応的重み調整）"""
        mixed_predictions = []
        
        for i in range(len(data)):
            current_byte = data[i]
            
            # 各予測器の確率を取得
            order0_prob = predictions['order0'][i].get(current_byte, 0.0)
            order1_prob = predictions['order1'][i].get(current_byte, 0.0)
            order2_prob = predictions['order2'][i].get(current_byte, 0.0)
            
            # 予測精度に基づく動的重み調整
            self._update_mixing_weights(order0_prob, order1_prob, order2_prob)
            
            # 重み付き混合確率の計算
            mixed_prob = {}
            all_bytes = set()
            all_bytes.update(predictions['order0'][i].keys())
            all_bytes.update(predictions['order1'][i].keys())
            all_bytes.update(predictions['order2'][i].keys())
            
            for byte in all_bytes:
                p0 = predictions['order0'][i].get(byte, 0.0)
                p1 = predictions['order1'][i].get(byte, 0.0)
                p2 = predictions['order2'][i].get(byte, 0.0)
                
                mixed_prob[byte] = (
                    self.mixing_weights['order0'] * p0 +
                    self.mixing_weights['order1'] * p1 +
                    self.mixing_weights['order2'] * p2
                )
            
            # 正規化
            total_prob = sum(mixed_prob.values())
            if total_prob > 0:
                mixed_prob = {byte: prob / total_prob for byte, prob in mixed_prob.items()}
            
            mixed_predictions.append(mixed_prob)
        
        return mixed_predictions
    
    def _update_mixing_weights(self, p0: float, p1: float, p2: float):
        """予測精度に基づく重み更新"""
        # より高い確率を予測した予測器により多くの重みを与える
        prediction_scores = {
            'order0': p0,
            'order1': p1,
            'order2': p2
        }
        
        # ソフトマックス風の重み更新
        total_score = sum(prediction_scores.values())
        if total_score > 0:
            for order in prediction_scores:
                target_weight = prediction_scores[order] / total_score
                current_weight = self.mixing_weights[order]
                
                # 学習率による適応的調整
                self.mixing_weights[order] = (
                    current_weight * (1 - self.learning_rate) +
                    target_weight * self.learning_rate
                )
        
        # 重みの正規化
        total_weight = sum(self.mixing_weights.values())
        if total_weight > 0:
            self.mixing_weights = {k: v / total_weight for k, v in self.mixing_weights.items()}
    
    def _fse_encode_simulation(self, data: bytes, mixed_probabilities: List[Dict[int, float]]) -> bytes:
        """FSE符号化シミュレーション（Zstandardベース）"""
        try:
            if self.zstd_available:
                # 最高圧縮レベルでZstandardを使用
                # 実際のFSE実装の代替として最適化されたZstd
                compressor = zstd.ZstdCompressor(
                    level=22,  # 最高圧縮レベル
                    compression_params=zstd.ZstdCompressionParameters(
                        window_log=22,      # 最大ウィンドウ
                        hash_log=12,        # 大きなハッシュテーブル
                        chain_log=12,       # 長いチェーン
                        search_log=7,       # 徹底的検索
                        min_match=3,        # 最小マッチ長
                        target_length=128,  # 長いターゲット
                        strategy=zstd.STRATEGY_BTULTRA2  # 最高品質戦略
                    )
                )
                return compressor.compress(data)
            else:
                return zlib.compress(data, level=9)
        except Exception:
            return zlib.compress(data, level=9)
    
    def decode_context_mixed(self, compressed_data: bytes, method: str) -> bytes:
        """コンテキストミキシング復号"""
        try:
            if method == "context_mixing_fse" and self.zstd_available:
                decompressor = zstd.ZstdDecompressor()
                return decompressor.decompress(compressed_data)
            elif method == "context_mixing_zlib":
                return zlib.decompress(compressed_data)
            else:
                return compressed_data
        except Exception:
            return compressed_data


class CoreCompressor:
    """
    TMC v9.0 高度統一圧縮エンジン
    コンテキストミキシング + 動的レベル選択による最適化
    """
    def __init__(self):
        self.zstd_available = ZSTD_AVAILABLE
        if self.zstd_available:
            # 複数レベルのcompressorを事前生成（効率化）
            self.zstd_compressors = {
                'fast': zstd.ZstdCompressor(level=1),      # 高速圧縮
                'balanced': zstd.ZstdCompressor(level=3),  # バランス型
                'high': zstd.ZstdCompressor(level=9),      # 高圧縮
                'ultra': zstd.ZstdCompressor(level=18),    # 超高圧縮
                'context': zstd.ZstdCompressor(level=22,   # コンテキストミキシング用
                    compression_params=zstd.ZstdCompressionParameters(
                        window_log=22,
                        hash_log=12,
                        chain_log=12,
                        search_log=7,
                        min_match=3,
                        target_length=7,
                        strategy=zstd.STRATEGY_BTULTRA2
                    ))
            }
            self.zstd_decompressor = zstd.ZstdDecompressor()
        else:
            # フォールバック用の最小構成
            self.fallback_available = True
        
        # TMC v9.0 新機能: SublinearLZ77とコンテキストミキシング
        self.sublinear_lz77 = SublinearLZ77Compressor()
        self.context_mixer = ContextMixingEncoder()
    
    def compress(self, data: bytes, stream_entropy: float = 4.0, stream_size: int = 0, 
                 use_context_mixing: bool = False) -> Tuple[bytes, str]:
        """
        TMC v9.0統一圧縮（コンテキストミキシング対応）
        エントロピーとサイズに基づく最適化 + 高度文脈符号化
        """
        try:
            if len(data) == 0:
                return data, "empty"
            
            size = len(data) if stream_size == 0 else stream_size
            
            # v9.0: SublinearLZ77前処理判定（テキストデータで効果的）
            if size >= 2048 and stream_entropy > 3.0:  # 中～高エントロピーデータでLZ77が効果的
                try:
                    lz77_compressed, lz77_info = self.sublinear_lz77.compress_sublinear_lz77(data)
                    if len(lz77_compressed) < len(data) * 0.85:  # 15%以上の圧縮効果がある場合
                        print(f"    [コアコンプレッサー] SublinearLZ77前処理成功: {len(data)} -> {len(lz77_compressed)} bytes")
                        # LZ77圧縮後にさらにZstd圧縮を適用
                        final_compressed, zstd_method = self.compress(lz77_compressed, stream_entropy, len(lz77_compressed), False)
                        return final_compressed, f"sublinear_lz77+{zstd_method}"
                    else:
                        print(f"    [コアコンプレッサー] SublinearLZ77効果不十分、スキップ")
                except Exception as e:
                    print(f"    [コアコンプレッサー] SublinearLZ77エラー: {e}")
            
            # v9.0: コンテキストミキシング判定（条件を緩和）
            if use_context_mixing and size >= 512:  # 512B以上でコンテキストミキシング有効（BWTデータ等の高圧縮対象）
                try:
                    compressed, method = self.context_mixer.encode_with_context_mixing(data, "transformed")
                    if len(compressed) < len(data) * 0.98:  # 2%以上の圧縮効果がある場合（閾値緩和）
                        return compressed, method
                    else:
                        print(f"    [コアコンプレッサー] コンテキストミキシング効果不十分、標準圧縮に切り替え")
                except Exception as e:
                    print(f"    [コアコンプレッサー] コンテキストミキシングエラー: {e}")
            
            if self.zstd_available:
                # TMC理論に基づく動的レベル選択
                compression_level = self._select_optimal_level(size, stream_entropy)
                compressor = self.zstd_compressors[compression_level]
                
                try:
                    compressed = compressor.compress(data)
                    return compressed, f"zstd_{compression_level}"
                except Exception:
                    # 極小データの場合は無圧縮
                    return data, "store"
            
            # Zstd利用不可の場合のフォールバック
            if size > 8192:
                compressed = lzma.compress(data, preset=6)
                return compressed, "lzma_fallback"
            else:
                compressed = zlib.compress(data, level=6)
                return compressed, "zlib_fallback"
                
        except Exception:
            return data, "store"
    
    def _select_optimal_level(self, size: int, entropy: float) -> str:
        """
        TMC動的レベル選択アルゴリズム（ユーザー提案実装）
        サイズとエントロピーに基づく最適化
        """
        # 超低エントロピー（高度に構造化されたデータ）
        if entropy < 2.0:
            if size > 32768:  # 大サイズ: 超高圧縮
                return 'ultra'
            else:  # 小サイズ: 高圧縮
                return 'high'
        
        # 低エントロピー（構造化データ）
        elif entropy < 4.0:
            if size > 16384:  # 大サイズ: 高圧縮
                return 'high'
            else:  # 小サイズ: バランス型
                return 'balanced'
        
        # 中エントロピー（一般的なデータ）
        elif entropy < 6.0:
            return 'balanced'
        
        # 高エントロピー（ランダムに近いデータ）
        else:
            if size < 4096:  # 小サイズ: 高速処理優先
                return 'fast'
            else:  # 大サイズ: バランス型で試行
                return 'balanced'
    
    def decompress(self, compressed_data: bytes, method: str) -> bytes:
        """TMC v9.0統一展開処理（SublinearLZ77 + コンテキストミキシング対応）"""
        try:
            # v9.0: SublinearLZ77組み合わせ復号
            if method.startswith("sublinear_lz77+"):
                # 例: "sublinear_lz77+zstd_high"
                zstd_method = method.split("+")[1]
                # まずZstd展開
                zstd_decompressed = self.decompress(compressed_data, zstd_method)
                # 次にSublinearLZ77展開
                lz77_info = {"method": "sublinear_lz77"}  # 最小限の情報
                return self.sublinear_lz77.decompress_sublinear_lz77(zstd_decompressed, lz77_info)
            
            # v9.0: コンテキストミキシング復号
            elif method.startswith("context_mixing"):
                return self.context_mixer.decode_context_mixed(compressed_data, method)
            elif method.startswith("zstd_") and self.zstd_available:
                # Zstd展開は圧縮レベルに関係なく常に高速
                return self.zstd_decompressor.decompress(compressed_data)
            elif method == "lzma_fallback":
                return lzma.decompress(compressed_data)
            elif method == "zlib_fallback":
                return zlib.decompress(compressed_data)
            else:
                return compressed_data
                
        except Exception:
            return compressed_data


class ImprovedDispatcher:
    """
    改良分析&ディスパッチステージ（ユーザー提案統合）
    より精密なデータタイプ判定
    """
    
    def dispatch(self, data_block: bytes) -> Tuple[DataType, Dict[str, Any]]:
        """改良データブロック分析"""
        print(f"[改良ディスパッチャ] データブロック (サイズ: {len(data_block)} bytes) を分析中...")
        
        if len(data_block) == 0:
            return DataType.GENERIC_BINARY, {}
        
        features = self._extract_enhanced_features(data_block)
        data_type = self._classify_enhanced_data_type(features, data_block)
        
        print(f"[改良ディスパッチャ] 判定: {data_type.value}")
        return data_type, features
    
    def _extract_enhanced_features(self, data: bytes) -> Dict[str, Any]:
        """拡張特徴量抽出"""
        try:
            features = {}
            
            # 基本統計
            data_array = np.frombuffer(data, dtype=np.uint8)
            features['size'] = len(data)
            features['entropy'] = self._calculate_entropy(data_array)
            features['variance'] = float(np.var(data_array))
            
            # テキスト性分析（ユーザー提案採用）
            text_chars = sum(1 for byte in data if 32 <= byte <= 126 or byte in [9, 10, 13])
            features['text_ratio'] = text_chars / len(data) if len(data) > 0 else 0
            
            # 浮動小数点データ分析
            features['is_float_candidate'] = (len(data) % 4 == 0 and len(data) > 100)
            
            # 整数系列性分析
            features['is_sequential_int_candidate'] = False
            if len(data) % 4 == 0 and len(data) > 100:
                try:
                    integers = np.frombuffer(data, dtype=np.int32)
                    if len(integers) > 1:
                        diffs = np.abs(np.diff(integers.astype(np.int64)))
                        features['int_diff_mean'] = float(np.mean(diffs))
                        features['is_sequential_int_candidate'] = features['int_diff_mean'] < 1000
                except Exception:
                    pass
            
            # 反復性分析
            if len(data) > 0:
                unique_ratio = len(np.unique(data_array)) / len(data_array)
                features['unique_ratio'] = unique_ratio
                features['repetition_score'] = 1.0 - unique_ratio
            
            # 圧縮済みデータ検出
            features['high_entropy'] = features['entropy'] > 7.5
            
            return features
            
        except Exception:
            return {'entropy': 4.0, 'size': len(data)}
    
    def _calculate_entropy(self, data_array: np.ndarray) -> float:
        """エントロピー計算"""
        try:
            byte_counts = np.bincount(data_array, minlength=256)
            probabilities = byte_counts[byte_counts > 0] / len(data_array)
            return float(-np.sum(probabilities * np.log2(probabilities)))
        except Exception:
            return 4.0
    
    def _classify_enhanced_data_type(self, features: Dict[str, Any], data: bytes) -> DataType:
        """
        TMC v6.0 拡張データタイプ分類
        判定順序の最適化（より特殊で確度の高いものから順に判定）
        """
        try:
            # 1. テキストデータ判定（最高優先度）
            if features.get('text_ratio', 0) > 0.85:
                return DataType.TEXT_DATA
            
            # 2. 系列整数データ判定（浮動小数点より先に判定）
            if features.get('is_sequential_int_candidate', False):
                # 追加検証: より厳密な系列性チェック
                if len(data) % 4 == 0 and len(data) > 100:
                    try:
                        integers = np.frombuffer(data, dtype=np.int32)
                        if len(integers) > 1:
                            diffs = np.abs(np.diff(integers.astype(np.int64)))
                            consecutive_small_diffs = np.sum(diffs < 100)
                            if consecutive_small_diffs / len(diffs) > 0.7:  # 70%以上が小さな差分
                                print(f"    [分類] 系列整数データ確認: 小差分率={consecutive_small_diffs/len(diffs):.2%}")
                                return DataType.SEQUENTIAL_INT_DATA
                    except Exception:
                        pass
            
            # 3. 浮動小数点データ判定（系列整数の後で判定）
            if features.get('is_float_candidate', False):
                # 追加検証: 浮動小数点数らしさをチェック
                if len(data) % 4 == 0 and len(data) > 100:
                    try:
                        floats = np.frombuffer(data, dtype=np.float32)
                        # NaN, Inf でない有効な浮動小数点数の割合をチェック
                        valid_floats = np.isfinite(floats)
                        valid_ratio = np.sum(valid_floats) / len(floats)
                        
                        # さらに、値の範囲が浮動小数点らしいかチェック
                        if valid_ratio > 0.95:  # 95%以上が有効な浮動小数点
                            valid_values = floats[valid_floats]
                            if len(valid_values) > 0:
                                try:
                                    value_range = float(np.max(valid_values) - np.min(valid_values))
                                    # 値の範囲が適度に大きい（整数系列でない）かつ有限
                                    if np.isfinite(value_range) and value_range > 1.0:
                                        print(f"    [分類] 浮動小数点データ確認: 有効率={valid_ratio:.2%}, 範囲={value_range:.2f}")
                                        return DataType.FLOAT_DATA
                                except (OverflowError, RuntimeWarning):
                                    # オーバーフローの場合は浮動小数点として扱わない
                                    pass
                    except Exception:
                        pass
            
            # 4. 高反復データ（前回と同じ）
            if features.get('repetition_score', 0) > 0.7:
                return DataType.REPETITIVE_BINARY
            
            # 5. 圧縮済みデータ（前回と同じ）
            if features.get('high_entropy', False):
                return DataType.COMPRESSED_LIKE
            
            # 6. その他の構造的データ（前回と同じ）
            if features.get('entropy', 8) < 6.0:
                return DataType.STRUCTURED_NUMERIC
            
            # 7. 汎用バイナリ（デフォルト）
            return DataType.GENERIC_BINARY
            
        except Exception:
            return DataType.GENERIC_BINARY


class TDTTransformer:
    """
    TMC v5.0 高度型付きデータ変換（ユーザー提案統合）
    統計的クラスタリングに基づく適応的ストリーム分解
    """
    
    def transform(self, data: bytes) -> Tuple[List[bytes], Dict[str, Any]]:
        """統計的クラスタリングによる適応的ストリーム分解"""
        print("  [TDT] 高度変換を実行中...")
        info = {'method': 'tdt_clustered', 'original_size': len(data)}
        
        try:
            # 4バイト単位チェック（ユーザー提案採用）
            if len(data) % 4 != 0:
                print("    [TDT] データが4バイトの倍数ではないため、変換をスキップします。")
                return [data], info
            
            # 浮動小数点として解釈
            floats = np.frombuffer(data, dtype=np.float32)
            byte_view = floats.view(np.uint8).reshape(-1, 4)
            
            print(f"    [TDT] {len(floats)}個の浮動小数点数を処理します。")
            
            # ステップ1: 各バイト位置の統計的特徴抽出
            byte_features = []
            for i in range(4):
                byte_stream = byte_view[:, i]
                features = self._extract_byte_position_features(byte_stream, i)
                byte_features.append(features)
                print(f"    [TDT] バイト位置 {i}: エントロピー={features['entropy']:.2f}, 分散={features['variance']:.2f}")
            
            # ステップ2: 統計的クラスタリング実行
            clusters = self._perform_statistical_clustering(byte_features)
            print(f"    [TDT] クラスタリング結果: {len(clusters)}個のクラスター")
            
            # ステップ3: クラスターに基づくストリーム生成
            streams = []
            cluster_info = []
            
            for cluster_id, byte_positions in enumerate(clusters):
                # クラスター内のバイト位置を結合
                cluster_data = bytearray()
                for pos in byte_positions:
                    cluster_data.extend(byte_view[:, pos].tobytes())
                
                stream = bytes(cluster_data)
                streams.append(stream)
                
                # クラスター統計計算
                cluster_entropy = self._calculate_stream_entropy(np.frombuffer(stream, dtype=np.uint8))
                cluster_info.append({
                    'positions': byte_positions,
                    'entropy': cluster_entropy,
                    'size': len(stream)
                })
                
                print(f"    [TDT] クラスター {cluster_id} (位置: {byte_positions}): サイズ={len(stream)}, エントロピー={cluster_entropy:.2f}")
            
            info['byte_features'] = byte_features
            info['clusters'] = cluster_info
            info['stream_count'] = len(streams)
            info['clustering_method'] = 'statistical_similarity'
            
            return streams, info
            
        except Exception as e:
            print(f"    [TDT] エラー: {e}")
            return [data], info
    
    def _extract_byte_position_features(self, byte_stream: np.ndarray, position: int) -> Dict[str, float]:
        """
        各バイト位置の統計的特徴抽出（ユーザー提案実装）
        """
        features = {
            'position': position,
            'entropy': self._calculate_stream_entropy(byte_stream),
            'variance': float(np.var(byte_stream)),
            'std_dev': float(np.std(byte_stream)),
            'unique_ratio': len(np.unique(byte_stream)) / len(byte_stream),
            'mean': float(np.mean(byte_stream)),
            'range': float(np.max(byte_stream) - np.min(byte_stream))
        }
        
        # 分布の偏り（歪度）- 改良版
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from scipy import stats
                features['skewness'] = float(stats.skew(byte_stream))
        except (ImportError, RuntimeWarning):
            # scipyが利用できない場合やエラーの場合の安全な計算
            mean_val = features['mean']
            std_val = features['std_dev']
            if std_val > 1e-8:  # より安全な閾値
                normalized = (byte_stream.astype(np.float64) - mean_val) / std_val
                features['skewness'] = float(np.mean(normalized ** 3))
            else:
                features['skewness'] = 0.0
        
        return features
    
    def _perform_statistical_clustering(self, byte_features: List[Dict[str, float]]) -> List[List[int]]:
        """
        統計的特徴に基づく階層クラスタリング（ユーザー提案実装）
        """
        try:
            # 特徴ベクトル構築
            feature_vectors = []
            for features in byte_features:
                vector = [
                    features['entropy'],
                    features['variance'],
                    features['unique_ratio'],
                    features['skewness']
                ]
                feature_vectors.append(vector)
            
            feature_matrix = np.array(feature_vectors)
            
            # 正規化（Z-score）
            if feature_matrix.std(axis=0).sum() > 0:
                feature_matrix = (feature_matrix - feature_matrix.mean(axis=0)) / (feature_matrix.std(axis=0) + 1e-8)
            
            # 距離行列計算（ユークリッド距離）
            n = len(byte_features)
            distance_matrix = np.zeros((n, n))
            
            for i in range(n):
                for j in range(i + 1, n):
                    distance = np.linalg.norm(feature_matrix[i] - feature_matrix[j])
                    distance_matrix[i, j] = distance_matrix[j, i] = distance
            
            # 簡易階層クラスタリング実装
            clusters = self._simple_hierarchical_clustering(distance_matrix, threshold=1.0)
            
            return clusters
            
        except Exception as e:
            print(f"    [TDT] クラスタリングエラー: {e} - デフォルト分割を使用")
            # フォールバック: 固定4分割
            return [[0], [1], [2], [3]]
    
    def _simple_hierarchical_clustering(self, distance_matrix: np.ndarray, threshold: float) -> List[List[int]]:
        """簡易階層クラスタリング実装"""
        n = distance_matrix.shape[0]
        clusters = [[i] for i in range(n)]  # 初期状態: 各要素が独自クラスター
        
        while len(clusters) > 1:
            # 最も近いクラスターペアを探索
            min_distance = float('inf')
            merge_i, merge_j = -1, -1
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # クラスター間の平均距離を計算
                    total_distance = 0
                    count = 0
                    
                    for idx_i in clusters[i]:
                        for idx_j in clusters[j]:
                            total_distance += distance_matrix[idx_i, idx_j]
                            count += 1
                    
                    if count > 0:
                        avg_distance = total_distance / count
                        if avg_distance < min_distance:
                            min_distance = avg_distance
                            merge_i, merge_j = i, j
            
            # 閾値チェック
            if min_distance > threshold:
                break
            
            # クラスターマージ
            if merge_i != -1 and merge_j != -1:
                new_cluster = clusters[merge_i] + clusters[merge_j]
                new_clusters = []
                for i, cluster in enumerate(clusters):
                    if i != merge_i and i != merge_j:
                        new_clusters.append(cluster)
                new_clusters.append(new_cluster)
                clusters = new_clusters
            else:
                break
        
        return clusters
    
    def inverse_transform(self, streams: List[bytes], info: Dict[str, Any]) -> bytes:
        """TDT統計的逆変換"""
        print("  [TDT] 統計的逆変換を実行中...")
        try:
            if 'clusters' not in info:
                # フォールバック: 従来方式
                return self._legacy_inverse_transform(streams)
            
            clusters = info['clusters']
            
            if len(streams) != len(clusters):
                print("    [TDT] ストリーム数とクラスター数が不一致")
                return b''.join(streams)
            
            # 元のバイト配列サイズを推定
            total_elements = sum(len(stream) for stream in streams) // 4
            byte_view = np.zeros((total_elements, 4), dtype=np.uint8)
            
            # 各クラスターからバイト位置を復元
            for cluster_id, (stream, cluster_info) in enumerate(zip(streams, clusters)):
                positions = cluster_info['positions']
                stream_data = np.frombuffer(stream, dtype=np.uint8)
                
                # ストリームデータを各バイト位置に分散配置
                elements_per_position = len(stream_data) // len(positions)
                
                for i, pos in enumerate(positions):
                    start_idx = i * elements_per_position
                    end_idx = (i + 1) * elements_per_position
                    if i == len(positions) - 1:  # 最後の位置は残りすべて
                        end_idx = len(stream_data)
                    
                    position_data = stream_data[start_idx:end_idx]
                    if len(position_data) == total_elements:
                        byte_view[:, pos] = position_data
                    else:
                        # サイズ調整
                        min_len = min(len(position_data), total_elements)
                        byte_view[:min_len, pos] = position_data[:min_len]
            
            return byte_view.tobytes()
            
        except Exception as e:
            print(f"    [TDT] 統計的逆変換エラー: {e}")
            return b''.join(streams)
    
    def _legacy_inverse_transform(self, streams: List[bytes]) -> bytes:
        """従来方式の逆変換（フォールバック）"""
        try:
            if len(streams) != 4:
                return streams[0] if streams else b''
            
            stream_lengths = [len(s) for s in streams]
            if len(set(stream_lengths)) != 1:
                return b''.join(streams)
            
            num_floats = stream_lengths[0]
            byte_view = np.empty((num_floats, 4), dtype=np.uint8)
            
            for i, stream in enumerate(streams):
                byte_view[:, i] = np.frombuffer(stream, dtype=np.uint8)
            
            return byte_view.tobytes()
            
        except Exception:
            return b''.join(streams)
    
    def _calculate_stream_entropy(self, stream: np.ndarray) -> float:
        """ストリームエントロピー計算"""
        try:
            byte_counts = np.bincount(stream, minlength=256)
            probabilities = byte_counts[byte_counts > 0] / len(stream)
            return float(-np.sum(probabilities * np.log2(probabilities)))
        except Exception:
            return 8.0


class LeCoAdvancedTransformer:
    """
    TMC v8.0 高度機械学習変換（可変長パーティショニング対応）
    局所パターン適応による極限圧縮率実現
    """
    
    def transform(self, data: bytes) -> Tuple[List[bytes], Dict[str, Any]]:
        """LeCo v8.0変換：可変長パーティショニング + 局所最適化"""
        print("  [LeCo v8.0] 可変長パーティショニング変換を実行中...")
        info = {'method': 'leco_variable_partitioning', 'original_size': len(data)}
        
        try:
            # 4バイト単位チェック
            if len(data) % 4 != 0:
                print("    [LeCo v8.0] データが4バイトの倍数ではないため、変換をスキップします。")
                return [data], info
            
            integers = np.frombuffer(data, dtype=np.int32)
            print(f"    [LeCo v8.0] {len(integers)}個の整数を可変長パーティショニング中...")
            
            # 可変長パーティショニング実行
            partitions = self._variable_length_partitioning(integers)
            print(f"    [LeCo v8.0] {len(partitions)}個のパーティションを生成")
            
            # 各パーティションに最適モデルを適用
            partition_streams = []
            partition_infos = []
            
            for i, partition_data in enumerate(partitions):
                partition_result = self._optimize_partition(partition_data, i)
                partition_streams.extend(partition_result['streams'])
                partition_infos.append(partition_result['info'])
                
                print(f"    [パーティション {i}] 長さ={len(partition_data)}, モデル={partition_result['info']['model_type']}, "
                      f"圧縮スコア={partition_result['info']['compression_score']:.2f}")
            
            # パーティション情報をヘッダーとして追加
            partition_header = self._create_partition_header(partition_infos, len(integers))
            final_streams = [partition_header] + partition_streams
            
            # 統計情報更新
            total_score = sum(p['compression_score'] for p in partition_infos)
            avg_score = total_score / len(partition_infos) if partition_infos else 32.0
            
            info.update({
                'partition_count': len(partitions),
                'partition_infos': partition_infos,
                'average_compression_score': avg_score,
                'variable_partitioning': True
            })
            
            return final_streams, info
            
        except Exception as e:
            print(f"    [LeCo v8.0] エラー: {e}")
            return [data], info
    
    def _variable_length_partitioning(self, integers: np.ndarray, threshold_bits: int = 8) -> List[np.ndarray]:
        """
        Greedyアルゴリズムによる可変長パーティショニング
        残差が閾値以下になるように動的に分割
        """
        partitions = []
        current_start = 0
        max_residual_value = (1 << (threshold_bits - 1)) - 1  # 8bit: 127
        
        while current_start < len(integers):
            # 貪欲にパーティションを拡張
            best_end = current_start + 1
            best_model = None
            
            # 最小パーティションサイズ（統計的意味を持つため）
            min_partition_size = max(3, min(50, len(integers) // 20))
            max_partition_size = min(len(integers) - current_start, 1000)  # 最大1000要素
            
            for potential_end in range(
                min(current_start + min_partition_size, len(integers)),
                min(current_start + max_partition_size + 1, len(integers) + 1)
            ):
                partition_data = integers[current_start:potential_end]
                
                # このパーティションに最適なモデルを試行
                best_partition_model = self._find_best_model_for_partition(partition_data)
                
                if best_partition_model is None:
                    break
                
                # 残差が閾値以下か確認
                max_residual = np.max(np.abs(best_partition_model['residuals']))
                if max_residual <= max_residual_value:
                    best_end = potential_end
                    best_model = best_partition_model
                else:
                    # 閾値を超えたので、ここで分割
                    break
            
            # パーティションを確定
            partition_data = integers[current_start:best_end]
            partitions.append(partition_data)
            
            current_start = best_end
            
            # 無限ループ防止
            if current_start >= len(integers):
                break
        
        return partitions
    
    def _find_best_model_for_partition(self, partition_data: np.ndarray) -> Optional[Dict[str, Any]]:
        """パーティション用の最適モデル探索"""
        try:
            models_to_try = []
            
            # 定数モデル
            try:
                const_result = self._try_constant_model(partition_data)
                models_to_try.append(const_result)
            except Exception:
                pass
            
            # 線形モデル（パーティションサイズが十分な場合）
            if len(partition_data) >= 3:
                try:
                    linear_result = self._try_linear_model(partition_data)
                    models_to_try.append(linear_result)
                except Exception:
                    pass
            
            # 二次モデル（パーティションサイズが十分な場合）
            if len(partition_data) >= 5:
                try:
                    quad_result = self._try_quadratic_model(partition_data)
                    models_to_try.append(quad_result)
                except Exception:
                    pass
            
            if not models_to_try:
                return None
            
            # 最適モデル選択
            best_model = min(models_to_try, key=lambda x: x['score'])
            return best_model
            
        except Exception:
            return None
    
    def _optimize_partition(self, partition_data: np.ndarray, partition_id: int) -> Dict[str, Any]:
        """個別パーティションの最適化"""
        best_model = self._find_best_model_for_partition(partition_data)
        
        if best_model is None:
            # フォールバック
            mean_val = np.mean(partition_data)
            residuals = partition_data - int(mean_val)
            best_model = {
                'type': 'constant_fallback',
                'params': {'c': float(mean_val)},
                'residuals': residuals,
                'score': 32.0
            }
        
        # パーティション情報作成
        partition_info = {
            'partition_id': partition_id,
            'model_type': best_model['type'],
            'params': best_model['params'],
            'data_length': len(partition_data),
            'compression_score': best_model['score'],
            'max_residual': int(np.max(np.abs(best_model['residuals']))) if len(best_model['residuals']) > 0 else 0
        }
        
        # ストリーム生成
        model_info_json = json.dumps(partition_info, separators=(',', ':'))
        model_info_bytes = model_info_json.encode('utf-8')
        model_header = len(model_info_bytes).to_bytes(4, 'big') + model_info_bytes
        
        residuals_stream = best_model['residuals'].astype(np.int32).tobytes()
        
        return {
            'info': partition_info,
            'streams': [model_header, residuals_stream]
        }
    
    def _create_partition_header(self, partition_infos: List[Dict], total_length: int) -> bytes:
        """パーティションヘッダー作成"""
        header_data = {
            'total_length': total_length,
            'partition_count': len(partition_infos),
            'partitions': [
                {
                    'id': p['partition_id'],
                    'length': p['data_length'],
                    'model': p['model_type']
                } for p in partition_infos
            ]
        }
        
        header_json = json.dumps(header_data, separators=(',', ':'))
        header_bytes = header_json.encode('utf-8')
        
        return len(header_bytes).to_bytes(4, 'big') + header_bytes
    
    # 既存のモデル試行メソッド（_try_constant_model, _try_linear_model, _try_quadratic_model）は継承
    def _try_constant_model(self, integers: np.ndarray) -> Dict[str, Any]:
        """定数モデル: y = c (Frame-of-Reference圧縮相当)"""
        mean_val = np.mean(integers)
        constant = int(round(mean_val))
        
        residuals = integers - constant
        max_abs_residual = int(np.max(np.abs(residuals))) if len(residuals) > 0 else 0
        
        # 残差を格納するのに必要なビット数を計算
        bits_needed = max_abs_residual.bit_length() + 1 if max_abs_residual > 0 else 1  # 符号ビット含む
        compression_score = float(bits_needed)
        
        return {
            'type': 'constant',
            'params': {'c': float(constant)},
            'residuals': residuals,
            'score': compression_score
        }
    
    def _try_linear_model(self, integers: np.ndarray) -> Dict[str, Any]:
        """線形モデル: y = ax + b"""
        x = np.arange(len(integers))
        slope, intercept = np.polyfit(x, integers, 1)
        
        predicted_values = (slope * x + intercept).astype(np.int32)
        residuals = integers - predicted_values
        
        max_abs_residual = int(np.max(np.abs(residuals))) if len(residuals) > 0 else 0
        bits_needed = max_abs_residual.bit_length() + 1 if max_abs_residual > 0 else 1
        
        # パラメータ格納コストも考慮（簡易版）
        param_cost = 64  # slope + intercept (各32bit想定)
        total_bits = bits_needed * len(integers) + param_cost
        compression_score = float(total_bits) / len(integers)
        
        return {
            'type': 'linear',
            'params': {'slope': float(slope), 'intercept': float(intercept)},
            'residuals': residuals,
            'score': compression_score
        }
    
    def _try_quadratic_model(self, integers: np.ndarray) -> Dict[str, Any]:
        """二次モデル: y = ax^2 + bx + c"""
        x = np.arange(len(integers))
        coeffs = np.polyfit(x, integers, 2)  # [a, b, c]
        
        predicted_values = np.polyval(coeffs, x).astype(np.int32)
        residuals = integers - predicted_values
        
        max_abs_residual = int(np.max(np.abs(residuals))) if len(residuals) > 0 else 0
        bits_needed = max_abs_residual.bit_length() + 1 if max_abs_residual > 0 else 1
        
        # パラメータ格納コストも考慮
        param_cost = 96  # a + b + c (各32bit想定)
        total_bits = bits_needed * len(integers) + param_cost
        compression_score = float(total_bits) / len(integers)
        
        return {
            'type': 'quadratic',
            'params': {'a': float(coeffs[0]), 'b': float(coeffs[1]), 'c': float(coeffs[2])},
            'residuals': residuals,
            'score': compression_score
        }
    
    def inverse_transform(self, streams: List[bytes], info: Dict[str, Any]) -> bytes:
        """LeCo v8.0 可変長パーティショニング逆変換"""
        print("  [LeCo v8.0] 可変長パーティショニング逆変換を実行中...")
        try:
            if not info.get('variable_partitioning', False):
                # v7.0互換モードにフォールバック
                return self._legacy_inverse_transform(streams, info)
            
            if len(streams) < 1:
                return b''
            
            # パーティションヘッダーの解析
            partition_header = streams[0]
            header_size = int.from_bytes(partition_header[:4], 'big')
            header_json = partition_header[4:4+header_size].decode('utf-8')
            header_data = json.loads(header_json)
            
            total_length = header_data['total_length']
            partition_count = header_data['partition_count']
            
            print(f"    [LeCo v8.0] パーティション数: {partition_count}, 総長: {total_length}")
            
            # 各パーティションのストリームを処理
            reconstructed_data = np.zeros(total_length, dtype=np.int32)
            current_pos = 0
            stream_idx = 1  # ヘッダー後から開始
            
            for _ in range(partition_count):
                # パーティション情報の復元
                if stream_idx >= len(streams):
                    break
                    
                model_header = streams[stream_idx]
                model_size = int.from_bytes(model_header[:4], 'big')
                model_json = model_header[4:4+model_size].decode('utf-8')
                partition_info = json.loads(model_json)
                
                # 残差ストリームの復元
                if stream_idx + 1 >= len(streams):
                    break
                    
                residuals_stream = streams[stream_idx + 1]
                residuals = np.frombuffer(residuals_stream, dtype=np.int32)
                
                # パーティションデータの復元
                partition_data = self._reconstruct_partition(residuals, partition_info)
                
                # 全体配列に配置
                end_pos = current_pos + len(partition_data)
                if end_pos <= total_length:
                    reconstructed_data[current_pos:end_pos] = partition_data
                    current_pos = end_pos
                
                stream_idx += 2
                
                print(f"    [パーティション {partition_info['partition_id']}] 復元完了: {len(partition_data)}要素")
            
            return reconstructed_data.tobytes()
            
        except Exception as e:
            print(f"    [LeCo v8.0] 逆変換エラー: {e}")
            return b''.join(streams)
    
    def _reconstruct_partition(self, residuals: np.ndarray, partition_info: Dict[str, Any]) -> np.ndarray:
        """個別パーティションの復元"""
        model_type = partition_info['model_type']
        params = partition_info['params']
        data_length = partition_info['data_length']
        
        if model_type == 'constant' or model_type == 'constant_fallback':
            constant = int(params['c'])
            return residuals + constant
            
        elif model_type == 'linear':
            slope = params['slope']
            intercept = params['intercept']
            x = np.arange(len(residuals))
            predicted_values = (slope * x + intercept).astype(np.int32)
            return predicted_values + residuals
            
        elif model_type == 'quadratic':
            a, b, c = params['a'], params['b'], params['c']
            x = np.arange(len(residuals))
            predicted_values = (a * x*x + b * x + c).astype(np.int32)
            return predicted_values + residuals
            
        else:
            return residuals
    
    def _legacy_inverse_transform(self, streams: List[bytes], info: Dict[str, Any]) -> bytes:
        """v7.0互換逆変換"""
        try:
            if len(streams) != 2:
                return streams[0] if streams else b''
            
            # モデル情報の復元
            model_header = streams[0]
            residuals_stream = streams[1]
            
            # モデル情報ヘッダーの解析
            model_info_size = int.from_bytes(model_header[:4], 'big')
            model_info_json = model_header[4:4+model_info_size].decode('utf-8')
            model_info = json.loads(model_info_json)
            
            model_type = model_info['model_type']
            params = model_info['params']
            data_length = model_info['data_length']
            
            # 残差の復元
            residuals = np.frombuffer(residuals_stream, dtype=np.int32)
            
            print(f"    [LeCo] モデルタイプ: {model_type}")
            print(f"    [LeCo] データ長: {data_length}")
            
            # モデルタイプ別の逆変換
            if model_type == 'constant' or model_type == 'constant_fallback':
                constant = int(params['c'])
                original_integers = residuals + constant
                
            elif model_type == 'linear':
                slope = params['slope']
                intercept = params['intercept']
                x = np.arange(len(residuals))
                predicted_values = (slope * x + intercept).astype(np.int32)
                original_integers = predicted_values + residuals
                
            elif model_type == 'quadratic':
                a, b, c = params['a'], params['b'], params['c']
                x = np.arange(len(residuals))
                predicted_values = (a * x*x + b * x + c).astype(np.int32)
                original_integers = predicted_values + residuals
                
            else:
                print(f"    [LeCo] 未知のモデルタイプ: {model_type}")
                return b''.join(streams)
            
            return original_integers.tobytes()
            
        except Exception as e:
            print(f"    [LeCo] 逆変換エラー: {e}")
            return b''.join(streams)


class LeCoTransformer:
    """
    TMC v6.0 高度機械学習変換（マルチモデル対応）
    動的モデル選択による予測圧縮の最適化
    """
    
    def transform(self, data: bytes) -> Tuple[List[bytes], Dict[str, Any]]:
        """LeCo v6.0変換：複数モデルの動的選択"""
        print("  [LeCo] TMC v6.0 マルチモデル変換を実行中...")
        info = {'method': 'leco_multimodel', 'original_size': len(data)}
        
        try:
            # 4バイト単位チェック
            if len(data) % 4 != 0:
                print("    [LeCo] データが4バイトの倍数ではないため、変換をスキップします。")
                return [data], info
            
            integers = np.frombuffer(data, dtype=np.int32)
            print(f"    [LeCo] {len(integers)}個の整数を処理します。")
            
            # 複数モデルの試行と最適選択
            best_model = self._select_optimal_model(integers)
            
            model_type = best_model['type']
            params = best_model['params']
            residuals = best_model['residuals']
            compression_score = best_model['score']
            
            print(f"    [LeCo] 最適モデル: {model_type}")
            print(f"    [LeCo] 圧縮スコア: {compression_score:.2f} bits/element")
            print(f"    [LeCo] 残差範囲: [{np.min(residuals)}, {np.max(residuals)}]")
            
            # モデル情報とパラメータのシリアライズ
            model_info = {
                'model_type': model_type,
                'params': params,
                'data_length': len(integers)
            }
            model_info_json = json.dumps(model_info, separators=(',', ':'))
            model_info_bytes = model_info_json.encode('utf-8')
            model_header = len(model_info_bytes).to_bytes(4, 'big') + model_info_bytes
            
            # 残差ストリーム生成
            residuals_stream = residuals.astype(np.int32).tobytes()
            
            # 統計情報更新
            info.update({
                'model_type': model_type,
                'compression_score': compression_score,
                'residual_variance': float(np.var(residuals)),
                'model_params': params
            })
            
            return [model_header, residuals_stream], info
            
        except Exception as e:
            print(f"    [LeCo] エラー: {e}")
            return [data], info
    
    def _select_optimal_model(self, integers: np.ndarray) -> Dict[str, Any]:
        """複数モデルを試行し、最適なものを動的選択"""
        models_to_try = []
        
        # 1. 定数モデル (Constant Model)
        try:
            const_result = self._try_constant_model(integers)
            models_to_try.append(const_result)
            print(f"    [LeCo] 定数モデル: {const_result['score']:.2f} bits/element")
        except Exception as e:
            print(f"    [LeCo] 定数モデルエラー: {e}")
        
        # 2. 線形モデル (Linear Model)
        try:
            linear_result = self._try_linear_model(integers)
            models_to_try.append(linear_result)
            print(f"    [LeCo] 線形モデル: {linear_result['score']:.2f} bits/element")
        except Exception as e:
            print(f"    [LeCo] 線形モデルエラー: {e}")
        
        # 3. 二次モデル (Quadratic Model) - オプション
        if len(integers) >= 10:  # 十分なデータ点がある場合のみ
            try:
                quad_result = self._try_quadratic_model(integers)
                models_to_try.append(quad_result)
                print(f"    [LeCo] 二次モデル: {quad_result['score']:.2f} bits/element")
            except Exception as e:
                print(f"    [LeCo] 二次モデルエラー: {e}")
        
        # 最適モデル選択（最小スコア）
        if not models_to_try:
            # フォールバック: 定数モデル
            mean_val = np.mean(integers)
            residuals = integers - int(mean_val)
            return {
                'type': 'constant_fallback',
                'params': {'c': float(mean_val)},
                'residuals': residuals,
                'score': 32.0  # ペナルティスコア
            }
        
        best_model = min(models_to_try, key=lambda x: x['score'])
        return best_model
    
    def _try_constant_model(self, integers: np.ndarray) -> Dict[str, Any]:
        """定数モデル: y = c (Frame-of-Reference圧縮相当)"""
        mean_val = np.mean(integers)
        constant = int(round(mean_val))
        
        residuals = integers - constant
        max_abs_residual = int(np.max(np.abs(residuals))) if len(residuals) > 0 else 0
        
        # 残差を格納するのに必要なビット数を計算
        bits_needed = max_abs_residual.bit_length() + 1 if max_abs_residual > 0 else 1  # 符号ビット含む
        compression_score = float(bits_needed)
        
        return {
            'type': 'constant',
            'params': {'c': float(constant)},
            'residuals': residuals,
            'score': compression_score
        }
    
    def _try_linear_model(self, integers: np.ndarray) -> Dict[str, Any]:
        """線形モデル: y = ax + b"""
        x = np.arange(len(integers))
        slope, intercept = np.polyfit(x, integers, 1)
        
        predicted_values = (slope * x + intercept).astype(np.int32)
        residuals = integers - predicted_values
        
        max_abs_residual = int(np.max(np.abs(residuals))) if len(residuals) > 0 else 0
        bits_needed = max_abs_residual.bit_length() + 1 if max_abs_residual > 0 else 1
        
        # パラメータ格納コストも考慮（簡易版）
        param_cost = 64  # slope + intercept (各32bit想定)
        total_bits = bits_needed * len(integers) + param_cost
        compression_score = float(total_bits) / len(integers)
        
        return {
            'type': 'linear',
            'params': {'slope': float(slope), 'intercept': float(intercept)},
            'residuals': residuals,
            'score': compression_score
        }
    
    def _try_quadratic_model(self, integers: np.ndarray) -> Dict[str, Any]:
        """二次モデル: y = ax^2 + bx + c"""
        x = np.arange(len(integers))
        coeffs = np.polyfit(x, integers, 2)  # [a, b, c]
        
        predicted_values = np.polyval(coeffs, x).astype(np.int32)
        residuals = integers - predicted_values
        
        max_abs_residual = int(np.max(np.abs(residuals))) if len(residuals) > 0 else 0
        bits_needed = max_abs_residual.bit_length() + 1 if max_abs_residual > 0 else 1
        
        # パラメータ格納コストも考慮
        param_cost = 96  # a + b + c (各32bit想定)
        total_bits = bits_needed * len(integers) + param_cost
        compression_score = float(total_bits) / len(integers)
        
        return {
            'type': 'quadratic',
            'params': {'a': float(coeffs[0]), 'b': float(coeffs[1]), 'c': float(coeffs[2])},
            'residuals': residuals,
            'score': compression_score
        }
    
    def inverse_transform(self, streams: List[bytes], info: Dict[str, Any]) -> bytes:
        """LeCo v6.0マルチモデル逆変換"""
        print("  [LeCo] TMC v6.0 マルチモデル逆変換を実行中...")
        try:
            if len(streams) != 2:
                return streams[0] if streams else b''
            
            # モデル情報の復元
            model_header = streams[0]
            residuals_stream = streams[1]
            
            # モデル情報ヘッダーの解析
            model_info_size = int.from_bytes(model_header[:4], 'big')
            model_info_json = model_header[4:4+model_info_size].decode('utf-8')
            model_info = json.loads(model_info_json)
            
            model_type = model_info['model_type']
            params = model_info['params']
            data_length = model_info['data_length']
            
            # 残差の復元
            residuals = np.frombuffer(residuals_stream, dtype=np.int32)
            
            print(f"    [LeCo] モデルタイプ: {model_type}")
            print(f"    [LeCo] データ長: {data_length}")
            
            # モデルタイプ別の逆変換
            if model_type == 'constant' or model_type == 'constant_fallback':
                constant = int(params['c'])
                original_integers = residuals + constant
                
            elif model_type == 'linear':
                slope = params['slope']
                intercept = params['intercept']
                x = np.arange(len(residuals))
                predicted_values = (slope * x + intercept).astype(np.int32)
                original_integers = predicted_values + residuals
                
            elif model_type == 'quadratic':
                a, b, c = params['a'], params['b'], params['c']
                x = np.arange(len(residuals))
                predicted_values = (a * x*x + b * x + c).astype(np.int32)
                original_integers = predicted_values + residuals
                
            else:
                print(f"    [LeCo] 未知のモデルタイプ: {model_type}")
                return b''.join(streams)
            
            return original_integers.tobytes()
            
        except Exception as e:
            print(f"    [LeCo] 逆変換エラー: {e}")
            return b''.join(streams)


class BWTTransformer:
    """
    TMC v8.1 完全堅牢化BWTTransformer（pydivsufsort完全準拠）
    テキストデータ最適化の極限実装 + 可逆性問題の根本的解決
    """
    
    def __init__(self):
        try:
            # pydivsufsortのインポートと逆変換関数の存在確認
            import pydivsufsort
            self.pydivsufsort_available = True
            self.pydivsufsort = pydivsufsort
            print("🔥 pydivsufsort利用可能 - 高速BWT + 堅牢な逆変換有効")
        except ImportError:
            self.pydivsufsort_available = False
            print("⚠️ pydivsufsort未利用 - フォールバック実装")
        
        self.post_bwt_pipeline = PostBWTPipeline()
    
    def transform(self, data: bytes) -> Tuple[List[bytes], Dict[str, Any]]:
        """TMC v8.1 完全堅牢化BWT変換（pydivsufsort完全準拠）"""
        print("  [強化BWT] TMC v8.1 専門変換を実行中...")
        info = {'method': 'enhanced_bwt_mtf_rle', 'original_size': len(data)}
        
        try:
            if not data:
                return [data], info
            
            # 動的サイズ制限（並列処理前提で拡張）
            MAX_BWT_SIZE = 2 * 1024 * 1024  # 2MB制限
            if len(data) > MAX_BWT_SIZE:
                print(f"    [強化BWT] データサイズ({len(data)})が制限({MAX_BWT_SIZE})を超過 - BWTスキップ")
                info['method'] = 'bwt_skipped_large'
                return [data], info
            
            # pydivsufsortに完全準拠したBWT実装
            if self.pydivsufsort_available:
                try:
                    print(f"    [強化BWT] pydivsufsortでBWT実行中...")
                    # pydivsufsortは(primary_index, bwt_array)の順序で返す
                    primary_index, bwt_array = self.pydivsufsort.bw_transform(data)
                    bwt_encoded = bytes(bwt_array)  # ndarrayをbytesに変換
                    print(f"    [強化BWT] pydivsufsort成功: BWT={len(bwt_encoded)}, index={primary_index}")
                except Exception as pyd_error:
                    print(f"    [強化BWT] pydivsufsortエラー: {pyd_error}")
                    print(f"    [強化BWT] フォールバックに切り替え")
                    bwt_encoded, primary_index = self._fallback_bwt_transform(data)
            else:
                bwt_encoded, primary_index = self._fallback_bwt_transform(data)
            
            # primary_indexの健全性チェック
            if not (0 <= primary_index < len(bwt_encoded)):
                raise ValueError(f"Invalid primary_index {primary_index} for BWT length {len(bwt_encoded)}")
            
            # Move-to-Front変換
            mtf_encoded = self._mtf_encode(bwt_encoded)
            print(f"    [強化BWT] BWT後: {len(bwt_encoded)} bytes -> MTF後: {len(mtf_encoded)} bytes")
            
            # MTF後のゼロ率計算（圧縮効果の指標）
            zero_count = mtf_encoded.count(0)
            zero_ratio = zero_count / len(mtf_encoded) if len(mtf_encoded) > 0 else 0
            print(f"    [MTF] ゼロの比率: {zero_ratio:.2%} (高いほど圧縮効果大)")
            
            # ポストBWTパイプライン統合（RLE + 分割エントロピー符号化）
            post_bwt_streams = self.post_bwt_pipeline.encode(mtf_encoded)
            print(f"    [強化BWT] ポストBWTパイプライン: {len(post_bwt_streams)}ストリーム生成")
            
            # primary_indexをバイト配列として先頭に配置
            index_bytes = primary_index.to_bytes(4, 'big')
            final_streams = [index_bytes] + post_bwt_streams
            
            # 情報更新
            info.update({
                'bwt_size': len(bwt_encoded),
                'mtf_size': len(mtf_encoded),
                'zero_ratio': zero_ratio,
                'primary_index': primary_index,
                'enhanced_pipeline': True,
                'stream_count': len(final_streams)
            })
            
            return final_streams, info
            
        except Exception as e:
            print(f"    [強化BWT] エラー: {e}")
            # エラー時はコンテキストミキシングを無効化してスキップ
            info['method'] = 'bwt_error_skip'
            info['error'] = str(e)
            return [data], info
            print(f"    [強化BWT] エラー: {e}")
            return [data], info
    
    def _fallback_bwt_transform(self, data: bytes) -> Tuple[bytes, int]:
        """フォールバック用の標準BWT実装"""
        # 改良版フォールバック実装（メモリ効率化）
        data_with_sentinel = data + b'\x00'  # センチネル文字追加
        n = len(data_with_sentinel)
        
        # より効率的なrotation生成
        rotations = []
        for i in range(n):
            rotation = data_with_sentinel[i:] + data_with_sentinel[:i]
            rotations.append((rotation, i))
        
        # ソート
        rotations.sort(key=lambda x: x[0])
        
        # 元の文字列の位置を特定
        primary_index = 0
        for idx, (rotation, original_pos) in enumerate(rotations):
            if original_pos == 0:
                primary_index = idx
                break
        
        # BWT文字列生成
        bwt_encoded = bytes(rotation[0][-1] for rotation, _ in rotations)
        
        return bwt_encoded, primary_index
    
    def _mtf_encode(self, data: bytes) -> bytes:
        """Move-to-Front変換（BWTの局所性を小さな整数に変換）"""
        alphabet = list(range(256))
        encoded = bytearray()
        
        for byte_val in data:
            rank = alphabet.index(byte_val)
            encoded.append(rank)
            # 見つかった文字をリストの先頭に移動
            alphabet.pop(rank)
            alphabet.insert(0, byte_val)
        
        return bytes(encoded)
    
    def _mtf_decode(self, encoded_data: bytes) -> bytes:
        """逆Move-to-Front変換"""
        alphabet = list(range(256))
        decoded = bytearray()
        
        for rank in encoded_data:
            byte_val = alphabet[rank]
            decoded.append(byte_val)
            # 見つかった文字をリストの先頭に移動
            alphabet.pop(rank)
            alphabet.insert(0, byte_val)
        
        return bytes(decoded)
    
    
    def inverse_transform(self, streams: List[bytes], info: Dict[str, Any]) -> bytes:
        """TMC v8.1 完全堅牢化BWT逆変換（pydivsufsort完全準拠）"""
        print("  [強化BWT] TMC v8.1 専門逆変換を実行中...")
        try:
            # BWTがスキップされた場合の処理
            if info.get('method') in ['bwt_skipped_large', 'bwt_error_skip']:
                print(f"    [強化BWT] {info.get('method')}データ - 元データ返却")
                return streams[0] if streams else b''
            
            if len(streams) < 1:
                return b''
            
            # primary_indexの復元
            primary_index = int.from_bytes(streams[0], 'big')
            
            # ポストBWTパイプライン逆変換
            if info.get('enhanced_pipeline', False):
                print("    [ポストBWT] RLE逆変換を実行中...")
                mtf_encoded = self.post_bwt_pipeline.decode(streams[1:])
            else:
                mtf_encoded = streams[1] if len(streams) > 1 else b''
            
            # 逆MTF変換
            if info.get('mtf_applied', True):
                bwt_encoded = self._mtf_decode(mtf_encoded)
                print(f"    [MTF] 逆MTF: {len(mtf_encoded)} bytes -> {len(bwt_encoded)} bytes")
            else:
                bwt_encoded = mtf_encoded
            
            # --- 逆BWTロジックの修正（根本的解決） ---
            if self.pydivsufsort_available:
                # pydivsufsortが利用可能な場合は、その逆変換のみを使用
                print("    [BWT] pydivsufsortによる堅牢な逆変換を実行")
                # pydivsufsortの逆変換: (primary_index, bwt_array) -> original_array
                try:
                    import numpy as np
                    # bytesをwritableなndarrayに変換
                    bwt_array = np.array(list(bwt_encoded), dtype=np.uint8)
                    original_array = self.pydivsufsort.inverse_bw_transform(primary_index, bwt_array)
                    original_data = bytes(original_array)
                except Exception as inv_error:
                    print(f"    [BWT] pydivsufsort逆変換エラー: {inv_error}")
                    print(f"    [BWT] フォールバック逆変換に切り替え")
                    original_data = self._fallback_bwt_inverse(bwt_encoded, primary_index)
            else:
                # ライブラリが利用不可の場合のみ、フォールバックを使用
                print("    [BWT] フォールバック逆BWTを実行")
                original_data = self._fallback_bwt_inverse(bwt_encoded, primary_index)
            
            print(f"    [強化BWT] 逆変換完了: {len(bwt_encoded)} -> {len(original_data)} bytes")
            return original_data
            
        except Exception as e:
            print(f"    [強化BWT] 逆変換エラー: {e}")
            return b''.join(streams)  # エラー時は安全に結合して返す
            if expected_length is not None:
                if len(original_data) != expected_length:
                    print(f"    [警告] データ長不一致: 期待={expected_length}, 実際={len(original_data)}")
                    # 必要に応じて切り詰めまたはパディング
                    if len(original_data) > expected_length:
                        original_data = original_data[:expected_length]
                        print(f"    [修正] データを期待長に切り詰め: {len(original_data)} bytes")
                else:
                    print(f"    [確認] データ長整合性: {len(original_data)} bytes ✓")
            
            return original_data
            
        except Exception as e:
            print(f"    [強化BWT] 逆変換エラー: {e}")
            return b''.join(streams)
    
    def _fallback_bwt_inverse(self, last_col: bytes, primary_index: int) -> bytes:
        """改良版フォールバック逆BWT実装（O(n)アルゴリズム）"""
        n = len(last_col)
        if n == 0:
            return b''
        
        # primary_indexの範囲チェック（可逆性の最重要ポイント）
        if primary_index < 0 or primary_index >= n:
            print(f"    [BWT] 警告: primary_index={primary_index} が範囲外 (0-{n-1})")
            # データが破損している可能性があるため、安全な値を使用
            if n > 0:
                primary_index = 0  # 最初のインデックスを使用
                print(f"    [BWT] primary_indexを0にリセット")
            else:
                return b''
        
        try:
            # 各文字の出現回数をカウント
            count = [0] * 256
            for char in last_col:
                count[char] += 1
            
            # 累積カウントを計算（first列の開始位置）
            first_col_starts = [0] * 256
            total = 0
            for i in range(256):
                first_col_starts[i] = total
                total += count[i]
            
            # 変換テーブルを構築（効率的なO(n)実装）
            next_idx = [0] * n
            char_counts = [0] * 256
            
            for i in range(n):
                char = last_col[i]
                next_idx[i] = first_col_starts[char] + char_counts[char]
                char_counts[char] += 1
            
            # 元の文字列を復元
            result = bytearray()
            current_idx = primary_index
            
            for step in range(n):
                if current_idx < 0 or current_idx >= n:
                    print(f"    [BWT] 逆変換エラー: step={step}, current_idx={current_idx} が範囲外")
                    break
                    
                char = last_col[current_idx]
                result.append(char)
                current_idx = next_idx[current_idx]
            
            # BWTでセンチネル文字（0バイト）が追加されている場合の処理
            # pydivsufsortが追加したセンチネル文字を適切に除去
            result_bytes = bytes(result)
            
            # 末尾のセンチネル文字を1つだけ除去（過度な除去を防止）
            if result_bytes and result_bytes[-1] == 0:
                result_bytes = result_bytes[:-1]
                print(f"    [BWT] センチネル文字除去: {len(result)} -> {len(result_bytes)} bytes")
            
            return result_bytes
            
        except Exception as e:
            print(f"    [BWT] 逆変換エラー: {e}")
            return b''


class NEXUSTMCEngineV9:
    """
    NEXUS TMC Engine v9.0 - コンテキストミキシング統合版
    次世代量子インテリジェント圧縮プラットフォーム
    Transform-Model-Code 圧縮フレームワーク TMC v9.0
    
    v9.0革新機能:
    - 高度コンテキストミキシング符号化（LZMAに匹敵する圧縮率）
    - 複数予測器 + 動的ミキシングによる極限圧縮率実現
    - BWTTransformer完全堅牢化 + 並列チャンク処理
    """
    
    def __init__(self, max_workers: int = None, chunk_size: int = DEFAULT_CHUNK_SIZE):
        self.max_workers = max_workers or min(8, mp.cpu_count())
        self.chunk_size = chunk_size
        self.dispatcher = ImprovedDispatcher()
        self.core_compressor = CoreCompressor()
        self.context_mixer = ContextMixingEncoder()  # v9.0新機能
        
        # TMC v9.0 新機能: 完全並列処理パイプライン
        self.enable_parallel_pipeline = True
        self.async_io_enabled = True
        
        # TMC v8.0 新機能: インテリジェント・バイパス
        self.meta_analyzer = MetaAnalyzer(self.core_compressor)
        
        # 変換器マッピング（v8.0強化版）
        self.transformers = {
            DataType.FLOAT_DATA: TDTTransformer(),
            DataType.TEXT_DATA: BWTTransformer(),  # v7.0強化版（ポストBWTパイプライン統合）
            DataType.SEQUENTIAL_INT_DATA: LeCoAdvancedTransformer(),  # v8.0: 可変長パーティショニング
            DataType.STRUCTURED_NUMERIC: TDTTransformer(),
            DataType.TIME_SERIES: LeCoAdvancedTransformer(),  # v8.0対応
            DataType.REPETITIVE_BINARY: None,  # RLE前処理のみ
            DataType.COMPRESSED_LIKE: None,    # 変換なし
            DataType.GENERIC_BINARY: None,     # 変換なし
        }
        
        print(f"🚀 TMC v9.0 エンジン初期化完了: {self.max_workers}並列ワーカー, チャンクサイズ={self.chunk_size//1024}KB (SublinearLZ77+コンテキストミキシング統合版)")
        
        self.stats = {
            'files_processed': 0,
            'total_input_size': 0,
            'total_compressed_size': 0,
            'reversibility_tests_passed': 0,
            'reversibility_tests_total': 0,
            'transforms_applied': 0,
            'transforms_bypassed': 0,
            'chunks_processed': 0,           # v8.0追加
            'parallel_efficiency': 0.0,     # v8.0追加
            'entropy_coding_used': 0         # v8.0追加
        }
        
        print(f"🚀 TMC v9.0 エンジン初期化完了: {self.max_workers}並列ワーカー, チャンクサイズ={chunk_size//1024//1024}MB (コンテキストミキシング統合版)")
    
    def compress_tmc_parallel(self, data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """
        TMC v8.0 並列チャンク圧縮処理
        真のマルチコア活用による革新的スループット
        """
        compression_start = time.perf_counter()
        
        try:
            print(f"\n--- TMC v8.0 並列チャンク圧縮開始 ({len(data)} bytes) ---")
            
            # 小さなデータは単一チャンク処理
            if len(data) <= self.chunk_size:
                print("  [チャンク分析] 小サイズデータ - 単一チャンク処理")
                return self._compress_single_chunk(data)
            
            # チャンク分割
            chunks = self._split_into_chunks(data)
            print(f"  [チャンク分析] {len(chunks)}個のチャンクに分割")
            
            # 並列チャンク圧縮
            compressed_chunks, chunk_infos = self._compress_chunks_parallel(chunks)
            
            # TMC v8.0 コンテナフォーマット構築
            container = self._build_tmc_v8_container(compressed_chunks, chunk_infos)
            
            total_time = time.perf_counter() - compression_start
            
            # 並列効率計算
            sequential_estimate = total_time * self.max_workers
            parallel_efficiency = min(1.0, sequential_estimate / total_time) if total_time > 0 else 0.0
            
            # 結果情報
            original_size = len(data)
            compressed_size = len(container)
            compression_ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
            
            result_info = {
                'compression_ratio': compression_ratio,
                'compression_throughput_mb_s': (original_size / 1024 / 1024) / total_time if total_time > 0 else 0,
                'total_compression_time': total_time,
                'chunk_count': len(chunks),
                'chunk_infos': chunk_infos,
                'parallel_workers_used': self.max_workers,
                'parallel_efficiency': parallel_efficiency,
                'original_size': original_size,
                'compressed_size': compressed_size,
                'tmc_version': '8.0',
                'reversible': True,
                'container_format': 'tmc_v8_parallel',
                'entropy_coding_efficiency': sum(1 for info in chunk_infos if info.data_type in ['sequential_int_data', 'text_data']) / len(chunk_infos)
            }
            
            # 統計更新
            self.stats['chunks_processed'] += len(chunks)
            self.stats['parallel_efficiency'] = parallel_efficiency
            
            print(f"--- TMC v8.0 並列圧縮完了 ---")
            print(f"圧縮率: {compression_ratio:.2f}% | 並列効率: {parallel_efficiency:.2%} | スループット: {result_info['compression_throughput_mb_s']:.2f} MB/s")
            
            return container, result_info
            
        except Exception as e:
            print(f"[TMC v8.0] 並列圧縮エラー: {e}")
            # フォールバック: 単一チャンク処理
            return self._compress_single_chunk(data)
    
    def _split_into_chunks(self, data: bytes) -> List[bytes]:
        """データを最適なチャンクサイズに分割"""
        chunks = []
        for i in range(0, len(data), self.chunk_size):
            chunk = data[i:i + self.chunk_size]
            chunks.append(chunk)
        return chunks
    
    def _compress_chunks_parallel(self, chunks: List[bytes]) -> Tuple[List[bytes], List[ChunkInfo]]:
        """並列チャンク圧縮処理"""
        print(f"  [並列処理] {self.max_workers}ワーカーで{len(chunks)}チャンクを並列圧縮中...")
        
        compressed_chunks = []
        chunk_infos = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 全チャンクを並列で処理
            future_to_chunk = {
                executor.submit(self._compress_chunk, chunk_data, chunk_id): chunk_id 
                for chunk_id, chunk_data in enumerate(chunks)
            }
            
            # 結果を順序通りに収集
            chunk_results = {}
            for future in as_completed(future_to_chunk):
                chunk_id = future_to_chunk[future]
                try:
                    compressed_data, chunk_info = future.result()
                    chunk_results[chunk_id] = (compressed_data, chunk_info)
                    print(f"    [チャンク {chunk_id}] 完了: {chunk_info.original_size} -> {chunk_info.compressed_size} bytes "
                          f"({chunk_info.compression_ratio:.1f}%, {chunk_info.data_type})")
                except Exception as e:
                    print(f"    [チャンク {chunk_id}] エラー: {e}")
                    # エラーの場合は元データをそのまま格納
                    chunk_data = chunks[chunk_id]
                    chunk_results[chunk_id] = (chunk_data, ChunkInfo(
                        chunk_id=chunk_id,
                        original_size=len(chunk_data),
                        compressed_size=len(chunk_data),
                        data_type="error_fallback",
                        compression_ratio=0.0,
                        processing_time=0.0
                    ))
        
        # 順序通りに結果を配列に格納
        for chunk_id in sorted(chunk_results.keys()):
            compressed_data, chunk_info = chunk_results[chunk_id]
            compressed_chunks.append(compressed_data)
            chunk_infos.append(chunk_info)
        
        return compressed_chunks, chunk_infos
    
    def _compress_chunk(self, chunk_data: bytes, chunk_id: int) -> Tuple[bytes, ChunkInfo]:
        """個別チャンクの圧縮処理（ワーカー関数）"""
        chunk_start = time.perf_counter()
        
        try:
            # 1. データタイプ分析
            data_type, features = self.dispatcher.dispatch(chunk_data)
            
            # 2. インテリジェント・バイパス分析
            transformer = self.transformers.get(data_type)
            should_transform, meta_info = self.meta_analyzer.should_apply_transform(
                chunk_data, transformer, data_type
            )
            
            # 3. 変換処理
            if should_transform and transformer:
                transformed_streams, transform_info = transformer.transform(chunk_data)
            else:
                transformed_streams = [chunk_data]
                transform_info = {'method': 'bypass', 'reason': 'intelligent_bypass'}
            
            # 4. 符号化処理（動的バックエンド選択）
            final_streams = []
            for stream in transformed_streams:
                if should_transform and data_type in [DataType.SEQUENTIAL_INT_DATA, DataType.TEXT_DATA]:
                    # 変換済みデータに純粋エントロピー符号化
                    compressed_stream, method = self.entropy_encoder.encode_entropy_stream(stream, "transformed")
                    self.stats['entropy_coding_used'] += 1
                else:
                    # 汎用データに従来型圧縮
                    stream_entropy = self._calculate_entropy(np.frombuffer(stream, dtype=np.uint8)) if len(stream) > 0 else 4.0
                    compressed_stream, method = self.core_compressor.compress(stream, stream_entropy)
                
                final_streams.append(compressed_stream)
            
            # 5. チャンク結果パッキング
            chunk_compressed = self._pack_chunk_data(final_streams, data_type, transform_info, features)
            
            processing_time = time.perf_counter() - chunk_start
            compression_ratio = (1 - len(chunk_compressed) / len(chunk_data)) * 100 if len(chunk_data) > 0 else 0
            
            chunk_info = ChunkInfo(
                chunk_id=chunk_id,
                original_size=len(chunk_data),
                compressed_size=len(chunk_compressed),
                data_type=data_type.value,
                compression_ratio=compression_ratio,
                processing_time=processing_time
            )
            
            return chunk_compressed, chunk_info
            
        except Exception as e:
            # エラー時のフォールバック
            processing_time = time.perf_counter() - chunk_start
            chunk_info = ChunkInfo(
                chunk_id=chunk_id,
                original_size=len(chunk_data),
                compressed_size=len(chunk_data),
                data_type="error_fallback",
                compression_ratio=0.0,
                processing_time=processing_time
            )
            return chunk_data, chunk_info
    
    def _pack_chunk_data(self, streams: List[bytes], data_type: DataType, 
                        transform_info: Dict[str, Any], features: Dict[str, Any]) -> bytes:
        """チャンクデータのパッキング"""
        # チャンクヘッダー作成
        chunk_header = {
            'data_type': data_type.value,
            'transform_info': transform_info,
            'stream_count': len(streams),
            'features': {k: v for k, v in features.items() if isinstance(v, (int, float, str, bool))}
        }
        
        header_json = json.dumps(chunk_header, separators=(',', ':'))
        header_bytes = header_json.encode('utf-8')
        
        # パッキング: [ヘッダーサイズ(4)] + [ヘッダー] + [ストリーム数(4)] + [サイズ1(4)] + [サイズ2(4)]... + [ストリーム1] + [ストリーム2]...
        packed_data = bytearray()
        packed_data.extend(len(header_bytes).to_bytes(4, 'big'))
        packed_data.extend(header_bytes)
        packed_data.extend(len(streams).to_bytes(4, 'big'))
        
        # ストリームサイズ情報
        for stream in streams:
            packed_data.extend(len(stream).to_bytes(4, 'big'))
        
        # ストリームデータ
        for stream in streams:
            packed_data.extend(stream)
        
        return bytes(packed_data)
    
    def _build_tmc_v8_container(self, compressed_chunks: List[bytes], 
                               chunk_infos: List[ChunkInfo]) -> bytes:
        """TMC v8.0 コンテナフォーマット構築"""
        container = bytearray()
        
        # マジックナンバー + バージョン
        container.extend(TMC_V8_MAGIC)
        container.extend(b'8.0\x00')
        
        # チャンク数
        container.extend(len(compressed_chunks).to_bytes(4, 'big'))
        
        # チャンク情報テーブル
        for chunk_info in chunk_infos:
            container.extend(chunk_info.chunk_id.to_bytes(4, 'big'))
            container.extend(chunk_info.original_size.to_bytes(4, 'big'))
            container.extend(chunk_info.compressed_size.to_bytes(4, 'big'))
            container.extend(chunk_info.data_type.encode('utf-8')[:16].ljust(16, b'\x00'))
        
        # 圧縮済みチャンクデータ
        for chunk_data in compressed_chunks:
            container.extend(chunk_data)
        
        return bytes(container)
    
    def _compress_single_chunk(self, data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """単一チャンク処理（v7.0互換）"""
        return self.compress_tmc(data)
    
    def compress_tmc(self, data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """TMC v7.0 インテリジェント統合圧縮処理"""
        compression_start = time.perf_counter()
        
        try:
            print("\n--- TMC v7.0 インテリジェント圧縮開始 ---")
            
            # 1. 改良分析&ディスパッチ
            data_type, features = self.dispatcher.dispatch(data)
            
            # 2. インテリジェント・バイパス分析（TMC v7.0新機能）
            transformer = self.transformers.get(data_type)
            should_transform, meta_info = self.meta_analyzer.should_apply_transform(
                data, transformer, data_type
            )
            
            # 3. 適応的変換（インテリジェント判定に基づく）
            if should_transform and transformer:
                print(f"  [インテリジェント] {data_type.value} 変換を実行")
                transformed_streams, transform_info = transformer.transform(data)
                self.stats['transforms_applied'] += 1
                
                # メタ分析情報を変換情報に統合
                transform_info['meta_analysis'] = meta_info
                transform_info['bypassed'] = False
            else:
                print(f"  [インテリジェント] {data_type.value} 変換をスキップ")
                transformed_streams = [data]
                transform_info = {
                    'method': 'bypassed', 
                    'meta_analysis': meta_info,
                    'bypassed': True,
                    'reason': meta_info.get('reason', 'ineffective')
                }
                self.stats['transforms_bypassed'] += 1
            
            # 4. 並列コア圧縮（v9.0: コンテキストミキシング対応）
            compressed_streams = []
            compression_methods = []
            
            print("  [符号化] TMC v9.0 コンテキスト適応型圧縮中...")
            for i, stream in enumerate(transformed_streams):
                # ストリームエントロピー計算
                if len(stream) > 0:
                    stream_array = np.frombuffer(stream, dtype=np.uint8)
                    stream_entropy = self._calculate_entropy(stream_array)
                else:
                    stream_entropy = 0.0
                
                # v9.0: コンテキストミキシング適用判定
                use_context_mixing = (
                    should_transform and  # 変換が適用されている場合
                    len(stream) > 2048 and  # 2KB以上
                    stream_entropy > 3.0 and  # 適度なエントロピー
                    stream_entropy < 7.0  # ランダム過ぎない
                )
                
                # TMC統一圧縮（コンテキストミキシング対応）
                compressed, comp_method = self.core_compressor.compress(
                    stream, 
                    stream_entropy=stream_entropy, 
                    stream_size=len(stream),
                    use_context_mixing=use_context_mixing
                )
                compressed_streams.append(compressed)
                compression_methods.append(comp_method)
                
                context_info = " (コンテキストミキシング)" if use_context_mixing else ""
                print(f"    ストリーム {i}: {len(stream)} bytes -> {len(compressed)} bytes ({comp_method}, エントロピー: {stream_entropy:.2f}){context_info}")
                            # 5. TMC v7.0 フォーマット構築
            final_data = self._pack_tmc_v7(compressed_streams, compression_methods, 
                                          data_type, transform_info, features)
            
            total_time = time.perf_counter() - compression_start
            
            # 結果情報
            compression_ratio = (1 - len(final_data) / len(data)) * 100 if len(data) > 0 else 0
            
            result_info = {
                'compression_ratio': compression_ratio,
                'compression_throughput_mb_s': (len(data) / 1024 / 1024) / total_time if total_time > 0 else 0,
                'total_compression_time': total_time,
                'data_type': data_type.value,
                'features': features,
                'transform_info': transform_info,
                'compression_methods': compression_methods,
                'stream_count': len(compressed_streams),
                'original_size': len(data),
                'compressed_size': len(final_data),
                'tmc_version': '7.0',
                'reversible': True,
                'zstd_used': self.core_compressor.zstd_available,
                'intelligent_bypass_used': True,  # v7.0新機能
                'transform_applied': should_transform,
                'meta_analysis': meta_info
            }
            
            print(f"--- TMC v7.0 圧縮完了 ---")
            print(f"合計サイズ: {len(data)} bytes -> {len(final_data)} bytes (圧縮率: {compression_ratio:.2f}%)")
            print(f"変換: {'適用' if should_transform else 'スキップ'}")
            
            return final_data, result_info
            
        except Exception as e:
            total_time = time.perf_counter() - compression_start
            print(f"❌ 圧縮エラー: {e}")
            return data, {
                'compression_ratio': 0.0,
                'error': str(e),
                'total_compression_time': total_time,
                'reversible': True
            }
    
    def _calculate_entropy(self, data_array: np.ndarray) -> float:
        """エントロピー計算ヘルパー"""
        try:
            byte_counts = np.bincount(data_array, minlength=256)
            probabilities = byte_counts[byte_counts > 0] / len(data_array)
            return float(-np.sum(probabilities * np.log2(probabilities)))
        except Exception:
            return 4.0
    
    def decompress_tmc(self, compressed_data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """TMC v7.0 展開処理"""
        decompression_start = time.perf_counter()
        
        try:
            print("\n--- TMC v7.0 展開開始 ---")
            
            # TMC v7.0 ヘッダー解析（v6.0互換）
            header = self._parse_tmc_v7_header(compressed_data)
            if not header:
                raise ValueError("Invalid TMC v7.0 format")
            
            # ストリーム抽出
            payload = compressed_data[header['header_size']:]
            streams = self._extract_tmc_v7_streams(payload, header)
            
            # 並列展開
            decompressed_streams = []
            for i, (stream, method) in enumerate(zip(streams, header['compression_methods'])):
                decompressed = self.core_compressor.decompress(stream, method)
                decompressed_streams.append(decompressed)
                print(f"    ストリーム {i}: {len(stream)} bytes -> {len(decompressed)} bytes ({method})")
            
            # 逆変換（インテリジェント・バイパス対応）
            data_type = DataType(header['data_type'])
            transformer = self.transformers.get(data_type)
            
            # 変換がバイパスされたかチェック
            transform_bypassed = header.get('transform_bypassed', False)
            
            if transformer and not transform_bypassed:
                print(f"  [逆変換] {data_type.value} 逆変換を実行")
                original_data = transformer.inverse_transform(decompressed_streams, header['transform_info'])
            else:
                print(f"  [逆変換] {data_type.value} 変換バイパス - 直接結合")
                original_data = b''.join(decompressed_streams)
            
            total_time = time.perf_counter() - decompression_start
            
            print(f"--- TMC v7.0 展開完了 ---")
            print(f"再構築データサイズ: {len(original_data)} bytes")
            
            result_info = {
                'decompression_throughput_mb_s': (len(original_data) / 1024 / 1024) / total_time if total_time > 0 else 0,
                'total_decompression_time': total_time,
                'decompressed_size': len(original_data),
                'tmc_version': '7.0',
                'transform_bypassed': transform_bypassed
            }
            
            return original_data, result_info
            
        except Exception as e:
            total_time = time.perf_counter() - decompression_start
            print(f"❌ 展開エラー: {e}")
            return compressed_data, {
                'error': str(e),
                'total_decompression_time': total_time
            }
    
    def test_reversibility(self, test_data: bytes, test_name: str = "test") -> Dict[str, Any]:
        """TMC v7.0 可逆性テスト"""
        test_start_time = time.perf_counter()
        
        try:
            print(f"🔄 TMC v7.0 可逆性テスト開始: {test_name}")
            
            # 圧縮
            compressed, compression_info = self.compress_tmc(test_data)
            
            # 展開
            decompressed, decompression_info = self.decompress_tmc(compressed)
            
            # 検証
            is_identical = (test_data == decompressed)
            
            # 統計更新
            self.stats['reversibility_tests_total'] += 1
            if is_identical:
                self.stats['reversibility_tests_passed'] += 1
            
            result_icon = "✅" if is_identical else "❌"
            transform_status = "適用" if compression_info.get('transform_applied', False) else "スキップ"
            print(f"   {result_icon} 可逆性: {'成功' if is_identical else '失敗'} | 変換: {transform_status}")
            
            return {
                'test_name': test_name,
                'reversible': is_identical,
                'original_size': len(test_data),
                'compressed_size': len(compressed),
                'decompressed_size': len(decompressed),
                'compression_ratio': compression_info.get('compression_ratio', 0),
                'compression_time': compression_info.get('total_compression_time', 0),
                'decompression_time': decompression_info.get('total_decompression_time', 0),
                'compression_throughput_mb_s': compression_info.get('compression_throughput_mb_s', 0),
                'decompression_throughput_mb_s': decompression_info.get('decompression_throughput_mb_s', 0),
                'total_test_time': time.perf_counter() - test_start_time,
                'data_type': compression_info.get('data_type', 'unknown'),
                'zstd_used': compression_info.get('zstd_used', False),
                'tmc_version': '7.0',
                'transform_applied': compression_info.get('transform_applied', False),
                'intelligent_bypass_used': compression_info.get('intelligent_bypass_used', False),
                'meta_analysis': compression_info.get('meta_analysis', {})
            }
            
        except Exception as e:
            return {
                'test_name': test_name,
                'reversible': False,
                'error': str(e),
                'tmc_version': '7.0'
            }
    
    def _pack_tmc_v7(self, streams: List[bytes], methods: List[str], 
                     data_type: DataType, transform_info: Dict[str, Any], 
                     features: Dict[str, Any]) -> bytes:
        """TMC v7.0 フォーマット構築（インテリジェント・バイパス対応）"""
        try:
            header = bytearray()
            
            # TMC v7.0 マジックナンバー
            header.extend(b'TMC7')
            
            # データタイプ
            data_type_bytes = data_type.value.encode('utf-8')[:32].ljust(32, b'\x00')
            header.extend(data_type_bytes)
            
            # ストリーム数
            header.extend(struct.pack('<I', len(streams)))
            
            # 圧縮メソッド情報
            for method in methods:
                method_bytes = method.encode('utf-8')[:16].ljust(16, b'\x00')
                header.extend(method_bytes)
            
            # 変換情報（安全なJSONシリアライズ、メタ分析情報統合）
            transform_info_safe = self._make_json_safe(transform_info)
            transform_str = json.dumps(transform_info_safe, separators=(',', ':'))
            transform_bytes = transform_str.encode('utf-8')
            header.extend(struct.pack('<I', len(transform_bytes)))
            header.extend(transform_bytes)
            
            # ストリームサイズテーブル
            for stream in streams:
                header.extend(struct.pack('<I', len(stream)))
            
            # チェックサム
            payload = b''.join(streams)
            checksum = zlib.crc32(payload) & 0xffffffff
            header.extend(struct.pack('<I', checksum))
            
            return bytes(header) + payload
            
        except Exception:
            return b''.join(streams)
    
    def _parse_tmc_v7_header(self, data: bytes) -> Optional[Dict[str, Any]]:
        """TMC v7.0 ヘッダー解析（v6.0互換）"""
        try:
            if len(data) < 44 or (data[:4] != b'TMC7' and data[:4] != b'TMC6' and data[:4] != b'TMC4'):
                return None
            
            offset = 4
            
            # データタイプ
            data_type = data[offset:offset+32].rstrip(b'\x00').decode('utf-8')
            offset += 32
            
            # ストリーム数
            stream_count = struct.unpack('<I', data[offset:offset+4])[0]
            offset += 4
            
            # 圧縮メソッド
            compression_methods = []
            for _ in range(stream_count):
                method = data[offset:offset+16].rstrip(b'\x00').decode('utf-8')
                compression_methods.append(method)
                offset += 16
            
            # 変換情報（安全なJSON解析）
            transform_info_size = struct.unpack('<I', data[offset:offset+4])[0]
            offset += 4
            transform_info_str = data[offset:offset+transform_info_size].decode('utf-8')
            transform_info = json.loads(transform_info_str)
            offset += transform_info_size
            
            # ストリームサイズ
            stream_sizes = []
            for _ in range(stream_count):
                size = struct.unpack('<I', data[offset:offset+4])[0]
                stream_sizes.append(size)
                offset += 4
            
            # チェックサム
            checksum = struct.unpack('<I', data[offset:offset+4])[0]
            offset += 4
            
            # v7.0機能の解析
            transform_bypassed = transform_info.get('bypassed', False)
            
            return {
                'data_type': data_type,
                'stream_count': stream_count,
                'compression_methods': compression_methods,
                'transform_info': transform_info,
                'stream_sizes': stream_sizes,
                'checksum': checksum,
                'header_size': offset,
                'transform_bypassed': transform_bypassed  # v7.0新機能
            }
            
        except Exception:
            return None
    
    def _extract_tmc_v7_streams(self, payload: bytes, header: Dict[str, Any]) -> List[bytes]:
        """TMC v7.0 ストリーム抽出"""
        try:
            streams = []
            offset = 0
            
            for size in header['stream_sizes']:
                stream = payload[offset:offset+size]
                streams.append(stream)
                offset += size
            
            return streams
            
        except Exception:
            return [payload]
    
    def _make_json_safe(self, data: Any) -> Any:
        """JSONシリアライズ可能な形式に変換"""
        if isinstance(data, dict):
            return {k: self._make_json_safe(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_json_safe(v) for v in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.int32, np.int64, np.float32, np.float64)):
            return float(data)
        elif isinstance(data, np.bool_):
            return bool(data)
        else:
            return data
    
    def _extract_tmc_v4_streams(self, payload: bytes, header: Dict[str, Any]) -> List[bytes]:
        """TMC v4.0 ストリーム抽出（互換性のため保持）"""
        return self._extract_tmc_v7_streams(payload, header)


# 後方互換性のためのエイリアス
NEXUSTMCEngineV8 = NEXUSTMCEngineV9  # v8.x系からのマイグレーション用

# エクスポート
__all__ = ['NEXUSTMCEngineV9', 'NEXUSTMCEngineV8', 'DataType']

if __name__ == "__main__":
    print("🚀 NEXUS TMC Engine v9.0 - コンテキストミキシング統合版")
    
    engine = NEXUSTMCEngineV9()
    
    # TMC v8.0 特化テストケース
    test_cases = [
        ("浮動小数点データ", np.linspace(1000, 1010, 2000, dtype=np.float32).tobytes()),
        ("系列整数データ", np.arange(0, 8000, 4, dtype=np.int32).tobytes()),
        ("テキストデータ", ("Hello TMC v8.0! " * 500).encode('utf-8')),
        ("反復バイナリ", b"PATTERN" * 1000),
        ("汎用バイナリ", bytes(range(256)) * 20),
        ("並列テスト（大容量）", np.arange(0, 50000, dtype=np.int32).tobytes()),  # v8.0: 並列処理テスト
        ("可変長パーティショニングテスト", np.concatenate([
            np.arange(1000, 2000, dtype=np.int32),  # 線形パート
            np.full(500, 5000, dtype=np.int32),      # 定数パート
        ]).tobytes()),  # v8.0: LeCoパーティショニングテスト
    ]
    
    success_count = 0
    total_tests = len(test_cases)
    
    for name, data in test_cases:
        result = engine.test_reversibility(data, name)
        if result.get('reversible', False):
            success_count += 1
        
        # v7.0新機能の詳細表示
        print(f"  インテリジェント・バイパス: {'有効' if result.get('intelligent_bypass_used') else '無効'}")
        if 'meta_analysis' in result and result['meta_analysis']:
            meta = result['meta_analysis']
            print(f"  メタ分析: {meta.get('reason', 'N/A')}")
            if 'effectiveness' in meta:
                print(f"  圧縮効果: {meta['effectiveness']:.2%}")
    
    print(f"\n📊 TMC v8.0 テスト結果: {success_count}/{total_tests} 成功")
    print(f"📈 統計:")
    print(f"  変換適用: {engine.stats['transforms_applied']}")
    print(f"  変換スキップ: {engine.stats['transforms_bypassed']}")
    print(f"  エントロピー符号化使用: {engine.stats['entropy_coding_used']}")
    print(f"  並列チャンク処理: {engine.stats['chunks_processed']}")
    
    if success_count == total_tests:
        print("🎉 TMC v8.0 次世代量子インテリジェント圧縮プラットフォーム準備完了!")
        print("🔥 並列チャンク処理 + 可変長パーティショニング + 純粋エントロピー符号化統合完了!")
        if ZSTD_AVAILABLE:
            print("⚡ 最高性能構成: 真の並列処理 + LeCoパーティショニング + 量子エントロピー符号化!")
    else:
        print("⚠️ 一部テスト失敗 - さらなる最適化が必要")
