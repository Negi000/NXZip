#!/usr/bin/env python3
"""
NEXUS TMC Engine v7.0 - インテリジェント圧縮プラットフォーム
Transform-Model-Code 革命的進化版 with インテリジェント・バイパス
"""

import os
import sys
import time
import struct
import zlib
import lzma
import bz2
import json
import numpy as np
from typing import Tuple, Dict, Any, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
import threading

# Zstandardのインポート（フォールバック付き）
try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
    print("🚀 Zstandard利用可能 - 高性能バックエンド有効")
except ImportError:
    ZSTD_AVAILABLE = False
    print("⚠️ Zstandard未利用 - 標準圧縮器を使用")

# pydivsufsortのインポート
try:
    import pydivsufsort
    PYDIVSUFSORT_AVAILABLE = True
    print("🔥 pydivsufsort利用可能 - 高速BWT有効")
except ImportError:
    PYDIVSUFSORT_AVAILABLE = False
    print("⚠️ pydivsufsort未利用 - フォールバック実装")


class DataType(Enum):
    """改良データタイプ分類"""
    FLOAT_DATA = "float_data"
    TEXT_DATA = "text_data"
    SEQUENTIAL_INT_DATA = "sequential_int_data"
    STRUCTURED_NUMERIC = "structured_numeric"
    TIME_SERIES = "time_series"
    REPETITIVE_BINARY = "repetitive_binary"
    COMPRESSED_LIKE = "compressed_like"
    GENERIC_BINARY = "generic_binary"


class MetaAnalyzer:
    """
    TMC v7.0 インテリジェント・バイパス - メタレイヤー分析器
    変換のコストパフォーマンスを予測・評価
    """
    
    def __init__(self, core_compressor):
        self.core_compressor = core_compressor
        self.cache = {}  # 分析結果キャッシュ
        self.sample_size = 8192  # 8KBサンプル
        
    def should_apply_transform(self, data: bytes, transformer, data_type: DataType) -> Tuple[bool, Dict[str, Any]]:
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
    
    def _get_effectiveness_threshold(self, data_type: DataType, data_size: int) -> float:
        """データタイプと サイズに基づく効果閾値"""
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


class ChunkManager:
    """
    TMC v7.0 チャンクベース並列処理マネージャー
    大規模データの並列圧縮・展開を管理
    """
    
    def __init__(self, chunk_size: int = 1024 * 1024, max_workers: int = None):
        self.chunk_size = chunk_size  # デフォルト1MB
        self.max_workers = max_workers or min(8, (os.cpu_count() or 1) + 4)
        
    def split_data(self, data: bytes) -> List[bytes]:
        """データをチャンクに分割"""
        if len(data) <= self.chunk_size:
            return [data]
        
        chunks = []
        for i in range(0, len(data), self.chunk_size):
            chunk = data[i:i + self.chunk_size]
            chunks.append(chunk)
        
        print(f"  [チャンク] データを{len(chunks)}個のチャンクに分割 (チャンクサイズ: {self.chunk_size:,} bytes)")
        return chunks
    
    def parallel_compress_chunks(self, chunks: List[bytes], compress_func) -> Tuple[List[bytes], List[Dict[str, Any]]]:
        """チャンクを並列圧縮"""
        print(f"  [並列圧縮] {len(chunks)}個のチャンクを{self.max_workers}スレッドで処理中...")
        
        compressed_chunks = []
        chunk_infos = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 各チャンクを並列処理に投入
            future_to_index = {}
            for i, chunk in enumerate(chunks):
                future = executor.submit(compress_func, chunk)
                future_to_index[future] = i
            
            # 結果を順序通りに収集
            results = [None] * len(chunks)
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    compressed_data, info = future.result()
                    results[index] = (compressed_data, info)
                    print(f"    [並列圧縮] チャンク {index + 1}/{len(chunks)} 完了")
                except Exception as e:
                    print(f"    [並列圧縮] チャンク {index + 1} エラー: {e}")
                    results[index] = (chunks[index], {'error': str(e)})
            
            # 結果を分離
            for compressed_data, info in results:
                compressed_chunks.append(compressed_data)
                chunk_infos.append(info)
        
        return compressed_chunks, chunk_infos
    
    def parallel_decompress_chunks(self, compressed_chunks: List[bytes], decompress_func) -> List[bytes]:
        """チャンクを並列展開"""
        print(f"  [並列展開] {len(compressed_chunks)}個のチャンクを{self.max_workers}スレッドで処理中...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {}
            for i, chunk in enumerate(compressed_chunks):
                future = executor.submit(decompress_func, chunk)
                future_to_index[future] = i
            
            # 結果を順序通りに収集
            results = [None] * len(compressed_chunks)
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    decompressed_data, _ = future.result()
                    results[index] = decompressed_data
                    print(f"    [並列展開] チャンク {index + 1}/{len(compressed_chunks)} 完了")
                except Exception as e:
                    print(f"    [並列展開] チャンク {index + 1} エラー: {e}")
                    results[index] = b''
            
            return results
    
    def pack_chunks(self, compressed_chunks: List[bytes], chunk_infos: List[Dict[str, Any]]) -> bytes:
        """圧縮済みチャンクをコンテナフォーマットにパック"""
        try:
            container = bytearray()
            
            # TMC v7.0 コンテナヘッダー
            container.extend(b'TMC7CONTAINER')
            
            # チャンク数
            container.extend(struct.pack('<I', len(compressed_chunks)))
            
            # チャンクサイズテーブル
            for chunk in compressed_chunks:
                container.extend(struct.pack('<I', len(chunk)))
            
            # メタデータ（JSON）
            metadata = {
                'chunk_count': len(compressed_chunks),
                'total_compressed_size': sum(len(chunk) for chunk in compressed_chunks),
                'chunk_infos': chunk_infos[:10]  # 最初の10個のチャンク情報のみ保存（サイズ制限）
            }
            metadata_json = json.dumps(metadata, separators=(',', ':')).encode('utf-8')
            container.extend(struct.pack('<I', len(metadata_json)))
            container.extend(metadata_json)
            
            # チャンクデータ
            for chunk in compressed_chunks:
                container.extend(chunk)
            
            return bytes(container)
            
        except Exception as e:
            print(f"  [チャンク] パックエラー: {e}")
            return b''.join(compressed_chunks)
    
    def unpack_chunks(self, container_data: bytes) -> Tuple[List[bytes], Dict[str, Any]]:
        """コンテナフォーマットからチャンクを抽出"""
        try:
            if not container_data.startswith(b'TMC7CONTAINER'):
                # 旧フォーマットまたは単一チャンク
                return [container_data], {}
            
            offset = 12  # ヘッダーサイズ
            
            # チャンク数
            chunk_count = struct.unpack('<I', container_data[offset:offset+4])[0]
            offset += 4
            
            # チャンクサイズテーブル
            chunk_sizes = []
            for _ in range(chunk_count):
                size = struct.unpack('<I', container_data[offset:offset+4])[0]
                chunk_sizes.append(size)
                offset += 4
            
            # メタデータ
            metadata_size = struct.unpack('<I', container_data[offset:offset+4])[0]
            offset += 4
            metadata_json = container_data[offset:offset+metadata_size].decode('utf-8')
            metadata = json.loads(metadata_json)
            offset += metadata_size
            
            # チャンクデータ抽出
            chunks = []
            for size in chunk_sizes:
                chunk = container_data[offset:offset+size]
                chunks.append(chunk)
                offset += size
            
            return chunks, metadata
            
        except Exception as e:
            print(f"  [チャンク] アンパックエラー: {e}")
            return [container_data], {}


# 以前のクラス群をインポート（CoreCompressor, ImprovedDispatcher, TDTTransformer, LeCoTransformer は v6.0 と同じ）
from .nexus_tmc_v4_unified import (
    CoreCompressor, ImprovedDispatcher, TDTTransformer, LeCoTransformer
)


class EnhancedBWTTransformer:
    """
    TMC v7.0 強化版BWTTransformer
    ポストBWTパイプライン統合とインテリジェント処理
    """
    
    def __init__(self):
        self.pydivsufsort_available = PYDIVSUFSORT_AVAILABLE
        self.post_bwt_pipeline = PostBWTPipeline()
        print(f"🔥 Enhanced BWT Transformer 初期化: pydivsufsort={'有効' if self.pydivsufsort_available else '無効'}")
    
    def transform(self, data: bytes) -> Tuple[List[bytes], Dict[str, Any]]:
        """TMC v7.0 強化BWT変換"""
        print("  [強化BWT] TMC v7.0 専門変換を実行中...")
        info = {'method': 'enhanced_bwt_mtf_rle', 'original_size': len(data)}
        
        try:
            if not data:
                return [data], info
            
            # 動的サイズ制限（並列処理前提）
            MAX_BWT_SIZE = 2 * 1024 * 1024  # 2MB制限（v7.0では拡張）
            if len(data) > MAX_BWT_SIZE:
                print(f"    [強化BWT] データサイズ({len(data)})が制限({MAX_BWT_SIZE})を超過 - BWTスキップ")
                info['method'] = 'bwt_skipped_large'
                return [data], info
            
            # 高速BWT変換
            if self.pydivsufsort_available:
                bwt_encoded, primary_index = self._fast_bwt_transform(data)
            else:
                bwt_encoded, primary_index = self._fallback_bwt_transform(data)
            
            # MTF変換
            mtf_encoded = self._mtf_encode(bwt_encoded)
            
            # ポストBWTパイプライン適用
            post_bwt_streams = self.post_bwt_pipeline.encode(mtf_encoded)
            
            print(f"    [強化BWT] パイプライン: BWT -> MTF -> ポストBWT ({len(post_bwt_streams)}ストリーム)")
            
            # インデックス情報
            index_bytes = primary_index.to_bytes(4, 'big')
            
            # 最終ストリーム構成
            final_streams = [index_bytes] + post_bwt_streams
            
            info.update({
                'primary_index': primary_index,
                'bwt_length': len(bwt_encoded),
                'mtf_length': len(mtf_encoded),
                'post_bwt_streams': len(post_bwt_streams),
                'enhanced_pipeline': True
            })
            
            return final_streams, info
            
        except Exception as e:
            print(f"    [強化BWT] エラー: {e}")
            return [data], info
    
    def inverse_transform(self, streams: List[bytes], info: Dict[str, Any]) -> bytes:
        """TMC v7.0 強化BWT逆変換"""
        print("  [強化BWT] TMC v7.0 専門逆変換を実行中...")
        
        try:
            if info.get('method') == 'bwt_skipped_large':
                return streams[0] if streams else b''
            
            if len(streams) < 1:
                return b''
            
            # インデックス復元
            primary_index = int.from_bytes(streams[0], 'big')
            post_bwt_streams = streams[1:]
            
            # ポストBWT逆変換
            if info.get('enhanced_pipeline', False):
                mtf_encoded = self.post_bwt_pipeline.decode(post_bwt_streams)
            else:
                # フォールバック: 従来方式
                mtf_encoded = streams[1] if len(streams) > 1 else b''
            
            # 逆MTF変換
            bwt_encoded = self._mtf_decode(mtf_encoded)
            
            # 逆BWT変換
            if self.pydivsufsort_available:
                original_data = self._fast_bwt_inverse(bwt_encoded, primary_index)
            else:
                original_data = self._fallback_bwt_inverse(bwt_encoded, primary_index)
            
            print("    [強化BWT] 逆変換パイプライン完了")
            return original_data
            
        except Exception as e:
            print(f"    [強化BWT] 逆変換エラー: {e}")
            return b''.join(streams)
    
    # BWT、MTF関連メソッドはv6.0と同じ（簡潔性のため省略）
    def _fast_bwt_transform(self, data: bytes) -> Tuple[bytes, int]:
        """高速BWT変換（pydivsufsort使用）"""
        try:
            from pydivsufsort import bw_transform
            result = bw_transform(data)
            
            if isinstance(result, tuple) and len(result) == 2:
                bwt_encoded, primary_index = result
            else:
                bwt_encoded = result
                primary_index = 0
            
            if isinstance(primary_index, (list, tuple, np.ndarray)):
                primary_index = int(primary_index[0]) if len(primary_index) > 0 else 0
            else:
                primary_index = int(primary_index)
            
            if not isinstance(bwt_encoded, bytes):
                bwt_encoded = bytes(bwt_encoded)
                
            return bwt_encoded, primary_index
        except Exception as e:
            print(f"    [強化BWT] 高速実装エラー: {e}")
            return self._fallback_bwt_transform(data)
    
    def _fallback_bwt_transform(self, data: bytes) -> Tuple[bytes, int]:
        """フォールバックBWT実装"""
        data_with_sentinel = data + b'\x00'
        n = len(data_with_sentinel)
        
        rotations = []
        for i in range(n):
            rotation = data_with_sentinel[i:] + data_with_sentinel[:i]
            rotations.append((rotation, i))
        
        rotations.sort(key=lambda x: x[0])
        
        primary_index = 0
        for idx, (rotation, original_pos) in enumerate(rotations):
            if original_pos == 0:
                primary_index = idx
                break
        
        bwt_encoded = bytes(rotation[0][-1] for rotation, _ in rotations)
        return bwt_encoded, primary_index
    
    def _mtf_encode(self, data: bytes) -> bytes:
        """Move-to-Front符号化"""
        alphabet = list(range(256))
        encoded = bytearray()
        
        for byte_val in data:
            rank = alphabet.index(byte_val)
            encoded.append(rank)
            alphabet.pop(rank)
            alphabet.insert(0, byte_val)
        
        return bytes(encoded)
    
    def _mtf_decode(self, encoded_data: bytes) -> bytes:
        """逆Move-to-Front符号化"""
        alphabet = list(range(256))
        decoded = bytearray()
        
        for rank in encoded_data:
            byte_val = alphabet[rank]
            decoded.append(byte_val)
            alphabet.pop(rank)
            alphabet.insert(0, byte_val)
        
        return bytes(decoded)
    
    def _fast_bwt_inverse(self, bwt_encoded: bytes, primary_index: int) -> bytes:
        """高速BWT逆変換"""
        try:
            from pydivsufsort import inverse_bwt
            primary_index = int(primary_index)
            reconstructed = inverse_bwt(bwt_encoded, primary_index)
            
            if isinstance(reconstructed, bytes) and reconstructed and reconstructed[-1] == 0:
                reconstructed = reconstructed[:-1]
            
            return reconstructed
        except Exception as e:
            print(f"    [強化BWT] 高速逆変換エラー: {e}")
            return self._fallback_bwt_inverse(bwt_encoded, primary_index)
    
    def _fallback_bwt_inverse(self, last_col: bytes, primary_index: int) -> bytes:
        """フォールバックBWT逆変換"""
        n = len(last_col)
        if n == 0:
            return b''
        
        count = [0] * 256
        for char in last_col:
            count[char] += 1
        
        first_col_starts = [0] * 256
        total = 0
        for i in range(256):
            first_col_starts[i] = total
            total += count[i]
        
        next_idx = [0] * n
        char_counts = [0] * 256
        
        for i in range(n):
            char = last_col[i]
            next_idx[i] = first_col_starts[char] + char_counts[char]
            char_counts[char] += 1
        
        result = bytearray()
        current_idx = primary_index
        
        for _ in range(n):
            char = last_col[current_idx]
            result.append(char)
            current_idx = next_idx[current_idx]
        
        if result and result[-1] == 0:
            result = result[:-1]
        
        return bytes(result)


class NEXUSTMCEngineV7:
    """
    NEXUS TMC Engine v7.0 - インテリジェント圧縮プラットフォーム
    メタ分析、ポストBWTパイプライン、並列チャンク処理統合
    """
    
    def __init__(self, chunk_size: int = 1024 * 1024, max_workers: int = None):
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        
        # コアコンポーネント
        self.dispatcher = ImprovedDispatcher()
        self.core_compressor = CoreCompressor()
        self.meta_analyzer = MetaAnalyzer(self.core_compressor)
        self.chunk_manager = ChunkManager(chunk_size, max_workers)
        
        # 変換器マッピング（v7.0強化版）
        self.transformers = {
            DataType.FLOAT_DATA: TDTTransformer(),
            DataType.TEXT_DATA: EnhancedBWTTransformer(),  # v7.0強化版
            DataType.SEQUENTIAL_INT_DATA: LeCoTransformer(),
            DataType.STRUCTURED_NUMERIC: TDTTransformer(),
            DataType.TIME_SERIES: LeCoTransformer(),
            DataType.REPETITIVE_BINARY: None,
            DataType.COMPRESSED_LIKE: None,
            DataType.GENERIC_BINARY: None
        }
        
        # 統計情報
        self.stats = {
            'files_processed': 0,
            'total_input_size': 0,
            'total_compressed_size': 0,
            'chunks_processed': 0,
            'transforms_applied': 0,
            'transforms_bypassed': 0
        }
    
    def compress(self, data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """
        TMC v7.0 最上位レベル圧縮
        チャンクベース並列処理 + インテリジェント最適化
        """
        compression_start = time.perf_counter()
        
        try:
            print("\n=== TMC v7.0 インテリジェント圧縮開始 ===")
            print(f"入力データサイズ: {len(data):,} bytes")
            
            # 小さなデータは従来方式
            if len(data) <= self.chunk_size:
                print("  [単一チャンク] 従来方式で処理")
                return self._compress_single_chunk(data)
            
            # 大きなデータは並列チャンク処理
            print("  [並列チャンク] 大規模データを並列処理")
            
            # 1. データをチャンクに分割
            chunks = self.chunk_manager.split_data(data)
            
            # 2. チャンクを並列圧縮
            compressed_chunks, chunk_infos = self.chunk_manager.parallel_compress_chunks(
                chunks, self._compress_single_chunk
            )
            
            # 3. コンテナフォーマットにパック
            final_data = self.chunk_manager.pack_chunks(compressed_chunks, chunk_infos)
            
            total_time = time.perf_counter() - compression_start
            
            # 統計情報集計
            self.stats['chunks_processed'] += len(chunks)
            total_original_size = sum(len(chunk) for chunk in chunks)
            total_compressed_size = len(final_data)
            
            compression_ratio = (1 - total_compressed_size / total_original_size) * 100 if total_original_size > 0 else 0
            
            result_info = {
                'compression_ratio': compression_ratio,
                'compression_throughput_mb_s': (total_original_size / 1024 / 1024) / total_time if total_time > 0 else 0,
                'total_compression_time': total_time,
                'original_size': total_original_size,
                'compressed_size': total_compressed_size,
                'chunk_count': len(chunks),
                'chunk_infos': chunk_infos[:5],  # 最初の5個のみ
                'tmc_version': '7.0',
                'parallel_processing': True,
                'transforms_applied': self.stats['transforms_applied'],
                'transforms_bypassed': self.stats['transforms_bypassed']
            }
            
            print(f"=== TMC v7.0 圧縮完了 ===")
            print(f"総圧縮率: {compression_ratio:.2f}%")
            print(f"並列処理: {len(chunks)}チャンク、スループット: {result_info['compression_throughput_mb_s']:.1f} MB/s")
            
            return final_data, result_info
            
        except Exception as e:
            total_time = time.perf_counter() - compression_start
            print(f"❌ TMC v7.0 圧縮エラー: {e}")
            return data, {
                'compression_ratio': 0.0,
                'error': str(e),
                'total_compression_time': total_time
            }
    
    def _compress_single_chunk(self, data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """単一チャンクの圧縮（インテリジェント・バイパス適用）"""
        try:
            # 1. データタイプ分析
            data_type, features = self.dispatcher.dispatch(data)
            
            # 2. インテリジェント・バイパス分析
            transformer = self.transformers.get(data_type)
            should_transform, meta_info = self.meta_analyzer.should_apply_transform(
                data, transformer, data_type
            )
            
            # 3. 適応的変換（インテリジェント判定に基づく）
            if should_transform and transformer:
                print(f"  [インテリジェント] {data_type.value} 変換を実行")
                transformed_streams, transform_info = transformer.transform(data)
                self.stats['transforms_applied'] += 1
            else:
                print(f"  [インテリジェント] {data_type.value} 変換をスキップ")
                transformed_streams = [data]
                transform_info = {'method': 'bypassed', 'meta_analysis': meta_info}
                self.stats['transforms_bypassed'] += 1
            
            # 4. コア圧縮
            compressed_streams = []
            compression_methods = []
            
            for i, stream in enumerate(transformed_streams):
                stream_entropy = self._calculate_entropy(stream) if len(stream) > 0 else 0.0
                compressed, comp_method = self.core_compressor.compress(
                    stream, stream_entropy=stream_entropy, stream_size=len(stream)
                )
                compressed_streams.append(compressed)
                compression_methods.append(comp_method)
            
            # 5. TMC v7.0 フォーマット構築
            final_data = self._pack_tmc_v7_chunk(
                compressed_streams, compression_methods, data_type, 
                transform_info, features, meta_info
            )
            
            compression_ratio = (1 - len(final_data) / len(data)) * 100 if len(data) > 0 else 0
            
            return final_data, {
                'compression_ratio': compression_ratio,
                'data_type': data_type.value,
                'transform_applied': should_transform,
                'meta_analysis': meta_info,
                'original_size': len(data),
                'compressed_size': len(final_data)
            }
            
        except Exception as e:
            print(f"  [チャンク圧縮] エラー: {e}")
            return data, {'error': str(e)}
    
    def decompress(self, compressed_data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """TMC v7.0 最上位レベル展開"""
        decompression_start = time.perf_counter()
        
        try:
            print("\n=== TMC v7.0 インテリジェント展開開始 ===")
            
            # コンテナフォーマット判定
            chunks, metadata = self.chunk_manager.unpack_chunks(compressed_data)
            
            if len(chunks) == 1 and not metadata:
                # 単一チャンク
                print("  [単一チャンク] 従来方式で展開")
                return self._decompress_single_chunk(chunks[0])
            
            # 並列チャンク展開
            print(f"  [並列展開] {len(chunks)}チャンクを並列処理")
            
            decompressed_chunks = self.chunk_manager.parallel_decompress_chunks(
                chunks, self._decompress_single_chunk
            )
            
            # チャンクを結合
            final_data = b''.join(decompressed_chunks)
            
            total_time = time.perf_counter() - decompression_start
            
            result_info = {
                'decompression_throughput_mb_s': (len(final_data) / 1024 / 1024) / total_time if total_time > 0 else 0,
                'total_decompression_time': total_time,
                'decompressed_size': len(final_data),
                'chunk_count': len(chunks),
                'tmc_version': '7.0',
                'parallel_processing': True
            }
            
            print(f"=== TMC v7.0 展開完了 ===")
            print(f"展開データサイズ: {len(final_data):,} bytes")
            
            return final_data, result_info
            
        except Exception as e:
            total_time = time.perf_counter() - decompression_start
            print(f"❌ TMC v7.0 展開エラー: {e}")
            return compressed_data, {
                'error': str(e),
                'total_decompression_time': total_time
            }
    
    def _decompress_single_chunk(self, compressed_chunk: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """単一チャンクの展開"""
        try:
            # TMC v7.0 ヘッダー解析
            header = self._parse_tmc_v7_header(compressed_chunk)
            if not header:
                # フォールバック: v6.0フォーマット
                from .nexus_tmc_v4_unified import NEXUSTMCEngineV4
                fallback_engine = NEXUSTMCEngineV4()
                return fallback_engine.decompress_tmc(compressed_chunk)
            
            # ストリーム抽出
            payload = compressed_chunk[header['header_size']:]
            streams = self._extract_streams(payload, header)
            
            # 並列展開
            decompressed_streams = []
            for stream, method in zip(streams, header['compression_methods']):
                decompressed = self.core_compressor.decompress(stream, method)
                decompressed_streams.append(decompressed)
            
            # 逆変換
            data_type = DataType(header['data_type'])
            transformer = self.transformers.get(data_type)
            
            if transformer and not header.get('transform_bypassed', False):
                original_data = transformer.inverse_transform(decompressed_streams, header['transform_info'])
            else:
                original_data = b''.join(decompressed_streams)
            
            return original_data, {
                'decompressed_size': len(original_data),
                'data_type': header['data_type']
            }
            
        except Exception as e:
            print(f"  [チャンク展開] エラー: {e}")
            return compressed_chunk, {'error': str(e)}
    
    def _calculate_entropy(self, data: bytes) -> float:
        """エントロピー計算"""
        if not data:
            return 0.0
        try:
            data_array = np.frombuffer(data, dtype=np.uint8)
            byte_counts = np.bincount(data_array, minlength=256)
            probabilities = byte_counts[byte_counts > 0] / len(data_array)
            return float(-np.sum(probabilities * np.log2(probabilities)))
        except Exception:
            return 4.0
    
    def _pack_tmc_v7_chunk(self, streams: List[bytes], methods: List[str], 
                          data_type: DataType, transform_info: Dict[str, Any], 
                          features: Dict[str, Any], meta_info: Dict[str, Any]) -> bytes:
        """TMC v7.0 チャンクフォーマット構築"""
        try:
            header = bytearray()
            
            # TMC v7.0 マジックナンバー
            header.extend(b'TMC7')
            
            # データタイプ
            data_type_bytes = data_type.value.encode('utf-8')[:32].ljust(32, b'\x00')
            header.extend(data_type_bytes)
            
            # ストリーム数
            header.extend(struct.pack('<I', len(streams)))
            
            # 圧縮メソッド
            for method in methods:
                method_bytes = method.encode('utf-8')[:16].ljust(16, b'\x00')
                header.extend(method_bytes)
            
            # 変換情報 + メタ分析情報
            combined_info = {
                'transform_info': self._make_json_safe(transform_info),
                'meta_analysis': self._make_json_safe(meta_info),
                'transform_bypassed': transform_info.get('method') == 'bypassed'
            }
            
            info_str = json.dumps(combined_info, separators=(',', ':'))
            info_bytes = info_str.encode('utf-8')
            header.extend(struct.pack('<I', len(info_bytes)))
            header.extend(info_bytes)
            
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
        """TMC v7.0 ヘッダー解析"""
        try:
            if len(data) < 44 or data[:4] != b'TMC7':
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
            
            # 変換情報
            info_size = struct.unpack('<I', data[offset:offset+4])[0]
            offset += 4
            info_str = data[offset:offset+info_size].decode('utf-8')
            combined_info = json.loads(info_str)
            offset += info_size
            
            # ストリームサイズ
            stream_sizes = []
            for _ in range(stream_count):
                size = struct.unpack('<I', data[offset:offset+4])[0]
                stream_sizes.append(size)
                offset += 4
            
            # チェックサム
            checksum = struct.unpack('<I', data[offset:offset+4])[0]
            offset += 4
            
            return {
                'data_type': data_type,
                'stream_count': stream_count,
                'compression_methods': compression_methods,
                'transform_info': combined_info.get('transform_info', {}),
                'meta_analysis': combined_info.get('meta_analysis', {}),
                'transform_bypassed': combined_info.get('transform_bypassed', False),
                'stream_sizes': stream_sizes,
                'checksum': checksum,
                'header_size': offset
            }
            
        except Exception:
            return None
    
    def _extract_streams(self, payload: bytes, header: Dict[str, Any]) -> List[bytes]:
        """ストリーム抽出"""
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
    
    def test_reversibility(self, test_data: bytes, test_name: str = "test") -> Dict[str, Any]:
        """TMC v7.0 可逆性テスト"""
        test_start_time = time.perf_counter()
        
        try:
            print(f"🔄 TMC v7.0 可逆性テスト開始: {test_name}")
            
            # 圧縮
            compressed, compression_info = self.compress(test_data)
            
            # 展開
            decompressed, decompression_info = self.decompress(compressed)
            
            # 検証
            is_identical = (test_data == decompressed)
            
            result_icon = "✅" if is_identical else "❌"
            print(f"   {result_icon} 可逆性: {'成功' if is_identical else '失敗'}")
            
            return {
                'test_name': test_name,
                'reversible': is_identical,
                'original_size': len(test_data),
                'compressed_size': len(compressed),
                'decompressed_size': len(decompressed),
                'compression_ratio': compression_info.get('compression_ratio', 0),
                'compression_time': compression_info.get('total_compression_time', 0),
                'decompression_time': decompression_info.get('total_decompression_time', 0),
                'total_test_time': time.perf_counter() - test_start_time,
                'parallel_processing': compression_info.get('parallel_processing', False),
                'transforms_applied': compression_info.get('transforms_applied', 0),
                'transforms_bypassed': compression_info.get('transforms_bypassed', 0),
                'tmc_version': '7.0'
            }
            
        except Exception as e:
            return {
                'test_name': test_name,
                'reversible': False,
                'error': str(e),
                'tmc_version': '7.0'
            }


# エクスポート
__all__ = ['NEXUSTMCEngineV7', 'DataType', 'MetaAnalyzer', 'PostBWTPipeline', 'ChunkManager']

if __name__ == "__main__":
    print("🚀 NEXUS TMC Engine v7.0 - インテリジェント圧縮プラットフォーム")
    
    engine = NEXUSTMCEngineV7(chunk_size=512*1024, max_workers=4)  # 512KB チャンク
    
    # v7.0 特化テストケース
    test_cases = [
        ("小サイズ浮動小数点", np.linspace(0, 100, 1000, dtype=np.float32).tobytes()),
        ("大サイズ系列整数", np.arange(0, 50000, dtype=np.int32).tobytes()),
        ("中サイズテキスト", ("TMC v7.0 is revolutionary! " * 1000).encode('utf-8')),
        ("大サイズ反復バイナリ", b"PATTERN" * 10000),
        ("並列処理テスト用大データ", bytes(range(256)) * 5000)  # 1.25MB
    ]
    
    success_count = 0
    total_tests = len(test_cases)
    
    for name, data in test_cases:
        result = engine.test_reversibility(data, name)
        if result.get('reversible', False):
            success_count += 1
            
        print(f"  並列処理: {'有効' if result.get('parallel_processing') else '無効'}")
        print(f"  変換適用/スキップ: {result.get('transforms_applied', 0)}/{result.get('transforms_bypassed', 0)}")
    
    print(f"\n📊 TMC v7.0 テスト結果: {success_count}/{total_tests} 成功")
    print(f"📈 統計: 変換適用={engine.stats['transforms_applied']}, 変換スキップ={engine.stats['transforms_bypassed']}")
    
    if success_count == total_tests:
        print("🎉 TMC v7.0 インテリジェント圧縮プラットフォーム準備完了!")
        print("🔥 インテリジェント・バイパス + ポストBWTパイプライン + 並列チャンク処理 統合完了!")
    else:
        print("⚠️ 一部テスト失敗 - さらなる最適化が必要")
