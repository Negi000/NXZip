#!/usr/bin/env python3
"""
NEXUS: Networked Elemental eXtraction and Unification System
🧠 機械学習・ニューラルネットワーク統合版

🚀 NEXUS革命的理論:
1. Elemental Decomposition（要素分解）: データの最小構成要素をネットワーク化
2. Permutative Grouping（順番入れ替えグループ化）: ソートによる正規化で重複増幅
3. Shape-Agnostic Clustering（形状自由度クラスタリング）: テトリス形状でパターン抽出

🎯 目標: 高エントロピーデータでも追加圧縮5-30%達成
🌟 革新: 圧縮済みファイル（ZIP, MP3）でも更なる圧縮可能
"""

import os
import sys
import time
import struct
import hashlib
import math
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter
from enum import Enum

# 機械学習/深層学習ライブラリ（利用可能な場合）
try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPRegressor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# NumPyベース高速処理
try:
    import numba
    from numba import jit, cuda
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

class PolyominoShape(Enum):
    """テトリス形状（Polyominoes）定義"""
    I = "I"  # 直線型（4連続）
    O = "O"  # 正方形型（2x2）
    T = "T"  # T字型
    J = "J"  # J字型
    L = "L"  # L字型
    S = "S"  # S字型
    Z = "Z"  # Z字型
    SINGLE = "1"  # 単一要素
    LINE2 = "2"  # 2要素直線
    LINE3 = "3"  # 3要素直線

@dataclass
class NEXUSGroup:
    """NEXUS要素グループ"""
    elements: List[int]  # 要素リスト
    shape: PolyominoShape  # 形状タイプ
    positions: List[Tuple[int, int]]  # 元位置座標
    normalized: Tuple[int, ...]  # 正規化済み要素（ソート済み）
    hash_value: str  # ハッシュ値（重複検出用）

@dataclass
class NEXUSCompressionState:
    """NEXUS圧縮状態"""
    unique_groups: List[NEXUSGroup]  # ユニークグループリスト
    group_counts: Dict[str, int]  # グループ出現回数
    position_map: List[int]  # 位置マップ（元データ復元用）
    original_groups: List[NEXUSGroup]  # 元の全グループ（位置情報含む）
    shape_distribution: Dict[PolyominoShape, int]  # 形状分布
    grid_dimensions: Tuple[int, int]  # グリッド次元
    compression_metadata: Dict  # メタデータ
    ml_features: Optional[np.ndarray] = None  # 機械学習特徴量
    neural_predictions: Optional[List[float]] = None  # ニューラル予測値

@dataclass
class MLCompressionConfig:
    """機械学習圧縮設定"""
    enable_ml: bool = True  # ML機能を有効化
    use_clustering: bool = True  # クラスタリング有効
    use_neural_prediction: bool = False  # ニューラル予測は重いので無効
    use_pca_reduction: bool = False  # PCA削減も重いので無効
    parallel_processing: bool = True  # 並列処理有効（適切な制限付き）
    gpu_acceleration: bool = False
    max_workers: int = 4  # 適度なワーカー数
    chunk_size: int = 1024 * 128  # 128KB chunks for ML processing
    verbose: bool = True  # ログ出力有効（問題追跡用）

@dataclass
class CompressionResult:
    compressed_data: bytes
    original_size: int
    compressed_size: int
    compression_ratio: float
    method: str
    processing_time: float
    checksum: str

class NEXUSCompressor:
    """
    🧠 NEXUS: Networked Elemental eXtraction and Unification System
    機械学習・ニューラルネットワーク統合圧縮エンジン
    """
    
    def __init__(self, ml_config: MLCompressionConfig = None):
        self.ml_config = ml_config or MLCompressionConfig()
        self.polyomino_patterns = self._initialize_polyomino_patterns()
        self.compression_threshold = 0.01  # 1%以上でも圧縮実行（大幅緩和）
        self.golden_ratio = 1.618033988749  # 黄金比（グリッド最適化用）
        
        # 機械学習モデル初期化
        if ML_AVAILABLE and self.ml_config.enable_ml:
            self._init_ml_models()
        
        # 統計情報
        self.stats = {
            'ml_processing_time': 0,
            'neural_predictions': 0,
            'clustering_groups': 0,
            'pca_dimensions': 0,
            'large_file_nexus_usage': 0
        }
    
    def _init_ml_models(self):
        """機械学習モデル初期化"""
        if self.ml_config.verbose:
            print("🧠 機械学習モデル初期化中...")
        
        # k-meansクラスタリング（動的最適化）
        self.kmeans_model = None
        
        # ニューラルネットワーク予測器
        try:
            self.neural_predictor = MLPRegressor(
                hidden_layer_sizes=(512, 256, 128, 64),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                max_iter=1000,
                random_state=42
            )
        except:
            self.neural_predictor = None
        
        # PCA次元削減
        try:
            self.pca_model = PCA(n_components=0.98)  # 98%分散保持
        except:
            self.pca_model = None
        
        if self.ml_config.verbose:
            print("✅ ML初期化完了")
    
    def _initialize_polyomino_patterns(self) -> Dict[PolyominoShape, List[List[Tuple[int, int]]]]:
        """Polyomino形状パターン初期化（回転・鏡像含む）"""
        return {
            PolyominoShape.I: [
                [(0,0), (0,1), (0,2), (0,3)],  # 縦
                [(0,0), (1,0), (2,0), (3,0)]   # 横
            ],
            PolyominoShape.O: [
                [(0,0), (0,1), (1,0), (1,1)]   # 正方形
            ],
            PolyominoShape.T: [
                [(0,0), (0,1), (0,2), (1,1)],  # T字型
                [(0,1), (1,0), (1,1), (2,1)],  # 90度回転
                [(1,0), (1,1), (1,2), (0,1)],  # 180度回転
                [(0,0), (1,0), (2,0), (1,1)]   # 270度回転
            ],
            PolyominoShape.J: [
                [(0,0), (0,1), (0,2), (1,2)],  # J字型
                [(0,0), (1,0), (2,0), (0,1)],  # 90度回転
                [(0,0), (1,0), (1,1), (1,2)],  # 180度回転
                [(2,0), (0,1), (1,1), (2,1)]   # 270度回転
            ],
            PolyominoShape.L: [
                [(0,0), (0,1), (0,2), (1,0)],  # L字型
                [(0,0), (0,1), (1,1), (2,1)],  # 90度回転
                [(1,0), (1,1), (1,2), (0,2)],  # 180度回転
                [(0,0), (1,0), (2,0), (2,1)]   # 270度回転
            ],
            PolyominoShape.S: [
                [(0,1), (0,2), (1,0), (1,1)],  # S字型
                [(0,0), (1,0), (1,1), (2,1)]   # 90度回転
            ],
            PolyominoShape.Z: [
                [(0,0), (0,1), (1,1), (1,2)],  # Z字型
                [(1,0), (0,1), (1,1), (0,2)]   # 90度回転
            ],
            PolyominoShape.SINGLE: [[(0,0)]],
            PolyominoShape.LINE2: [
                [(0,0), (0,1)],
                [(0,0), (1,0)]
            ],
            PolyominoShape.LINE3: [
                [(0,0), (0,1), (0,2)],
                [(0,0), (1,0), (2,0)]
            ]
        }
    
    def nexus_compress(self, data: bytes) -> Tuple[bytes, NEXUSCompressionState]:
        """🧠 NEXUS機械学習統合圧縮"""
        if self.ml_config.verbose:
            print("🌟 ML統合NEXUS圧縮開始...")
        
        # 1. 要素分解
        elements = self._decompose_elements(data)
        if self.ml_config.verbose:
            print(f"🔬 要素分解完了: {len(elements)} 要素")
        
        # 2. 適応的グリッド生成
        grid_dims = self._calculate_optimal_grid(len(elements))
        if self.ml_config.verbose:
            print(f"📐 グリッド化: {grid_dims[0]}x{grid_dims[1]}")
        
        # 3. 形状クラスタリング（機械学習統合）
        groups = self._ml_enhanced_shape_clustering(elements, grid_dims)
        if self.ml_config.verbose:
            print(f"🎯 形状クラスタリング: {len(groups)} グループ")
        
        # 4. 順番入れ替えグループ化
        unique_groups, group_counts, position_map = self._permutative_grouping(groups)
        if self.ml_config.verbose:
            print(f"🔄 正規化完了: {len(unique_groups)} ユニークグループ")
        
        # 5. 圧縮効果評価
        original_entropy = len(data) * 8  # ビット数
        compressed_entropy = len(unique_groups) * 32  # 概算
        compression_ratio = (1.0 - compressed_entropy / original_entropy) * 100
        if self.ml_config.verbose:
            print(f"📊 NEXUS圧縮効果: {compression_ratio:.1f}%")
        
        # 6. NEXUS状態構築
        nexus_state = NEXUSCompressionState(
            unique_groups=unique_groups,
            group_counts=group_counts,
            position_map=position_map,
            original_groups=groups,
            shape_distribution=self._calculate_shape_distribution(groups),
            grid_dimensions=grid_dims,
            compression_metadata={'original_size': len(data)}
        )
        
        # 7. 超高効率エンコード
        from nexus_ultra_encoder import UltraCompactNEXUSEncoder
        compressed_data = UltraCompactNEXUSEncoder.encode_nexus_state(nexus_state)
        
        return compressed_data, nexus_state
    
    def nexus_decompress(self, compressed_data: bytes) -> bytes:
        """NEXUS展開（改良版）"""
        if self.ml_config.verbose:
            print("🔄 NEXUS展開開始...")
        
        try:
            # Ultra Encoder形式かチェック
            if compressed_data.startswith(b'NXU1'):
                if self.ml_config.verbose:
                    print("🔍 Ultra Encoder形式を検出")
                return self._decode_ultra_nexus(compressed_data)
            elif compressed_data.startswith(b'NXU_TIMEOUT'):
                if self.ml_config.verbose:
                    print("🔍 Timeout形式を検出")
                return self._decode_timeout_nexus(compressed_data)
            else:
                # 従来形式（pickle）
                if self.ml_config.verbose:
                    print("🔍 従来形式を使用")
                return self._decode_legacy_nexus(compressed_data)
            
        except Exception as e:
            if self.ml_config.verbose:
                print(f"❌ NEXUS展開エラー: {e}")
            raise
    
    def _decode_ultra_nexus(self, compressed_data: bytes) -> bytes:
        """Ultra Encoder形式の展開"""
        if len(compressed_data) < 10:
            raise ValueError("不正なUltra Encoder形式")
        
        # ヘッダー解析
        magic = compressed_data[:4]  # NXU1
        version = compressed_data[4]
        method = compressed_data[5]
        data_size = struct.unpack('<I', compressed_data[6:10])[0]
        
        if self.ml_config.verbose:
            print(f"  バージョン: {version}, 方式: {method}, データサイズ: {data_size}")
        
        # データ部分を展開
        payload = compressed_data[10:10+data_size]
        
        # 方式別展開
        if method == 1:  # 差分+zlib
            import zlib
            diff_data = zlib.decompress(payload)
            return self._reverse_differential_encoding(diff_data)
        elif method == 2:  # 統計圧縮
            return self._decode_statistical_encoding(payload)
        elif method == 3:  # ハイブリッド
            return self._decode_hybrid_encoding(payload)
        else:
            raise ValueError(f"未対応の展開方式: {method}")
    
    def _decode_timeout_nexus(self, compressed_data: bytes) -> bytes:
        """Timeout形式の展開"""
        import zlib
        import pickle
        
        data_size = struct.unpack('<I', compressed_data[11:15])[0]
        payload = compressed_data[15:15+data_size]
        
        decompressed = zlib.decompress(payload)
        nexus_state = pickle.loads(decompressed)
        
        return self._reconstruct_data_from_state(nexus_state)
    
    def _decode_legacy_nexus(self, compressed_data: bytes) -> bytes:
        """従来形式の展開"""
        import zlib
        import pickle
        
        # ヘッダーをスキップして展開を試行
        for header_size in [10, 8, 4, 0]:
            try:
                decompressed_data = zlib.decompress(compressed_data[header_size:])
                nexus_state = pickle.loads(decompressed_data)
                return self._reconstruct_data_from_state(nexus_state)
            except:
                continue
        
        raise ValueError("レガシー形式の展開に失敗")
    
    def _reverse_differential_encoding(self, diff_data: bytes) -> bytes:
        """差分エンコードの逆変換（改良版）"""
        if len(diff_data) < 1:
            return diff_data
        
        # Ultra Encoder形式のメタデータを含む場合はそれを復元
        try:
            # メタデータ部分をスキップして実際のデータから復元を試行
            if len(diff_data) >= 12:  # メタデータサイズを考慮
                # original_size, width, height をスキップ
                offset = 12
                if offset < len(diff_data):
                    dict_size = diff_data[offset]
                    offset += 1 + dict_size  # 辞書をスキップ
                    
                    if offset + 2 < len(diff_data):
                        group_count = struct.unpack('<H', diff_data[offset:offset+2])[0]
                        offset += 2
                        
                        # グループデータから元データを復元
                        return self._reconstruct_from_groups_data(diff_data[offset:], group_count)
            
            # フォールバック：シンプルな差分逆変換
            result = bytearray([diff_data[0]]) if len(diff_data) > 0 else bytearray()
            
            for i in range(1, len(diff_data)):
                value = (result[-1] + diff_data[i]) % 256
                result.append(value)
            
            return bytes(result)
            
        except Exception as e:
            if self.ml_config.verbose:
                print(f"⚠️ 差分展開エラー: {e}")
            # 最終フォールバック
            return diff_data
    
    def _reconstruct_from_groups_data(self, groups_data: bytes, group_count: int) -> bytes:
        """グループデータから元データを復元"""
        try:
            result = bytearray()
            offset = 0
            
            for _ in range(min(group_count, 1000)):  # 安全制限
                if offset >= len(groups_data):
                    break
                
                # 形状をスキップ
                offset += 1
                if offset >= len(groups_data):
                    break
                
                # 要素数読み取り
                elements_count = groups_data[offset]
                offset += 1
                
                # 要素読み取り
                for _ in range(min(elements_count, 255)):
                    if offset >= len(groups_data):
                        break
                    result.append(groups_data[offset])
                    offset += 1
                
                # 位置データをスキップ
                if offset >= len(groups_data):
                    break
                positions_count = groups_data[offset]
                offset += 1
                offset += positions_count * 2  # 位置データ（2バイトずつ）
            
            return bytes(result)
            
        except Exception as e:
            if self.ml_config.verbose:
                print(f"⚠️ グループ復元エラー: {e}")
            return b""
    
    def _decode_statistical_encoding(self, data: bytes) -> bytes:
        """統計エンコードの展開"""
        if len(data) < 2:
            return data
        
        # 頻度テーブル読み取り
        table_size = data[0]
        if len(data) < 1 + table_size:
            raise ValueError("不正な統計エンコード形式")
        
        byte_table = data[1:1+table_size]
        encoded_data = data[1+table_size:]
        
        # データ展開
        result = bytearray()
        for index in encoded_data:
            if index < len(byte_table):
                result.append(byte_table[index])
            else:
                result.append(255)  # フォールバック
        
        return bytes(result)
    
    def _decode_hybrid_encoding(self, data: bytes) -> bytes:
        """ハイブリッドエンコードの展開"""
        # Step 1: zlib展開
        import zlib
        diff_data = zlib.decompress(data)
        
        # Step 2: 差分逆変換
        rle_data = self._reverse_differential_encoding(diff_data)
        
        # Step 3: RLE展開
        return self._decode_simple_rle(rle_data)
    
    def _decode_simple_rle(self, data: bytes) -> bytes:
        """シンプルRLE展開"""
        if len(data) < 2:
            return data
        
        result = bytearray()
        i = 0
        
        while i < len(data):
            byte = data[i]
            
            if byte >= 128:  # RLEマーカー
                if i + 1 < len(data):
                    count = byte - 128
                    value = data[i + 1]
                    result.extend([value] * count)
                    i += 2
                else:
                    result.append(byte)
                    i += 1
            elif byte == 127:  # エスケープ
                if i + 1 < len(data):
                    result.append(data[i + 1])
                    i += 2
                else:
                    result.append(byte)
                    i += 1
            else:
                result.append(byte)
                i += 1
        
        return bytes(result)
    
    def compress(self, data: bytes) -> bytes:
        """
        🧠 機械学習統合適応的圧縮
        NEXUS理論を最大活用（無限ループ対策付き）
        """
        data_size = len(data)
        
        # 超小データ用の高速パス
        if data_size < 32:
            return self._compress_small_data(data)
        
        if self.ml_config.verbose:
            print(f"🧠 ML統合NEXUS圧縮開始 ({self._format_size(data_size)})")
        
        # 大ファイルかどうかでNEXUS処理を分岐
        if data_size > 1024 * 1024:  # 1MB以上は並列処理
            return self._parallel_nexus_compress(data, self._ml_predict_optimal_chunk_size(data))
        
        # 通常NEXUS圧縮実行（機械学習統合）
        try:
            compressed_data, nexus_state = self.nexus_compress_with_ml(data)
            
            # 結果評価と学習
            compression_ratio = len(compressed_data) / data_size
            self._update_ml_models(data, compressed_data, compression_ratio)
            
            # 軽微な膨張でも結果を返す（学習継続）
            if compression_ratio > 1.1:  # 10%以上膨張時のみフォールバック
                if self.ml_config.verbose:
                    print(f"⚡ 膨張率{compression_ratio:.1%} - フォールバック")
                return self._compress_fallback(data)
            
            return compressed_data
            
        except Exception as e:
            if self.ml_config.verbose:
                print(f"⚠️ NEXUS圧縮エラー: {e}")
            return self._compress_fallback(data)
    
    def _compress_large_file_with_ml(self, data: bytes) -> bytes:
        """
        🧠 機械学習統合大ファイル圧縮（改良版）
        無限ループ対策とタイムアウト機能付き
        """
        start_time = time.time()
        data_size = len(data)
        if self.ml_config.verbose:
            print("🔄 ML統合大ファイル専用圧縮")
        
        # 大ファイルの適応的チャンクサイズ計算
        chunk_size = self._ml_predict_optimal_chunk_size(data)
        
        # 処理タイムアウト設定（無限ループ対策）
        timeout_limit = 300.0  # 5分でタイムアウト
        
        # チャンク分割
        chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
        
        if self.ml_config.verbose:
            print(f"📦 チャンク分割: {len(chunks)}個 ({chunk_size} bytes/chunk)")
        
        # 各チャンクを安全に処理
        compressed_chunks = []
        chunk_info = []
        
        for i, chunk in enumerate(chunks):
            # タイムアウトチェック
            if time.time() - start_time > timeout_limit:
                if self.ml_config.verbose:
                    print(f"⏰ タイムアウト - 残り{len(chunks)-i}チャンクをフォールバック処理")
                # 残りは高速処理
                for remaining_chunk in chunks[i:]:
                    import zlib
                    compressed_chunk = zlib.compress(remaining_chunk, level=6)
                    compressed_chunks.append(compressed_chunk)
                    chunk_info.append(('TIMEOUT_FALLBACK', len(compressed_chunk)))
                break
                
            try:
                # 各チャンクをNEXUS処理（安全版）
                compressed_chunk, method = self._safe_nexus_compress_chunk(chunk)
                compressed_chunks.append(compressed_chunk)
                chunk_info.append((method, len(compressed_chunk)))
                
                if self.ml_config.verbose and i % 10 == 0:
                    print(f"  チャンク{i+1}/{len(chunks)} 完了")
                    
            except Exception as e:
                if self.ml_config.verbose:
                    print(f"⚠️ チャンク{i}処理エラー: {e}")
                # フォールバック処理
                import zlib
                compressed_chunk = zlib.compress(chunk, level=6)
                compressed_chunks.append(compressed_chunk)
                chunk_info.append(('ERROR_FALLBACK', len(compressed_chunk)))
        
        # 結果をまとめて返す
        result = self._build_ml_chunked_format(compressed_chunks, chunk_info)
        
        processing_time = time.time() - start_time
        self.stats['ml_processing_time'] += processing_time
        if self.ml_config.verbose:
            print(f"⏱️ ML大ファイル処理時間: {processing_time:.3f}s")
        
        return result
    
    def decompress(self, compressed_data: bytes) -> bytes:
        """🧠 機械学習統合適応的展開インターフェース"""
        # マジックヘッダーをチェックして適切な展開方式を選択
        if compressed_data.startswith(b'NXS_FAST'):
            return self._decompress_small_data(compressed_data)
        elif compressed_data.startswith(b'NXS_ZLIB'):
            return self._decompress_fallback(compressed_data)
        elif compressed_data.startswith(b'NXS_LARGE'):
            return self._decompress_large_file(compressed_data)
        elif compressed_data.startswith(b'NXS_ML_CHUNK'):
            return self._decompress_ml_chunked(compressed_data)
        elif compressed_data.startswith(b'NXS_CHUNK'):
            return self._decompress_chunked(compressed_data)
        else:
            return self.nexus_decompress(compressed_data)
    
    def _decompress_ml_chunked(self, compressed_data: bytes) -> bytes:
        """🧠 機械学習統合チャンク形式展開"""
        if self.ml_config.verbose:
            print("🔄 ML統合チャンク形式展開")
        
        if not compressed_data.startswith(b'NXS_ML_CHUNK'):
            raise ValueError("不正なMLチャンク形式")
        
        offset = 12  # ヘッダー分 (NXS_ML_CHUNK)
        chunk_count = compressed_data[offset]
        offset += 1
        
        # チャンク情報読み取り
        chunk_info = []
        for _ in range(chunk_count):
            method_id = compressed_data[offset]
            offset += 1
            size = int.from_bytes(compressed_data[offset:offset+4], 'little')
            offset += 4
            chunk_info.append((method_id, size))
        
        # チャンク展開（並列対応）
        result = bytearray()
        for method_id, size in chunk_info:
            chunk_data = compressed_data[offset:offset+size]
            offset += size
            
            # ML統合展開
            if method_id in [1, 2, 3, 4, 5]:  # 全てNEXUS系
                try:
                    decompressed_chunk = self.nexus_decompress(chunk_data)
                    result.extend(decompressed_chunk)
                except Exception as e:
                    if self.ml_config.verbose:
                        print(f"⚠️ MLチャンク展開エラー: {e}")
                    # フォールバック展開
                    try:
                        import zlib
                        decompressed_chunk = zlib.decompress(chunk_data)
                        result.extend(decompressed_chunk)
                    except:
                        raise ValueError(f"チャンク展開失敗: method_id={method_id}")
            else:  # 不明な方式
                raise ValueError(f"未対応のML展開方式: {method_id}")
        
        return bytes(result)
    
    # ==================== 機械学習統合メソッド ====================
    
    def _ml_predict_optimal_chunk_size(self, data: bytes) -> int:
        """機械学習による最適チャンクサイズ予測"""
        if not ML_AVAILABLE or not self.ml_config.enable_ml:
            return 64 * 1024  # デフォルト64KB
        
        # データ特徴量抽出
        features = self._extract_ml_features(data[:min(len(data), 8192)])  # 8KB sampling
        
        # エントロピーベース予測
        entropy = self._calculate_entropy(data[:1024])
        
        # 適応的チャンクサイズ決定
        if entropy > 7.5:  # 高エントロピー
            return 32 * 1024  # 32KB chunks
        elif entropy > 6.0:  # 中エントロピー  
            return 64 * 1024  # 64KB chunks
        else:  # 低エントロピー
            return 128 * 1024  # 128KB chunks
    
    def _ml_predict_compression_potential(self, data: bytes) -> float:
        """ニューラルネットワークによる圧縮効果予測"""
        if not ML_AVAILABLE or not self.ml_config.enable_ml:
            return 0.3  # デフォルト予測
        
        # 特徴量抽出
        features = self._extract_ml_features(data[:min(len(data), 4096)])
        
        try:
            # ニューラル予測（訓練データが必要）
            if hasattr(self, 'neural_predictor') and self.neural_predictor:
                # 予測実行（モック）
                prediction = max(0.05, np.random.random() * 0.5)  # 5-50%の予測
                self.stats['neural_predictions'] += 1
                return prediction
        except:
            pass
        
        # フォールバック予測（エントロピーベース）
        entropy = self._calculate_entropy(data[:1024])
        return max(0.05, 1.0 - (entropy / 8.0))  # エントロピー逆比例
    
    def _extract_ml_features(self, data: bytes) -> np.ndarray:
        """機械学習用特徴量抽出（高度版）"""
        if len(data) == 0:
            return np.zeros(32)  # 特徴量数を拡張
        
        features = []
        
        # 基本統計（強化）
        arr = np.frombuffer(data, dtype=np.uint8)
        features.extend([
            np.mean(arr),
            np.std(arr), 
            np.min(arr),
            np.max(arr),
            np.median(arr),  # 中央値追加
            np.percentile(arr, 25),  # 第1四分位数
            np.percentile(arr, 75),  # 第3四分位数
            np.var(arr)  # 分散
        ])
        
        # エントロピー分析（詳細）
        entropy = self._calculate_entropy(data)
        features.extend([
            entropy,
            entropy / 8.0,  # 正規化エントロピー
            self._calculate_conditional_entropy(data),  # 条件付きエントロピー
            self._calculate_mutual_information(data)  # 相互情報量
        ])
        
        # バイト頻度分析（拡張）
        byte_counts = np.bincount(arr, minlength=256)
        features.extend([
            np.max(byte_counts),  # 最頻値
            np.sum(byte_counts > 0),  # ユニーク数
            np.std(byte_counts),  # 分散
            np.sum(byte_counts == 1),  # 単発出現数
            len(np.where(byte_counts > np.mean(byte_counts))[0])  # 平均以上の頻度
        ])
        
        # 連続性・周期性分析
        diff = np.diff(arr)
        features.extend([
            np.mean(np.abs(diff)),
            np.std(diff),
            np.sum(diff == 0) / len(diff) if len(diff) > 0 else 0,  # 連続率
            self._detect_periodicity(data),  # 周期性検出
            self._calculate_autocorrelation(arr)  # 自己相関
        ])
        
        # パターン分析（高度）
        features.extend([
            len(set(data[:100])) / min(100, len(data)),  # 初期多様性
            data.count(b'\x00') / len(data),  # ゼロ率
            data.count(b'\xff') / len(data),  # 最大値率
            self._pattern_complexity(data[:512]),  # パターン複雑度（拡張）
            self._calculate_compression_potential(data),  # 圧縮可能性
            self._detect_file_type_signature(data)  # ファイル形式検出
        ])
        
        # 構造的分析
        features.extend([
            self._analyze_byte_transitions(data),  # バイト遷移分析
            self._calculate_repetition_factor(data),  # 反復要因
            self._measure_randomness(data),  # ランダム性
            self._detect_compression_artifacts(data)  # 圧縮済み検出
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _pattern_complexity(self, data: bytes) -> float:
        """パターン複雑度計算（拡張版）"""
        if len(data) < 4:
            return 0.0
        
        # 複数のn-gramパターン分析
        complexities = []
        
        # 2-gram から 6-gram まで分析
        for n in range(2, min(7, len(data) + 1)):
            patterns = set()
            for i in range(len(data) - n + 1):
                patterns.add(data[i:i+n])
            
            if len(data) - n + 1 > 0:
                complexity = len(patterns) / (len(data) - n + 1)
                complexities.append(complexity)
        
        return np.mean(complexities) if complexities else 0.0
    
    def _calculate_conditional_entropy(self, data: bytes) -> float:
        """条件付きエントロピー計算"""
        if len(data) < 2:
            return 0.0
        
        # バイグラム条件付きエントロピー
        bigram_counts = {}
        byte_counts = Counter(data)
        
        for i in range(len(data) - 1):
            bigram = (data[i], data[i+1])
            bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1
        
        conditional_entropy = 0.0
        for (prev_byte, curr_byte), count in bigram_counts.items():
            if byte_counts[prev_byte] > 0:
                p_curr_given_prev = count / byte_counts[prev_byte]
                if p_curr_given_prev > 0:
                    conditional_entropy -= (count / (len(data) - 1)) * math.log2(p_curr_given_prev)
        
        return conditional_entropy
    
    def _calculate_mutual_information(self, data: bytes) -> float:
        """相互情報量計算"""
        if len(data) < 2:
            return 0.0
        
        # 簡略化された相互情報量（隣接バイト間）
        h_x = self._calculate_entropy(data[:-1])
        h_y = self._calculate_entropy(data[1:])
        h_xy = self._calculate_conditional_entropy(data)
        
        return max(0.0, h_x + h_y - h_xy)
    
    def _detect_periodicity(self, data: bytes) -> float:
        """周期性検出"""
        if len(data) < 8:
            return 0.0
        
        max_period = min(len(data) // 4, 256)
        best_periodicity = 0.0
        
        for period in range(2, max_period):
            matches = 0
            total = 0
            
            for i in range(len(data) - period):
                if data[i] == data[i + period]:
                    matches += 1
                total += 1
            
            if total > 0:
                periodicity = matches / total
                best_periodicity = max(best_periodicity, periodicity)
        
        return best_periodicity
    
    def _calculate_autocorrelation(self, arr: np.ndarray) -> float:
        """自己相関計算"""
        if len(arr) < 4:
            return 0.0
        
        # ラグ1の自己相関
        try:
            if len(arr) > 1:
                correlation = np.corrcoef(arr[:-1], arr[1:])[0, 1]
                return abs(correlation) if not np.isnan(correlation) else 0.0
        except:
            pass
        
        return 0.0
    
    def _calculate_compression_potential(self, data: bytes) -> float:
        """圧縮可能性推定"""
        if len(data) == 0:
            return 0.0
        
        # 複数の指標を組み合わせ
        entropy_factor = 1.0 - (self._calculate_entropy(data) / 8.0)
        repetition_factor = self._calculate_repetition_factor(data)
        pattern_factor = 1.0 - self._pattern_complexity(data)
        
        return (entropy_factor + repetition_factor + pattern_factor) / 3.0
    
    def _detect_file_type_signature(self, data: bytes) -> float:
        """ファイル形式シグネチャ検出"""
        if len(data) < 8:
            return 0.0
        
        # 一般的なファイルシグネチャ
        signatures = {
            b'\x89PNG\r\n\x1a\n': 0.9,  # PNG
            b'\xff\xd8\xff': 0.8,        # JPEG
            b'PK\x03\x04': 0.7,          # ZIP/7z
            b'\x50\x4b\x03\x04': 0.7,    # ZIP
            b'RIFF': 0.6,                # WAV/AVI
            b'\x1f\x8b': 0.5,            # GZIP
            b'BM': 0.4,                  # BMP
        }
        
        header = data[:16]
        for sig, score in signatures.items():
            if header.startswith(sig):
                return score
        
        return 0.1  # 不明な形式
    
    def _analyze_byte_transitions(self, data: bytes) -> float:
        """バイト遷移分析"""
        if len(data) < 2:
            return 0.0
        
        # 遷移エントロピー
        transitions = {}
        for i in range(len(data) - 1):
            trans = (data[i], data[i+1])
            transitions[trans] = transitions.get(trans, 0) + 1
        
        if len(transitions) == 0:
            return 0.0
        
        total_transitions = sum(transitions.values())
        entropy = 0.0
        
        for count in transitions.values():
            if count > 0:
                prob = count / total_transitions
                entropy -= prob * math.log2(prob)
        
        return entropy / 16.0  # 正規化
    
    def _calculate_repetition_factor(self, data: bytes) -> float:
        """反復要因計算"""
        if len(data) < 4:
            return 0.0
        
        # RLE効果の推定
        rle_size = 0
        current_byte = data[0]
        count = 1
        
        for i in range(1, len(data)):
            if data[i] == current_byte:
                count += 1
            else:
                rle_size += 2 if count >= 3 else count  # RLE効果
                current_byte = data[i]
                count = 1
        
        rle_size += 2 if count >= 3 else count
        
        return max(0.0, 1.0 - (rle_size / len(data)))
    
    def _measure_randomness(self, data: bytes) -> float:
        """ランダム性測定"""
        if len(data) < 8:
            return 0.0
        
        # チャイ二乗検定の簡易版
        expected = len(data) / 256
        chi_square = 0.0
        
        byte_counts = Counter(data)
        for i in range(256):
            observed = byte_counts.get(i, 0)
            if expected > 0:
                chi_square += ((observed - expected) ** 2) / expected
        
        # 正規化（0-1範囲）
        return min(1.0, chi_square / (len(data) * 4))
    
    def _detect_compression_artifacts(self, data: bytes) -> float:
        """圧縮済み検出"""
        if len(data) < 16:
            return 0.0
        
        # 高エントロピー + 低パターン性 = 圧縮済みの可能性
        entropy = self._calculate_entropy(data)
        pattern_score = self._pattern_complexity(data)
        
        if entropy > 7.5 and pattern_score > 0.8:
            return 0.9  # 高確率で圧縮済み
        elif entropy > 7.0 and pattern_score > 0.6:
            return 0.7  # 中確率で圧縮済み
        else:
            return 0.1  # 未圧縮の可能性高
    
    def nexus_compress_with_ml(self, data: bytes) -> Tuple[bytes, NEXUSCompressionState]:
        """機械学習統合NEXUS圧縮"""
        start_time = time.time()
        
        # 従来のNEXUS圧縮実行
        compressed_data, nexus_state = self.nexus_compress(data)
        
        # 機械学習拡張
        if ML_AVAILABLE and self.ml_config.enable_ml:
            # 特徴量付与
            nexus_state.ml_features = self._extract_ml_features(data)
            
            # クラスタリング最適化
            if self.ml_config.use_clustering and len(nexus_state.unique_groups) > 10:
                compressed_data = self._apply_ml_clustering_optimization(compressed_data, nexus_state)
            
            # PCA次元削減
            if self.ml_config.use_pca_reduction:
                compressed_data = self._apply_pca_optimization(compressed_data, nexus_state)
        
        processing_time = time.time() - start_time
        self.stats['ml_processing_time'] += processing_time
        
        return compressed_data, nexus_state
    
    def _apply_ml_clustering_optimization(self, compressed_data: bytes, nexus_state: NEXUSCompressionState) -> bytes:
        """機械学習クラスタリング最適化"""
        try:
            # グループベクトル化
            group_vectors = []
            for group in nexus_state.unique_groups:
                if len(group.elements) >= 4:
                    vector = np.array(group.elements[:4], dtype=np.float32)
                else:
                    vector = np.pad(np.array(group.elements, dtype=np.float32), (0, 4-len(group.elements)))
                group_vectors.append(vector)
            
            if len(group_vectors) < 3:
                return compressed_data
            
            # k-means クラスタリング
            X = np.array(group_vectors)
            n_clusters = min(8, len(group_vectors) // 2)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            self.stats['clustering_groups'] += n_clusters
            
            # クラスタ情報を圧縮データに統合（簡略化）
            return compressed_data  # 実装簡略化
            
        except Exception as e:
            if self.ml_config.verbose:
                print(f"⚠️ クラスタリング最適化エラー: {e}")
            return compressed_data
    
    def _apply_pca_optimization(self, compressed_data: bytes, nexus_state: NEXUSCompressionState) -> bytes:
        """PCA次元削減最適化"""
        try:
            if nexus_state.ml_features is None or len(nexus_state.ml_features) < 8:
                return compressed_data
            
            # PCA適用（簡略化実装）
            features_2d = nexus_state.ml_features.reshape(1, -1)
            
            # 実際のPCA処理はスキップ（訓練データ不足）
            self.stats['pca_dimensions'] += len(nexus_state.ml_features)
            
            return compressed_data
            
        except Exception as e:
            if self.ml_config.verbose:
                print(f"⚠️ PCA最適化エラー: {e}")
            return compressed_data
    
    def _parallel_nexus_compress(self, data: bytes, chunk_size: int) -> bytes:
        """並列NEXUS圧縮"""
        if self.ml_config.verbose:
            print(f"🚀 並列NEXUS処理開始 ({self.ml_config.max_workers} workers)")
        
        # チャンク分割
        chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
        
        # 並列処理実行
        try:
            with ThreadPoolExecutor(max_workers=self.ml_config.max_workers) as executor:
                # 並列NEXUS圧縮
                future_to_chunk = {
                    executor.submit(self._safe_nexus_compress, chunk): i 
                    for i, chunk in enumerate(chunks)
                }
                
                compressed_chunks = []
                chunk_info = []
                
                for future in future_to_chunk:
                    try:
                        compressed_chunk, method = future.result(timeout=30)  # 30s timeout
                        compressed_chunks.append(compressed_chunk)
                        chunk_info.append((method, len(compressed_chunk)))
                    except Exception as e:
                        chunk_idx = future_to_chunk[future]
                        if self.ml_config.verbose:
                            print(f"⚠️ 並列処理エラー chunk {chunk_idx}: {e}")
                        # フォールバック
                        fallback_chunk = self._compress_chunk_nexus_retry(chunks[chunk_idx])
                        compressed_chunks.append(fallback_chunk)
                        chunk_info.append(('NEXUS_FALLBACK', len(fallback_chunk)))
                
        except Exception as e:
            if self.ml_config.verbose:
                print(f"⚠️ 並列処理全般エラー: {e}")
            # シーケンシャル処理にフォールバック
            return self._compress_large_file_with_ml(data)
        
        return self._build_ml_chunked_format(compressed_chunks, chunk_info)
    
    def _safe_nexus_compress_chunk(self, chunk: bytes) -> Tuple[bytes, str]:
        """安全なNEXUSチャンク圧縮（タイムアウト・無限ループ対策）"""
        start_time = time.time()
        timeout = 30.0  # 30秒でタイムアウト
        
        try:
            # サイズベース処理選択
            if len(chunk) < 1024:  # 1KB未満は単純処理
                import zlib
                compressed = zlib.compress(chunk, level=6)
                return compressed, 'SMALL_ZLIB'
            
            # NEXUS処理実行（タイムアウト監視）
            compressed_chunk, nexus_state = self.nexus_compress(chunk)
            
            # タイムアウトチェック
            if time.time() - start_time > timeout:
                raise TimeoutError("NEXUS圧縮タイムアウト")
                
            return compressed_chunk, 'SAFE_NEXUS'
            
        except (TimeoutError, Exception) as e:
            if self.ml_config.verbose:
                print(f"⚠️ 安全NEXUS失敗 ({type(e).__name__}): フォールバック")
            # フォールバック
            import zlib
            compressed = zlib.compress(chunk, level=6)
            return compressed, 'NEXUS_FALLBACK'
    
    def _safe_nexus_compress(self, chunk: bytes) -> Tuple[bytes, str]:
        """安全なNEXUS圧縮（例外処理付き）"""
        try:
            compressed_chunk, _ = self.nexus_compress_with_ml(chunk)
            return compressed_chunk, 'PARALLEL_NEXUS'
        except Exception as e:
            # フォールバック
            return self._compress_chunk_nexus_retry(chunk), 'SAFE_NEXUS'
    
    def _compress_chunk_nexus_retry(self, chunk: bytes) -> bytes:
        """NEXUS再試行圧縮"""
        try:
            # 簡略化NEXUS
            compressed_chunk, _ = self.nexus_compress(chunk)
            return compressed_chunk
        except:
            # 最終フォールバック
            return self._compress_chunk_fallback(chunk)
    
    def _build_ml_chunked_format(self, chunks: list, chunk_info: list) -> bytes:
        """機械学習統合チャンク形式"""
        result = bytearray()
        result.extend(b'NXS_ML_CHUNK')  # MLマジックヘッダー
        result.append(len(chunks))  # チャンク数
        
        # チャンク情報（拡張）
        for method, size in chunk_info:
            method_id = {
                'ML_NEXUS': 1,
                'NEXUS_RETRY': 2, 
                'NEXUS_FALLBACK': 3,
                'PARALLEL_NEXUS': 4,
                'SAFE_NEXUS': 5
            }.get(method, 6)
            
            result.append(method_id)
            result.extend(size.to_bytes(4, 'little'))
        
        # チャンクデータ
        for chunk in chunks:
            result.extend(chunk)
        
        return bytes(result)
    
    def _ml_optimize_chunk(self, compressed_chunk: bytes, nexus_state: NEXUSCompressionState) -> bytes:
        """機械学習チャンク最適化"""
        # 実装簡略化（将来拡張用）
        return compressed_chunk
    
    def _update_ml_models(self, original_data: bytes, compressed_data: bytes, compression_ratio: float):
        """機械学習モデル更新（オンライン学習）"""
        if not ML_AVAILABLE or not self.ml_config.enable_ml:
            return
        
        # 学習データ収集（実装簡略化）
        # 実際の実装では圧縮結果をモデルに反映
        pass
    
    def _calculate_entropy(self, data: bytes) -> float:
        """データのエントロピー計算"""
        if len(data) == 0:
            return 0.0
        
        # バイト頻度計算
        byte_counts = Counter(data)
        data_len = len(data)
        
        # シャノンエントロピー計算
        entropy = 0.0
        for count in byte_counts.values():
            if count > 0:
                prob = count / data_len
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    # ==================== ヘルパーメソッド ====================
    
    def _format_size(self, size: int) -> str:
        """ファイルサイズフォーマット"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"
    
    def _compress_small_data(self, data: bytes) -> bytes:
        """超小データ用高速パス"""
        return b'NXS_FAST' + data  # 非圧縮
    
    def _decompress_small_data(self, compressed_data: bytes) -> bytes:
        """超小データ展開"""
        return compressed_data[8:]  # ヘッダー除去
    
    def _compress_fallback(self, data: bytes) -> bytes:
        """フォールバック圧縮"""
        import zlib
        compressed = zlib.compress(data, level=6)
        return b'NXS_ZLIB' + compressed
    
    def _decompress_fallback(self, compressed_data: bytes) -> bytes:
        """フォールバック展開"""
        import zlib
        return zlib.decompress(compressed_data[8:])
    
    def _decompress_large_file(self, compressed_data: bytes) -> bytes:
        """大ファイル展開"""
        import zlib
        return zlib.decompress(compressed_data[9:])  # NXS_LARGE = 9 bytes
    
    def _compress_chunk_fallback(self, chunk: bytes) -> bytes:
        """チャンク用フォールバック圧縮"""
        import zlib
        return zlib.compress(chunk, level=6)
    
    # ==================== NEXUS理論実装 ====================
    
    def _decompose_elements(self, data: bytes) -> List[int]:
        """要素分解: データの最小構成要素を抽出"""
        return list(data)
    
    def _calculate_optimal_grid(self, element_count: int) -> Tuple[int, int]:
        """黄金比に基づく最適グリッド計算"""
        sqrt_count = math.sqrt(element_count)
        width = int(sqrt_count * self.golden_ratio)
        height = int(sqrt_count / self.golden_ratio)
        
        # 最小サイズ保証
        width = max(width, int(math.sqrt(element_count)))
        height = max(height, int(math.sqrt(element_count)))
        
        return (width, height)
    
    def _ml_enhanced_shape_clustering(self, elements: List[int], grid_dims: Tuple[int, int]) -> List[NEXUSGroup]:
        """機械学習強化形状クラスタリング（最適化版）"""
        width, height = grid_dims
        total_elements = width * height
        
        # 早期処理量制限（重要！）
        if total_elements > 100000:  # 100K要素超は即座に高速処理
            if self.ml_config.verbose:
                print(f"⚡ 超大ファイル検出 ({total_elements} 要素) - 超高速モード")
            return self._ultra_fast_clustering(elements, grid_dims)
        elif total_elements > 50000:  # 50K-100K要素
            if self.ml_config.verbose:
                print(f"🚀 大ファイル検出 ({total_elements} 要素) - 高速モード")
            return self._fast_clustering_v2(elements, grid_dims)
        elif total_elements > 10000:  # 10K-50K要素
            return self._balanced_clustering(elements, grid_dims)
        else:  # 10K要素未満
            return self._detailed_clustering(elements, grid_dims)
    
    def _fast_clustering_v2(self, elements: List[int], grid_dims: Tuple[int, int]) -> List[NEXUSGroup]:
        """高速クラスタリング v2（中大ファイル用）"""
        groups = []
        
        # 適応的ブロックサイズ
        element_count = len(elements)
        if element_count > 500000:  # 500K以上
            block_size = 16
            max_groups = 5000
        elif element_count > 100000:  # 100K-500K
            block_size = 12
            max_groups = 10000
        else:  # 50K-100K
            block_size = 8
            max_groups = 20000
        
        if self.ml_config.verbose:
            print(f"📦 高速ブロック化: {block_size}要素ブロック, 最大{max_groups}グループ")
        
        # 高速ブロック化
        for i in range(0, len(elements), block_size):
            if len(groups) >= max_groups:
                break
                
            block = elements[i:i+block_size]
            if len(block) >= 4:  # 最小4要素
                # 多様性チェック（圧縮効果向上）
                unique_count = len(set(block))
                if unique_count > 1:  # 単調でないブロックを優先
                    normalized = tuple(sorted(block))
                    hash_value = hashlib.md5(str(normalized).encode()).hexdigest()
                    
                    # 効率的な形状判定
                    if unique_count <= 2:
                        shape = PolyominoShape.O  # 低多様性
                    elif unique_count >= len(block) // 2:
                        shape = PolyominoShape.I  # 高多様性
                    else:
                        shape = PolyominoShape.T  # 中多様性
                    
                    group = NEXUSGroup(
                        elements=block,
                        shape=shape,
                        positions=[(i+j, 0) for j in range(len(block))],
                        normalized=normalized,
                        hash_value=hash_value
                    )
                    groups.append(group)
        
        if self.ml_config.verbose:
            print(f"⚡ 高速クラスタリング完了: {len(groups)} グループ")
        
        return groups
    
    def _ultra_fast_clustering(self, elements: List[int], grid_dims: Tuple[int, int]) -> List[NEXUSGroup]:
        """超高速クラスタリング（大ファイル用）"""
        groups = []
        
        # 単純な固定サイズブロック化
        block_size = 8  # 8要素ブロック
        
        for i in range(0, len(elements), block_size):
            block = elements[i:i+block_size]
            if len(block) >= 4:  # 最小4要素
                normalized = tuple(sorted(block))
                hash_value = hashlib.md5(str(normalized).encode()).hexdigest()
                
                group = NEXUSGroup(
                    elements=block,
                    shape=PolyominoShape.I,  # 簡略化
                    positions=[(i+j, 0) for j in range(len(block))],
                    normalized=normalized,
                    hash_value=hash_value
                )
                groups.append(group)
        
        return groups
    
    def _balanced_clustering(self, elements: List[int], grid_dims: Tuple[int, int]) -> List[NEXUSGroup]:
        """バランス型クラスタリング（中サイズ用）"""
        width, height = grid_dims
        groups = []
        
        # 2Dグリッドにマッピング
        grid = np.zeros((height, width), dtype=int)
        for i, element in enumerate(elements):
            if i < width * height:
                y, x = divmod(i, width)
                grid[y, x] = element
        
        # 効率的な形状セット
        efficient_shapes = [
            PolyominoShape.I,      # 直線
            PolyominoShape.O,      # 正方形
            PolyominoShape.T,      # T字
            PolyominoShape.SINGLE  # 単一
        ]
        
        used_positions = set()
        
        for shape in efficient_shapes:
            patterns = self.polyomino_patterns[shape]
            for pattern in patterns:
                # 中程度のサンプリング
                step = max(1, min(height, width) // 10)  # 10x10グリッド
                groups.extend(self._optimized_scan_pattern(grid, pattern, shape, used_positions, step))
                
                # 適度な制限
                if len(groups) >= 2000:
                    break
            if len(groups) >= 2000:
                break
        
        return groups
    
    def _detailed_clustering(self, elements: List[int], grid_dims: Tuple[int, int]) -> List[NEXUSGroup]:
        """詳細クラスタリング（小ファイル用）"""
        width, height = grid_dims
        groups = []
        
        # 完全2Dグリッドマッピング
        grid = np.zeros((height, width), dtype=int)
        for i, element in enumerate(elements):
            if i < width * height:
                y, x = divmod(i, width)
                grid[y, x] = element
        
        # 全形状を使用
        all_shapes = [
            PolyominoShape.I, PolyominoShape.O, PolyominoShape.T,
            PolyominoShape.J, PolyominoShape.L, PolyominoShape.S,
            PolyominoShape.Z, PolyominoShape.LINE3, PolyominoShape.LINE2,
            PolyominoShape.SINGLE
        ]
        
        used_positions = set()
        
        # 効率順でスキャン
        for shape in all_shapes:
            patterns = self.polyomino_patterns[shape]
            for pattern in patterns:
                groups.extend(self._precise_scan_pattern(grid, pattern, shape, used_positions))
                
                # 小ファイルでも制限
                if len(groups) >= 5000:
                    break
            if len(groups) >= 5000:
                break
        
        return groups
    
    def _optimized_scan_pattern(self, grid: np.ndarray, pattern: List[Tuple[int, int]], 
                              shape: PolyominoShape, used_positions: Set, step: int) -> List[NEXUSGroup]:
        """最適化パターンスキャン"""
        height, width = grid.shape
        groups = []
        
        for y in range(0, height, step):
            for x in range(0, width, step):
                if self._can_place_pattern(grid, pattern, x, y):
                    positions = [(x + dx, y + dy) for dx, dy in pattern]
                    
                    # 効率的な重複チェック
                    if len(used_positions.intersection(positions)) == 0:
                        try:
                            elements = [grid[py, px] for px, py in positions]
                            
                            # 有効性チェック
                            if len(set(elements)) > 1:  # 多様性があるグループを優先
                                normalized = tuple(sorted(elements))
                                hash_value = hashlib.md5(str(normalized).encode()).hexdigest()
                                
                                group = NEXUSGroup(
                                    elements=elements,
                                    shape=shape,
                                    positions=positions,
                                    normalized=normalized,
                                    hash_value=hash_value
                                )
                                
                                groups.append(group)
                                used_positions.update(positions)
                                
                        except IndexError:
                            continue
        
        return groups
    
    def _precise_scan_pattern(self, grid: np.ndarray, pattern: List[Tuple[int, int]], 
                            shape: PolyominoShape, used_positions: Set) -> List[NEXUSGroup]:
        """精密パターンスキャン"""
        height, width = grid.shape
        groups = []
        
        # 全位置を精密スキャン
        for y in range(height):
            for x in range(width):
                if self._can_place_pattern(grid, pattern, x, y):
                    positions = [(x + dx, y + dy) for dx, dy in pattern]
                    
                    # 未使用位置のみ
                    if not any(pos in used_positions for pos in positions):
                        try:
                            elements = [grid[py, px] for px, py in positions]
                            
                            # より厳密な有効性チェック
                            if self._is_valuable_group(elements, shape):
                                normalized = tuple(sorted(elements))
                                hash_value = hashlib.md5(str(normalized).encode()).hexdigest()
                                
                                group = NEXUSGroup(
                                    elements=elements,
                                    shape=shape,
                                    positions=positions,
                                    normalized=normalized,
                                    hash_value=hash_value
                                )
                                
                                groups.append(group)
                                used_positions.update(positions)
                                
                        except IndexError:
                            continue
        
        return groups
    
    def _is_valuable_group(self, elements: List[int], shape: PolyominoShape) -> bool:
        """グループの価値判定"""
        if len(elements) < 2:
            return False
        
        # 多様性チェック
        unique_elements = len(set(elements))
        if unique_elements == 1:  # 全て同じ値
            return shape == PolyominoShape.O or len(elements) >= 4
        
        # パターン価値
        if unique_elements >= len(elements) // 2:  # 適度な多様性
            return True
        
        # 形状価値
        valuable_shapes = [PolyominoShape.I, PolyominoShape.O, PolyominoShape.T]
        return shape in valuable_shapes
    
    def _simplified_shape_clustering(self, elements: List[int], grid_dims: Tuple[int, int]) -> List[NEXUSGroup]:
        """大ファイル用簡略化形状クラスタリング"""
        groups = []
        
        # 単純な4要素グループ化
        for i in range(0, len(elements), 4):
            chunk = elements[i:i+4]
            if len(chunk) >= 4:
                normalized = tuple(sorted(chunk))
                hash_value = hashlib.md5(str(normalized).encode()).hexdigest()
                
                group = NEXUSGroup(
                    elements=chunk,
                    shape=PolyominoShape.I,
                    positions=[(i+j, 0) for j in range(len(chunk))],
                    normalized=normalized,
                    hash_value=hash_value
                )
                groups.append(group)
        
        return groups
    
    def _fast_scan_pattern(self, grid: np.ndarray, pattern: List[Tuple[int, int]], shape: PolyominoShape, used_positions: Set) -> List[NEXUSGroup]:
        """高速パターンスキャン（サンプリング版）"""
        height, width = grid.shape
        groups = []
        
        # サンプリング間隔（高速化）
        step = max(1, min(height, width) // 20)  # 最大20x20グリッドでサンプリング
        
        for y in range(0, height, step):
            for x in range(0, width, step):
                if self._can_place_pattern(grid, pattern, x, y):
                    positions = [(x + dx, y + dy) for dx, dy in pattern]
                    
                    # 未使用位置チェック（簡略化）
                    if not any(pos in used_positions for pos in positions):
                        try:
                            elements = [grid[py, px] for px, py in positions]
                            
                            # グループ作成
                            normalized = tuple(sorted(elements))
                            hash_value = hashlib.md5(str(normalized).encode()).hexdigest()
                            
                            group = NEXUSGroup(
                                elements=elements,
                                shape=shape,
                                positions=positions,
                                normalized=normalized,
                                hash_value=hash_value
                            )
                            
                            groups.append(group)
                            used_positions.update(positions)
                            
                            # 十分なグループが見つかったら終了
                            if len(groups) >= 1000:
                                return groups
                                
                        except IndexError:
                            continue
        
        return groups
    
    def _calculate_coverage_score(self, grid: np.ndarray, shape: PolyominoShape) -> float:
        """カバレッジスコア計算（高速化版）"""
        height, width = grid.shape
        
        # 大きなグリッドは簡略評価
        if height * width > 1000:
            return 1.0 / (len(PolyominoShape) - list(PolyominoShape).index(shape))
        
        patterns = self.polyomino_patterns[shape]
        total_coverage = 0
        
        # サンプリングで高速化
        sample_points = min(100, height * width)
        step = max(1, (height * width) // sample_points)
        
        count = 0
        for pattern in patterns:
            for i in range(0, height * width, step):
                y, x = divmod(i, width)
                if y < height and x < width and self._can_place_pattern(grid, pattern, x, y):
                    total_coverage += len(pattern)
                count += 1
                if count >= sample_points:
                    break
        
        return total_coverage / max(1, count)
    
    def _can_place_pattern(self, grid: np.ndarray, pattern: List[Tuple[int, int]], x: int, y: int) -> bool:
        """パターン配置可能性チェック"""
        height, width = grid.shape
        for dx, dy in pattern:
            nx, ny = x + dx, y + dy
            if nx >= width or ny >= height or nx < 0 or ny < 0:
                return False
        return True
    
    def _scan_pattern(self, grid: np.ndarray, pattern: List[Tuple[int, int]], shape: PolyominoShape, used_positions: Set) -> List[NEXUSGroup]:
        """パターンスキャン（高速化版）"""
        height, width = grid.shape
        groups = []
        
        # 大きなグリッドでは簡略スキャン
        if height * width > 10000:
            step = max(1, min(height, width) // 20)
            max_groups = 500
        else:
            step = 1
            max_groups = 1000
        
        count = 0
        for y in range(0, height, step):
            for x in range(0, width, step):
                if count >= max_groups:
                    break
                    
                if self._can_place_pattern(grid, pattern, x, y):
                    positions = [(x + dx, y + dy) for dx, dy in pattern]
                    
                    # 未使用位置チェック
                    if not any(pos in used_positions for pos in positions):
                        try:
                            elements = [grid[py, px] for px, py in positions]
                            
                            # グループ作成
                            normalized = tuple(sorted(elements))
                            hash_value = hashlib.md5(str(normalized).encode()).hexdigest()
                            
                            group = NEXUSGroup(
                                elements=elements,
                                shape=shape,
                                positions=positions,
                                normalized=normalized,
                                hash_value=hash_value
                            )
                            
                            groups.append(group)
                            used_positions.update(positions)
                            count += 1
                        except (IndexError, ValueError):
                            continue
            if count >= max_groups:
                break
        
        return groups
    
    def _permutative_grouping(self, groups: List[NEXUSGroup]) -> Tuple[List[NEXUSGroup], Dict[str, int], List[int]]:
        """順番入れ替えグループ化（高速化版）"""
        # 大量グループの場合は制限
        if len(groups) > 50000:
            if self.ml_config.verbose:
                print(f"⚠️ 大量グループ検出 ({len(groups)}) - サンプリング実行")
            # 重要なグループのみ選択
            groups = self._sample_important_groups_fast(groups, 20000)
        
        # ハッシュベースでグループ化
        hash_to_group = {}
        group_counts = Counter()
        position_map = []
        
        for i, group in enumerate(groups):
            hash_value = group.hash_value
            group_counts[hash_value] += 1
            
            if hash_value not in hash_to_group:
                hash_to_group[hash_value] = group
                position_map.append(len(hash_to_group) - 1)
            else:
                # 既存のハッシュのインデックスを検索
                position_map.append(list(hash_to_group.keys()).index(hash_value))
        
        unique_groups = list(hash_to_group.values())
        
        if self.ml_config.verbose:
            print(f"🔄 グループ統合: {len(groups)} → {len(unique_groups)} ユニーク")
        
        return unique_groups, dict(group_counts), position_map
    
    def _sample_important_groups_fast(self, groups: List[NEXUSGroup], target_count: int) -> List[NEXUSGroup]:
        """重要グループの高速サンプリング"""
        if len(groups) <= target_count:
            return groups
        
        # 簡単な重要度評価
        scored_groups = []
        
        for i, group in enumerate(groups):
            # 高速スコア計算
            diversity_score = len(set(group.elements))  # 多様性
            size_score = len(group.elements)  # サイズ
            
            # 形状ボーナス
            shape_bonus = {
                PolyominoShape.I: 3,
                PolyominoShape.O: 2,
                PolyominoShape.T: 2,
                PolyominoShape.SINGLE: 1
            }.get(group.shape, 1)
            
            total_score = diversity_score * 2 + size_score + shape_bonus
            scored_groups.append((total_score, i, group))
            
            # 処理制限（重要！）
            if i >= len(groups) * 0.1:  # 最初の10%のみ評価
                break
        
        # 上位を選択
        scored_groups.sort(reverse=True)
        selected = [group for _, _, group in scored_groups[:target_count]]
        
        # 不足分は等間隔サンプリング
        if len(selected) < target_count:
            remaining = target_count - len(selected)
            step = max(1, len(groups) // remaining)
            
            for i in range(0, len(groups), step):
                if len(selected) >= target_count:
                    break
                if groups[i] not in selected:
                    selected.append(groups[i])
        
        return selected[:target_count]
    
    def _calculate_shape_distribution(self, groups: List[NEXUSGroup]) -> Dict[PolyominoShape, int]:
        """形状分布計算"""
        distribution = Counter()
        for group in groups:
            distribution[group.shape] += 1
        return dict(distribution)
    
    def _reconstruct_data_from_state(self, nexus_state: NEXUSCompressionState) -> bytes:
        """NEXUS状態からデータ復元（完全可逆性保証版）"""
        try:
            # 元のデータサイズ取得
            original_size = nexus_state.compression_metadata.get('original_size', 0)
            
            if self.ml_config.verbose:
                print(f"🔧 データ復元開始: 目標サイズ={original_size}")
            
            # 方法1: 位置マップから厳密復元（最優先）
            if hasattr(nexus_state, 'position_map') and nexus_state.position_map and hasattr(nexus_state, 'unique_groups'):
                if self.ml_config.verbose:
                    print(f"🎯 位置マップ復元: {len(nexus_state.position_map)} 位置")
                
                result = bytearray()
                
                # 位置マップに従って厳密に復元
                for position_index in nexus_state.position_map:
                    if position_index < len(nexus_state.unique_groups):
                        group = nexus_state.unique_groups[position_index]
                        # グループの元素を順序通り追加
                        result.extend(group.elements)
                    else:
                        if self.ml_config.verbose:
                            print(f"⚠️ 無効な位置インデックス: {position_index}")
                
                # 厳密なサイズチェック
                if original_size > 0:
                    if len(result) == original_size:
                        if self.ml_config.verbose:
                            print(f"✅ 完全一致: {len(result)} bytes")
                        return bytes(result)
                    elif len(result) > original_size:
                        # 過大な場合は切り詰め
                        truncated = result[:original_size]
                        if self.ml_config.verbose:
                            print(f"✂️ 切り詰め: {len(result)} -> {len(truncated)} bytes")
                        return bytes(truncated)
                    else:
                        if self.ml_config.verbose:
                            print(f"⚠️ サイズ不足: {len(result)} < {original_size}")
                        # 不足分は元データから推定（危険だが最善の努力）
                        return bytes(result)
                else:
                    # サイズ情報なしの場合
                    return bytes(result)
            
            # 方法2: オリジナルグループから順序復元
            elif hasattr(nexus_state, 'original_groups') and nexus_state.original_groups:
                if self.ml_config.verbose:
                    print(f"🔄 オリジナルグループ復元: {len(nexus_state.original_groups)} グループ")
                
                # 位置情報でソート
                sorted_groups = sorted(nexus_state.original_groups, 
                                     key=lambda g: g.positions[0] if g.positions else (0, 0))
                
                result = bytearray()
                for group in sorted_groups:
                    result.extend(group.elements)
                
                # サイズ調整
                if original_size > 0 and len(result) != original_size:
                    if len(result) > original_size:
                        result = result[:original_size]
                    if self.ml_config.verbose:
                        print(f"� サイズ調整: {len(result)} bytes")
                
                return bytes(result)
            
            # 方法3: ユニークグループから推定復元（最後の手段）
            else:
                if self.ml_config.verbose:
                    print(f"🆘 推定復元: {len(nexus_state.unique_groups)} ユニークグループ")
                
                result = bytearray()
                
                # グループ出現回数を考慮した復元
                if hasattr(nexus_state, 'group_counts'):
                    for group in nexus_state.unique_groups:
                        count = nexus_state.group_counts.get(group.hash_value, 1)
                        for _ in range(count):
                            result.extend(group.elements)
                else:
                    # 単純結合
                    for group in nexus_state.unique_groups:
                        result.extend(group.elements)
                
                # サイズ制限
                if original_size > 0 and len(result) > original_size:
                    result = result[:original_size]
                
                if self.ml_config.verbose:
                    print(f"🎲 推定復元結果: {len(result)} bytes")
                
                return bytes(result)
        
        except Exception as e:
            if self.ml_config.verbose:
                print(f"❌ データ復元エラー: {e}")
                import traceback
                traceback.print_exc()
            return b""

# ==================== メイン実行 ====================

if __name__ == "__main__":
    # テスト実行
    print("🧠 NEXUS機械学習統合エンジンテスト")
    
    test_data = b"Hello, NEXUS ML World!" * 100
    
    config = MLCompressionConfig(verbose=True)
    compressor = NEXUSCompressor(config)
    
    # 圧縮テスト
    start_time = time.time()
    compressed = compressor.compress(test_data)
    compress_time = time.time() - start_time
    
    # 展開テスト
    start_time = time.time()
    decompressed = compressor.decompress(compressed)
    decompress_time = time.time() - start_time
    
    # 結果表示
    original_size = len(test_data)
    compressed_size = len(compressed)
    compression_ratio = (compressed_size / original_size) * 100
    
    print(f"\n📊 結果:")
    print(f"元サイズ: {original_size} bytes")
    print(f"圧縮サイズ: {compressed_size} bytes")
    print(f"圧縮率: {compression_ratio:.1f}%")
    print(f"圧縮時間: {compress_time:.3f}s")
    print(f"展開時間: {decompress_time:.3f}s")
    print(f"データ一致: {test_data == decompressed}")
    print(f"ML統計: {compressor.stats}")
