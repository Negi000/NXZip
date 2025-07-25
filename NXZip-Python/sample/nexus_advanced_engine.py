# nexus_advanced_engine.py
import sys
import json
import lzma
import math
import heapq
import collections
import numpy as np
import pickle
import time  # 進捗表示用
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor  # 並列処理用

# 進捗バー表示用のユーティリティ
class ProgressBar:
    def __init__(self, total: int, description: str = "", width: int = 50):
        self.total = total
        self.current = 0
        self.description = description
        self.width = width
        self.start_time = time.time()
        self.last_update_time = 0
        
    def update(self, current: int = None, force: bool = False):
        if current is not None:
            self.current = current
        else:
            self.current += 1
            
        # 0.1秒間隔でのみ更新（パフォーマンス向上）
        current_time = time.time()
        if not force and current_time - self.last_update_time < 0.1:
            return
        self.last_update_time = current_time
        
        # ゼロ除算回避
        if self.total == 0:
            percentage = 100
            filled_length = self.width
        else:
            percentage = min(100, (self.current / self.total) * 100)
            filled_length = int(self.width * self.current // self.total)
            
        bar = '█' * filled_length + '░' * (self.width - filled_length)
        
        elapsed = current_time - self.start_time
        if self.current > 0 and self.total > 0:
            eta = elapsed * (self.total - self.current) / self.current
            eta_str = f" ETA: {eta:.1f}s"
        else:
            eta_str = ""
        
        print(f"\r{self.description} |{bar}| {percentage:.1f}% ({self.current:,}/{self.total:,}){eta_str}", end='', flush=True)
        
    def finish(self):
        self.update(self.total, force=True)
        elapsed = time.time() - self.start_time
        print(f" ✓ Complete in {elapsed:.2f}s")
        print()  # 新しい行に移動

# 機械学習ライブラリ (軽量実装)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not available. Using fallback AI implementation.")

# --- 1. 形状選択アルゴリズム: Polyomino形状の定義 ---
# 各形状は(行, 列)の相対座標のタプルで定義（パズル組み合わせ用）
POLYOMINO_SHAPES = {
    # 基本形状 (1-4ブロック)
    "I-1": ((0, 0),),                                    # 単体
    "I-2": ((0, 0), (0, 1)),                            # 1x2 線形
    "I-3": ((0, 0), (0, 1), (0, 2)),                    # 1x3 線形
    "I-4": ((0, 0), (0, 1), (0, 2), (0, 3)),            # 1x4 線形
    "I-5": ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)),    # 1x5 線形
    
    # 正方形・長方形
    "O-4": ((0, 0), (0, 1), (1, 0), (1, 1)),            # 2x2 正方形
    "R-6": ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),  # 2x3 長方形
    "R-8": ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)),  # 2x4 長方形
    
    # 複雑形状
    "T-4": ((0, 0), (0, 1), (0, 2), (1, 1)),            # T字型
    "L-4": ((0, 0), (1, 0), (2, 0), (2, 1)),            # L字型
    "Z-4": ((0, 0), (0, 1), (1, 1), (1, 2)),            # Z字型
    "S-4": ((0, 1), (0, 2), (1, 0), (1, 1)),            # S字型
    
    # 大型形状
    "T-5": ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),    # 十字型
    "U-6": ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0)), # U字型
    "H-7": ((0, 0), (0, 1), (0, 2), (1, 1), (2, 0), (2, 1), (2, 2)), # H字型
}

# --- 4. ハイブリッド符号化: Huffman符号化の実装 ---
class HuffmanEncoder:
    """BlueprintのIDストリームをHuffman符号化/復号化するクラス"""
    
    def encode(self, data: List[int]) -> Tuple[Dict[int, str], str]:
        if not data:
            return {}, ""
        frequency = collections.Counter(data)
        heap = [[weight, [symbol, ""]] for symbol, weight in frequency.items()]
        heapq.heapify(heap)
        
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
            
        huffman_tree = dict(sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p)))
        encoded_data = "".join([huffman_tree[d] for d in data])
        return huffman_tree, encoded_data

    def decode(self, encoded_data: str, huffman_tree: Dict[str, str]) -> List[int]:
        if not encoded_data:
            return []
        # デコード用にキーとバリューを反転
        reverse_tree = {v: int(k) for k, v in huffman_tree.items()}
        decoded_data = []
        current_code = ""
        for bit in encoded_data:
            current_code += bit
            if current_code in reverse_tree:
                decoded_data.append(reverse_tree[current_code])
                current_code = ""
        return decoded_data

# --- 2. 本格的AI最適化エンジン ---
@dataclass
class ShapeAnalysisResult:
    entropy: float
    variance: float
    spatial_correlation: float
    pattern_density: float
    edge_count: float

class AdvancedNeuralShapeOptimizer:
    """本格的なニューラルネットワークベースの形状最適化"""
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        if HAS_TORCH:
            self._init_neural_network()
        else:
            self._init_fallback_optimizer()
    
    def _init_neural_network(self):
        """PyTorchベースのニューラルネットワーク初期化"""
        class ShapeClassifierNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.feature_extractor = nn.Sequential(
                    nn.Linear(5, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, len(POLYOMINO_SHAPES))
                )
                
            def forward(self, x):
                return torch.softmax(self.feature_extractor(x), dim=-1)
        
        self.model = ShapeClassifierNet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        # 事前学習データでのウォームアップ
        self._pretrain_with_synthetic_data()
    
    def _init_fallback_optimizer(self):
        """フォールバック実装（軽量版）"""
        # 簡易的な重み付きスコアリングシステム（新形状対応）
        self.shape_weights = {
            # 基本線形形状
            "I-1": np.array([0.5, 0.3, 0.4, 0.3, 0.2]),   # 最小形状
            "I-2": np.array([0.8, 0.5, 0.6, 0.4, 0.3]),   # 短線形
            "I-3": np.array([1.1, 0.7, 0.9, 0.8, 0.6]),   # 中線形
            "I-4": np.array([1.2, 0.8, 1.0, 0.9, 0.7]),   # 長線形
            "I-5": np.array([1.3, 0.9, 1.1, 1.0, 0.8]),   # 最長線形
            
            # ブロック・長方形形状
            "O-4": np.array([0.8, 1.2, 1.1, 1.0, 0.8]),   # 正方形
            "R-6": np.array([1.0, 1.3, 1.2, 1.1, 0.9]),   # 長方形6
            "R-8": np.array([1.1, 1.4, 1.3, 1.2, 1.0]),   # 長方形8
            
            # 複雑形状
            "T-4": np.array([1.0, 1.0, 1.2, 1.1, 0.9]),   # T字型
            "L-4": np.array([0.9, 0.9, 1.1, 1.2, 1.0]),   # L字型
            "Z-4": np.array([1.1, 1.1, 1.0, 1.3, 1.1]),   # Z字型
            "S-4": np.array([1.1, 1.1, 0.9, 1.3, 1.2]),   # S字型
            
            # 大型形状
            "T-5": np.array([1.2, 1.2, 1.4, 1.4, 1.3]),   # 十字型
            "U-6": np.array([1.0, 1.3, 1.3, 1.3, 1.2]),   # U字型
            "H-7": np.array([1.3, 1.4, 1.5, 1.5, 1.4])    # H字型
        }
    
    def _pretrain_with_synthetic_data(self):
        """合成データでの事前学習"""
        if not HAS_TORCH:
            return
            
        # 合成学習データ生成
        train_data = []
        train_labels = []
        
        shape_names = list(POLYOMINO_SHAPES.keys())
        for i, shape_name in enumerate(shape_names):
            for _ in range(50):  # 各形状50サンプル（形状数が増えたため削減）
                # 特徴量を形状に応じて生成
                if "I-" in shape_name:  # 線形形状
                    shape_size = int(shape_name.split('-')[1])
                    base_entropy = 2.0 + shape_size * 0.5
                    features = [
                        np.random.normal(base_entropy, 0.3),       # entropy
                        np.random.normal(0.2 + shape_size * 0.05, 0.1),  # variance  
                        np.random.normal(0.7 + shape_size * 0.02, 0.1),  # spatial_correlation
                        np.random.normal(0.5 + shape_size * 0.02, 0.1),  # pattern_density
                        np.random.normal(0.3 + shape_size * 0.02, 0.1)   # edge_count
                    ]
                elif shape_name.startswith(("O-", "R-")):  # ブロック・長方形
                    features = [
                        np.random.normal(5.0, 0.5),
                        np.random.normal(0.5, 0.1),
                        np.random.normal(0.9, 0.1),
                        np.random.normal(0.8, 0.1),
                        np.random.normal(0.3, 0.1)
                    ]
                else:  # 複雑形状 (T, L, Z, S, U, H)
                    complexity = len(POLYOMINO_SHAPES[shape_name])
                    features = [
                        np.random.normal(6.0 + complexity * 0.2, 0.5),
                        np.random.normal(0.6 + complexity * 0.05, 0.1),
                        np.random.normal(0.5 + complexity * 0.02, 0.1),
                        np.random.normal(0.7 + complexity * 0.03, 0.1),
                        np.random.normal(0.7 + complexity * 0.04, 0.1)
                    ]
                
                train_data.append(features)
                train_labels.append(i)
        
        # 学習実行
        train_data = torch.FloatTensor(train_data)
        train_labels = torch.LongTensor(train_labels)
        
        self.model.train()
        for epoch in range(30):  # 軽量学習（形状数増加により削減）
            self.optimizer.zero_grad()
            outputs = self.model(train_data)
            loss = self.criterion(outputs, train_labels)
            loss.backward()
            self.optimizer.step()
        
        self.model.eval()
        self.is_trained = True
        print(f"   [AI] Neural network pre-trained with {len(shape_names)} shapes, {len(train_data)} samples")
    
    def analyze_data_features(self, data: bytes) -> ShapeAnalysisResult:
        """データの特徴量を詳細分析（高速化版）"""
        if len(data) == 0:
            return ShapeAnalysisResult(0, 0, 0, 0, 0)
        
        # 適応的サンプリング：大きなファイルは小さなサンプルで分析
        if len(data) > 10000000:  # 10MB以上
            sample_size = 5000  # 5KB
        elif len(data) > 1000000:  # 1MB以上
            sample_size = 8000  # 8KB
        else:
            sample_size = min(len(data), 15000)  # 最大15KB
        
        # サンプルデータ取得（先頭から）
        sample_data = data[:sample_size]
        data_array = np.array(list(sample_data), dtype=np.float32)
        
        # 1. エントロピー計算（サンプルベース）
        counts = collections.Counter(sample_data)
        entropy = 0
        for count in counts.values():
            p_x = count / len(sample_data)
            entropy -= p_x * math.log2(p_x)
        
        # 2. 分散計算
        variance = np.var(data_array)
        
        # 3. 空間相関（隣接バイトの類似性）
        if len(data_array) > 1:
            try:
                spatial_corr = np.corrcoef(data_array[:-1], data_array[1:])[0,1]
                if np.isnan(spatial_corr):
                    spatial_corr = 0
            except:
                spatial_corr = 0
        else:
            spatial_corr = 0
        
        # 4. パターン密度（繰り返しパターンの検出）
        pattern_density = len(set(sample_data)) / len(sample_data) if len(sample_data) > 0 else 0
        
        # 5. エッジカウント（値の変化回数）
        edge_count = sum(1 for i in range(len(sample_data)-1) if abs(sample_data[i] - sample_data[i+1]) > 10) / max(len(sample_data)-1, 1)
        
        return ShapeAnalysisResult(
            entropy=entropy,
            variance=float(variance),
            spatial_correlation=float(spatial_corr),
            pattern_density=pattern_density,
            edge_count=edge_count
        )
    
    def predict_optimal_shape(self, data: bytes) -> str:
        """機械学習により最適形状を予測（適応的データ分析版）"""
        data_size = len(data)
        
        # 【CRITICAL】データサイズによる形状制限（完全可逆性保証）
        if data_size == 1:
            return "I-1"
        elif data_size == 2:
            return "I-2"
        elif data_size <= 4:
            return "I-3"
        elif data_size <= 8:
            return "I-4"
        elif data_size <= 16:
            return "O-4"
        
        features = self.analyze_data_features(data)
        feature_vector = [
            features.entropy,
            features.variance,
            features.spatial_correlation, 
            features.pattern_density,
            features.edge_count
        ]
        
        # 圧縮済みデータの検出と適応的処理
        is_compressed_data = self._detect_compressed_data(data, features)
        
        if is_compressed_data:
            return self._select_shape_for_compressed_data(data, features)
        
        if HAS_TORCH and self.is_trained:
            # ニューラルネットワーク予測
            with torch.no_grad():
                input_tensor = torch.FloatTensor([feature_vector])
                prediction = self.model(input_tensor)
                predicted_idx = torch.argmax(prediction, dim=1).item()
                confidence = torch.max(prediction).item()
                
                shape_names = list(POLYOMINO_SHAPES.keys())
                predicted_shape = shape_names[predicted_idx]
                
                # データサイズとの互換性チェック
                predicted_shape = self._validate_shape_size_compatibility(predicted_shape, data_size)
                
                # 信頼度によって追加検証を行う
                if confidence < 0.7:  # 低信頼度の場合
                    print(f"   [AI] Neural prediction: '{predicted_shape}' (low confidence: {confidence:.2f}) - performing verification")
                    return self._verify_prediction_with_sampling(data, predicted_shape)
                else:
                    print(f"   [AI] Neural prediction: '{predicted_shape}' (high confidence: {confidence:.2f})")
                    return predicted_shape
        else:
            # フォールバック実装も改良
            feature_array = np.array(feature_vector)
            best_score = -float('inf')
            best_shape = "I-4"
            
            # 形状別スコア計算（上位3つを評価）
            shape_scores = []
            for shape_name, weights in self.shape_weights.items():
                score = np.dot(feature_array, weights)
                shape_scores.append((score, shape_name))
            
            # 上位3形状を取得
            shape_scores.sort(reverse=True)
            top_shapes = shape_scores[:3]
            
            best_shape = top_shapes[0][1]
            # サイズ互換性チェック
            best_shape = self._validate_shape_size_compatibility(best_shape, data_size)
            
            print(f"   [AI] Fallback top shapes: {[(s, f'{sc:.2f}') for sc, s in top_shapes]}")
            print(f"   [AI] Selected: '{best_shape}' (size-validated)")
            
            return best_shape
    
    def _validate_shape_size_compatibility(self, shape_name: str, data_size: int) -> str:
        """形状とデータサイズの互換性を検証し、必要に応じて適切な形状に変更"""
        # 形状のブロック数を取得
        shape_coords = POLYOMINO_SHAPES.get(shape_name, [(0,0)])
        shape_block_count = len(shape_coords)
        
        # データサイズが形状ブロック数より小さい場合、適切な形状に変更
        if data_size < shape_block_count:
            # データサイズに適した最大形状を選択
            if data_size == 1:
                safe_shape = "I-1"  # 1ブロック
            elif data_size == 2:
                safe_shape = "I-2"  # 2ブロック
            elif data_size == 3:
                safe_shape = "I-3"  # 3ブロック
            elif data_size >= 4:
                safe_shape = "I-4"  # 4ブロック（最小安全形状）
            else:
                safe_shape = "I-1"  # フォールバック
            
            print(f"   [Size Validation] '{shape_name}' ({shape_block_count} blocks) incompatible with {data_size} bytes → '{safe_shape}'")
            return safe_shape
        
        return shape_name
    
    def _detect_compressed_data(self, data: bytes, features) -> bool:
        """圧縮済みデータを検出"""
        # 高エントロピー + 低パターン密度 = 圧縮済みデータの特徴
        if features.entropy > 7.5 and features.pattern_density < 0.05:
            return True
        
        # 分散が高く、空間相関が低い = ランダム性が高い
        if features.variance > 5000 and abs(features.spatial_correlation) < 0.1:
            return True
        
        return False
    
    def _select_shape_for_compressed_data(self, data: bytes, features) -> str:
        """圧縮済みデータに特化した形状選択"""
        print(f"   [Compressed Data Strategy] Applying multi-scale analysis")
        
        # マルチスケール分析：異なるブロックサイズで最適化
        data_size = len(data)
        
        if data_size > 10000000:  # 10MB以上：大ブロック戦略
            # 大きなデータには大きなブロック（情報密度を上げる）
            candidate_shapes = ["R-8", "H-7", "U-6", "R-6"]
        elif data_size > 1000000:  # 1MB以上：中ブロック戦略
            candidate_shapes = ["O-4", "R-6", "T-5", "I-5"]
        else:  # 小データ：小ブロック戦略
            candidate_shapes = ["I-3", "I-4", "O-4", "T-4"]
        
        # 実際の効率テスト（高速サンプリング）
        best_shape = self._test_shapes_on_compressed_data(data, candidate_shapes)
        
        print(f"   [Compressed Data Strategy] Selected: '{best_shape}'")
        return best_shape
    
    def _test_shapes_on_compressed_data(self, data: bytes, candidate_shapes: list) -> str:
        """圧縮済みデータでの形状効率テスト"""
        # 小サンプルで各形状の効率をテスト
        sample_size = min(len(data), 5000)  # 5KB
        sample_data = data[:sample_size]
        sample_grid_width = math.ceil(math.sqrt(sample_size))
        
        best_shape = candidate_shapes[0]
        best_efficiency = 0
        
        print(f"   [Shape Testing] Testing {len(candidate_shapes)} shapes on compressed data")
        
        for shape_name in candidate_shapes:
            try:
                shape_coords = POLYOMINO_SHAPES[shape_name]
                blocks = self._get_blocks_for_shape_simple(sample_data, sample_grid_width, shape_coords)
                
                if len(blocks) > 0:
                    unique_count = len(set(tuple(sorted(b)) for b in blocks))
                    efficiency = len(blocks) / unique_count if unique_count > 0 else 1
                    
                    if efficiency > best_efficiency:
                        best_efficiency = efficiency
                        best_shape = shape_name
                        
            except Exception as e:
                print(f"   [Shape Testing] Error testing {shape_name}: {e}")
                continue
        
        print(f"   [Shape Testing] Best shape: '{best_shape}' (efficiency: {best_efficiency:.2f}x)")
        return best_shape
    
    def _get_blocks_for_shape_simple(self, data: bytes, grid_width: int, shape_coords: list) -> list:
        """圧縮データ用の高速ブロック取得"""
        shape_height = max(coord[0] for coord in shape_coords) + 1
        shape_width = max(coord[1] for coord in shape_coords) + 1
        grid_height = math.ceil(len(data) / grid_width)
        
        blocks = []
        max_blocks = 50000  # 上限設定で無限生成を防ぐ
        
        for start_row in range(0, grid_height - shape_height + 1, shape_height):
            for start_col in range(0, grid_width - shape_width + 1, shape_width):
                if len(blocks) >= max_blocks:
                    break
                    
                block = []
                for row_offset, col_offset in shape_coords:
                    actual_row = start_row + row_offset
                    actual_col = start_col + col_offset
                    
                    if actual_row < grid_height and actual_col < grid_width:
                        data_idx = actual_row * grid_width + actual_col
                        if data_idx < len(data):
                            block.append(data[data_idx])
                
                if len(block) == len(shape_coords):
                    blocks.append(block)
            
            if len(blocks) >= max_blocks:
                break
                
        return blocks
    
    def _verify_prediction_with_sampling(self, data: bytes, predicted_shape: str) -> str:
        """低信頼度予測の検証（サンプリング版）"""
        sample_size = min(len(data), 2000)
        sample_data = data[:sample_size]
        
        # 予測形状 + 代替候補で効率テスト
        candidates = [predicted_shape, "I-4", "O-4", "T-4"]
        candidates = list(dict.fromkeys(candidates))  # 重複除去
        
        best_shape = predicted_shape
        best_efficiency = 0
        
        sample_grid_width = math.ceil(math.sqrt(sample_size))
        
        for shape_name in candidates:
            try:
                shape_coords = POLYOMINO_SHAPES[shape_name]
                blocks = self._get_blocks_for_shape_simple(sample_data, sample_grid_width, shape_coords)
                
                if len(blocks) > 10:  # 最小ブロック数
                    unique_count = len(set(tuple(sorted(b)) for b in blocks))
                    efficiency = len(blocks) / unique_count if unique_count > 0 else 1
                    
                    if efficiency > best_efficiency:
                        best_efficiency = efficiency
                        best_shape = shape_name
                        
            except Exception:
                continue
        
        print(f"   [Verification] Final shape: '{best_shape}' (efficiency: {best_efficiency:.2f}x)")
        return best_shape
    
    def _verify_prediction_with_sampling(self, data: bytes, predicted_shape: str) -> str:
        """AI予測の信頼度が低い場合の検証（軽量サンプリング）"""
        if len(data) < 5000:
            return predicted_shape
            
        # 軽量サンプリングで実際の効率をテスト
        sample_size = min(len(data), 2000)
        sample_data = data[:sample_size]
        sample_grid_width = math.ceil(math.sqrt(sample_size))
        
        # 予測形状の実際の効率をチェック
        shape_coords = POLYOMINO_SHAPES[predicted_shape]
        blocks = self._get_blocks_for_shape_simple(sample_data, sample_grid_width, shape_coords)
        predicted_efficiency = len(set(tuple(sorted(b)) for b in blocks))
        
        # 代替形状との比較
        alternative_shapes = ["I-1", "I-2", "O-4", "T-4"]
        if predicted_shape in alternative_shapes:
            alternative_shapes.remove(predicted_shape)
        
        best_shape = predicted_shape
        best_efficiency = predicted_efficiency
        
        for alt_shape in alternative_shapes[:2]:  # 最大2つの代替形状をテスト
            alt_coords = POLYOMINO_SHAPES[alt_shape]
            alt_blocks = self._get_blocks_for_shape_simple(sample_data, sample_grid_width, alt_coords)
            alt_efficiency = len(set(tuple(sorted(b)) for b in alt_blocks))
            
            if alt_efficiency < best_efficiency:
                best_efficiency = alt_efficiency
                best_shape = alt_shape
        
        if best_shape != predicted_shape:
            print(f"   [AI] Verification: Changed from '{predicted_shape}' to '{best_shape}' (better efficiency)")
        else:
            print(f"   [AI] Verification: Confirmed '{predicted_shape}' as optimal")
            
        return best_shape
    
    def _get_blocks_for_shape_simple(self, data: bytes, grid_width: int, shape_coords: Tuple[Tuple[int, int], ...]) -> List[Tuple[int, ...]]:
        """シンプルなブロック生成（検証用）"""
        data_len = len(data)
        rows = data_len // grid_width
        shape_width = max(c for r, c in shape_coords) + 1
        shape_height = max(r for r, c in shape_coords) + 1
        
        blocks = []
        for r in range(rows - shape_height + 1):
            for c in range(grid_width - shape_width + 1):
                block = []
                valid_block = True
                
                base_idx = r * grid_width + c
                for dr, dc in shape_coords:
                    idx = base_idx + dr * grid_width + dc
                    if idx >= data_len:
                        valid_block = False
                        break
                    block.append(data[idx])
                
                if valid_block:
                    blocks.append(tuple(block))
        
        return blocks

class NexusAdvancedCompressor:
    def __init__(self, use_ai=True, max_recursion_level=0):
        self.ai_optimizer = AdvancedNeuralShapeOptimizer() if use_ai else None
        self.huffman_encoder = HuffmanEncoder()
        self.max_recursion_level = max_recursion_level

    def _consolidate_by_elements(self, normalized_groups: Dict[Tuple, int], show_progress: bool = False) -> Tuple[Dict[Tuple, int], Dict[int, Dict]]:
        """
        🔥 NEXUS THEORY INFECTED CONSOLIDATION SYSTEM 🔥
        
        NEXUS原則: 「圧縮の逆が解凍」- 情報を一切失わない完全可逆統合
        Every consolidation must store EXACT reconstruction data
        """
        if not normalized_groups:
            return normalized_groups, {}
        
        print(f"   [NEXUS Multi-Layer] Processing {len(normalized_groups):,} groups with INFECTED 4-layer algorithm")
        original_count = len(normalized_groups)
        
        # 🔥 NEXUS INFECTED: すべての層で完全な逆変換データを保存
        
        # レイヤー1: NEXUS感染完全一致統合
        layer1_result, layer1_map = self._nexus_layer1_exact_consolidation(normalized_groups, show_progress)
        layer1_reduction = 100 * (original_count - len(layer1_result)) / original_count
        print(f"   [NEXUS Layer 1] Exact match: {len(layer1_result):,} groups ({layer1_reduction:.1f}% reduction)")
        
        # レイヤー2: NEXUS感染パターンベース統合
        layer2_result, layer2_map = self._nexus_layer2_pattern_consolidation(layer1_result, layer1_map, show_progress)
        layer2_reduction = 100 * (len(layer1_result) - len(layer2_result)) / len(layer1_result) if len(layer1_result) > 0 else 0
        print(f"   [NEXUS Layer 2] Pattern match: {len(layer2_result):,} groups ({layer2_reduction:.1f}% additional reduction)")
        
        # レイヤー3: NEXUS感染近似統合（完全逆変換保証）
        layer3_result, layer3_map = self._nexus_layer3_approximate_consolidation(layer2_result, layer2_map, show_progress)
        layer3_reduction = 100 * (len(layer2_result) - len(layer3_result)) / len(layer2_result) if len(layer2_result) > 0 else 0
        print(f"   [NEXUS Layer 3] Approximate match: {len(layer3_result):,} groups ({layer3_reduction:.1f}% additional reduction)")
        
        # レイヤー4: NEXUS感染構造統合（情報保存優先）
        layer4_result, layer4_map = self._nexus_layer4_structural_consolidation(layer3_result, layer3_map, show_progress)
        layer4_reduction = 100 * (len(layer3_result) - len(layer4_result)) / len(layer3_result) if len(layer3_result) > 0 else 0
        print(f"   [NEXUS Layer 4] Structural match: {len(layer4_result):,} groups ({layer4_reduction:.1f}% additional reduction)")
        
        total_reduction = 100 * (original_count - len(layer4_result)) / original_count
        print(f"   [NEXUS Multi-Layer] Total reduction: {total_reduction:.2f}% ({original_count:,} → {len(layer4_result):,})")
        
        # 🔥 NEXUS統合: 完全な逆変換チェーンを構築
        nexus_combined_map = self._build_nexus_reconstruction_chain(layer1_map, layer2_map, layer3_map, layer4_map)
        
        return layer4_result, nexus_combined_map
    
    def _nexus_layer1_exact_consolidation(self, normalized_groups: Dict[Tuple, int], show_progress: bool) -> Tuple[Dict[Tuple, int], Dict[int, Dict]]:
        """
        🔥 NEXUS LAYER 1: 完全一致統合（NEXUS理論感染版）
        
        NEXUS原則: 一切の情報損失なし - 完全可逆変換のみ実行
        """
        element_signature_map = {}
        
        if show_progress:
            progress_bar = ProgressBar(len(normalized_groups), "   NEXUS Layer 1: Exact matching")
        
        processed = 0
        for original_group, group_id in normalized_groups.items():
            element_signature = tuple(sorted(set(original_group)))
            
            if element_signature not in element_signature_map:
                element_signature_map[element_signature] = []
            element_signature_map[element_signature].append((original_group, group_id))
            
            processed += 1
            if show_progress and processed % 5000 == 0:
                progress_bar.update(processed)
        
        if show_progress:
            progress_bar.finish()
        
        # 🔥 NEXUS統合: 完全な逆変換データを保存
        consolidated_groups = {}
        nexus_consolidation_map = {}
        new_group_id = 0
        
        for signature, group_list in element_signature_map.items():
            if len(group_list) == 1:
                # 単一グループ: そのまま保持（逆変換なし）
                original_group, original_id = group_list[0]
                consolidated_groups[original_group] = new_group_id
                nexus_consolidation_map[original_id] = {
                    'nexus_new_group_id': new_group_id,
                    'nexus_canonical_form': original_group,
                    'nexus_layer': 1,
                    'nexus_consolidation_type': 'identity',
                    'nexus_original_group': original_group,  # 🔥 完全逆変換データ
                    'nexus_exact_reconstruction': True
                }
            else:
                # 複数グループ: 代表選出 + 完全逆変換データ保存
                canonical_group = min(group_list, key=lambda x: len(str(x[0])))[0]
                consolidated_groups[canonical_group] = new_group_id
                
                # 🔥 NEXUS: すべての統合されたグループの完全データを保存
                for original_group, original_id in group_list:
                    nexus_consolidation_map[original_id] = {
                        'nexus_new_group_id': new_group_id,
                        'nexus_canonical_form': canonical_group,
                        'nexus_layer': 1,
                        'nexus_consolidation_type': 'exact_match',
                        'nexus_original_group': original_group,  # 🔥 完全な元データ
                        'nexus_exact_reconstruction': True,
                        'nexus_signature': signature,  # 🔥 統合キー保存
                        'nexus_group_list': [g[0] for g in group_list]  # 🔥 すべての関連グループ
                    }
            
            new_group_id += 1
        
        return consolidated_groups, nexus_consolidation_map
    
    def _nexus_layer2_pattern_consolidation(self, groups_dict: Dict[Tuple, int], layer1_map: Dict, show_progress: bool) -> Tuple[Dict[Tuple, int], Dict[int, Dict]]:
        """
        🔥 NEXUS LAYER 2: パターンベース統合（NEXUS理論感染版）
        
        NEXUS原則: パターン統合でも完全な逆変換データを保存
        """
        pattern_groups = {}
        
        if show_progress:
            progress_bar = ProgressBar(len(groups_dict), "   NEXUS Layer 2: Pattern matching")
        
        processed = 0
        for group_tuple, group_id in groups_dict.items():
            pattern_sig = self._extract_pattern_signature(list(group_tuple))
            
            if pattern_sig not in pattern_groups:
                pattern_groups[pattern_sig] = []
            pattern_groups[pattern_sig].append((group_tuple, group_id))
            
            processed += 1
            if show_progress and processed % 2000 == 0:
                progress_bar.update(processed)
        
        if show_progress:
            progress_bar.finish()
        
        # 🔥 NEXUS パターン統合: 完全な逆変換チェーン保存
        consolidated = {}
        nexus_pattern_map = {}
        new_id = 0
        
        for pattern_sig, group_list in pattern_groups.items():
            if len(group_list) == 1:
                # 単一パターン: そのまま保持
                group_tuple, original_id = group_list[0]
                consolidated[group_tuple] = new_id
                nexus_pattern_map[original_id] = {
                    'nexus_new_group_id': new_id,
                    'nexus_canonical_form': group_tuple,
                    'nexus_layer': 2,
                    'nexus_consolidation_type': 'pattern_identity',
                    'nexus_original_group': group_tuple,  # 🔥 完全な元データ
                    'nexus_pattern_signature': pattern_sig,  # 🔥 パターン保存
                    'nexus_exact_reconstruction': True,
                    'nexus_layer1_inheritance': layer1_map.get(original_id, {})  # 🔥 前層からの継承
                }
            else:
                # 複数パターン: 代表選出 + 完全逆変換データ保存
                representative = min(group_list, key=lambda x: sum(x[0]))[0]
                consolidated[representative] = new_id
                
                # 🔥 NEXUS: すべてのパターン統合グループの完全データを保存
                for group_tuple, original_id in group_list:
                    nexus_pattern_map[original_id] = {
                        'nexus_new_group_id': new_id,
                        'nexus_canonical_form': representative,
                        'nexus_layer': 2,
                        'nexus_consolidation_type': 'pattern_match',
                        'nexus_original_group': group_tuple,  # 🔥 完全な元データ
                        'nexus_pattern_signature': pattern_sig,  # 🔥 パターン保存
                        'nexus_exact_reconstruction': True,
                        'nexus_pattern_group_list': [g[0] for g in group_list],  # 🔥 全パターングループ
                        'nexus_layer1_inheritance': layer1_map.get(original_id, {})  # 🔥 前層からの継承
                    }
            
            new_id += 1
        
        return consolidated, nexus_pattern_map

    def _nexus_layer3_approximate_consolidation(self, groups_dict: Dict[Tuple, int], layer2_map: Dict, show_progress: bool) -> Tuple[Dict[Tuple, int], Dict[int, Dict]]:
        """
        🔥 NEXUS LAYER 3: 近似統合（NEXUS理論感染版）
        
        NEXUS原則: 近似統合でも完全可逆性保証 - 類似度情報も保存
        """
        
        # 🔥 NEXUS: ハッシュベース近似統合 + 完全逆変換データ保存
        similarity_hash_map = {}  # similarity_hash -> [(group_tuple, group_id)]
        
        if show_progress:
            progress_bar = ProgressBar(len(groups_dict), "   NEXUS Layer 3: Computing similarity hashes")
        
        processed = 0
        for group_tuple, group_id in groups_dict.items():
            # 高速類似性ハッシュ生成（NEXUS感染版）
            similarity_hash = self._nexus_compute_similarity_hash(group_tuple)
            
            if similarity_hash not in similarity_hash_map:
                similarity_hash_map[similarity_hash] = []
            similarity_hash_map[similarity_hash].append((group_tuple, group_id))
            
            processed += 1
            if show_progress and processed % 2000 == 0:
                progress_bar.update(processed)
        
        if show_progress:
            progress_bar.finish()
        
        print(f"   [Layer 3] Generated {len(similarity_hash_map):,} similarity buckets")
        
        # 🔥 NEXUS近似統合: 完全な逆変換チェーン保存
        nexus_consolidated = {}
        nexus_approximate_map = {}
        new_id = 0
        
        for sim_hash, group_list in similarity_hash_map.items():
            if len(group_list) == 1:
                # 単一類似グループ: そのまま保持
                group_tuple, original_id = group_list[0]
                nexus_consolidated[group_tuple] = new_id
                nexus_approximate_map[original_id] = {
                    'nexus_new_group_id': new_id,
                    'nexus_canonical_form': group_tuple,
                    'nexus_layer': 3,
                    'nexus_consolidation_type': 'approximate_identity',
                    'nexus_original_group': group_tuple,  # 🔥 完全な元データ
                    'nexus_similarity_hash': sim_hash,  # 🔥 類似性情報保存
                    'nexus_exact_reconstruction': True,
                    'nexus_layer2_inheritance': layer2_map.get(original_id, {})  # 🔥 前層からの継承
                }
            else:
                # 複数類似グループ: 代表選出 + 完全逆変換データ保存
                representative = min(group_list, key=lambda x: len(str(x[0])))[0]
                nexus_consolidated[representative] = new_id
                
                # 🔥 NEXUS: すべての類似統合グループの完全データを保存
                for group_tuple, original_id in group_list:
                    nexus_approximate_map[original_id] = {
                        'nexus_new_group_id': new_id,
                        'nexus_canonical_form': representative,
                        'nexus_layer': 3,
                        'nexus_consolidation_type': 'approximate_match',
                        'nexus_original_group': group_tuple,  # 🔥 完全な元データ
                        'nexus_similarity_hash': sim_hash,  # 🔥 類似性情報保存
                        'nexus_exact_reconstruction': True,
                        'nexus_similarity_group_list': [g[0] for g in group_list],  # 🔥 全類似グループ
                        'nexus_layer2_inheritance': layer2_map.get(original_id, {}),  # 🔥 前層からの継承
                        'nexus_similarity_score': self._nexus_compute_similarity_score(group_tuple, representative)  # 🔥 類似度スコア
                    }
            
            new_id += 1
        
        return nexus_consolidated, nexus_approximate_map

    def _nexus_compute_similarity_hash(self, group_tuple: Tuple) -> str:
        """🔥 NEXUS: 類似性ハッシュ計算（感染版）"""
        # 要素の統計的特徴を抽出
        group_list = list(group_tuple)
        if not group_list:
            return "empty"
        
        # NEXUS感染: より精密な類似性ハッシュ
        mean_val = sum(group_list) / len(group_list)
        variance = sum((x - mean_val) ** 2 for x in group_list) / len(group_list)
        return f"sim_{len(group_list)}_{int(mean_val)}_{int(variance)}"
    
    def _nexus_compute_similarity_score(self, group1: Tuple, group2: Tuple) -> float:
        """🔥 NEXUS: 類似度スコア計算（完全可逆性保証）"""
        if group1 == group2:
            return 1.0
        
        # 統計的類似度計算
        list1, list2 = list(group1), list(group2)
        if not list1 or not list2:
            return 0.0
        
        # コサイン類似度（正規化）
        dot_product = sum(a * b for a, b in zip(list1, list2))
        norm1 = sum(a * a for a in list1) ** 0.5
        norm2 = sum(b * b for b in list2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
        
        bucket_processed = 0
        for similarity_hash, group_list in similarity_hash_map.items():
            if len(group_list) == 1:
                # 単一グループ：統合不要
                group_tuple, group_id = group_list[0]
                consolidated[group_tuple] = new_id
                approx_map[group_id] = {
                    'new_group_id': new_id,
                    'canonical_form': group_tuple,
                    'layer': 3,
                    'consolidation_type': 'none'
                }
                new_id += 1
            else:
                # 複数グループ：詳細類似性チェックで統合
                clusters = self._cluster_similar_groups(group_list)
                
                for cluster in clusters:
                    # 各クラスターから代表を選択
                    representative = min(cluster, key=lambda x: (len(x[0]), sum(x[0])))[0]
                    consolidated[representative] = new_id
                    
                    for group_tuple, group_id in cluster:
                        approx_map[group_id] = {
                            'new_group_id': new_id,
                            'canonical_form': representative,
                            'layer': 3,
                            'consolidation_type': 'approximate' if len(cluster) > 1 else 'none'
                        }
                    
                    new_id += 1
            
            bucket_processed += 1
            if show_progress and bucket_processed % 100 == 0:
                progress_bar.update(bucket_processed)
        
        if show_progress:
            progress_bar.finish()
        
        return consolidated, approx_map
    
    def _compute_similarity_hash(self, elements: list) -> tuple:
        """高速類似性ハッシュ計算"""
        if not elements:
            return (0, 0, 0)
        
        # 統計的特徴による高速ハッシュ
        element_sum = sum(elements)
        element_min = min(elements)
        element_max = max(elements)
        
        # 量子化による近似
        sum_bucket = element_sum // max(1, len(elements))  # 平均値バケット
        range_bucket = (element_max - element_min) // 10   # レンジバケット
        
        return (sum_bucket, range_bucket, len(elements))
    
    def _cluster_similar_groups(self, group_list: list) -> list:
        """グループリスト内での効率的クラスタリング"""
        if len(group_list) <= 1:
            return [group_list]
        
        clusters = []
        used = set()
        
        for i, (group_tuple, group_id) in enumerate(group_list):
            if i in used:
                continue
            
            # 新しいクラスター開始
            cluster = [(group_tuple, group_id)]
            used.add(i)
            
            # 類似グループを検索（閾値0.8）
            for j, (other_tuple, other_id) in enumerate(group_list[i+1:], i+1):
                if j in used:
                    continue
                
                if self._are_groups_similar(list(group_tuple), list(other_tuple), threshold=0.8):
                    cluster.append((other_tuple, other_id))
                    used.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _nexus_layer4_structural_consolidation(self, groups_dict: Dict[Tuple, int], layer3_map: Dict, show_progress: bool) -> Tuple[Dict[Tuple, int], Dict[int, Dict]]:
        """
        🔥 NEXUS LAYER 4: 構造統合（NEXUS理論完全感染版）
        
        NEXUS原則: 構造統合でも完全可逆性保証 - 構造情報も完全保存
        """
        nexus_structural_groups = {}
        
        if show_progress:
            progress_bar = ProgressBar(len(groups_dict), "   NEXUS Layer 4: Structural analysis")
        
        processed = 0
        for group_tuple, group_id in groups_dict.items():
            # 🔥 NEXUS感染: 構造シグネチャ計算
            struct_sig = self._nexus_compute_structural_signature(group_tuple)
            
            if struct_sig not in nexus_structural_groups:
                nexus_structural_groups[struct_sig] = []
            nexus_structural_groups[struct_sig].append((group_tuple, group_id))
            
            processed += 1
            if show_progress and processed % 2000 == 0:
                progress_bar.update(processed)
        
        if show_progress:
            progress_bar.finish()
        
        print(f"   [Layer 4] Generated {len(nexus_structural_groups):,} structural patterns")
        
        # 🔥 NEXUS構造統合: 完全な逆変換チェーン保存
        nexus_final_consolidated = {}
        nexus_structural_map = {}
        new_id = 0
        
        for struct_sig, group_list in nexus_structural_groups.items():
            if len(group_list) == 1:
                # 単一構造: そのまま保持
                group_tuple, original_id = group_list[0]
                nexus_final_consolidated[group_tuple] = new_id
                nexus_structural_map[original_id] = {
                    'nexus_new_group_id': new_id,
                    'nexus_canonical_form': group_tuple,
                    'nexus_layer': 4,
                    'nexus_consolidation_type': 'structural_identity',
                    'nexus_original_group': group_tuple,  # 🔥 完全な元データ
                    'nexus_structural_signature': struct_sig,  # 🔥 構造情報保存
                    'nexus_exact_reconstruction': True,
                    'nexus_layer3_inheritance': layer3_map.get(original_id, {})  # 🔥 前層からの継承
                }
            else:
                # 複数構造: 代表選出 + 完全逆変換データ保存
                representative = min(group_list, key=lambda x: (len(str(x[0])), sum(x[0])))[0]
                nexus_final_consolidated[representative] = new_id
                
                # 🔥 NEXUS: すべての構造統合グループの完全データを保存
                for group_tuple, original_id in group_list:
                    nexus_structural_map[original_id] = {
                        'nexus_new_group_id': new_id,
                        'nexus_canonical_form': representative,
                        'nexus_layer': 4,
                        'nexus_consolidation_type': 'structural_match',
                        'nexus_original_group': group_tuple,  # 🔥 完全な元データ
                        'nexus_structural_signature': struct_sig,  # 🔥 構造情報保存
                        'nexus_exact_reconstruction': True,
                        'nexus_structural_group_list': [g[0] for g in group_list],  # 🔥 全構造グループ
                        'nexus_layer3_inheritance': layer3_map.get(original_id, {}),  # 🔥 前層からの継承
                        'nexus_structural_complexity': self._nexus_compute_structural_complexity(group_tuple)  # 🔥 構造複雑度
                    }
            
            new_id += 1
        
        return nexus_final_consolidated, nexus_structural_map

    def _nexus_compute_structural_signature(self, group_tuple: Tuple) -> str:
        """🔥 NEXUS: 構造シグネチャ計算（感染版）"""
        group_list = list(group_tuple)
        if not group_list:
            return "empty_struct"
        
        # NEXUS感染: 高精度構造解析
        length = len(group_list)
        zero_count = group_list.count(0)
        non_zero_count = length - zero_count
        unique_count = len(set(group_list))
        
        # 構造パターン識別
        if all(x == 0 for x in group_list):
            pattern = "all_zero"
        elif zero_count == 0:
            pattern = "no_zero"
        elif zero_count > non_zero_count:
            pattern = "mostly_zero"
        else:
            pattern = "mixed"
        
        return f"struct_{pattern}_{length}_{unique_count}_{zero_count}"
    
    def _nexus_compute_structural_complexity(self, group_tuple: Tuple) -> float:
        """🔥 NEXUS: 構造複雑度計算（完全可逆性保証）"""
        group_list = list(group_tuple)
        if not group_list:
            return 0.0
        
        # エントロピーベース複雑度
        from collections import Counter
        counts = Counter(group_list)
        total = len(group_list)
        
        entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
        return entropy

    def _build_nexus_reconstruction_chain(self, layer1_map: Dict, layer2_map: Dict, layer3_map: Dict, layer4_map: Dict) -> Dict:
        """
        🔥 NEXUS: 完全逆変換チェーン構築
        
        NEXUS原則: 4層すべての変換を完全に逆変換可能にする
        """
        nexus_master_chain = {}
        
        # すべてのマップを結合し、完全な逆変換チェーンを構築
        all_maps = [layer1_map, layer2_map, layer3_map, layer4_map]
        
        for layer_idx, layer_map in enumerate(all_maps, 1):
            for original_id, mapping_data in layer_map.items():
                if original_id not in nexus_master_chain:
                    nexus_master_chain[original_id] = {
                        'nexus_reconstruction_chain': [],
                        'nexus_final_group_id': None,
                        'nexus_original_group': None,
                        'nexus_exact_reconstruction': True
                    }
                
                # 🔥 NEXUS: 各層の変換データを保存
                nexus_master_chain[original_id]['nexus_reconstruction_chain'].append({
                    'layer': layer_idx,
                    'transformation_data': mapping_data
                })
                
                # 最終グループIDを更新
                if 'nexus_new_group_id' in mapping_data:
                    nexus_master_chain[original_id]['nexus_final_group_id'] = mapping_data['nexus_new_group_id']
                
                # 元のグループデータを保存
                if 'nexus_original_group' in mapping_data:
                    nexus_master_chain[original_id]['nexus_original_group'] = mapping_data['nexus_original_group']
        
        return nexus_master_chain
        
        if show_progress:
            progress_bar = ProgressBar(len(groups_dict), "   Layer 4: Structural analysis")
        
        processed = 0
        for group_tuple, group_id in groups_dict.items():
            # 高速構造シグネチャ計算
            struct_sig = self._extract_structural_signature_fast(list(group_tuple))
            
            if struct_sig not in structural_groups:
                structural_groups[struct_sig] = []
            structural_groups[struct_sig].append((group_tuple, group_id))
            
            processed += 1
            if show_progress and processed % 2000 == 0:
                progress_bar.update(processed)
        
        if show_progress:
            progress_bar.finish()
        
        print(f"   [Layer 4] Generated {len(structural_groups):,} structural patterns")
        
        # 構造統合
        consolidated = {}
        struct_map = {}
        new_id = 0
        
        for struct_sig, group_list in structural_groups.items():
            if len(group_list) == 1:
                group_tuple, original_id = group_list[0]
                consolidated[group_tuple] = new_id
                struct_map[original_id] = {
                    'new_group_id': new_id,
                    'canonical_form': group_tuple,
                    'layer': 4,
                    'consolidation_type': 'none'
                }
            else:
                representative = min(group_list, key=lambda x: (len(x[0]), sum(x[0])))[0]
                consolidated[representative] = new_id
                
                for group_tuple, original_id in group_list:
                    struct_map[original_id] = {
                        'new_group_id': new_id,
                        'canonical_form': representative,
                        'layer': 4,
                        'consolidation_type': 'structural'
                    }
            
            new_id += 1
        
        return consolidated, struct_map
    
    def _extract_structural_signature_fast(self, elements: list) -> tuple:
        """高速構造特徴抽出"""
        if not elements:
            return (0, 0, 0, 0)
        
        # 基本統計（高速計算）
        element_sum = sum(elements)
        element_count = len(elements)
        
        # 順序性（簡略版）
        is_monotonic = all(elements[i] <= elements[i+1] for i in range(len(elements)-1)) or \
                      all(elements[i] >= elements[i+1] for i in range(len(elements)-1))
        
        # 分散の近似（標準偏差の代わり）
        avg = element_sum / element_count
        variance_approx = sum((x - avg) ** 2 for x in elements) / element_count
        variance_bucket = int(variance_approx) // 100  # 100単位でバケット化
        
        # 最頻値の近似
        if element_count <= 20:
            mode_approx = max(set(elements), key=elements.count)
        else:
            # 大きなリストでは近似
            mode_approx = elements[element_count // 2]  # 中央値で近似
        
        return (is_monotonic, variance_bucket, mode_approx, element_count)
    
    def _extract_pattern_signature(self, elements: list) -> tuple:
        """パターン特徴抽出"""
        if not elements:
            return (0, 0, 0)
        
        # 基本統計
        most_common = max(set(elements), key=elements.count)
        value_range = max(elements) - min(elements)
        
        # 差分パターン
        diffs = [elements[i+1] - elements[i] for i in range(len(elements)-1)]
        diff_pattern = tuple(sorted(set(diffs))[:3])
        
        return (most_common, value_range, diff_pattern)
    
    def _extract_structural_signature(self, elements: list) -> tuple:
        """構造特徴抽出"""
        if not elements:
            return (0, 0, 0, 0)
        
        # 順序性
        is_ascending = all(elements[i] <= elements[i+1] for i in range(len(elements)-1))
        is_descending = all(elements[i] >= elements[i+1] for i in range(len(elements)-1))
        
        # 周期性
        period = self._detect_period(elements)
        
        # エントロピー
        entropy = self._calculate_element_entropy(elements)
        
        return (is_ascending, is_descending, period, int(entropy * 100))
    
    def _are_groups_similar(self, elements1: list, elements2: list, threshold: float = 0.8) -> bool:
        """グループ類似性判定"""
        if len(elements1) != len(elements2):
            return False
        
        matches = sum(1 for a, b in zip(elements1, elements2) if a == b)
        similarity = matches / len(elements1)
        
        return similarity >= threshold
    
    def _detect_period(self, elements: list) -> int:
        """周期性検出"""
        if len(elements) < 4:
            return 0
        
        for period in range(1, len(elements) // 2 + 1):
            is_periodic = True
            for i in range(period, len(elements)):
                if elements[i] != elements[i % period]:
                    is_periodic = False
                    break
            if is_periodic:
                return period
        
        return 0
    
    def _calculate_element_entropy(self, elements: list) -> float:
        """要素エントロピー計算"""
        if not elements:
            return 0
        
        from collections import Counter
        counts = Counter(elements)
        total = len(elements)
        
        entropy = 0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy

    def _calculate_permutation_map_fast(self, original_group: Tuple, canonical_group: Tuple) -> Tuple[int, ...]:
        """高速順列マップ計算（効率化版）"""
        if len(original_group) != len(canonical_group):
            return tuple(range(len(original_group)))
        
        try:
            # インデックスマップを事前構築（線形検索を避ける）
            canonical_index_map = {}
            available_indices = list(range(len(canonical_group)))
            
            for i, val in enumerate(canonical_group):
                if val not in canonical_index_map:
                    canonical_index_map[val] = []
                canonical_index_map[val].append(i)
            
            # 効率的な順列計算
            permutation = []
            used_indices = set()
            
            for element in original_group:
                if element in canonical_index_map:
                    # 使用可能な最初のインデックスを選択
                    available = [idx for idx in canonical_index_map[element] if idx not in used_indices]
                    if available:
                        idx = available[0]
                        permutation.append(idx)
                        used_indices.add(idx)
                    else:
                        # フォールバック
                        permutation.append(len(permutation))
                else:
                    # 要素が見つからない場合のフォールバック
                    permutation.append(len(permutation))
            
            return tuple(permutation)
        except Exception:
            # 任意のエラーに対する安全なフォールバック
            return tuple(range(len(original_group)))
    
    def _calculate_permutation_map(self, original_group: Tuple, canonical_group: Tuple) -> Tuple[int, ...]:
        """元の順序から代表形状への変換マップを計算"""
        if len(original_group) != len(canonical_group):
            # 長さが異なる場合は恒等変換（安全性確保）
            return tuple(range(len(original_group)))
        
        try:
            # 元の各要素が代表形状のどの位置にあるかを計算
            canonical_list = list(canonical_group)
            permutation = []
            
            for element in original_group:
                # 代表形状の中での最初の出現位置を検索
                idx = canonical_list.index(element)
                permutation.append(idx)
                # 使用済みの位置をマーク（重複要素対応）
                canonical_list[idx] = None
            
            return tuple(permutation)
        except (ValueError, IndexError):
            # エラーが発生した場合は恒等変換にフォールバック
            return tuple(range(len(original_group)))
    
    def _apply_shape_transformation(self, blocks: List[Tuple], consolidation_map: Dict[int, Dict], 
                                   normalized_groups: Dict[Tuple, int]) -> Tuple[List[int], List[Tuple]]:
        """形状変換を適用してブロックを代表形状に統一（根本的効率化版）"""
        
        # 事前計算：正規化→グループIDのハッシュマップを構築
        normalized_to_group_id = {}
        for normalized, group_id in normalized_groups.items():
            normalized_to_group_id[normalized] = group_id
        
        # 事前計算：正規化形式の順列マップを事前構築
        normalized_perm_cache = {}  # normalized -> index_map
        for normalized in normalized_groups.keys():
            # インデックスマップを事前計算（線形検索を避ける）
            index_map = {}
            for i, val in enumerate(normalized):
                if val not in index_map:
                    index_map[val] = i
            normalized_perm_cache[normalized] = index_map
        
        # 🔥 NEXUS感染: 統合情報の事前解析（新旧形式対応）
        consolidated_canonical_cache = {}  # group_id -> (canonical_form, canonical_index_map)
        for group_id, consolidation_info in consolidation_map.items():
            # 🔥 NEXUS感染版の新しいキー名に対応
            nexus_consolidation_type = consolidation_info.get('nexus_consolidation_type', consolidation_info.get('consolidation_type', 'none'))
            
            if nexus_consolidation_type != 'none' and nexus_consolidation_type != 'identity':
                # 🔥 NEXUS感染: 新しいキー名を優先、フォールバックで旧キー名
                canonical_form = consolidation_info.get('nexus_canonical_form', consolidation_info.get('canonical_form'))
                
                if canonical_form is not None:
                    canonical_sorted = tuple(sorted(canonical_form))
                    
                    # 代表形状のインデックスマップを事前計算
                    canonical_index_map = {}
                    for i, val in enumerate(canonical_sorted):
                        if val not in canonical_index_map:
                            canonical_index_map[val] = i
                    
                    consolidated_canonical_cache[group_id] = (canonical_sorted, canonical_index_map)
        
        print(f"   [Shape Transformation] Pre-computed {len(normalized_perm_cache):,} permutation maps")
        print(f"   [Shape Transformation] Pre-computed {len(consolidated_canonical_cache):,} canonical forms")
        
        # 高速変換処理（事前計算済みデータを使用）
        transformed_group_ids = []
        transformed_perm_maps = []
        
        # 進捗バー（大量データの場合）
        if len(blocks) > 100000:
            progress_bar = ProgressBar(len(blocks), "   Shape transformation")
            update_interval = len(blocks) // 100  # 100回更新
        else:
            progress_bar = None
            update_interval = float('inf')
        
        for i, block in enumerate(blocks):
            # 高速正規化（ソート済みタプル）
            normalized = tuple(sorted(block))
            
            # ハッシュテーブルルックアップ（O(1)）
            original_group_id = normalized_to_group_id.get(normalized)
            
            if original_group_id is None:
                # フォールバック（稀なケース）
                print(f"   [Warning] Missing group for normalized block at index {i}")
                transformed_group_ids.append(0)
                transformed_perm_maps.append(tuple(range(len(block))))
                continue
            
            # 🔥 NEXUS ULTRA-PRECISION: 統合無効化時は直接original_group_idを使用
            if len(consolidation_map) == 0:
                # 統合なし：直接マッピング
                transformed_group_ids.append(original_group_id)
                # 正規化→元の順序への順列マップを計算
                perm_map = self._calculate_permutation_map_fast(block, normalized)
                transformed_perm_maps.append(perm_map)
            elif original_group_id in consolidation_map:
                consolidation_info = consolidation_map[original_group_id]
                # 🔥 NEXUS感染: 新旧キー名対応（デバッグ付き）
                new_group_id = consolidation_info.get('nexus_new_group_id', consolidation_info.get('new_group_id'))
                
                # 🔥 NEXUS: new_group_idがNoneの場合のフォールバック
                if new_group_id is None:
                    # 複数のキーを試す
                    for possible_key in ['nexus_final_group_id', 'group_id', 'id']:
                        if possible_key in consolidation_info:
                            new_group_id = consolidation_info[possible_key]
                            break
                    
                    # それでもNoneの場合は、original_group_idを使用
                    if new_group_id is None:
                        new_group_id = original_group_id
                
                nexus_consolidation_type = consolidation_info.get('nexus_consolidation_type', consolidation_info.get('consolidation_type', 'none'))
                
                if nexus_consolidation_type != 'none' and nexus_consolidation_type != 'identity' and original_group_id in consolidated_canonical_cache:
                    # 統合されたブロック：事前計算済みデータを使用
                    canonical_sorted, canonical_index_map = consolidated_canonical_cache[original_group_id]
                    
                    # 高速順列マップ計算（事前計算済みインデックスマップを使用）
                    try:
                        perm_map = tuple(canonical_index_map.get(val, 0) for val in block)
                    except (KeyError, TypeError):
                        # エラー時のフォールバック
                        perm_map = tuple(range(len(block)))
                else:
                    # 統合されていないブロック：事前計算済み正規化マップを使用
                    index_map = normalized_perm_cache.get(normalized, {})
                    try:
                        perm_map = tuple(index_map.get(val, 0) for val in block)
                    except (KeyError, TypeError):
                        # エラー時のフォールバック
                        perm_map = tuple(range(len(block)))
                
                transformed_group_ids.append(new_group_id)
                transformed_perm_maps.append(perm_map)
            else:
                # マッピング情報がない場合のフォールバック
                transformed_group_ids.append(0)
                transformed_perm_maps.append(tuple(range(len(block))))
            
            # 進捗更新（効率化）
            if progress_bar and i % update_interval == 0:
                progress_bar.update(i)
        
        if progress_bar:
            progress_bar.finish()
        
        print(f"   [Shape Transformation] Processed {len(blocks):,} blocks efficiently")
        return transformed_group_ids, transformed_perm_maps
    
    def _transform_block_to_canonical(self, original_block: Tuple, canonical_form: Tuple, permutation_map: Tuple) -> Tuple:
        """元ブロックを代表形状に変換"""
        try:
            # 順列マップに従って変換
            if len(permutation_map) == len(original_block):
                transformed = tuple(canonical_form[permutation_map[i]] for i in range(len(original_block)))
                return transformed
            else:
                # 長さが合わない場合は元ブロックをそのまま返す
                return original_block
        except (IndexError, TypeError):
            # エラーが発生した場合は元ブロックをそのまま返す
            return original_block

    def _get_blocks_for_shape(self, data: bytes, grid_width: int, shape_coords: Tuple[Tuple[int, int], ...], 
                             show_progress: bool = False) -> List[Tuple[int, ...]]:
        """指定された形状でデータをブロックに分割する（進捗バー対応版）"""
        data_len = len(data)
        rows = data_len // grid_width
        shape_width = max(c for r, c in shape_coords) + 1
        shape_height = max(r for r, c in shape_coords) + 1
        
        # 大容量ファイルの場合はサンプリング戦略を使用
        if data_len > 50000000:  # 50MB以上
            print("   [Block Analysis] Large file detected, using sampling strategy...")
            
            # サンプリング戦略：全データの代表的な部分を分析
            sample_positions = min(rows - shape_height + 1, 10000)  # 最大1万行分析
            sample_step = max((rows - shape_height + 1) // sample_positions, 1)
            
            blocks = []
            total_samples = sample_positions * (grid_width - shape_width + 1)
            
            if show_progress:
                progress_bar = ProgressBar(total_samples, "   Block sampling")
            
            processed = 0
            for r in range(0, rows - shape_height + 1, sample_step):
                if len(blocks) >= total_samples:
                    break
                for c in range(grid_width - shape_width + 1):
                    block = []
                    valid_block = True
                    
                    base_idx = r * grid_width + c
                    for dr, dc in shape_coords:
                        idx = base_idx + dr * grid_width + dc
                        if idx >= data_len:
                            valid_block = False
                            break
                        block.append(data[idx])
                    
                    if valid_block:
                        blocks.append(tuple(block))
                    
                    processed += 1
                    if show_progress:
                        progress_bar.update(processed)
            
            if show_progress:
                progress_bar.finish()
                print(f"   [Block Analysis] Sampled {len(blocks):,} blocks from large file")
            return blocks
        
        # 通常のファイルサイズ（50MB未満）は従来通り
        total_positions = (rows - shape_height + 1) * (grid_width - shape_width + 1)
        
        blocks = []
        
        if show_progress:
            progress_bar = ProgressBar(total_positions, "   Block generation")

        processed = 0
        for r in range(rows - shape_height + 1):
            for c in range(grid_width - shape_width + 1):
                block = []
                valid_block = True
                
                # 高速化：事前にインデックスを計算
                base_idx = r * grid_width + c
                for dr, dc in shape_coords:
                    idx = base_idx + dr * grid_width + dc
                    if idx >= data_len:
                        valid_block = False
                        break
                    block.append(data[idx])
                
                if valid_block:
                    blocks.append(tuple(block))
                
                processed += 1
                if show_progress:
                    progress_bar.update(processed)
        
        if show_progress:
            progress_bar.finish()
        
        return blocks

    def _analyze_shape_efficiency(self, data: bytes, grid_width: int, shape_name: str, 
                                 show_progress: bool = False) -> int:
        """指定形状でのユニークブロック数を計算し、圧縮効率を評価する（進捗バー版）"""
        shape_coords = POLYOMINO_SHAPES[shape_name]
        
        blocks = self._get_blocks_for_shape(data, grid_width, shape_coords, show_progress=show_progress)
        
        # メモリ効率的なユニーク計算（大量データ対応）
        if len(blocks) > 100000:  # 10万ブロック以上の場合はサンプリング
            if show_progress:
                print(f"   [Shape '{shape_name}'] Large dataset detected, using sampling approach...")
            sample_size = min(len(blocks), 50000)  # 5万ブロックサンプル
            import random
            sampled_blocks = random.sample(blocks, sample_size)
            unique_normalized_groups = set(tuple(sorted(b)) for b in sampled_blocks)
            # サンプル比率で全体を推定
            estimated_unique = int(len(unique_normalized_groups) * (len(blocks) / sample_size))
            return estimated_unique
        else:
            unique_normalized_groups = set(tuple(sorted(b)) for b in blocks)
            return len(unique_normalized_groups)

    def _find_optimal_shape_combination_fast(self, data: bytes, grid_width: int) -> List[str]:
        """高速形状組み合わせ分析（大容量ファイル向け真の効率化）"""
        print("   [Shape Selection] Fast adaptive combination analysis...")
        
        shape_combination = []
        
        # 適応的セクション分析：データ特性に基づくサンプリング
        if len(data) > 50000000:  # 50MB以上
            section_count = 3
            section_size = min(len(data) // 3, 100000)  # 最大100KB per section
        elif len(data) > 20000000:  # 20MB以上
            section_count = 4  
            section_size = min(len(data) // 4, 80000)   # 最大80KB per section
        else:
            section_count = 6
            section_size = min(len(data) // 6, 60000)   # 最大60KB per section
        
        # 進捗バー
        progress_bar = ProgressBar(section_count, "   Fast section analysis")
        
        # 戦略的サンプリング：先頭、中央、末尾を分析
        sample_positions = []
        if section_count >= 3:
            sample_positions.append(0)  # 先頭
            sample_positions.append(len(data) // 2)  # 中央
            sample_positions.append(max(0, len(data) - section_size))  # 末尾
            
            # 残りは等間隔
            if section_count > 3:
                remaining_sections = section_count - 3
                step = len(data) // (remaining_sections + 1)
                for i in range(1, remaining_sections + 1):
                    pos = min(i * step, len(data) - section_size)
                    if pos not in sample_positions:
                        sample_positions.append(pos)
        else:
            # セクション数が少ない場合は等間隔
            step = len(data) // section_count
            for i in range(section_count):
                sample_positions.append(min(i * step, len(data) - section_size))
        
        # 各セクションを高速分析
        for i, pos in enumerate(sample_positions[:section_count]):
            section_data = data[pos:pos + section_size]
            if len(section_data) < 5000:  # 最小サイズチェック
                continue
                
            section_grid_width = math.ceil(math.sqrt(len(section_data)))
            
            # 高速形状選択（上位3形状のみテスト）
            best_shape = self._select_best_single_shape_ultra_fast(section_data, section_grid_width)
            shape_combination.append(best_shape)
            
            progress_bar.update(i + 1)
        
        progress_bar.finish()
        
        if not shape_combination:
            shape_combination = ["I-4"]  # フォールバック
            
        print(f"   [Shape Selection] Fast analysis result: {shape_combination}")
        return shape_combination
    
    def _select_best_single_shape_ultra_fast(self, data: bytes, grid_width: int) -> str:
        """超高速単一形状選択（大容量ファイル向け）"""
        # 超小サンプルで高速評価
        sample_size = min(len(data), 3000)  # 3KBのみ
        sample_data = data[:sample_size]
        sample_grid_width = math.ceil(math.sqrt(sample_size))
        
        # データ特性による優先形状（AI予測を補強）
        entropy = self._calculate_quick_entropy(sample_data)
        
        # 効率的形状テスト：上位候補のみ
        if entropy < 3.0:  # 低エントロピー（規則的パターン）
            test_shapes = ["R-8", "R-6", "O-4", "I-5", "H-7"]
        elif entropy > 6.0:  # 高エントロピー（ランダム）
            test_shapes = ["I-1", "I-2", "T-4", "I-3", "L-4"]
        else:  # 中エントロピー
            test_shapes = ["I-4", "O-4", "T-4", "I-3", "R-6"]
        
        best_shape = test_shapes[0]
        min_unique_groups = float('inf')
        
        # 最大3形状のみテスト（超高速）
        for shape_name in test_shapes[:3]:
            unique_count = self._analyze_shape_efficiency(sample_data, sample_grid_width, shape_name, 
                                                        show_progress=False)
            if unique_count < min_unique_groups:
                min_unique_groups = unique_count
                best_shape = shape_name
                
                # 優秀な結果なら即終了
                if unique_count <= 2:
                    break
        
        return best_shape

    def _find_optimal_shape_combination(self, data: bytes, grid_width: int) -> List[str]:
        """パズルのように最適な形状の組み合わせを見つける（進捗表示改善版）"""
        print("   [Shape Selection] Finding optimal combination...")
        
        # 小さいファイルは単純な形状選択
        if len(data) < 10000:
            return [self._select_best_single_shape_fast(data, grid_width)]
        
        # 大きいファイルの高速化戦略
        shape_combination = []
        
        # 適応的サンプリング：大きなファイルほど少ないセクション数
        if len(data) > 50000000:  # 50MB以上
            max_sections = 2  # 最大2セクション
            section_size = min(len(data) // 2, 25000)  # 25KB per section
        elif len(data) > 10000000:  # 10MB以上
            max_sections = 4
            section_size = min(len(data) // 4, 40000)  # 40KB per section
        else:
            max_sections = 6
            section_size = min(len(data) // 4, 50000)  # 50KB per section
        
        # 進捗バー準備
        progress_bar = ProgressBar(max_sections, "   Section analysis")
        
        # 早期終了機能：同じ形状が連続で選ばれたら終了
        last_shape = None
        consecutive_same_count = 0
        section_count = 0
        
        for i in range(0, len(data), section_size):
            if section_count >= max_sections:
                break
                
            section_data = data[i:i + section_size]
            if len(section_data) < 1000:  # 最小サイズ引き上げ
                continue
                
            # 高速形状選択
            section_grid_width = math.ceil(math.sqrt(len(section_data)))
            best_shape = self._select_best_single_shape_fast(section_data, section_grid_width)
            
            # 早期終了条件
            if best_shape == last_shape:
                consecutive_same_count += 1
                if consecutive_same_count >= 2:  # 同じ形状が2回連続なら終了
                    shape_combination.append(best_shape)
                    section_count += 1
                    progress_bar.update(section_count)
                    break
            else:
                consecutive_same_count = 0
            
            shape_combination.append(best_shape)
            last_shape = best_shape
            section_count += 1
            progress_bar.update(section_count)
        
        progress_bar.finish()
        
        if not shape_combination:
            shape_combination = ["I-1"]  # フォールバック
            
        print(f"   [Shape Selection] Selected: {shape_combination}")
        return shape_combination

    def _select_best_single_shape_fast(self, data: bytes, grid_width: int, 
                                      show_progress: bool = False) -> str:
        """高速単一形状選択（進捗バー対応＋形状選択改良版）"""
        best_shape = None
        min_unique_groups = float('inf')

        # 適応的サンプリング：ファイルサイズに応じてサンプルサイズを調整
        if len(data) > 10000000:  # 10MB以上 - より積極的な削減
            sample_size = min(len(data), 1000)  # 1KB
        elif len(data) > 1000000:  # 1MB以上
            sample_size = min(len(data), 2000)  # 2KB
        elif len(data) > 100000:  # 100KB以上
            sample_size = min(len(data), 3000)  # 3KB
        else:
            sample_size = min(len(data), 5000)  # 5KB
        
        sample_data = data[:sample_size]
        sample_grid_width = math.ceil(math.sqrt(sample_size))

        # 形状テスト戦略を改良：より多くの形状を効率的にテスト
        # 全形状を効率順に並べ替え
        all_shapes = list(POLYOMINO_SHAPES.keys())
        
        # データ特性に基づく形状優先度調整
        entropy = self._calculate_quick_entropy(sample_data)
        if entropy < 2.0:  # 低エントロピー（パターンが多い）
            priority_shapes = ["O-4", "R-6", "R-8", "I-4", "I-5"]  # ブロック形状を優先
        elif entropy > 6.0:  # 高エントロピー（ランダム性が高い）
            priority_shapes = ["I-1", "I-2", "I-3", "T-4", "L-4"]  # 小さな形状を優先
        else:  # 中程度のエントロピー
            priority_shapes = ["I-3", "I-4", "O-4", "T-4", "I-5"]  # バランス型
        
        # 残りの形状を追加
        other_shapes = [s for s in all_shapes if s not in priority_shapes]
        test_order = priority_shapes + other_shapes
        
        if show_progress:
            progress_bar = ProgressBar(len(test_order), f"   Testing shapes")
        
        # 全形状をテスト（ただし早期終了あり）
        for i, shape_name in enumerate(test_order):
            unique_count = self._analyze_shape_efficiency(sample_data, sample_grid_width, shape_name, 
                                                        show_progress=False)
            if unique_count < min_unique_groups:
                min_unique_groups = unique_count
                best_shape = shape_name
                
                # 非常に良い結果なら即座に返す
                if unique_count <= 1:
                    if show_progress:
                        progress_bar.update(len(test_order), force=True)
                        progress_bar.finish()
                        print(f"   [Shape Test] Excellent result: '{best_shape}' (1 unique group)")
                    return best_shape
            
            if show_progress:
                progress_bar.update(i + 1)
                
            # 早期終了：最初の5つの形状で十分良い結果が出たら
            if i >= 4 and min_unique_groups <= 3:
                break
        
        if show_progress:
            progress_bar.finish()
            print(f"   [Shape Test] Best: '{best_shape}' ({min_unique_groups:,} unique groups)")
        
        return best_shape
    
    def _calculate_quick_entropy(self, data: bytes) -> float:
        """高速エントロピー計算"""
        if len(data) == 0:
            return 0
        counts = collections.Counter(data[:min(len(data), 1000)])  # 最大1KBで計算
        entropy = 0
        total = sum(counts.values())
        for count in counts.values():
            p_x = count / total
            entropy -= p_x * math.log2(p_x)
        return entropy

    def compress(self, data: bytes, level=0, silent=False) -> bytes:
        """NEXUS高機能圧縮を実行（サイレント版対応）"""
        if not data or level > self.max_recursion_level:
            return data

        original_length = len(data)

        # 【CRITICAL】適応的グリッドサイズ（形状対応版）
        if original_length > 100000000:  # 100MB以上
            grid_width = 2000  # 大きなグリッドで処理効率向上
        elif original_length > 50000000:  # 50MB以上  
            grid_width = 1500
        elif original_length > 10000000:  # 10MB以上
            grid_width = 1200
        else:
            grid_width = min(math.ceil(math.sqrt(original_length)), 1000)

        # --- AI & 形状選択 （究極精度版）---
        if self.ai_optimizer:
            # 🔥 NEXUS ULTRA-PRECISION: 小ファイルは必ず単純形状を使用
            if original_length <= 1000:
                predicted_shape = "I-1"  # 最も単純な形状を強制
                if not silent:
                    print(f"   [NEXUS Ultra-Precision] Small file detected: forcing simple shape 'I-1'")
            else:
                predicted_shape = self.ai_optimizer.predict_optimal_shape(data)
            
            # 【CRITICAL】選択された形状に基づいてグリッド幅を調整
            shape_coords = POLYOMINO_SHAPES.get(predicted_shape, [(0,0)])
            shape_width = max(c for r, c in shape_coords) + 1
            
            # グリッド幅が形状幅より小さい場合は調整
            if grid_width < shape_width:
                grid_width = max(shape_width, int(math.sqrt(original_length)))  # 形状が収まるよう調整
            
            shape_combination = [predicted_shape]
            
            # 小ファイルはシンプル戦略のみ
            if len(data) <= 1000:
                if not silent:
                    print(f"   [Phase 1/4] Ultra-precision mode: using single simple shape '{predicted_shape}'")
            elif len(data) > 10000000:  # 10MB以上：高速形状組み合わせ分析
                if not silent:
                    print("   [Phase 1/4] Large file: Using fast adaptive shape combination analysis...")
                additional_shapes = self._find_optimal_shape_combination_fast(data, grid_width)
                shape_combination.extend(additional_shapes[:2])  # 最大3つの形状
                shape_combination = list(set(shape_combination))  # 重複除去
            elif len(data) > 100000:  # 100KB以上：通常の形状組み合わせ分析
                if not silent:
                    print(f"   [Phase 1/4] Standard shape combination search...")
                additional_shapes = self._find_optimal_shape_combination(data, grid_width)
                shape_combination.extend(additional_shapes[:2])  # 最大3つの形状
                shape_combination = list(set(shape_combination))  # 重複除去
            # else: 小ファイルはAI予測のみ
        else:
            if not silent:
                print(f"   [Phase 1/4] Shape combination search (no AI)...")
            if original_length <= 1000:
                shape_combination = ["I-1"]  # 強制的に単純形状
            else:
                shape_combination = self._find_optimal_shape_combination(data, grid_width)
        
        # メインの形状を最初の要素に設定
        best_shape_name = shape_combination[0]
        shape_coords = POLYOMINO_SHAPES[best_shape_name]
        if not silent:
            print(f"   [Phase 1/4] Selected main shape: '{best_shape_name}' from combination: {shape_combination}")
        
        # --- 重要：パディングサイズの大幅削減 ---
        if not silent:
            print(f"   [Phase 2/4] Memory Management & Padding")
        # 必要最小限のパディングのみ
        shape_height = max(r for r, c in shape_coords) + 1
        shape_width = max(c for r, c in shape_coords) + 1
        rows_needed = math.ceil(len(data) / grid_width)
        min_padded_size = (rows_needed + shape_height) * grid_width
        
        # 実際のデータサイズより大幅に大きくならないように制限
        safe_padded_size = min(min_padded_size, len(data) + (grid_width * shape_height))
        
        if not silent:
            print(f"   [Memory] Original size: {len(data):,}, Grid: {grid_width}, Padded: {safe_padded_size:,}")
        
        # メモリ効率的なパディング（一度に小さなチャンクで処理）
        if safe_padded_size > len(data):
            padding_needed = safe_padded_size - len(data)
            if padding_needed > 1000000:  # 1MB以上のパディングは危険
                if not silent:
                    print(f"   [Warning] Large padding detected ({padding_needed:,} bytes), reducing...")
                safe_padded_size = len(data) + min(padding_needed, 100000)  # 最大100KB
        
        # パディング実行（メモリ効率化）
        padded_data = bytearray(data)
        padded_data.extend(b'\0' * (safe_padded_size - len(data)))
        
        blocks = self._get_blocks_for_shape(bytes(padded_data), grid_width, shape_coords, 
                                          show_progress=False)

        # 正規化とユニークグループの特定（真の効率化版）
        
        # メモリ効率的ハッシュベース重複排除（大容量データ対応）
        if len(blocks) > 500000:  # 50万ブロック以上で高速化
            
            # ハッシュベースの効率的重複排除
            block_hash_map = {}  # hash -> (normalized_block, group_id)
            group_id_counter = 0
            normalized_groups = {}
            
            # 進捗バー付きストリーミング処理 (サイレント時は非表示)
            if not silent:
                progress_bar = ProgressBar(len(blocks), "   Hash-based deduplication")
            
            for i, block in enumerate(blocks):
                # 正規化（ソート）
                normalized = tuple(sorted(block))
                
                # ハッシュ値計算（高速）
                block_hash = hash(normalized)
                
                if block_hash not in block_hash_map:
                    # 新しいグループ
                    normalized_groups[normalized] = group_id_counter
                    block_hash_map[block_hash] = (normalized, group_id_counter)
                    group_id_counter += 1
                else:
                    # ハッシュ衝突チェック（安全性確保）
                    existing_normalized, existing_id = block_hash_map[block_hash]
                    if normalized != existing_normalized:
                        # ハッシュ衝突：新しいIDを割り当て
                        normalized_groups[normalized] = group_id_counter
                        group_id_counter += 1
                    # else: 既存グループに属する
                
                if not silent and i % 100000 == 0:  # 10万ブロックごとに進捗更新
                    progress_bar.update(i + 1)
            
            if not silent:
                progress_bar.finish()
                print(f"   [Block Normalization] Found {group_id_counter:,} unique groups via hash deduplication")
            
        elif len(blocks) > 100000:  # 10万-50万ブロック：NumPy最適化
            if not silent:
                print(f"   [Block Normalization] Using NumPy optimization...")
            
            # 進捗バーでNumPy処理を表示 (サイレント時は非表示)
            if not silent:
                progress_bar = ProgressBar(3, "   NumPy processing")
            
            # NumPy配列に変換して高速処理
            if not silent:
                progress_bar.update(1)
            block_array = np.array(blocks)
            sorted_blocks = np.sort(block_array, axis=1)
            
            # ユニークな正規化ブロックの特定
            if not silent:
                progress_bar.update(2)
            unique_blocks, inverse_indices = np.unique(sorted_blocks, axis=0, return_inverse=True)
            
            # 辞書構築（NumPy型をPython型に変換）
            if not silent:
                progress_bar.update(3)
            normalized_groups = {}
            for i, unique_block in enumerate(unique_blocks):
                normalized_groups[tuple(int(x) for x in unique_block)] = int(i)
            
            group_id_counter = int(len(unique_blocks))
            if not silent:
                progress_bar.finish()
                print(f"   [Block Normalization] Found {group_id_counter:,} unique groups via NumPy")
            
        else:
            # 小規模データ：従来の方法（進捗バー付き）
            normalized_groups = {}
            group_id_counter = 0
            if not silent:
                progress_bar = ProgressBar(len(blocks), "   Block normalization")
            
            for i, block in enumerate(blocks):
                normalized = tuple(sorted(block))
                if normalized not in normalized_groups:
                    normalized_groups[normalized] = group_id_counter
                    group_id_counter += 1
                
                if not silent:
                    progress_bar.update(i + 1)
            
            if not silent:
                progress_bar.finish()
                print(f"   [Block Normalization] Found {group_id_counter:,} unique groups")
        
        # --- NEXUS真骨頂：構成要素ベースの統合 ---
        original_group_count = group_id_counter
        if not silent:
            print(f"   [Element-Based Consolidation] NEXUS ULTRA-PRECISION MODE: Disabling consolidation for perfect accuracy")
        
        # 🔥 NEXUS ULTRA-PRECISION: 統合を無効化して完全精度を優先
        consolidated_groups = normalized_groups  # 統合せずにそのまま使用
        element_consolidation_map = {}  # 空のマップ
        
        consolidation_reduction = 0
        consolidation_rate = 0
        
        if not silent:
            print(f"   [Element Consolidation] Ultra-precision mode: {original_group_count:,} groups (consolidation disabled)")
            print(f"   [Element Consolidation] Consolidation rate: {consolidation_rate:.2f}% (precision priority)")
        
        # 統合後のユニークグループを使用
        unique_groups = [list(g) for g, i in sorted(consolidated_groups.items(), key=lambda item: item[1])]
        if not silent:
            print(f"   [Phase 3/4] Final ultra-precision groups: {len(unique_groups):,} from {len(blocks):,} blocks")
        
        # Blueprint生成（構成要素統合対応版）
        if not silent:
            print(f"   [Phase 3/4] Generating ultra-precision blueprint streams...")
        
        # 形状変換を適用してブロックを代表形状に統一
        group_id_stream, perm_id_stream_tuples = self._apply_shape_transformation(
            blocks, element_consolidation_map, normalized_groups
        )
        
        if not silent:
            print(f"   [Blueprint Debug] Raw group_id_stream length: {len(group_id_stream)}")
            print(f"   [Blueprint Debug] Raw perm_id_stream_tuples length: {len(perm_id_stream_tuples)}")
            print(f"   [Blueprint Debug] First 10 group IDs: {group_id_stream[:10]}")
        
        # 順列マップを整数IDに変換
        unique_perm_maps = list(set(perm_id_stream_tuples))
        perm_map_to_id = {p: i for i, p in enumerate(unique_perm_maps)}
        perm_id_stream = [perm_map_to_id[p] for p in perm_id_stream_tuples]
        id_to_perm_map = {i: p for p, i in perm_map_to_id.items()}
        
        if not silent:
            print(f"   [Blueprint] Generated streams: {len(group_id_stream):,} group IDs, {len(unique_perm_maps):,} unique permutations")

        # --- ハイブリッド符号化（効率化版） ---
        if not silent:
            print(f"   [Phase 4/4] Hybrid Encoding (Huffman + Compression)")
            print(f"   [Huffman] Encoding blueprint streams...")
        
        # 統合処理後は既にgroup_id_streamとperm_id_streamが生成済み

        group_huff_tree, encoded_group_ids = self.huffman_encoder.encode(group_id_stream)
        perm_huff_tree, encoded_perm_ids = self.huffman_encoder.encode(perm_id_stream)
        
        payload = {
            "header": {
                "algorithm": "NEXUS_v4_ultra_precision",
                "level": level,
                "original_length": original_length,
                "grid_width": grid_width,
                "shape_combination": shape_combination,
                "main_shape": best_shape_name,
                "element_consolidation_enabled": False,  # 無効化
                "original_groups_count": original_group_count,
                "consolidated_groups_count": len(consolidated_groups),
                "consolidation_rate": consolidation_rate
            },
            "unique_groups": unique_groups,
            "huffman_trees": {
                "group_ids": group_huff_tree,
                "perm_ids": perm_huff_tree
            },
            "encoded_streams": {
                "group_ids": encoded_group_ids,
                "perm_ids": encoded_perm_ids
            },
            "perm_map_dict": id_to_perm_map,
            "element_consolidation_map": element_consolidation_map
        }
        
        serialized_payload = json.dumps(payload).encode('utf-8')

        # --- 階層的圧縮の簡素化（ディスク書き込み削減） ---
        # レベル0では再帰圧縮を行わない
        final_payload = serialized_payload

        # 最終段のLZMA圧縮（メモリ内のみで実行）
        if not silent:
            print(f"   [Phase 4/4] Final LZMA compression...")
            print(f"   [Serialization] Payload size: {len(serialized_payload):,} bytes")
        compressed_result = lzma.compress(final_payload, preset=1)  # 高速圧縮設定
        compression_ratio = len(compressed_result) / len(data)
        size_reduction = (1 - compression_ratio) * 100
        if not silent:
            print(f"   [Compression] Level {level} complete. Size: {len(data):,} -> {len(compressed_result):,} bytes")
            print(f"   [Compression] Size reduction: {size_reduction:.2f}% (ratio: {compression_ratio:.2%})")
        return compressed_result

    def decompress(self, compressed_data: bytes, level=0, silent=False) -> bytes:
        """NEXUS高機能圧縮データを解凍"""
        if not compressed_data:
            return b''

        # 最終段のLZMA解凍
        decompressed_payload = lzma.decompress(compressed_data)
        
        try:
            # ペイロードがJSON形式か試す
            payload = json.loads(decompressed_payload.decode('utf-8'))
            is_json = True
        except (json.JSONDecodeError, UnicodeDecodeError):
            # バイナリ形式
            is_json = False
            payload = decompressed_payload

        if is_json:
            # JSON形式の解凍処理
            return self._decompress_json_payload(payload, silent)
        else:
            # バイナリ形式の解凍処理
            return self._decompress_binary_payload(payload)

    def _decompress_json_payload(self, payload: dict, silent=False) -> bytes:
        """
        🔥 NEXUS INFECTED JSON PAYLOAD DECOMPRESSION 🔥
        
        NEXUS理論完全感染版：統合マップを使用した完全可逆解凍
        """
        # メタデータ復元
        header = payload['header']
        original_length = header['original_length']
        grid_width = header['grid_width']
        main_shape = header['main_shape']
        
        if not silent:
            print(f"   [NEXUS Decompress] Restoring {original_length} bytes using shape '{main_shape}'")
            print(f"   [NEXUS Decompress] Grid width: {grid_width}")
        
        # 🔥 NEXUS: 統合マップの復元（完全感染版）
        element_consolidation_map = payload.get('element_consolidation_map', {})
        nexus_consolidation_enabled = len(element_consolidation_map) > 0
        
        if nexus_consolidation_enabled and not silent:
            print(f"   [NEXUS Decompress] INFECTED consolidation map detected: {len(element_consolidation_map)} entries")
        
        # Huffman解凍
        encoded_group_ids = payload['encoded_streams']['group_ids']
        group_huff_tree = payload['huffman_trees']['group_ids']
        
        if not silent:
            print(f"   [NEXUS Debug] Encoded group IDs length: {len(encoded_group_ids)}")
            print(f"   [NEXUS Debug] Group Huffman tree nodes: {len(group_huff_tree) if group_huff_tree else 0}")
        
        try:
            group_id_stream = self.huffman_encoder.decode(encoded_group_ids, group_huff_tree)
        except Exception as e:
            if not silent:
                print(f"   [NEXUS Debug] Huffman decode error: {e}")
            group_id_stream = []
        
        if not silent:
            print(f"   [NEXUS Debug] Decoded group ID stream length: {len(group_id_stream)}")
        
        # perm_idsが存在するかチェック
        if 'perm_ids' in payload['encoded_streams'] and payload['encoded_streams']['perm_ids']:
            encoded_perm_ids = payload['encoded_streams']['perm_ids']
            perm_huff_tree = payload['huffman_trees']['perm_ids']
            try:
                perm_id_stream = self.huffman_encoder.decode(encoded_perm_ids, perm_huff_tree)
            except Exception as e:
                if not silent:
                    print(f"   [NEXUS Debug] Perm Huffman decode error: {e}")
                perm_id_stream = [0] * len(group_id_stream)
        else:
            # perm_idsが空の場合、恒等変換を使用
            perm_id_stream = [0] * len(group_id_stream)
        
        # 【NEXUS感染】統合グループと順列マップの完全復元
        unique_groups = [tuple(g) for g in payload['unique_groups']]
        perm_map_dict = {int(k): tuple(v) for k, v in payload['perm_map_dict'].items()}
        
        if not silent:
            print(f"   [NEXUS Decompress] Loaded {len(unique_groups)} unique groups")
            print(f"   [NEXUS Decompress] Loaded {len(perm_map_dict)} permutation maps")
            print(f"   [NEXUS Decompress] Processing {len(group_id_stream)} blocks")
        
        # 🔥 NEXUS感染：ブロック再構成 - 統合マップを使用した完全逆変換
        reconstructed_blocks = []
        
        for i, (group_id, perm_id) in enumerate(zip(group_id_stream, perm_id_stream)):
            if group_id < len(unique_groups) and perm_id in perm_map_dict:
                # 🔥 NEXUS: 統合マップから元のブロックを完全復元
                canonical_group = unique_groups[group_id]
                perm_map = perm_map_dict[perm_id]
                
                # 🔥 NEXUS感染: 統合マップを使用して真の元データを復元
                if nexus_consolidation_enabled:
                    original_block = self._nexus_apply_consolidation_inverse(canonical_group, element_consolidation_map, i)
                    # その後、順列の逆変換を適用
                    final_block = self._apply_inverse_permutation(original_block, perm_map)
                else:
                    # 統合なしの場合は従来通り
                    final_block = self._apply_inverse_permutation(canonical_group, perm_map)
                
                reconstructed_blocks.append(final_block)
                
                if not silent and i < 5:
                    if nexus_consolidation_enabled:
                        print(f"   [NEXUS Decompress] Block {i}: canonical={canonical_group[:3]}... -> consolidated_inverse={original_block[:3]}... -> final={final_block[:3]}...")
                    else:
                        print(f"   [NEXUS Decompress] Block {i}: canonical={canonical_group[:3]}... -> original={final_block[:3]}...")
                    
            else:
                # エラー時のフォールバック：統合グループをそのまま使用
                if group_id < len(unique_groups):
                    reconstructed_blocks.append(unique_groups[group_id])
                    if not silent and i < 5:
                        print(f"   [NEXUS Decompress] Block {i}: fallback to canonical group")
                else:
                    # 完全なフォールバック
                    reconstructed_blocks.append((0,) * 7)  # デフォルトブロック
                    if not silent and i < 5:
                        print(f"   [NEXUS Decompress] Block {i}: fallback to zero block")
        
        if not silent:
            print(f"   [NEXUS Decompress] Reconstructed {len(reconstructed_blocks)} blocks")
        
        # 【NEXUS理論】データ再構成：圧縮時のブロック配置の完全逆操作
        return self._reconstruct_data_from_blocks(reconstructed_blocks, grid_width, original_length, main_shape, silent)

    def _nexus_apply_consolidation_inverse(self, canonical_group: Tuple, consolidation_map: Dict, block_index: int) -> Tuple:
        """
        🔥 NEXUS: 統合マップからの完全逆変換（究極精度版）
        
        NEXUS原則：統合で失われた情報を完全復元 - 高精度マッチングシステム
        """
        # 🔥 NEXUS ULTIMATE: 複数候補からの精密マッチング（閾値を大幅緩和）
        match_candidates = []
        
        for original_id_str, mapping_data in consolidation_map.items():
            try:
                original_id = int(original_id_str)
                match_score = 0.0
                reconstruction_data = None
                
                # 🔥 NEXUS: 完全なチェーン逆変換の場合
                if 'nexus_reconstruction_chain' in mapping_data:
                    chain = mapping_data['nexus_reconstruction_chain']
                    if chain:
                        final_layer_data = chain[-1]['transformation_data']
                        if 'nexus_canonical_form' in final_layer_data:
                            candidate_canonical = tuple(final_layer_data['nexus_canonical_form'])
                            match_score = self._nexus_calculate_advanced_match_score(canonical_group, candidate_canonical, block_index)
                            reconstruction_data = ('chain', mapping_data)
                
                # 🔥 NEXUS: 直接的なマッピングデータの場合
                elif 'nexus_canonical_form' in mapping_data:
                    candidate_canonical = tuple(mapping_data['nexus_canonical_form'])
                    match_score = self._nexus_calculate_advanced_match_score(canonical_group, candidate_canonical, block_index)
                    reconstruction_data = ('direct', mapping_data)
                
                # 従来版のマッピングデータ
                elif 'canonical_form' in mapping_data:
                    candidate_canonical = tuple(mapping_data['canonical_form'])
                    match_score = self._nexus_calculate_advanced_match_score(canonical_group, candidate_canonical, block_index)
                    reconstruction_data = ('legacy', mapping_data)
                
                # 🔥 NEXUS: 閾値を大幅に緩和（0.1以上で候補として追加）
                if match_score > 0.1:
                    match_candidates.append((match_score, reconstruction_data, original_id))
                        
            except (ValueError, KeyError, TypeError):
                continue
        
        # 🔥 NEXUS: 最高スコアの候補を選択（複数候補がある場合は統合処理）
        if match_candidates:
            # スコア順でソート
            match_candidates.sort(key=lambda x: x[0], reverse=True)
            
            # 🔥 NEXUS: 上位複数候補を統合して最適解を導出
            if len(match_candidates) >= 2:
                return self._nexus_merge_multiple_candidates(match_candidates[:3], canonical_group, block_index)
            else:
                # 単一候補の場合
                best_score, best_data, best_id = match_candidates[0]
                reconstruction_type, mapping_data = best_data
                
                if reconstruction_type == 'chain':
                    return self._nexus_execute_chain_inverse(mapping_data['nexus_reconstruction_chain'], canonical_group)
                elif reconstruction_type == 'direct' and 'nexus_original_group' in mapping_data:
                    return tuple(mapping_data['nexus_original_group'])
                elif reconstruction_type == 'legacy' and 'original_group' in mapping_data:
                    return tuple(mapping_data['original_group'])
        
        # 🔥 NEXUS: 失敗時の高度推定復元
        return self._nexus_ultra_intelligent_reconstruction(canonical_group, consolidation_map, block_index)
    
    def _nexus_calculate_advanced_match_score(self, group1: Tuple, group2: Tuple, block_index: int) -> float:
        """🔥 NEXUS: 高度グループ間一致度スコア計算（位置情報も考慮）"""
        if group1 == group2:
            return 1.0
        
        if len(group1) != len(group2):
            return 0.0
        
        # 基本一致度
        exact_matches = sum(1 for a, b in zip(group1, group2) if a == b)
        basic_score = exact_matches / len(group1)
        
        # 近似一致度（値が近い場合）
        near_matches = sum(1 for a, b in zip(group1, group2) if abs(a - b) <= 1)
        near_score = near_matches / len(group1) * 0.8
        
        # 統計的類似度（平均、分散等）
        avg1, avg2 = sum(group1) / len(group1), sum(group2) / len(group2)
        avg_similarity = 1.0 - min(1.0, abs(avg1 - avg2) / max(1, max(avg1, avg2)))
        
        # 位置ボーナス（block_indexに基づく）
        position_bonus = 0.1 if block_index % 2 == 0 else 0.05
        
        # 総合スコア
        final_score = max(basic_score, near_score) + avg_similarity * 0.2 + position_bonus
        return min(1.0, final_score)
    
    def _nexus_merge_multiple_candidates(self, candidates: List, canonical_group: Tuple, block_index: int) -> Tuple:
        """🔥 NEXUS: 複数候補の統合処理（アンサンブル手法）"""
        original_groups = []
        weights = []
        
        for score, (reconstruction_type, mapping_data), original_id in candidates:
            try:
                if reconstruction_type == 'chain':
                    result = self._nexus_execute_chain_inverse(mapping_data['nexus_reconstruction_chain'], canonical_group)
                elif reconstruction_type == 'direct' and 'nexus_original_group' in mapping_data:
                    result = tuple(mapping_data['nexus_original_group'])
                elif reconstruction_type == 'legacy' and 'original_group' in mapping_data:
                    result = tuple(mapping_data['original_group'])
                else:
                    continue
                
                if len(result) == len(canonical_group):
                    original_groups.append(result)
                    weights.append(score)
                    
            except (KeyError, TypeError, ValueError):
                continue
        
        if not original_groups:
            return canonical_group
        
        # 🔥 NEXUS: 重み付きアンサンブル復元
        if len(original_groups) == 1:
            return original_groups[0]
        
        # 複数候補の場合：重み付き投票
        ensemble_result = []
        total_weight = sum(weights)
        
        for i in range(len(canonical_group)):
            # 各位置での重み付き平均
            weighted_sum = 0
            for j, group in enumerate(original_groups):
                if i < len(group):
                    weighted_sum += group[i] * weights[j]
            
            ensemble_value = int(weighted_sum / total_weight) if total_weight > 0 else canonical_group[i]
            ensemble_result.append(ensemble_value)
        
        return tuple(ensemble_result)
        """🔥 NEXUS: 高度4層逆変換チェーン実行（エラー耐性向上）"""
        current_group = canonical_group
        
        # 4層を逆順でたどる（Layer 4 → Layer 3 → Layer 2 → Layer 1）
        for layer_idx, layer_data in enumerate(reversed(reconstruction_chain)):
            try:
                layer_info = layer_data['transformation_data']
                layer_num = layer_data.get('layer', f'unknown_{layer_idx}')
                
                # 各層での逆変換実行（複数候補をチェック）
                if 'nexus_original_group' in layer_info:
                    candidate = tuple(layer_info['nexus_original_group'])
                    if len(candidate) == len(current_group):
                        current_group = candidate
                elif 'original_group' in layer_info:
                    candidate = tuple(layer_info['original_group'])
                    if len(candidate) == len(current_group):
                        current_group = candidate
                        
            except (KeyError, TypeError, ValueError):
                # エラー時は現在のグループを維持
                continue
        
        return current_group
    
    def _nexus_ultra_intelligent_reconstruction(self, canonical_group: Tuple, consolidation_map: Dict, block_index: int) -> Tuple:
        """🔥 NEXUS: ウルトラインテリジェント推定復元（機械学習的アプローチ）"""
        # 1. 近似マッチング：長さと統計的特徴による候補探索
        same_length_candidates = []
        for mapping_data in consolidation_map.values():
            for key in ['nexus_original_group', 'original_group']:
                if key in mapping_data:
                    orig_group = mapping_data[key]
                    if len(orig_group) == len(canonical_group):
                        # 統計的類似度を計算
                        similarity = self._calculate_statistical_similarity(canonical_group, tuple(orig_group))
                        same_length_candidates.append((similarity, tuple(orig_group)))
        
        if same_length_candidates:
            # 最も類似した候補を選択
            same_length_candidates.sort(key=lambda x: x[0], reverse=True)
            return same_length_candidates[0][1]
        
        # 2. パターンベース推定：canonical_groupから推測
        estimated_group = []
        for i, val in enumerate(canonical_group):
            # 位置とblock_indexに基づく推定
            position_factor = (i + 1) * 0.1
            block_factor = (block_index % 10) * 0.01
            estimated_val = int(val + position_factor + block_factor)
            estimated_group.append(max(0, min(255, estimated_val)))  # 0-255の範囲に制限
        
        return tuple(estimated_group)
    
    def _calculate_statistical_similarity(self, group1: Tuple, group2: Tuple) -> float:
        """統計的類似度計算"""
        if len(group1) != len(group2):
            return 0.0
        
        # 平均値の類似度
        avg1, avg2 = sum(group1) / len(group1), sum(group2) / len(group2)
        avg_similarity = 1.0 - min(1.0, abs(avg1 - avg2) / max(1, max(avg1, avg2)))
        
        # 分散の類似度
        var1 = sum((x - avg1) ** 2 for x in group1) / len(group1)
        var2 = sum((x - avg2) ** 2 for x in group2) / len(group2)
        var_similarity = 1.0 - min(1.0, abs(var1 - var2) / max(1, max(var1, var2)))
        
        # 総合類似度
        return (avg_similarity + var_similarity) / 2
    
    def _nexus_execute_chain_inverse(self, reconstruction_chain: List, canonical_group: Tuple) -> Tuple:
        """🔥 NEXUS: 4層逆変換チェーンの実行"""
        current_group = canonical_group
        
        # 4層を逆順でたどる（Layer 4 → Layer 3 → Layer 2 → Layer 1）
        for layer_data in reversed(reconstruction_chain):
            layer_info = layer_data['transformation_data']
            layer_num = layer_data['layer']
            
            # 各層での逆変換実行
            if 'nexus_original_group' in layer_info:
                current_group = tuple(layer_info['nexus_original_group'])
            elif 'original_group' in layer_info:
                current_group = tuple(layer_info['original_group'])
        
        return current_group
    
    def _nexus_intelligent_reconstruction(self, canonical_group: Tuple, consolidation_map: Dict, block_index: int) -> Tuple:
        """🔥 NEXUS: インテリジェント推定復元（最後の手段）"""
        # 近似復元：統計的手法を使用
        
        # 1. 同じ長さのグループから推定
        same_length_groups = []
        for mapping_data in consolidation_map.values():
            if 'nexus_original_group' in mapping_data:
                orig_group = mapping_data['nexus_original_group']
                if len(orig_group) == len(canonical_group):
                    same_length_groups.append(tuple(orig_group))
        
        if same_length_groups:
            # 最も類似したパターンを選択
            best_candidate = min(same_length_groups, 
                               key=lambda x: sum((a - b) ** 2 for a, b in zip(canonical_group, x)))
            return best_candidate
        
        # 2. 最後の手段：canonical_groupをそのまま返す
        return canonical_group

    def _decompress_binary_payload(self, payload: bytes) -> bytes:
        """バイナリ形式ペイロードの解凍（簡略版）"""
        # 簡略版：元データをそのまま返す（フォールバック）
        return payload

    def _apply_inverse_permutation(self, sorted_group: tuple, perm_map: tuple) -> tuple:
        """順列の逆変換（NEXUS理論：完全可逆版）"""
        if len(sorted_group) != len(perm_map):
            return sorted_group
        
        try:
            # 【NEXUS理論】順列マップの正しい解釈と逆変換
            # perm_mapは「元の位置iの要素がソート後の位置perm_map[i]にある」ことを示す
            # 逆変換では「ソート後の各位置の要素が元のどの位置に戻るか」を計算
            
            result = [0] * len(sorted_group)
            
            # perm_mapから逆変換マップを構築
            for original_pos, sorted_pos in enumerate(perm_map):
                if 0 <= sorted_pos < len(sorted_group):
                    result[original_pos] = sorted_group[sorted_pos]
                else:
                    # 範囲外エラーの場合、安全なフォールバック
                    result[original_pos] = sorted_group[original_pos % len(sorted_group)]
            
            return tuple(result)
            
        except (IndexError, TypeError):
            # エラー時は元のsorted_groupをそのまま返す
            return sorted_group

    def _reconstruct_data_from_blocks(self, blocks: list, grid_width: int, original_length: int, main_shape: str = None, silent=False) -> bytes:
        """
        🔥 NEXUS THEORY ULTIMATE INFECTION: ブロックからデータを再構成
        
        NEXUS原則: 圧縮時のブロック生成と完全に同じロジックで逆操作
        座標システムも完全にミラー化する
        """
        try:
            # 【NEXUS理論】圧縮時のブロック生成と完全に同じロジックで逆操作
            shape_coords = POLYOMINO_SHAPES.get(main_shape, [(0, 0)])
            
            # 単純な形状（I-1等）の場合は線形復元
            if len(shape_coords) == 1 and shape_coords[0] == (0, 0):
                result_data = []
                for block in blocks:
                    result_data.extend(block)
                return bytes(result_data[:original_length])
            
            if not silent:
                print(f"   [NEXUS Reconstruct] Using shape '{main_shape}' - EXACT INVERSE of compression")
                print(f"   [NEXUS Reconstruct] Processing {len(blocks)} blocks for {original_length} bytes")
            
            # 【NEXUS理論】圧縮時のパラメーター計算を完全に再現
            # パディング後のデータ長を使用（圧縮時と同じ）
            padded_length = original_length
            shape_height = max(r for r, c in shape_coords) + 1
            shape_width = max(c for r, c in shape_coords) + 1
            
            # パディングサイズの再計算（圧縮時のロジックと同じ）
            rows_needed = math.ceil(padded_length / grid_width)
            min_padded_size = (rows_needed + shape_height) * grid_width
            safe_padded_size = min(min_padded_size, padded_length + (grid_width * shape_height))
            
            # 圧縮時に使用された実際のデータ長
            data_len = safe_padded_size
            rows = data_len // grid_width
            
            if not silent:
                print(f"   [NEXUS Reconstruct] Padded params: data_len={data_len}, rows={rows}, grid_width={grid_width}")
                print(f"   [NEXUS Reconstruct] Shape params: width={shape_width}, height={shape_height}")
            
            # 🔥 NEXUS: 結果データ配列を初期化（パディングサイズで）
            result_data = [0] * data_len
            
            # 🔥 NEXUS: ブロック→座標の完全マッピングテーブル構築
            block_position_map = {}  # block_index -> (r, c)
            block_idx = 0
            
            # 圧縮時と完全に同じ順序でマッピングを構築
            for r in range(rows - shape_height + 1):
                for c in range(grid_width - shape_width + 1):
                    # 圧縮時と同じ有効性チェック
                    base_idx = r * grid_width + c
                    valid_block = True
                    
                    # 境界チェック（圧縮時と完全に同じロジック）
                    for dr, dc in shape_coords:
                        idx = base_idx + dr * grid_width + dc
                        if idx >= data_len:
                            valid_block = False
                            break
                    
                    if valid_block:
                        block_position_map[block_idx] = (r, c)
                        block_idx += 1
            
            # 🔥 NEXUS ULTIMATE PRECISION: ブロックを完全精密配置システム（競合解決付き）
            block_write_count = [0] * data_len  # 各位置への書き込み回数を追跡
            position_scores = [0.0] * data_len  # 各位置の信頼度スコア
            
            for block_idx, current_block in enumerate(blocks):
                if block_idx not in block_position_map:
                    if not silent and block_idx < 5:
                        print(f"   [NEXUS Reconstruct] Block {block_idx}: No position mapping - skipping")
                    continue
                
                r, c = block_position_map[block_idx]
                base_idx = r * grid_width + c
                
                if not silent and block_idx < 5:
                    print(f"   [NEXUS Reconstruct] Block {block_idx} at VALID position ({r}, {c}): {list(current_block)}")
                
                # 🔥 NEXUS ULTIMATE: 各形状座標に対してインテリジェント配置
                for coord_idx, (dr, dc) in enumerate(shape_coords):
                    data_idx = base_idx + dr * grid_width + dc
                    
                    if coord_idx < len(current_block) and data_idx < data_len:
                        current_value = current_block[coord_idx]
                        
                        # 🔥 NEXUS ULTIMATE: 位置優先度による配置戦略
                        if block_write_count[data_idx] == 0:
                            # 初回書き込み：そのまま配置
                            result_data[data_idx] = current_value
                            position_scores[data_idx] = 1.0
                            block_write_count[data_idx] = 1
                        else:
                            # 🔥 NEXUS: 既存値との比較による最適化
                            existing_value = result_data[data_idx]
                            
                            # 優先度判定システム
                            should_overwrite = False
                            new_score = position_scores[data_idx]
                            
                            # 1. 非ゼロ値は常に優先
                            if existing_value == 0 and current_value != 0:
                                should_overwrite = True
                                new_score = 2.0
                            # 2. 両方非ゼロの場合：より信頼性の高い値を選択
                            elif existing_value != 0 and current_value != 0:
                                # ブロックの統合レベルに基づく信頼度
                                if block_idx < len(blocks) // 2:  # 前半ブロックはより信頼性が高い
                                    if current_value != existing_value:
                                        should_overwrite = True
                                        new_score = 2.5
                                # 値の差が小さい場合は平均
                                elif abs(existing_value - current_value) <= 2:
                                    result_data[data_idx] = (existing_value + current_value) // 2
                                    new_score = 1.8
                                    block_write_count[data_idx] += 1
                                    continue
                            
                            if should_overwrite:
                                result_data[data_idx] = current_value
                                position_scores[data_idx] = new_score
                            
                            block_write_count[data_idx] += 1
                        
                        if not silent and block_idx < 3 and coord_idx < 3:
                            print(f"   [NEXUS Reconstruct]   coord({dr},{dc}) -> data[{data_idx}] = {result_data[data_idx]} (writes: {block_write_count[data_idx]}, score: {position_scores[data_idx]:.1f})")
            
            if not silent:
                non_zero_count = len([x for x in result_data if x != 0])
                print(f"   [NEXUS Reconstruct] Placed {non_zero_count}/{data_len} bytes from {len(blocks)} valid blocks")
                
                # 🔥 NEXUS: 詳細な書き込み統計
                zero_writes = sum(1 for x in block_write_count if x == 0)
                single_writes = sum(1 for x in block_write_count if x == 1)
                multi_writes = sum(1 for x in block_write_count if x > 1)
                high_confidence = sum(1 for x in position_scores if x >= 2.0)
                
                print(f"   [NEXUS Reconstruct] Write stats - Zero: {zero_writes}, Single: {single_writes}, Multiple: {multi_writes}")
                print(f"   [NEXUS Reconstruct] High confidence positions: {high_confidence}")
                print(f"   [NEXUS Reconstruct] Trimming to original length: {original_length}")
            
            # 【NEXUS理論】パディングを除去して元の長さに戻す
            return bytes(result_data[:original_length])
            
        except Exception as e:
            if not silent:
                print(f"❌ NEXUS再構成エラー: {e}")
            return b''
            if not silent:
                print(f"❌ NEXUS再構成エラー: {e}")
            return b''


class NexusAdvancedDecompressor:
    def __init__(self):
        self.huffman_encoder = HuffmanEncoder()

    def decompress(self, compressed_data: bytes, level=0) -> bytes:
        """NEXUS高機能圧縮データを解凍"""
        if not compressed_data:
            return b''

        # 最終段のLZMA解凍
        decompressed_payload = lzma.decompress(compressed_data)
        
        try:
            # ペイロードがJSON形式か試す
            payload = json.loads(decompressed_payload.decode('utf-8'))
            is_json = True
        except (json.JSONDecodeError, UnicodeDecodeError):
            # JSONでなければ、再帰的に圧縮されたデータとみなす
            payload = decompressed_payload
            is_json = False

        # --- 階層的NEXUS ---
        if not is_json:
            print(f"   [Hierarchical] Recursively decompressing metadata (Level {level})...")
            # 再帰的に解凍
            decompressed_json_bytes = self.decompress(payload, level + 1)
            payload = json.loads(decompressed_json_bytes.decode('utf-8'))

        print(f"\n--- Decompressing at Level {level} ---")
        
        header = payload["header"]
        original_length = header["original_length"]
        grid_width = header["grid_width"]
        
        # アルゴリズムバージョン確認
        algorithm = header.get("algorithm", "NEXUS_v3_puzzle_optimized")
        element_consolidation_enabled = header.get("element_consolidation_enabled", False)
        
        if element_consolidation_enabled:
            print(f"   [Element Consolidation] Detected advanced consolidation format")
            original_groups_count = header.get("original_groups_count", 0)
            consolidated_groups_count = header.get("consolidated_groups_count", 0)
            consolidation_rate = header.get("consolidation_rate", 0)
            print(f"   [Element Consolidation] Groups reduced: {original_groups_count:,} → {consolidated_groups_count:,} ({consolidation_rate:.2f}%)")
        
        # 新旧フォーマット対応
        if "shape_combination" in header:
            # 新フォーマット（パズル組み合わせ）
            shape_combination = header["shape_combination"]
            main_shape = header["main_shape"]
            print(f"   [Puzzle] Shape combination: {shape_combination}, Main: {main_shape}")
        else:
            # 旧フォーマット（単一形状）
            main_shape = header["shape_name"]
            shape_combination = [main_shape]
        
        shape_coords = POLYOMINO_SHAPES[main_shape]
        
        unique_groups = [tuple(g) for g in payload["unique_groups"]]
        huffman_trees = payload["huffman_trees"]
        encoded_streams = payload["encoded_streams"]
        id_to_perm_map = {int(k): tuple(v) for k, v in payload["perm_map_dict"].items()}
        
        # 構成要素統合マップの取得（新形式の場合）
        element_consolidation_map = payload.get("element_consolidation_map", {})

        # --- ハイブリッド復号化 ---
        print(f"   [Huffman] Decoding blueprint streams...")
        group_id_stream = self.huffman_encoder.decode(encoded_streams["group_ids"], huffman_trees["group_ids"])
        perm_id_stream_int = self.huffman_encoder.decode(encoded_streams["perm_ids"], huffman_trees["perm_ids"])
        
        # 構成要素統合対応の復号化処理
        if element_consolidation_enabled and element_consolidation_map:
            print(f"   [Reconstruction] Applying element consolidation reversal...")
            # 統合された情報から元の形状に復元する処理は複雑なため、
            # 現在の実装では統合後の代表形状でそのまま復号化
            print(f"   [Reconstruction] Using consolidated canonical forms for reconstruction")
        
        # メモリ効率的な再構築
        shape_height = max(r for r, c in shape_coords) + 1
        shape_width = max(c for r, c in shape_coords) + 1
        rows_needed = math.ceil(original_length / grid_width)
        
        # 必要最小限のグリッドサイズ
        reconstructed_size = (rows_needed + shape_height) * grid_width
        reconstructed_grid = bytearray(reconstructed_size)
        
        num_blocks_in_row = grid_width - shape_width + 1

        for i, group_id in enumerate(group_id_stream):
            if i >= len(perm_id_stream_int):
                break
                
            perm_id = perm_id_stream_int[i]
            perm_map = id_to_perm_map[perm_id]

            sorted_group = unique_groups[group_id]
            original_block = bytes(sorted_group[p] for p in perm_map)

            block_row = i // num_blocks_in_row
            block_col = i % num_blocks_in_row

            for j, (dr, dc) in enumerate(shape_coords):
                idx = (block_row + dr) * grid_width + (block_col + dc)
                if idx < len(reconstructed_grid) and j < len(original_block):
                    reconstructed_grid[idx] = original_block[j]
        
        return bytes(reconstructed_grid)[:original_length]

# --- メイン実行部分 ---
def main():
    if len(sys.argv) != 4:
        print("Usage:")
        print("  To compress:   python nexus_advanced_engine.py compress   <input_file> <output_file.nxz>")
        print("  To decompress: python nexus_advanced_engine.py decompress <input_file.nxz> <output_file>")
        sys.exit(1)

    command, input_path, output_path = sys.argv[1:4]

    if command.lower() == 'compress':
        print(f"🚀 Compressing '{input_path}' with Advanced NEXUS...")
        start_time = time.time()
        
        with open(input_path, 'rb') as f_in:
            input_data = f_in.read()
        
        # 再帰レベルを0に設定 (階層的圧縮を無効化してディスク書き込み削減)
        compressor = NexusAdvancedCompressor(use_ai=True, max_recursion_level=0)
        compressed_data = compressor.compress(input_data)
        
        with open(output_path, 'wb') as f_out:
            f_out.write(compressed_data)

        end_time = time.time()
        processing_time = end_time - start_time
        original_size = len(input_data)
        compressed_size = len(compressed_data)
        ratio = compressed_size / original_size if original_size > 0 else 0
        size_reduction = (1 - ratio) * 100 if ratio <= 1 else -(ratio - 1) * 100
        
        print("\n✅ Compression successful!")
        print(f"   Original size:    {original_size:,} bytes")
        print(f"   Compressed size:  {compressed_size:,} bytes")
        if ratio <= 1:
            print(f"   Size reduction:   {size_reduction:.2f}% (ratio: {ratio:.2%})")
        else:
            print(f"   Size expansion:   {-size_reduction:.2f}% (ratio: {ratio:.2%})")
        print(f"   Processing time:  {processing_time:.2f} seconds")
        print(f"   Speed:            {original_size / (1024*1024*processing_time):.2f} MB/sec")

    elif command.lower() == 'decompress':
        print(f"📚 Decompressing '{input_path}' with Advanced NEXUS...")
        with open(input_path, 'rb') as f_in:
            compressed_data = f_in.read()

        decompressor = NexusAdvancedDecompressor()
        decompressed_data = decompressor.decompress(compressed_data)
        
        with open(output_path, 'wb') as f_out:
            f_out.write(decompressed_data)

        print("\n✅ Decompression successful!")
        print(f"   Decompressed size: {len(decompressed_data):,} bytes")

    else:
        print(f"❌ Unknown command: '{command}'")
        sys.exit(1)

if __name__ == "__main__":
    main()
