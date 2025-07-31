"""
NEXUS TMC Engine - TDT Transform Module

This module provides advanced TDT (Typed Data Transform) transformation
with statistical clustering-based adaptive stream decomposition.
"""

import numpy as np
from typing import List, Tuple, Dict, Any

__all__ = ['TDTTransformer']


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
            'mean': float(np.mean(byte_stream))
        }
        
        # 範囲計算の安全な実装
        try:
            max_val = np.max(byte_stream)
            min_val = np.min(byte_stream)
            if np.isfinite(max_val) and np.isfinite(min_val):
                features['range'] = float(max_val - min_val)
            else:
                features['range'] = 0.0
        except (OverflowError, ValueError):
            features['range'] = 0.0
        
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
