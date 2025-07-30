#!/usr/bin/env python3
"""
NEXUS理論完全実装エンジン v6.1 改良版
v6.1の良好な性能を基盤として、具体的な問題点のみを修正

修正点:
1. データ膨張問題の解決（圧縮率マイナス対策）
2. 速度向上のための軽量化
3. フォールバック機能の強化
"""

import numpy as np
import os
import hashlib
import time
import threading
import queue
import lzma
import zlib
import bz2
from typing import Dict, List, Tuple, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from dataclasses import dataclass
from enum import Enum


class CompressionStrategy(Enum):
    """圧縮戦略"""
    ULTRA_VISUAL = "ultra_visual"          # 画像・動画超最適化
    DEEP_PATTERN = "deep_pattern"          # 深層パターン解析
    QUANTUM_ENTROPY = "quantum_entropy"    # 量子エントロピー最適化
    MEGA_REDUNDANCY = "mega_redundancy"    # 超冗長性除去
    ADAPTIVE_FUSION = "adaptive_fusion"    # 適応的融合圧縮


@dataclass
class FastAnalysisResult:
    """高速解析結果"""
    entropy_score: float
    pattern_coherence: float
    compression_potential: float
    optimal_strategy: CompressionStrategy
    visual_features: Dict[str, float]


class FastPatternAnalyzer:
    """高速パターン解析器 - v6.1ベース軽量化版"""
    
    def __init__(self):
        self.golden_ratio = 1.618033988749895
        
    def analyze_fast(self, data: bytes) -> FastAnalysisResult:
        """高速解析実行 - 軽量化"""
        try:
            if len(data) == 0:
                return self._create_default_result()
            
            # より軽量なサンプリング
            sample_size = min(len(data), 32 * 1024)  # 32KB上限に削減
            if len(data) > sample_size:
                # 3箇所からサンプリング（軽量化）
                step = len(data) // 3
                samples = []
                for i in range(0, len(data), step):
                    end = min(i + sample_size // 3, len(data))
                    samples.append(data[i:end])
                sample_data = b''.join(samples)[:sample_size]
            else:
                sample_data = data
            
            data_array = np.frombuffer(sample_data, dtype=np.uint8)
            
            # 軽量化されたエントロピー計算
            entropy = self._fast_entropy(data_array)
            
            # 軽量化パターン検出
            coherence = self._fast_coherence(data_array)
            
            # 圧縮ポテンシャル推定
            potential = self._estimate_potential(entropy, coherence)
            
            # ビジュアル特徴軽量検出
            visual_features = self._fast_visual_analysis(data_array)
            
            # 戦略決定（改良版）
            strategy = self._improved_strategy_selection(potential, coherence, visual_features, len(data))
            
            return FastAnalysisResult(
                entropy_score=entropy,
                pattern_coherence=coherence,
                compression_potential=potential,
                optimal_strategy=strategy,
                visual_features=visual_features
            )
            
        except Exception as e:
            print(f"高速解析エラー: {e}")
            return self._create_default_result()
    
    def _fast_entropy(self, data: np.ndarray) -> float:
        """軽量化エントロピー計算"""
        if len(data) < 16:
            return 0.5
        
        # ヒストグラム計算（変更なし - v6.1で良好）
        hist = np.bincount(data, minlength=256)
        prob = hist / len(data)
        prob = prob[prob > 0]
        
        # シャノンエントロピー
        entropy = -np.sum(prob * np.log2(prob))
        return entropy / 8.0  # 正規化
    
    def _fast_coherence(self, data: np.ndarray) -> float:
        """軽量化コヒーレンス計算"""
        if len(data) < 32:
            return 0.5
        
        try:
            # さらに軽量化した自己相関（32点上限）
            max_lag = min(32, len(data) // 8)  # 1/8に削減
            if max_lag < 2:
                return 0.5
            
            sample = data[:max_lag * 8]
            autocorr = np.correlate(sample.astype(float), sample.astype(float), mode='full')
            center = len(autocorr) // 2
            autocorr = autocorr[center:center + max_lag]
            
            if len(autocorr) > 1 and autocorr[0] != 0:
                autocorr = autocorr / autocorr[0]
                coherence = np.mean(np.abs(autocorr[1:]))
            else:
                coherence = 0.5
            
            return np.clip(coherence, 0.0, 1.0)
            
        except Exception:
            return 0.5
    
    def _estimate_potential(self, entropy: float, coherence: float) -> float:
        """圧縮ポテンシャル推定 - v6.1ベース"""
        # 理論最大圧縮率
        theoretical_max = 1.0 - entropy
        
        # コヒーレンス補正
        coherence_bonus = coherence * 0.3
        
        # NEXUS理論補正
        nexus_multiplier = 1.0 + coherence * 0.1
        
        potential = (theoretical_max + coherence_bonus) * nexus_multiplier
        return np.clip(potential, 0.0, 0.99)
    
    def _fast_visual_analysis(self, data: np.ndarray) -> Dict[str, float]:
        """軽量化ビジュアル特徴解析"""
        if len(data) < 64:
            return {'gradient': 0.0, 'repetition': 0.0, 'texture': 0.0}
        
        # より軽量なサンプリング
        sample_size = min(512, len(data))  # 512に削減
        sample = data[:sample_size]
        
        features = {}
        
        # グラデーション検出（変更なし - v6.1で良好）
        diff = np.abs(np.diff(sample.astype(int)))
        features['gradient'] = np.sum(diff <= 2) / len(diff) if len(diff) > 0 else 0.0
        
        # 反復検出（変更なし - v6.1で良好）
        pattern_len = min(16, len(sample) // 4)
        if pattern_len > 0:
            pattern = sample[:pattern_len]
            matches = 0
            checks = min(4, len(sample) // pattern_len)
            for i in range(checks):
                start = i * pattern_len
                end = start + pattern_len
                if end <= len(sample) and np.array_equal(pattern, sample[start:end]):
                    matches += 1
            features['repetition'] = matches / checks if checks > 0 else 0.0
        else:
            features['repetition'] = 0.0
        
        # テクスチャ（軽量化）
        if len(sample) >= 32:  # 閾値を下げる
            chunks = [sample[i:i+8] for i in range(0, len(sample)-8, 8)][:4]  # チャンクサイズ削減
            if len(chunks) > 1:
                variances = [np.var(chunk) for chunk in chunks]
                features['texture'] = 1.0 - (np.std(variances) / (np.mean(variances) + 1e-6))
            else:
                features['texture'] = 0.0
        else:
            features['texture'] = 0.0
        
        return features
    
    def _improved_strategy_selection(self, potential: float, coherence: float, 
                                   visual_features: Dict[str, float], data_size: int) -> CompressionStrategy:
        """改良された戦略選択 - データ膨張対策"""
        
        # 小さなファイル（1KB未満）は軽量戦略のみ
        if data_size < 1024:
            return CompressionStrategy.ADAPTIVE_FUSION
        
        # 大きなファイル（10MB以上）は高速戦略優先
        if data_size > 10 * 1024 * 1024:
            if visual_features.get('repetition', 0) > 0.7:
                return CompressionStrategy.MEGA_REDUNDANCY
            else:
                return CompressionStrategy.ADAPTIVE_FUSION
        
        # 中サイズファイルの戦略選択（v6.1ベース）
        # ビジュアル特徴が強い場合
        if (visual_features.get('gradient', 0) > 0.6 or 
            visual_features.get('repetition', 0) > 0.4):
            return CompressionStrategy.ULTRA_VISUAL
        
        # 高い圧縮ポテンシャル
        if potential > 0.8 and coherence > 0.7:
            return CompressionStrategy.MEGA_REDUNDANCY
        
        # 中程度のパターン
        if potential > 0.6 and coherence > 0.5:
            return CompressionStrategy.DEEP_PATTERN
        
        # エントロピー最適化
        if coherence > 0.6:
            return CompressionStrategy.QUANTUM_ENTROPY
        
        # デフォルト（最も安全）
        return CompressionStrategy.ADAPTIVE_FUSION
    
    def _create_default_result(self) -> FastAnalysisResult:
        """デフォルト結果作成"""
        return FastAnalysisResult(
            entropy_score=0.5,
            pattern_coherence=0.5,
            compression_potential=0.5,
            optimal_strategy=CompressionStrategy.ADAPTIVE_FUSION,
            visual_features={'gradient': 0.0, 'repetition': 0.0, 'texture': 0.0}
        )


class ImprovedVisualCompressor:
    """改良版ビジュアル圧縮器 - データ膨張対策強化"""
    
    def compress_visual_optimized(self, data: bytes, analysis: FastAnalysisResult) -> bytes:
        """最適化ビジュアル圧縮 - 膨張対策"""
        if len(data) < 1024:
            return self._safe_compress(data)  # 小ファイルは安全な圧縮
        
        try:
            # ビジュアル前処理（軽量化）
            processed = self._safe_visual_preprocess(data, analysis.visual_features)
            
            # 段階的圧縮（安全性強化）
            candidates = []
            
            # 候補1: 前処理 + LZMA
            try:
                candidate1 = lzma.compress(processed, preset=6, check=lzma.CHECK_NONE)
                if len(candidate1) < len(data) * 0.98:  # 2%以上削減なら候補
                    candidates.append(candidate1)
            except:
                pass
            
            # 候補2: 前処理 + ZLIB
            try:
                candidate2 = zlib.compress(processed, level=6)
                if len(candidate2) < len(data) * 0.98:
                    candidates.append(candidate2)
            except:
                pass
            
            # 候補3: 元データ + LZMA（フォールバック）
            try:
                candidate3 = lzma.compress(data, preset=4)  # 軽量設定
                candidates.append(candidate3)
            except:
                pass
            
            # 最良の結果を選択
            if candidates:
                best = min(candidates, key=len)
                if len(best) < len(data):  # 必ず縮小を確認
                    return best
            
            # 最終フォールバック
            return self._safe_compress(data)
            
        except Exception:
            return self._safe_compress(data)
    
    def _safe_compress(self, data: bytes) -> bytes:
        """安全な圧縮（膨張回避保証）"""
        # 複数の圧縮手法を試して最良の結果を使用
        candidates = []
        
        try:
            lzma_result = lzma.compress(data, preset=1)  # 最軽量
            candidates.append(lzma_result)
        except:
            pass
        
        try:
            zlib_result = zlib.compress(data, level=6)
            candidates.append(zlib_result)
        except:
            pass
        
        try:
            bz2_result = bz2.compress(data, compresslevel=1)  # 最軽量
            candidates.append(bz2_result)
        except:
            pass
        
        if candidates:
            best = min(candidates, key=len)
            # 膨張チェック
            if len(best) < len(data):
                return best
        
        # 最悪の場合は元データを返す（膨張防止）
        return data
    
    def _safe_visual_preprocess(self, data: bytes, features: Dict[str, float]) -> bytes:
        """安全なビジュアル前処理 - 膨張リスク軽減"""
        if len(data) < 64:
            return data
        
        try:
            data_array = np.frombuffer(data, dtype=np.uint8)
            
            # グラデーション最適化（安全版）
            if features.get('gradient', 0) > 0.8:  # 閾値を上げて安全性向上
                # 差分エンコーディング（安全チェック付き）
                diff = np.diff(data_array.astype(int))
                diff_encoded = np.clip(diff + 128, 0, 255).astype(np.uint8)
                processed = np.concatenate([[data_array[0]], diff_encoded])
                
                # 膨張チェック
                if len(processed.tobytes()) <= len(data):
                    return processed.tobytes()
            
            # 反復最適化（安全版）
            if features.get('repetition', 0) > 0.7:  # 閾値を上げる
                rle_result = self._safe_rle(data_array)
                if len(rle_result) < len(data):
                    return rle_result
            
            return data
            
        except Exception:
            return data
    
    def _safe_rle(self, data: np.ndarray) -> bytes:
        """安全なRun-Length Encoding"""
        if len(data) < 8:
            return data.tobytes()
        
        try:
            compressed = []
            i = 0
            
            while i < len(data):
                count = 1
                while (i + count < len(data) and 
                       data[i] == data[i + count] and 
                       count < 127):
                    count += 1
                
                if count >= 6:  # より高い閾値で安全性向上
                    compressed.extend([255, count, data[i]])
                else:
                    compressed.extend(data[i:i+count])
                
                i += count
            
            result = bytes(compressed)
            # 膨張チェック
            if len(result) < len(data):
                return result
            else:
                return data.tobytes()
                
        except Exception:
            return data.tobytes()


class NEXUSUltimateEngineImproved:
    """NEXUS理論完全実装エンジン v6.1 改良版"""
    
    def __init__(self, max_threads: int = None):
        self.max_threads = max_threads or min(mp.cpu_count(), 4)
        self.analyzer = FastPatternAnalyzer()
        self.visual_compressor = ImprovedVisualCompressor()
        
        # 統計
        self.stats = {
            'files_processed': 0,
            'total_input_size': 0,
            'total_output_size': 0,
            'total_time': 0.0,
            'strategy_usage': {strategy.value: 0 for strategy in CompressionStrategy},
            'fallback_usage': 0
        }
    
    def compress_nexus_improved(self, data: bytes, file_type: str = "unknown") -> Tuple[bytes, Dict[str, Any]]:
        """改良版NEXUS圧縮 - v6.1ベース"""
        start_time = time.perf_counter()
        
        if len(data) == 0:
            return data, {'compression_ratio': 0.0, 'strategy': 'none', 'throughput_mb_s': 0.0}
        
        try:
            # 高速解析（v6.1ベース）
            analysis = self.analyzer.analyze_fast(data)
            
            # 戦略別圧縮実行（改良版 - 膨張対策強化）
            if analysis.optimal_strategy == CompressionStrategy.ULTRA_VISUAL:
                compressed = self.visual_compressor.compress_visual_optimized(data, analysis)
            elif analysis.optimal_strategy == CompressionStrategy.DEEP_PATTERN:
                compressed = self._compress_deep_pattern_safe(data, analysis)
            elif analysis.optimal_strategy == CompressionStrategy.QUANTUM_ENTROPY:
                compressed = self._compress_quantum_entropy_safe(data, analysis)
            elif analysis.optimal_strategy == CompressionStrategy.MEGA_REDUNDANCY:
                compressed = self._compress_mega_redundancy_safe(data, analysis)
            else:  # ADAPTIVE_FUSION
                compressed = self._compress_adaptive_fusion_safe(data)
            
            # 必須: 膨張チェック
            if len(compressed) >= len(data):
                compressed = self._guaranteed_safe_compress(data)
                strategy_used = 'safe_fallback'
                self.stats['fallback_usage'] += 1
            else:
                strategy_used = analysis.optimal_strategy.value
            
            # 統計更新
            compression_time = time.perf_counter() - start_time
            self._update_stats(data, compressed, compression_time, analysis.optimal_strategy)
            
            # 結果
            compression_ratio = (1 - len(compressed) / len(data)) * 100 if len(data) > 0 else 0.0
            throughput = (len(data) / 1024 / 1024) / compression_time if compression_time > 0 else 0.0
            
            result_info = {
                'compression_ratio': compression_ratio,
                'throughput_mb_s': throughput,
                'strategy': strategy_used,
                'nexus_analysis': {
                    'entropy_score': analysis.entropy_score,
                    'pattern_coherence': analysis.pattern_coherence,
                    'compression_potential': analysis.compression_potential,
                    'processing_mode': 'improved_safe'
                },
                'input_size': len(data),
                'output_size': len(compressed),
                'compression_time': compression_time
            }
            
            return compressed, result_info
            
        except Exception as e:
            print(f"改良圧縮エラー: {e}")
            fallback = self._guaranteed_safe_compress(data)
            compression_time = time.perf_counter() - start_time
            throughput = (len(data) / 1024 / 1024) / compression_time if compression_time > 0 else 0.0
            
            return fallback, {
                'compression_ratio': (1 - len(fallback) / len(data)) * 100,
                'throughput_mb_s': throughput,
                'strategy': 'emergency_fallback',
                'compression_time': compression_time,
                'error': str(e)
            }
    
    def _compress_deep_pattern_safe(self, data: bytes, analysis: FastAnalysisResult) -> bytes:
        """安全な深層パターン圧縮 - 速度優先版"""
        try:
            # 大きなファイルは軽量処理のみ
            if len(data) > 5 * 1024 * 1024:  # 5MB超は高速処理
                return zlib.compress(data, level=3)
            
            if len(data) > 1024:
                # 差分エンコーディング（軽量版）
                data_array = np.frombuffer(data, dtype=np.uint8)
                diff = np.diff(data_array.astype(int))
                processed = np.concatenate([[data_array[0]], 
                                          np.clip(diff + 128, 0, 255).astype(np.uint8)])
                
                # 軽量圧縮のみ
                try:
                    result = lzma.compress(processed.tobytes(), preset=2)  # preset削減
                    if len(result) < len(data):
                        return result
                except:
                    pass
            
            # フォールバック
            return lzma.compress(data, preset=2)  # 軽量化
                
        except Exception:
            return self._guaranteed_safe_compress(data)
    
    def _compress_quantum_entropy_safe(self, data: bytes, analysis: FastAnalysisResult) -> bytes:
        """安全な量子エントロピー圧縮 - 速度優先版"""
        # 大きなファイルは分割しない（速度重視）
        if len(data) > 10 * 1024 * 1024:  # 10MB超
            return zlib.compress(data, level=1)  # 最高速
        
        if len(data) < 2048:
            return self._guaranteed_safe_compress(data)
        
        try:
            # エントロピーベース単一圧縮選択（分割なし）
            if analysis.entropy_score > 0.7:
                # 高エントロピー: 軽い圧縮
                return zlib.compress(data, level=1)  # 最高速
            else:
                # 低エントロピー: 軽量LZMA
                return lzma.compress(data, preset=1)  # 最軽量
            
        except Exception:
            return self._guaranteed_safe_compress(data)
    
    def _compress_mega_redundancy_safe(self, data: bytes, analysis: FastAnalysisResult) -> bytes:
        """安全な超冗長性除去圧縮"""
        try:
            if len(data) < 512:
                return lzma.compress(data, preset=6)
            
            data_array = np.frombuffer(data, dtype=np.uint8)
            
            # 安全な冗長性除去
            processed = self._safe_redundancy_removal(data_array)
            
            # 膨張チェック付き最終圧縮
            if len(processed.tobytes()) < len(data):
                final_result = lzma.compress(processed.tobytes(), preset=6)
                if len(final_result) < len(data):
                    return final_result
            
            # フォールバック
            return self._guaranteed_safe_compress(data)
            
        except Exception:
            return self._guaranteed_safe_compress(data)
    
    def _compress_adaptive_fusion_safe(self, data: bytes) -> bytes:
        """安全な適応的融合圧縮 - 速度最優先版"""
        # 速度優先の軽量手法選択
        methods = [
            (lambda d: zlib.compress(d, level=3), "zlib3"),  # 最優先: 高速
            (lambda d: lzma.compress(d, preset=1), "lzma1"), # 次点: 軽量LZMA
            (lambda d: bz2.compress(d, compresslevel=1), "bz2-1")  # 最後: 軽量BZ2
        ]
        
        # 大きなファイルは最初の手法のみ（速度重視）
        if len(data) > 5 * 1024 * 1024:  # 5MB超
            try:
                return zlib.compress(data, level=1)  # 最高速
            except:
                return data
        
        # 中サイズファイルは2手法まで
        if len(data) > 1024 * 1024:  # 1MB超
            methods = methods[:2]
        
        best_result = None
        best_size = len(data)
        
        for method, name in methods:
            try:
                result = method(data)
                if len(result) < best_size:
                    best_result = result
                    best_size = len(result)
                    # 十分な圧縮が得られたら即座に終了（速度重視）
                    if best_size < len(data) * 0.9:
                        break
            except Exception:
                continue
        
        if best_result is not None:
            return best_result
        else:
            return data  # 最悪の場合は元データ
    
    def _safe_redundancy_removal(self, data: np.ndarray) -> np.ndarray:
        """安全な冗長性除去"""
        if len(data) < 16:
            return data
        
        try:
            result = []
            i = 0
            
            while i < len(data):
                if i + 4 < len(data) and np.all(data[i:i+4] == data[i]):
                    # 4個以上の連続（安全な閾値）
                    count = 4
                    while (i + count < len(data) and 
                           data[i] == data[i + count] and 
                           count < 255):
                        count += 1
                    result.extend([254, count, data[i]])
                    i += count
                else:
                    result.append(data[i])
                    i += 1
            
            result_array = np.array(result, dtype=np.uint8)
            # 膨張チェック
            if len(result_array) < len(data):
                return result_array
            else:
                return data
                
        except Exception:
            return data
    
    def _guaranteed_safe_compress(self, data: bytes) -> bytes:
        """保証された安全な圧縮（膨張絶対回避） - 高速版"""
        # 速度重視の安全な圧縮手法
        speed_methods = [
            lambda d: zlib.compress(d, level=1),   # 最高速zlib
            lambda d: lzma.compress(d, preset=0),  # 最軽量lzma
        ]
        
        for method in speed_methods:
            try:
                result = method(data)
                if len(result) < len(data):
                    return result
            except Exception:
                continue
        
        # 全て失敗した場合は元データを返す
        return data
    
    def _update_stats(self, input_data: bytes, output_data: bytes, 
                     compression_time: float, strategy: CompressionStrategy):
        """統計更新"""
        self.stats['files_processed'] += 1
        self.stats['total_input_size'] += len(input_data)
        self.stats['total_output_size'] += len(output_data)
        self.stats['total_time'] += compression_time
        self.stats['strategy_usage'][strategy.value] += 1
    
    def get_nexus_stats(self) -> Dict[str, Any]:
        """統計レポート"""
        if self.stats['files_processed'] == 0:
            return {'status': 'no_data'}
        
        total_ratio = (1 - self.stats['total_output_size'] / self.stats['total_input_size']) * 100
        avg_throughput = (self.stats['total_input_size'] / 1024 / 1024) / self.stats['total_time']
        
        return {
            'files_processed': self.stats['files_processed'],
            'total_compression_ratio': total_ratio,
            'average_throughput_mb_s': avg_throughput,
            'total_time': self.stats['total_time'],
            'strategy_distribution': self.stats['strategy_usage'],
            'fallback_usage': self.stats['fallback_usage'],
            'total_input_mb': self.stats['total_input_size'] / 1024 / 1024,
            'total_output_mb': self.stats['total_output_size'] / 1024 / 1024,
            'performance_grade': self._grade_performance(avg_throughput, total_ratio)
        }
    
    def _grade_performance(self, throughput: float, compression: float) -> str:
        """性能グレード評価"""
        if throughput >= 25 and compression >= 30:
            return "EXCELLENT"
        elif throughput >= 15 and compression >= 20:
            return "VERY_GOOD"
        elif throughput >= 10 and compression >= 15:
            return "GOOD"
        elif throughput >= 5 and compression >= 10:
            return "ACCEPTABLE"
        else:
            return "NEEDS_IMPROVEMENT"


if __name__ == "__main__":
    # 改良版テスト
    test_data = b"NEXUS Ultimate Engine Improved Test Data " * 1000
    engine = NEXUSUltimateEngineImproved()
    
    start_time = time.perf_counter()
    compressed, info = engine.compress_nexus_improved(test_data)
    total_time = time.perf_counter() - start_time
    
    print(f"圧縮率: {info['compression_ratio']:.2f}%")
    print(f"戦略: {info['strategy']}")
    print(f"スループット: {info['throughput_mb_s']:.2f}MB/s")
    print(f"処理時間: {total_time:.3f}秒")
    print(f"膨張回避: {'成功' if len(compressed) < len(test_data) else '失敗'}")
