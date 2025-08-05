"""
NEXUS TMC Engine - Meta Analyzer Module

This module provides intelligent meta-analysis capabilities for determining
optimal compression strategies based on data characteristics and predictive
entropy analysis.
"""

import numpy as np
from typing import Dict, Any, Tuple, List
from .entropy_calculator import (
    calculate_entropy, 
    estimate_temporal_similarity,
    estimate_repetition_density, 
    estimate_context_predictability,
    calculate_theoretical_compression_gain,
    generate_sample_key
)

__all__ = ['MetaAnalyzer']


class MetaAnalyzer:
    """
    TMC v9.0 革新的予測型メタ分析器
    残差エントロピー予測による高速・正確な変換効果判定
    """
    
    def __init__(self, core_compressor, lightweight_mode: bool = False):
        self.core_compressor = core_compressor
        self.lightweight_mode = lightweight_mode
        
        # 改良キャッシュシステム（モード別最適化）
        self.cache = {}  # 分析結果キャッシュ
        if lightweight_mode:
            # 軽量モード: 高速処理優先
            self.cache_max_size = 100  # 小さなキャッシュ
            self.sample_size = 256  # 高速サンプリング
            self.entropy_threshold = 0.95  # 厳しい閾値（変換を減らして高速化）
            print("🔍 予測型MetaAnalyzer初期化完了（軽量モード: 高速処理優先）")
        else:
            # 通常モード: 精度・圧縮率優先
            self.cache_max_size = 1000  # 大きなキャッシュ
            self.sample_size = 1024  # 詳細サンプリング
            self.entropy_threshold = 0.85  # 標準閾値
            print("🔍 予測型MetaAnalyzer初期化完了（通常モード: 高精度分析）")
        
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        
    def should_apply_transform(self, data: bytes, transformer, data_type) -> Tuple[bool, Dict[str, Any]]:
        """
        残差エントロピー予測による高速変換効果分析（モード別最適化）
        Returns: (should_transform, analysis_info)
        """
        print(f"  [予測メタ分析] {data_type if isinstance(data_type, str) else data_type.value} の変換効果を理論予測中...")
        
        if not transformer or len(data) < 512:
            return False, {'reason': 'no_transformer_or_tiny_data'}
        
        # 軽量モード: 超高速判定（分析をスキップ）
        if self.lightweight_mode:
            # 軽量モードでは変換を原則スキップして高速化
            if len(data) < 4096:  # 4KB未満は変換しない
                return False, {'reason': 'lightweight_mode_skip_small', 'entropy_improvement': 0.0}
            
            # 最低限のエントロピーチェックのみ
            basic_entropy = calculate_entropy(data[:256])  # 最初の256バイトのみ
            if basic_entropy > 7.5:  # 高エントロピーなら変換スキップ
                return False, {'reason': 'lightweight_mode_high_entropy', 'entropy_improvement': 0.0}
            
            # 簡易的な判定（詳細分析なし）
            simple_improvement = (8.0 - basic_entropy) / 8.0
            return simple_improvement > 0.2, {
                'entropy_improvement': simple_improvement,
                'theoretical_compression_gain': simple_improvement * 50,  # 簡易推定
                'reason': 'lightweight_mode_simple_check'
            }
        
        try:
            # 通常モード: 詳細分析
            sample = data[:min(self.sample_size, len(data))]
            sample_key = hash(sample) + hash(str(data_type))
            
            # キャッシュチェック
            if sample_key in self.cache:
                self.cache_hit_count += 1
                cached_result = self.cache[sample_key]
                print(f"    [予測メタ分析] キャッシュヒット: 残差エントロピー改善={cached_result['entropy_improvement']:.2%}")
                return cached_result['should_transform'], cached_result
            
            # キャッシュミス
            self.cache_miss_count += 1
            
            # 残差エントロピー予測による効果判定
            original_entropy = calculate_entropy(sample)
            predicted_residual_entropy, header_cost = self._predict_residual_entropy(sample, data_type, len(data))
            
            # 情報理論的利得計算
            theoretical_gain = calculate_theoretical_compression_gain(
                original_entropy, predicted_residual_entropy, header_cost, len(data)
            )
            
            # 変換判定（理論的利得が正の場合のみ変換）
            should_transform = theoretical_gain > 0
            entropy_improvement = (original_entropy - predicted_residual_entropy) / original_entropy if original_entropy > 0 else 0
            
            analysis_info = {
                'sample_size': len(sample),
                'original_entropy': original_entropy,
                'predicted_residual_entropy': predicted_residual_entropy,
                'theoretical_header_cost': header_cost,
                'entropy_improvement': entropy_improvement,
                'theoretical_gain': theoretical_gain,
                'should_transform': should_transform,
                'method': 'residual_entropy_prediction'
            }
            
            # キャッシュに保存（サイズ制限付き）
            self._update_cache(sample_key, analysis_info)
            
            print(f"    [予測メタ分析] 残差エントロピー改善: {entropy_improvement:.2%}, 理論利得: {theoretical_gain:.1f}% -> {'変換実行' if should_transform else '変換スキップ'}")
            
            return should_transform, analysis_info
            
        except Exception as e:
            # デバッグ用詳細エラー情報
            error_detail = str(e)
            if 'nxzip' in error_detail.lower():
                print(f"    [予測メタ分析] インポートエラー (無害): {error_detail[:50]}... - 保守的判定を使用")
            else:
                print(f"    [予測メタ分析] 予測エラー: {error_detail[:50]}... - 保守的判定でスキップ")
            return False, {'reason': 'prediction_error', 'error': str(e), 'fallback': 'conservative'}
    
    def _predict_residual_entropy(self, sample: bytes, data_type, full_data_size: int) -> Tuple[float, int]:
        """データタイプ別残差エントロピー予測"""
        # データタイプに応じた予測
        if hasattr(data_type, 'value'):
            data_type_str = data_type.value
        else:
            data_type_str = str(data_type)
        
        if 'sequential_int' in data_type_str.lower():
            # LeCo変換の残差エントロピー予測
            residual_entropy = self._predict_leco_residual_entropy(sample)
            header_cost = 32  # LeCo辞書サイズ
            
        elif 'float' in data_type_str.lower():
            # TDT変換の残差エントロピー予測
            residual_entropy = self._predict_tdt_residual_entropy(sample)
            header_cost = 24  # TDT変換パラメータ
            
        elif 'text' in data_type_str.lower() or 'repetitive' in data_type_str.lower():
            # BWT+MTF変換の残差エントロピー予測
            residual_entropy = self._predict_bwt_residual_entropy(sample)
            header_cost = 16  # BWT変換インデックス
            
        else:
            # 一般的変換（コンテキストミキシング）の残差エントロピー予測
            residual_entropy = self._predict_contextmixing_residual_entropy(sample)
            header_cost = 40  # コンテキストミキシングモデル
        
        return residual_entropy, header_cost
    
    def _predict_leco_residual_entropy(self, sample: bytes) -> float:
        """LeCo変換後の残差エントロピー予測（整数系列特化）"""
        if len(sample) < 16:
            return calculate_entropy(sample)
        
        try:
            # 4バイト整数として解釈し、1次差分の分散を予測
            int_values = []
            for i in range(0, len(sample) - 3, 4):
                val = int.from_bytes(sample[i:i+4], 'little', signed=True)
                int_values.append(val)
            
            if len(int_values) < 2:
                return calculate_entropy(sample) * 0.9
            
            # 1次差分のエントロピー（LeCoの残差に相当）
            differences = [int_values[i+1] - int_values[i] for i in range(len(int_values)-1)]
            diff_bytes = b''.join(val.to_bytes(4, 'little', signed=True) for val in differences)
            residual_entropy = calculate_entropy(diff_bytes)
            
            # 系列整数データは通常70-85%のエントロピー削減が期待できる
            return residual_entropy * 0.75
            
        except:
            return calculate_entropy(sample) * 0.9
    
    def _predict_tdt_residual_entropy(self, sample: bytes) -> float:
        """TDT変換後の残差エントロピー予測（時系列特化）"""
        original_entropy = calculate_entropy(sample)
        
        # 浮動小数点データの時系列変換効果を予測
        similarity_factor = estimate_temporal_similarity(sample)
        
        # 高い時系列相関があるほど大きなエントロピー削減
        entropy_reduction = similarity_factor * 0.6  # 最大60%削減
        return original_entropy * (1.0 - entropy_reduction)
    
    def _predict_bwt_residual_entropy(self, sample: bytes) -> float:
        """BWT+MTF変換後の残差エントロピー予測（繰り返しパターン特化）"""
        original_entropy = calculate_entropy(sample)
        
        # 繰り返しパターンの密度を推定
        repetition_factor = estimate_repetition_density(sample)
        
        # 繰り返しが多いほどBWT+MTFの効果は大きい
        entropy_reduction = repetition_factor * 0.7  # 最大70%削減
        return original_entropy * (1.0 - entropy_reduction)
    
    def _predict_contextmixing_residual_entropy(self, sample: bytes) -> float:
        """コンテキストミキシング変換後の残差エントロピー予測"""
        original_entropy = calculate_entropy(sample)
        
        # コンテキスト予測可能性を推定
        context_predictability = estimate_context_predictability(sample)
        
        # 予測可能性が高いほどエントロピー削減効果が大きい
        entropy_reduction = context_predictability * 0.4  # 最大40%削減
        return original_entropy * (1.0 - entropy_reduction)
    
    def _update_cache(self, key: str, value: dict):
        """キャッシュを更新（サイズ制限付き）"""
        # キャッシュサイズ制限チェック
        if len(self.cache) >= self.cache_max_size:
            # 最も古いエントリを削除（FIFO）
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            print(f"    [キャッシュ管理] 最大サイズ到達により古いエントリを削除: {self.cache_max_size}")
        
        # 新しいエントリを追加
        self.cache[key] = value
    
    def get_cache_stats(self) -> dict:
        """キャッシュ統計を取得"""
        total_requests = self.cache_hit_count + self.cache_miss_count
        hit_rate = self.cache_hit_count / total_requests if total_requests > 0 else 0.0
        
        return {
            "total_entries": len(self.cache),
            "max_size": self.cache_max_size,
            "hits": self.cache_hit_count,
            "misses": self.cache_miss_count,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }
    
    def clear_cache(self):
        """キャッシュをクリア"""
        self.cache.clear()
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        print("🧹 MetaAnalyzerキャッシュをクリアしました")
