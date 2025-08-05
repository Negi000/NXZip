"""
NEXUS TMC Engine - Entropy Calculator Module (Numba Optimized)

This module provides high-performance entropy calculation functions
optimized with Numba JIT compilation for maximum performance.
"""

import numpy as np
from typing import List, Tuple
import hashlib

# Numba imports with fallback
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
    print("🚀 Numba利用可能 - エントロピー計算の高速化有効")
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠️ Numba未利用 - 標準実装を使用")
    # Fallback decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args else decorator
    
    def prange(x):
        return range(x)

__all__ = ['calculate_entropy', 'calculate_entropy_vectorized', 'calculate_entropy_batch_numba']


@jit(nopython=True, cache=True)
def _calculate_entropy_numba_core(byte_counts: np.ndarray, data_length: int) -> float:
    """
    Numba最適化されたエントロピー計算のコア関数
    """
    if data_length == 0:
        return 0.0
    
    entropy = 0.0
    for count in byte_counts:
        if count > 0:
            probability = count / data_length
            entropy -= probability * np.log2(probability)
    
    return entropy


@jit(nopython=True, cache=True)  
def _count_bytes_numba(data_array: np.ndarray) -> np.ndarray:
    """
    Numba最適化されたバイトカウント
    """
    counts = np.zeros(256, dtype=np.int64)
    for byte_val in data_array:
        counts[byte_val] += 1
    return counts


def calculate_entropy(data: bytes) -> float:
    """
    シャノンエントロピー計算（Numba JIT最適化版）
    
    Args:
        data: 分析対象のバイトデータ
        
    Returns:
        float: 計算されたシャノンエントロピー (0.0-8.0)
    """
    if len(data) == 0:
        return 0.0
    
    if NUMBA_AVAILABLE:
        # Numba最適化パス
        data_array = np.frombuffer(data, dtype=np.uint8)
        byte_counts = _count_bytes_numba(data_array)
        return _calculate_entropy_numba_core(byte_counts, len(data))
    else:
        # フォールバック: NumPy実装
        byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        nonzero_counts = byte_counts[byte_counts > 0]
        if len(nonzero_counts) == 0:
            return 0.0
        
        probabilities = nonzero_counts / len(data)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy


@jit(nopython=True, cache=True)
def _calculate_entropy_batch_numba(data_arrays: list, lengths: np.ndarray) -> np.ndarray:
    """
    Numba最適化されたバッチエントロピー計算
    """
    result = np.zeros(len(data_arrays), dtype=np.float64)
    
    for i in prange(len(data_arrays)):
        if lengths[i] > 0:
            counts = _count_bytes_numba(data_arrays[i])
            result[i] = _calculate_entropy_numba_core(counts, lengths[i])
    
    return result


def calculate_entropy_batch_numba(data_chunks: List[bytes]) -> List[float]:
    """
    複数データチャンクのエントロピーを並列計算（Numba最適化）
    
    Args:
        data_chunks: バイトデータのリスト
        
    Returns:
        List[float]: 各チャンクのエントロピーリスト
    """
    if not data_chunks:
        return []
    
    if NUMBA_AVAILABLE and len(data_chunks) > 4:
        # Numba並列処理パス（大量データ用）
        try:
            data_arrays = [np.frombuffer(chunk, dtype=np.uint8) for chunk in data_chunks]
            lengths = np.array([len(chunk) for chunk in data_chunks], dtype=np.int64)
            
            results = _calculate_entropy_batch_numba(data_arrays, lengths)
            return results.tolist()
        except Exception:
            # フォールバック
            pass
    
    # 標準実装
    entropies = []
    for chunk in data_chunks:
        entropies.append(calculate_entropy(chunk))
    
    return entropies


def calculate_entropy_vectorized(data_chunks: List[bytes]) -> List[float]:
    """
    複数データチャンクのエントロピーを並列計算（エイリアス）
    """
    return calculate_entropy_batch_numba(data_chunks)


@jit(nopython=True, cache=True)
def _temporal_similarity_numba(data_array: np.ndarray) -> float:
    """
    Numba最適化された時系列類似性計算
    """
    if len(data_array) < 2:
        return 0.0
    
    total_diff = 0.0
    for i in range(len(data_array) - 1):
        total_diff += abs(int(data_array[i+1]) - int(data_array[i]))
    
    avg_diff = total_diff / (len(data_array) - 1)
    return max(0.0, min(1.0, 1.0 - (avg_diff / 128.0)))


def estimate_temporal_similarity(sample: bytes) -> float:
    """
    時系列類似性推定（Numba最適化版）
    
    Args:
        sample: 分析対象のサンプルデータ
        
    Returns:
        float: 時系列類似性スコア (0.0-1.0)
    """
    if len(sample) < 8:
        return 0.0
    
    if NUMBA_AVAILABLE:
        data_array = np.frombuffer(sample, dtype=np.uint8)
        return _temporal_similarity_numba(data_array)
    
    # フォールバック実装
    differences = [abs(sample[i+1] - sample[i]) for i in range(len(sample)-1)]
    avg_diff = sum(differences) / len(differences) if differences else 255
    return max(0.0, min(1.0, 1.0 - (avg_diff / 128)))


@jit(nopython=True, cache=True)
def _repetition_density_numba(data_array: np.ndarray) -> float:
    """
    Numba最適化された繰り返しパターン密度計算
    """
    if len(data_array) < 4:
        return 0.0
    
    max_count = 0
    data_len = len(data_array)
    
    # 2バイトパターンの検出
    for i in range(data_len - 1):
        pattern = (int(data_array[i]) << 8) | int(data_array[i+1])
        count = 1
        
        for j in range(i + 2, data_len - 1):
            test_pattern = (int(data_array[j]) << 8) | int(data_array[j+1])
            if test_pattern == pattern:
                count += 1
        
        if count > max_count:
            max_count = count
    
    repetition_ratio = max_count / (data_len // 2) if data_len > 2 else 0
    return min(1.0, repetition_ratio)


def estimate_repetition_density(sample: bytes) -> float:
    """
    繰り返しパターン密度推定（Numba最適化版）
    
    Args:
        sample: 分析対象のサンプルデータ
        
    Returns:
        float: 繰り返しパターン密度スコア (0.0-1.0)
    """
    if len(sample) < 4:
        return 0.0
    
    if NUMBA_AVAILABLE:
        data_array = np.frombuffer(sample, dtype=np.uint8)
        return _repetition_density_numba(data_array)
    
    # フォールバック実装
    pattern_counts = {}
    for pattern_len in [2, 3, 4]:
        for i in range(len(sample) - pattern_len + 1):
            pattern = sample[i:i+pattern_len]
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    max_count = max(pattern_counts.values()) if pattern_counts else 1
    repetition_ratio = max_count / (len(sample) // 2) if len(sample) > 2 else 0
    
    return min(1.0, repetition_ratio)


def estimate_context_predictability(sample: bytes) -> float:
    """
    コンテキスト予測可能性推定（0.0-1.0）
    
    Args:
        sample: 分析対象のサンプルデータ
        
    Returns:
        float: コンテキスト予測可能性スコア (0.0-1.0)
    """
    if len(sample) < 3:
        return 0.0
    
    # 2-gram予測精度で推定
    bigram_counts = {}
    for i in range(len(sample) - 1):
        bigram = sample[i:i+2]
        bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1
    
    # 高頻度bigramの割合
    total_bigrams = len(sample) - 1
    high_freq_count = sum(1 for count in bigram_counts.values() if count > 1)
    
    return high_freq_count / total_bigrams if total_bigrams > 0 else 0.0


@jit(nopython=True, cache=True)
def _calculate_theoretical_compression_gain_numba(original_entropy: float, residual_entropy: float, 
                                                 header_cost: float, data_size: float) -> float:
    """
    Numba最適化された理論的圧縮利得計算
    """
    if original_entropy <= 0 or data_size <= 0:
        return 0.0
    
    # より実用的な圧縮サイズ推定
    implementation_efficiency = 0.85  # 実装効率 (85%)
    
    # 理論的圧縮サイズ（バイト単位）
    original_size_bytes = data_size
    theoretical_residual_size = (residual_entropy / 8.0) * data_size * implementation_efficiency
    header_size_bytes = header_cost
    
    # 総圧縮サイズ
    total_compressed_size = theoretical_residual_size + header_size_bytes
    
    # 利得計算（負の値を防ぐ）
    if original_size_bytes > total_compressed_size:
        gain_percentage = ((original_size_bytes - total_compressed_size) / original_size_bytes) * 100
        return min(95.0, max(0.0, gain_percentage))  # 理論上限95%
    
    return 0.0


def calculate_theoretical_compression_gain(original_entropy: float, residual_entropy: float, 
                                         header_cost: int, data_size: int) -> float:
    """
    改良版理論的圧縮利得計算（パーセンテージ）
    
    Args:
        original_entropy: 元データのエントロピー
        residual_entropy: 変換後の残差エントロピー
        header_cost: ヘッダーコスト（バイト）
        data_size: データサイズ（バイト）
        
    Returns:
        float: 理論的圧縮利得（パーセンテージ）
    """
    if NUMBA_AVAILABLE:
        return _calculate_theoretical_compression_gain_numba(
            float(original_entropy), float(residual_entropy), 
            float(header_cost), float(data_size)
        )
    
    # フォールバック実装
    if original_entropy <= 0 or data_size <= 0:
        return 0.0
    
    implementation_efficiency = 0.85
    original_size_bytes = data_size
    theoretical_residual_size = (residual_entropy / 8.0) * data_size * implementation_efficiency
    header_size_bytes = header_cost
    
    total_compressed_size = theoretical_residual_size + header_size_bytes
    
    if original_size_bytes > total_compressed_size:
        gain_percentage = ((original_size_bytes - total_compressed_size) / original_size_bytes) * 100
        return min(95.0, max(0.0, gain_percentage))
    
    return 0.0


def generate_sample_key(data: bytes, offset: int = 0, size: int = None) -> str:
    """
    サンプルデータのハッシュキーを生成
    
    Args:
        data: 対象データ
        offset: オフセット
        size: サイズ
        
    Returns:
        str: ハッシュキー
    """
    if size is None:
        size = len(data)
    
    hasher = hashlib.md5()
    hasher.update(data[offset:offset+size])
    hasher.update(f"{offset}:{size}".encode())
    return hasher.hexdigest()


def calculate_theoretical_compression_gain(original_entropy: float, residual_entropy: float, 
                                         header_cost: int, data_size: int) -> float:
    """
    改良版理論的圧縮利得計算（パーセンテージ）
    
    Args:
        original_entropy: 元データのエントロピー
        residual_entropy: 変換後の残差エントロピー
        header_cost: ヘッダーコスト（バイト）
        data_size: データサイズ（バイト）
        
    Returns:
        float: 理論的圧縮利得（パーセンテージ）
    """
    if original_entropy <= 0 or data_size <= 0:
        return 0.0
    
    # より実用的な圧縮サイズ推定
    # Shannon限界に実装効率を考慮
    implementation_efficiency = 0.85  # 実装効率 (85%)
    
    # 理論的圧縮サイズ（バイト単位）
    original_size_bytes = data_size
    theoretical_residual_size = (residual_entropy / 8.0) * data_size * implementation_efficiency
    header_size_bytes = header_cost
    
    # 総圧縮サイズ
    total_compressed_size = theoretical_residual_size + header_size_bytes
    
    # 利得計算（負の値を防ぐ）
    if original_size_bytes > total_compressed_size:
        gain_percentage = ((original_size_bytes - total_compressed_size) / original_size_bytes) * 100
        return min(95.0, max(0.0, gain_percentage))  # 理論上限95%
    
    return 0.0


def generate_sample_key(data: bytes, offset: int = 0, size: int = None) -> str:
    """
    サンプルデータのハッシュキーを生成
    
    Args:
        data: 対象データ
        offset: オフセット
        size: サイズ
        
    Returns:
        str: ハッシュキー
    """
    if size is None:
        size = len(data)
    
    hasher = hashlib.md5()
    hasher.update(data[offset:offset+size])
    hasher.update(f"{offset}:{size}".encode())
    return hasher.hexdigest()
