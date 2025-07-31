#!/usr/bin/env python3
"""
NEXUS TMC Phase 1 最適化実装
エントロピー計算のNumba最適化
"""

import numpy as np
import numba
import time
import sys
from typing import Tuple

# 元のNEXUS TMCからエントロピー計算を抽出してNumba最適化

@numba.jit(nopython=True, cache=True)
def calculate_entropy_numba(data_bytes: np.ndarray) -> float:
    """シャノンエントロピー計算（Numba最適化版）"""
    # バイトカウント（Numba対応）
    byte_counts = np.zeros(256, dtype=np.int64)
    
    for byte_val in data_bytes:
        byte_counts[byte_val] += 1
    
    # 確率計算
    data_length = len(data_bytes)
    entropy = 0.0
    
    for count in byte_counts:
        if count > 0:
            probability = count / data_length
            entropy -= probability * np.log2(probability)
    
    return entropy

@numba.jit(nopython=True, cache=True)
def analyze_byte_patterns_numba(data_bytes: np.ndarray) -> Tuple[float, float, float]:
    """バイトパターン分析（Numba最適化版）"""
    if len(data_bytes) < 4:
        return 1.0, 1.0, 1.0
    
    # 差分計算
    diff_sum = 0.0
    for i in range(len(data_bytes) - 1):
        diff_sum += abs(int(data_bytes[i+1]) - int(data_bytes[i]))
    
    avg_difference = diff_sum / (len(data_bytes) - 1)
    
    # 連続性計算
    consecutive_count = 0
    for i in range(len(data_bytes) - 1):
        if abs(int(data_bytes[i+1]) - int(data_bytes[i])) <= 1:
            consecutive_count += 1
    
    consecutive_ratio = consecutive_count / (len(data_bytes) - 1)
    
    # 繰り返しパターン検出
    repeat_count = 0
    for i in range(len(data_bytes) - 1):
        if data_bytes[i] == data_bytes[i+1]:
            repeat_count += 1
    
    repeat_ratio = repeat_count / (len(data_bytes) - 1)
    
    return avg_difference, consecutive_ratio, repeat_ratio

@numba.jit(nopython=True, cache=True)
def rle_compress_numba(data_bytes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Run-Length Encoding（Numba最適化版）"""
    if len(data_bytes) == 0:
        empty_array = np.zeros(0, dtype=np.uint8)
        return empty_array, empty_array
    
    # 最大サイズでバッファを準備
    literals = np.zeros(len(data_bytes), dtype=np.uint8)
    run_lengths = np.zeros(len(data_bytes), dtype=np.uint8)
    
    literals_idx = 0
    runs_idx = 0
    
    current_byte = data_bytes[0]
    run_length = 1
    
    for i in range(1, len(data_bytes)):
        if data_bytes[i] == current_byte and run_length < 255:
            run_length += 1
        else:
            # 現在のランを記録
            literals[literals_idx] = current_byte
            run_lengths[runs_idx] = run_length
            literals_idx += 1
            runs_idx += 1
            
            # 新しいランを開始
            current_byte = data_bytes[i]
            run_length = 1
    
    # 最後のランを記録
    literals[literals_idx] = current_byte
    run_lengths[runs_idx] = run_length
    literals_idx += 1
    runs_idx += 1
    
    # 実際のサイズにトリム
    return literals[:literals_idx], run_lengths[:runs_idx]

@numba.jit(nopython=True, cache=True)
def mtf_transform_numba(data_bytes: np.ndarray) -> np.ndarray:
    """Move-to-Front変換（Numba最適化版）"""
    # アルファベット初期化（0-255）
    alphabet = np.arange(256, dtype=np.uint8)
    result = np.zeros(len(data_bytes), dtype=np.uint8)
    
    for i in range(len(data_bytes)):
        byte_val = data_bytes[i]
        
        # アルファベット内での位置を見つける
        pos = 0
        for j in range(256):
            if alphabet[j] == byte_val:
                pos = j
                break
        
        result[i] = pos
        
        # Move-to-Front: 見つけた要素を先頭に移動
        if pos > 0:
            # 位置posの要素を一時保存
            temp = alphabet[pos]
            # pos位置から先頭まで一つずつ後ろにずらす
            for k in range(pos, 0, -1):
                alphabet[k] = alphabet[k-1]
            # 先頭に移動
            alphabet[0] = temp
    
    return result

class NumbaOptimizedEngine:
    """Numba最適化エンジン（Phase 1）"""
    
    def __init__(self):
        self.name = "NEXUS TMC Numba Optimized"
        # JITコンパイルのウォームアップ
        self._warmup_jit()
    
    def _warmup_jit(self):
        """JITコンパイルのウォームアップ"""
        print("🔥 Numba JITコンパイル ウォームアップ中...")
        dummy_data = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
        
        # 各関数を一度実行してコンパイル
        calculate_entropy_numba(dummy_data)
        analyze_byte_patterns_numba(dummy_data)
        rle_compress_numba(dummy_data)
        mtf_transform_numba(dummy_data)
        
        print("✅ JITコンパイル完了")
    
    def process_data_optimized(self, data: bytes) -> dict:
        """最適化されたデータ処理"""
        # NumPy配列に変換
        data_array = np.frombuffer(data, dtype=np.uint8)
        
        start_time = time.time()
        
        # 1. エントロピー計算（Numba最適化）
        entropy = calculate_entropy_numba(data_array)
        entropy_time = time.time() - start_time
        
        # 2. パターン分析（Numba最適化）
        start_time = time.time()
        avg_diff, consecutive_ratio, repeat_ratio = analyze_byte_patterns_numba(data_array)
        pattern_time = time.time() - start_time
        
        # 3. RLE圧縮（Numba最適化）
        start_time = time.time()
        rle_literals, rle_runs = rle_compress_numba(data_array)
        rle_time = time.time() - start_time
        
        # 4. MTF変換（Numba最適化）
        start_time = time.time()
        mtf_result = mtf_transform_numba(data_array)
        mtf_time = time.time() - start_time
        
        return {
            'entropy': entropy,
            'avg_difference': avg_diff,
            'consecutive_ratio': consecutive_ratio,
            'repeat_ratio': repeat_ratio,
            'rle_compression_ratio': len(rle_literals) / len(data),
            'mtf_size': len(mtf_result),
            'timing': {
                'entropy_time': entropy_time,
                'pattern_time': pattern_time,
                'rle_time': rle_time,
                'mtf_time': mtf_time,
                'total_time': entropy_time + pattern_time + rle_time + mtf_time
            }
        }

def benchmark_optimization():
    """最適化効果のベンチマーク"""
    print("🚀 Numba最適化効果ベンチマーク")
    print("=" * 50)
    
    # テストデータの準備
    test_sizes = [1024, 10240, 102400, 1024000]  # 1KB, 10KB, 100KB, 1MB
    
    engine = NumbaOptimizedEngine()
    
    for size in test_sizes:
        print(f"\n📊 データサイズ: {size:,} bytes")
        
        # テストデータ生成（実際のファイルパターンに近い）
        np.random.seed(42)  # 再現可能性のため
        test_data = bytes(np.random.randint(0, 256, size, dtype=np.uint8))
        
        # ベンチマーク実行
        start_time = time.time()
        result = engine.process_data_optimized(test_data)
        total_time = time.time() - start_time
        
        # 処理速度計算（最小時間で除算エラーを防ぐ）
        total_time = max(total_time, 1e-6)  # 最小1マイクロ秒
        speed_mbps = (size / (1024 * 1024)) / total_time
        
        print(f"  ⚡ 総処理速度: {speed_mbps:.1f} MB/s")
        print(f"  📈 エントロピー: {result['entropy']:.2f}")
        print(f"  ⏱️ 処理時間詳細:")
        for operation, time_taken in result['timing'].items():
            if operation != 'total_time':
                time_taken = max(time_taken, 1e-6)  # 最小時間を設定
                op_speed = (size / (1024 * 1024)) / time_taken
                print(f"    {operation}: {time_taken*1000:.2f}ms ({op_speed:.1f} MB/s)")
        
        print(f"  🔧 RLE圧縮率: {result['rle_compression_ratio']:.2%}")

def main():
    """メイン実行"""
    print("NEXUS TMC Phase 1: Numba最適化実装")
    print("=" * 50)
    
    benchmark_optimization()
    
    print("\n" + "=" * 50)
    print("📊 Phase 1 最適化完了")
    print("次のステップ: NEXUS TMC本体への統合")
    print("=" * 50)

if __name__ == "__main__":
    main()
