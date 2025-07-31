#!/usr/bin/env python3
"""
NEXUS TMC Engine - Optimization Performance Test

Phase 2: Numba/Cython最適化後のパフォーマンステスト
- entropy_calculator.py: Numba JIT最適化 (期待: 3-5x improvement)
- bwt_transform.py: Numba JIT最適化 (期待: 2-3x improvement)
- 全体目標: 150+ MB/s compression speed (5-7x total improvement)
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# プロジェクトパスを追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from nxzip.engine.nexus_tmc_v91_modular import NEXUSTMCEngineV91
from nxzip.engine.analyzers.entropy_calculator import calculate_entropy, calculate_theoretical_compression_gain
from nxzip.engine.transforms.bwt_transform import BWTTransformer


def generate_test_data(size: int = 1024 * 1024) -> bytes:
    """テスト用データ生成 (1MB)"""
    # 圧縮しやすいパターンを含むデータ
    patterns = [
        b"The quick brown fox jumps over the lazy dog. " * 50,
        b"1234567890" * 100,
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 40,
        bytes(range(256)) * 10,
        os.urandom(size // 4)  # ランダムデータも混在
    ]
    
    data = b""
    while len(data) < size:
        for pattern in patterns:
            data += pattern
            if len(data) >= size:
                break
    
    return data[:size]


def test_entropy_calculator_performance():
    """entropy_calculator.py Numba最適化のパフォーマンステスト"""
    print("=" * 60)
    print("🔬 Entropy Calculator Performance Test (Numba JIT)")
    print("=" * 60)
    
    # テストデータ生成
    test_sizes = [64*1024, 256*1024, 1024*1024]  # 64KB, 256KB, 1MB
    
    for size in test_sizes:
        print(f"\n📊 Test Size: {size // 1024}KB")
        data = generate_test_data(size)
        
        # 複数回実行して平均を測定
        times = []
        for i in range(5):
            start_time = time.perf_counter()
            
            # エントロピー計算
            entropy = calculate_entropy(data)
            compression_gain = calculate_theoretical_compression_gain(entropy, entropy * 0.7, 256, len(data))
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        throughput = size / avg_time / (1024 * 1024)  # MB/s
        
        print(f"  Average Time: {avg_time:.4f}s")
        print(f"  Throughput: {throughput:.2f} MB/s")
        print(f"  Entropy: {entropy:.4f}")
        print(f"  Compression Gain: {compression_gain:.2f}%")


def test_bwt_performance():
    """bwt_transform.py Numba最適化のパフォーマンステスト"""
    print("\n" + "=" * 60)
    print("🔄 BWT Transform Performance Test (Numba JIT)")
    print("=" * 60)
    
    # 小さなサイズでテスト（BWTは計算量が大きいため）
    test_sizes = [8*1024, 32*1024, 64*1024]  # 8KB, 32KB, 64KB
    transformer = BWTTransformer()
    
    for size in test_sizes:
        print(f"\n📊 Test Size: {size // 1024}KB")
        data = generate_test_data(size)
        
        # 複数回実行して平均を測定
        times = []
        for i in range(3):  # BWTは重いので3回のみ
            start_time = time.perf_counter()
            
            # BWT変換
            try:
                streams, info = transformer.transform(data)
                recovered = transformer.inverse_transform(streams, info)
                
                # データ整合性チェック
                if recovered != data:
                    print(f"  ⚠️ Data integrity check failed!")
                    continue
                    
            except Exception as e:
                print(f"  ❌ BWT Transform failed: {e}")
                continue
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        if times:
            avg_time = sum(times) / len(times)
            throughput = size / avg_time / (1024 * 1024)  # MB/s
            
            print(f"  Average Time: {avg_time:.4f}s")
            print(f"  Throughput: {throughput:.2f} MB/s")
            print(f"  Stream Count: {info.get('stream_count', 'N/A')}")
            print(f"  Zero Ratio: {info.get('zero_ratio', 0):.2%}")


def test_overall_compression_performance():
    """TMC v9.1エンジン全体のパフォーマンステスト"""
    print("\n" + "=" * 60)
    print("🚀 Overall TMC v9.1 Engine Performance Test")
    print("=" * 60)
    
    engine = NEXUSTMCEngineV91()
    test_sizes = [256*1024, 512*1024, 1024*1024]  # 256KB, 512KB, 1MB
    
    for size in test_sizes:
        print(f"\n📊 Test Size: {size // 1024}KB")
        data = generate_test_data(size)
        
        # 圧縮テスト
        times = []
        for i in range(3):
            start_time = time.perf_counter()
            
            try:
                compressed_data, info = engine.compress(data)
                decompressed_data = engine.decompress(compressed_data, info)
                
                # データ整合性チェック
                if decompressed_data != data:
                    print(f"  ⚠️ Data integrity check failed!")
                    continue
                    
            except Exception as e:
                print(f"  ❌ Compression failed: {e}")
                continue
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        if times:
            avg_time = sum(times) / len(times)
            throughput = size / avg_time / (1024 * 1024)  # MB/s
            compression_ratio = len(compressed_data) / len(data)
            
            print(f"  Average Time: {avg_time:.4f}s")
            print(f"  Throughput: {throughput:.2f} MB/s")
            print(f"  Compression Ratio: {compression_ratio:.4f}")
            print(f"  Space Saving: {(1-compression_ratio)*100:.2f}%")


def main():
    """最適化パフォーマンステストのメイン実行"""
    print("🔥 NEXUS TMC v9.1 - Phase 2 Optimization Performance Test")
    print("Expected improvements:")
    print("  - entropy_calculator.py: 3-5x faster (Numba JIT)")
    print("  - bwt_transform.py: 2-3x faster (Numba JIT)")
    print("  - Overall target: 150+ MB/s compression speed")
    
    try:
        # 個別モジュールテスト
        test_entropy_calculator_performance()
        test_bwt_performance()
        
        # 全体パフォーマンステスト
        test_overall_compression_performance()
        
        print("\n" + "=" * 60)
        print("✅ Performance Test Completed Successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
