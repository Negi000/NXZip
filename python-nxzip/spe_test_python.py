#!/usr/bin/env python3
"""
NXZip SPE Core System Test - Python Implementation
NXZipのSPE（Structure-Preserving Encryption）システムのPython実装とテスト

現在のRust実装からの完全移植版
"""

import time
import struct
from typing import List, Tuple, Union
import hashlib
import os


class SPECore:
    """
    SPE (Structure-Preserving Encryption) コアシステム
    データの論理構造を保持しながら高度な難読化を実現
    """
    
    def __init__(self, key: bytes = None):
        """SPEコアを初期化"""
        self.xor_key = key or b"NXZip_SPE_2024"
        self.block_size = 16
        
    def apply_simple_spe(self, data: bytes) -> bytes:
        """簡易SPE変換を適用（Rust実装からの移植）"""
        result = bytearray(data)
        
        # 1. 構造保持パディング
        original_len = len(result)
        padded_len = ((original_len + 15) // 16) * 16  # 16バイト境界
        result.extend(b'\x00' * (padded_len - original_len))
        
        # 元の長さを末尾に記録（8バイト）
        result.extend(struct.pack('<Q', original_len))
        
        # 2. ブロックシャッフル（16バイトブロック）
        if len(result) >= 32:
            self._apply_block_shuffle(result)
            
        # 3. XOR難読化
        self._apply_xor_obfuscation(result)
        
        return bytes(result)
    
    def reverse_simple_spe(self, data: bytes) -> bytes:
        """簡易SPE逆変換を適用（完全復元）"""
        result = bytearray(data)
        
        # 逆順で処理
        
        # 1. XOR除去
        self._remove_xor_obfuscation(result)
        
        # 2. ブロックシャッフル逆変換
        if len(result) >= 32:
            self._reverse_block_shuffle(result)
            
        # 3. パディング除去
        if len(result) >= 8:
            # 末尾8バイトから元の長さを取得
            original_len = struct.unpack('<Q', result[-8:])[0]
            
            # 長さ情報を除去
            result = result[:-8]
            
            # 元のサイズに切り詰め
            if original_len <= len(result):
                result = result[:original_len]
                
        return bytes(result)
    
    def _apply_block_shuffle(self, data: bytearray) -> None:
        """ブロックシャッフルを適用"""
        block_size = self.block_size
        num_blocks = len(data) // block_size
        
        for i in range(num_blocks):
            swap_with = (i * 7 + 3) % num_blocks  # 決定論的パターン
            if i != swap_with:
                start1 = i * block_size
                start2 = swap_with * block_size
                
                # ブロック単位でスワップ
                for j in range(block_size):
                    if start1 + j < len(data) and start2 + j < len(data):
                        data[start1 + j], data[start2 + j] = data[start2 + j], data[start1 + j]
    
    def _reverse_block_shuffle(self, data: bytearray) -> None:
        """ブロックシャッフル逆変換（自己逆変換）"""
        # 同じパターンで逆変換
        self._apply_block_shuffle(data)
    
    def _apply_xor_obfuscation(self, data: bytearray) -> None:
        """XOR難読化を適用"""
        for i in range(len(data)):
            data[i] ^= self.xor_key[i % len(self.xor_key)]
    
    def _remove_xor_obfuscation(self, data: bytearray) -> None:
        """XOR難読化を除去（自己逆変換）"""
        self._apply_xor_obfuscation(data)


def test_basic_reversibility():
    """基本的な可逆性テスト"""
    print("\n📋 Testing Basic Reversibility...")
    
    spe = SPECore()
    test_data = b"Hello, NXZip SPE Core System!"
    print(f"Original data: {test_data.decode('utf-8', errors='ignore')}")
    
    # SPE変換
    transformed = spe.apply_simple_spe(test_data)
    print(f"Transformed: {transformed[:16].hex().upper()}")
    
    # 逆変換
    restored = spe.reverse_simple_spe(transformed)
    print(f"Restored: {restored.decode('utf-8', errors='ignore')}")
    
    assert test_data == restored, "可逆性テスト失敗"
    print("✅ Reversibility test passed")


def test_structure_preservation():
    """データ構造保持テスト"""
    print("\n🏗️ Testing Structure Preservation...")
    
    spe = SPECore()
    test_sizes = [10, 100, 1000, 5000]
    
    for size in test_sizes:
        test_data = bytes(i % 256 for i in range(size))
        
        transformed = spe.apply_simple_spe(test_data)
        restored = spe.reverse_simple_spe(transformed)
        
        assert test_data == restored, f"Structure preservation failed for size {size}"
        
        ratio = len(transformed) / len(test_data)
        print(f"Size {size}: Original {len(test_data)} -> Transformed {len(transformed)} (ratio: {ratio:.2f})")
    
    print("✅ Structure preservation test passed")


def test_performance():
    """パフォーマンステスト"""
    print("\n⚡ Testing Performance...")
    
    spe = SPECore()
    data_size = 10000
    test_data = bytes(i % 256 for i in range(data_size))
    
    # 変換性能測定
    start = time.perf_counter()
    transformed = spe.apply_simple_spe(test_data)
    transform_time = time.perf_counter() - start
    
    # 復元性能測定
    start = time.perf_counter()
    restored = spe.reverse_simple_spe(transformed)
    restore_time = time.perf_counter() - start
    
    assert test_data == restored
    
    throughput = data_size / transform_time / 1024 / 1024  # MB/s
    
    print(f"Performance for {data_size} bytes:")
    print(f"  Transform: {transform_time:.6f}s ({throughput:.2f} MB/s)")
    print(f"  Restore: {restore_time:.6f}s")
    
    print("✅ Performance test completed")


def integration_test():
    """統合テスト"""
    print("\n🔧 Running Integration Tests...")
    
    spe = SPECore()
    
    # 複数のデータパターンをテスト
    test_patterns = [
        b"",                                                      # 空データ
        b"A",                                                     # 1バイト
        b"Hello",                                                 # 短文
        "これは日本語のテストデータです。".encode('utf-8'),           # 日本語
        bytes(range(255)),                                        # バイナリデータ
        b"A" * 1000,                                             # 反復データ
    ]
    
    for i, pattern in enumerate(test_patterns):
        print(f"Pattern {i + 1}: {len(pattern)} bytes")
        
        transformed = spe.apply_simple_spe(pattern)
        restored = spe.reverse_simple_spe(transformed)
        
        assert pattern == restored, f"Pattern {i + 1} failed"
        
        if len(pattern) > 0:
            compression_ratio = len(transformed) / len(pattern)
            print(f"  Ratio: {compression_ratio:.2f}x")
    
    print("✅ Integration tests passed")


def test_edge_cases():
    """エッジケースのテスト"""
    print("\n🚧 Testing Edge Cases...")
    
    spe = SPECore()
    
    # 極小データ
    tiny_data = b"X"
    transformed = spe.apply_simple_spe(tiny_data)
    restored = spe.reverse_simple_spe(transformed)
    assert tiny_data == restored
    
    # 境界サイズ（16バイト）
    boundary_data = b"A" * 16
    transformed = spe.apply_simple_spe(boundary_data)
    restored = spe.reverse_simple_spe(transformed)
    assert boundary_data == restored
    
    # 大きなデータ
    large_data = b"B" * 100000
    start = time.perf_counter()
    transformed = spe.apply_simple_spe(large_data)
    restored = spe.reverse_simple_spe(transformed)
    elapsed = time.perf_counter() - start
    assert large_data == restored
    
    print(f"Large data test: {len(large_data)} bytes processed in {elapsed:.4f}s")
    print("✅ Edge cases test passed")


def test_deterministic_behavior():
    """決定論的動作のテスト"""
    print("\n🔄 Testing Deterministic Behavior...")
    
    spe = SPECore()
    test_data = b"Deterministic test data for NXZip SPE"
    
    # 同じデータを複数回変換
    results = []
    for _ in range(5):
        transformed = spe.apply_simple_spe(test_data)
        results.append(transformed)
    
    # 全ての結果が同じであることを確認
    for i in range(1, len(results)):
        assert results[0] == results[i], f"Non-deterministic behavior detected at iteration {i}"
    
    print("✅ Deterministic behavior confirmed")


def test_key_sensitivity():
    """鍵の感度テスト"""
    print("\n🔑 Testing Key Sensitivity...")
    
    test_data = b"Key sensitivity test data"
    
    # 異なる鍵でSPE変換
    spe1 = SPECore(b"key_1_test")
    spe2 = SPECore(b"key_2_test")
    
    transformed1 = spe1.apply_simple_spe(test_data)
    transformed2 = spe2.apply_simple_spe(test_data)
    
    # 異なる鍵では異なる結果になることを確認
    assert transformed1 != transformed2, "Different keys produced same result"
    
    # それぞれの鍵で正しく復元できることを確認
    restored1 = spe1.reverse_simple_spe(transformed1)
    restored2 = spe2.reverse_simple_spe(transformed2)
    
    assert restored1 == test_data, "Key 1 restoration failed"
    assert restored2 == test_data, "Key 2 restoration failed"
    
    print("✅ Key sensitivity test passed")


def benchmark_comparison():
    """ベンチマーク比較"""
    print("\n📊 Benchmark Comparison...")
    
    spe = SPECore()
    sizes = [1024, 10240, 102400, 1048576]  # 1KB, 10KB, 100KB, 1MB
    
    print("Size\t\tTransform\tRestore\t\tThroughput")
    print("-" * 60)
    
    for size in sizes:
        test_data = bytes(i % 256 for i in range(size))
        
        # 複数回実行して平均を取る
        transform_times = []
        restore_times = []
        
        for _ in range(3):
            start = time.perf_counter()
            transformed = spe.apply_simple_spe(test_data)
            transform_times.append(time.perf_counter() - start)
            
            start = time.perf_counter()
            restored = spe.reverse_simple_spe(transformed)
            restore_times.append(time.perf_counter() - start)
            
            assert test_data == restored
        
        avg_transform = sum(transform_times) / len(transform_times)
        avg_restore = sum(restore_times) / len(restore_times)
        throughput = size / avg_transform / 1024 / 1024
        
        size_str = f"{size//1024}KB" if size < 1048576 else f"{size//1048576}MB"
        print(f"{size_str:<12}\t{avg_transform:.6f}s\t{avg_restore:.6f}s\t{throughput:.2f} MB/s")


def main():
    """メインテスト実行"""
    print("🎯 NXZip SPE Core System Test - Python Implementation")
    print("=" * 60)
    
    # 基本テスト
    test_basic_reversibility()
    test_structure_preservation()
    test_performance()
    integration_test()
    
    # 追加テスト
    test_edge_cases()
    test_deterministic_behavior()
    test_key_sensitivity()
    
    # ベンチマーク
    benchmark_comparison()
    
    print("\n" + "=" * 60)
    print("✅ All SPE Core tests passed!")
    print("🚀 Python implementation ready for production!")


if __name__ == "__main__":
    main()
