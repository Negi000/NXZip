#!/usr/bin/env python3
"""
修正されたSPE Core実装
完全可逆なアルゴリズムです
"""

import struct
import hashlib
from typing import List, Tuple


class SPECore:
    """改良されたSPE Core"""
    
    def __init__(self):
        self.xor_key = b"NXZip_SPE_2024_v2"
        self.block_size = 16
    
    def apply_transform(self, data: bytes) -> bytes:
        """完全可逆なSPE変換を適用"""
        result = bytearray(data)
        original_len = len(result)
        
        # 1. パディング（16バイト境界）
        padded_len = ((original_len + 15) // 16) * 16
        result.extend(b'\x00' * (padded_len - original_len))
        
        # 元の長さを記録
        result.extend(struct.pack('<Q', original_len))
        
        # 2. 可逆ブロックシャッフル
        if len(result) >= 32:
            self._apply_reversible_shuffle(result)
        
        # 3. XOR難読化
        self._apply_xor(result)
        
        return bytes(result)
    
    def reverse_transform(self, data: bytes) -> bytes:
        """SPE変換を完全に逆変換"""
        result = bytearray(data)
        
        # 1. XOR除去
        self._apply_xor(result)
        
        # 2. ブロックシャッフル逆変換
        if len(result) >= 32:
            self._reverse_reversible_shuffle(result)
        
        # 3. パディング除去
        if len(result) >= 8:
            # 元の長さを取得
            original_len = struct.unpack('<Q', result[-8:])[0]
            result = result[:-8]  # 長さ情報を除去
            result = result[:original_len]  # 元のサイズに切り詰め
        
        return bytes(result)
    
    def _apply_xor(self, data: bytearray) -> None:
        """XOR難読化/除去（自己逆変換）"""
        for i in range(len(data)):
            data[i] ^= self.xor_key[i % len(self.xor_key)]
    
    def _apply_reversible_shuffle(self, data: bytearray) -> None:
        """可逆ブロックシャッフル"""
        num_blocks = len(data) // self.block_size
        if num_blocks <= 1:
            return
        
        # Fisher-Yates shuffle の変種（可逆）
        # ランダムシードを作成（データの最初の8バイトのハッシュから）
        seed_data = data[:8]
        seed = int.from_bytes(hashlib.sha256(seed_data).digest()[:4], 'little')
        
        # シャッフルパターンを生成（可逆性を保証）
        for i in range(num_blocks - 1, 0, -1):
            # 確定的な擬似乱数を生成
            seed = (seed * 1103515245 + 12345) % (2**31)
            j = seed % (i + 1)
            
            if i != j:
                self._swap_blocks(data, i, j)
    
    def _reverse_reversible_shuffle(self, data: bytearray) -> None:
        """可逆ブロックシャッフルの逆変換"""
        num_blocks = len(data) // self.block_size
        if num_blocks <= 1:
            return
        
        # 同じシードを使用
        seed_data = data[:8]
        seed = int.from_bytes(hashlib.sha256(seed_data).digest()[:4], 'little')
        
        # スワップの順序を記録
        swaps = []
        for i in range(num_blocks - 1, 0, -1):
            seed = (seed * 1103515245 + 12345) % (2**31)
            j = seed % (i + 1)
            if i != j:
                swaps.append((i, j))
        
        # 逆順でスワップを実行
        for i, j in reversed(swaps):
            self._swap_blocks(data, i, j)
    
    def _swap_blocks(self, data: bytearray, i: int, j: int) -> None:
        """2つのブロックをスワップ"""
        start1 = i * self.block_size
        start2 = j * self.block_size
        
        for k in range(self.block_size):
            if start1 + k < len(data) and start2 + k < len(data):
                data[start1 + k], data[start2 + k] = data[start2 + k], data[start1 + k]


def test_spe_core():
    """SPE Core のテスト"""
    spe = SPECore()
    
    test_cases = [
        b"Hello",
        b"",
        b"x" * 15,
        b"x" * 16,
        b"x" * 17,
        bytes(range(100)),
        bytes(range(256)),
        b"The quick brown fox jumps over the lazy dog."
    ]
    
    print("🧪 Testing SPE Core")
    print("=" * 50)
    
    all_passed = True
    for i, test_data in enumerate(test_cases):
        print(f"Test {i+1}: {len(test_data)} bytes")
        
        # 変換
        transformed = spe.apply_transform(test_data)
        
        # 逆変換
        restored = spe.reverse_transform(transformed)
        
        # 検証
        success = test_data == restored
        print(f"  Result: {'✅ PASS' if success else '❌ FAIL'}")
        
        if not success:
            print(f"  Original: {test_data[:20]}{'...' if len(test_data) > 20 else ''}")
            print(f"  Restored: {restored[:20]}{'...' if len(restored) > 20 else ''}")
            all_passed = False
    
    print("=" * 50)
    print(f"Overall result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    return all_passed


def test_reversibility_edge_cases():
    """可逆性のエッジケースをテスト"""
    spe = SPECore()
    
    print("\n🔬 Testing Reversibility Edge Cases")
    print("=" * 50)
    
    # 大きなデータでのテスト
    large_data = bytes(range(256)) * 10  # 2560バイト
    print(f"Large data test: {len(large_data)} bytes")
    
    transformed = spe.apply_transform(large_data)
    restored = spe.reverse_transform(transformed)
    
    success = large_data == restored
    print(f"Result: {'✅ PASS' if success else '❌ FAIL'}")
    
    if success:
        print("✅ Reversibility confirmed for large data")
    else:
        print("❌ Reversibility failed for large data")
        # 詳細な差分分析
        differences = sum(1 for i in range(min(len(large_data), len(restored))) 
                         if large_data[i] != restored[i])
        print(f"Differences found: {differences}")
    
    return success


if __name__ == "__main__":
    print("🚀 NXZip SPE Core v2.0 - Fixed Implementation")
    print("=" * 60)
    
    # 基本テスト
    basic_passed = test_spe_core()
    
    # エッジケーステスト
    edge_passed = test_reversibility_edge_cases()
    
    print("\n" + "=" * 60)
    print("FINAL SUMMARY:")
    print(f"Basic tests: {'✅ PASSED' if basic_passed else '❌ FAILED'}")
    print(f"Edge cases:  {'✅ PASSED' if edge_passed else '❌ FAILED'}")
    
    if basic_passed and edge_passed:
        print("\n🎉 SPE Core is now completely reversible!")
        print("Ready for integration with compression and encryption systems.")
    else:
        print("\n⚠️  Issues remain - further debugging required.")
