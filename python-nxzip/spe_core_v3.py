#!/usr/bin/env python3
"""
SPE Core v3.0 - 完全修正版
シンプルで確実な可逆アルゴリズム
"""

import struct
from typing import List


class SPECore:
    """完全可逆なSPE Core v3.0"""
    
    def __init__(self):
        self.xor_key = b"NXZip_SPE_2024_v3"
        self.block_size = 16
    
    def apply_transform(self, data: bytes) -> bytes:
        """完全可逆なSPE変換"""
        if not data:
            return data
            
        result = bytearray(data)
        original_len = len(result)
        
        # 1. パディング（16バイト境界に調整）
        padded_len = ((original_len + 15) // 16) * 16
        result.extend(b'\x00' * (padded_len - original_len))
        
        # 元の長さを最後に追加
        result.extend(struct.pack('<Q', original_len))
        
        # 2. 確定的なブロック循環シフト（完全可逆）
        if len(result) >= 32:
            self._apply_cyclic_shift(result)
        
        # 3. バイトレベル変換（完全可逆）
        self._apply_byte_transform(result)
        
        # 4. XOR難読化（自己逆変換）
        self._apply_xor(result)
        
        return bytes(result)
    
    def reverse_transform(self, data: bytes) -> bytes:
        """SPE変換を完全に逆変換"""
        if not data:
            return data
            
        result = bytearray(data)
        
        # 4. XOR除去
        self._apply_xor(result)
        
        # 3. バイトレベル変換を逆変換
        self._reverse_byte_transform(result)
        
        # 2. ブロック循環シフトを逆変換
        if len(result) >= 32:
            self._reverse_cyclic_shift(result)
        
        # 1. パディング除去
        if len(result) >= 8:
            original_len = struct.unpack('<Q', result[-8:])[0]
            result = result[:-8]
            result = result[:original_len]
        
        return bytes(result)
    
    def _apply_xor(self, data: bytearray) -> None:
        """XOR難読化（自己逆変換）"""
        for i in range(len(data)):
            data[i] ^= self.xor_key[i % len(self.xor_key)]
    
    def _apply_byte_transform(self, data: bytearray) -> None:
        """バイトレベル変換（可逆）"""
        for i in range(len(data)):
            # 可逆なバイト変換（ビット反転 + 加算）
            data[i] = ((data[i] ^ 0xFF) + 0x5A) & 0xFF
    
    def _reverse_byte_transform(self, data: bytearray) -> None:
        """バイトレベル変換を逆変換"""
        for i in range(len(data)):
            # 逆変換（減算 + ビット反転）
            data[i] = ((data[i] - 0x5A) & 0xFF) ^ 0xFF
    
    def _apply_cyclic_shift(self, data: bytearray) -> None:
        """ブロック循環シフト（完全可逆）"""
        num_blocks = len(data) // self.block_size
        if num_blocks <= 1:
            return
        
        # シンプルな循環シフト（右シフト）
        shift_amount = 1  # 固定シフト量
        self._cyclic_shift_blocks(data, shift_amount, num_blocks)
    
    def _reverse_cyclic_shift(self, data: bytearray) -> None:
        """ブロック循環シフトを逆変換"""
        num_blocks = len(data) // self.block_size
        if num_blocks <= 1:
            return
        
        # 左シフト（右シフトの逆）
        shift_amount = -1  # 逆方向シフト
        self._cyclic_shift_blocks(data, shift_amount, num_blocks)
    
    def _cyclic_shift_blocks(self, data: bytearray, shift: int, num_blocks: int) -> None:
        """ブロックを循環シフト"""
        if shift == 0 or num_blocks <= 1:
            return
        
        # 正規化
        shift = shift % num_blocks
        if shift == 0:
            return
        
        # 循環シフトを実行（メモリ効率的）
        temp_blocks = []
        
        # シフト分のブロックを一時保存
        for i in range(shift):
            start = i * self.block_size
            end = start + self.block_size
            temp_blocks.append(data[start:end])
        
        # 残りのブロックを前方に移動
        for i in range(shift, num_blocks):
            src_start = i * self.block_size
            src_end = src_start + self.block_size
            dst_start = (i - shift) * self.block_size
            dst_end = dst_start + self.block_size
            
            for j in range(self.block_size):
                if src_start + j < len(data) and dst_start + j < len(data):
                    data[dst_start + j] = data[src_start + j]
        
        # 一時保存したブロックを末尾に配置
        for i, temp_block in enumerate(temp_blocks):
            dst_start = (num_blocks - shift + i) * self.block_size
            for j in range(len(temp_block)):
                if dst_start + j < len(data):
                    data[dst_start + j] = temp_block[j]


def comprehensive_test():
    """包括的テスト"""
    spe = SPECore()
    
    test_cases = [
        (b"", "Empty data"),
        (b"A", "Single byte"),
        (b"Hello", "Short string"),
        (b"x" * 15, "15 bytes"),
        (b"x" * 16, "Exactly 16 bytes"),
        (b"x" * 17, "17 bytes"),
        (b"x" * 32, "32 bytes"),
        (bytes(range(100)), "100 sequential bytes"),
        (bytes(range(256)), "256 sequential bytes"),
        (b"The quick brown fox jumps over the lazy dog.", "Sentence"),
        (b"\x00" * 100, "100 null bytes"),
        (b"\xFF" * 100, "100 max bytes"),
    ]
    
    print("🧪 Comprehensive SPE Core Test v3.0")
    print("=" * 50)
    
    all_passed = True
    for i, (test_data, description) in enumerate(test_cases):
        print(f"Test {i+1}: {description} ({len(test_data)} bytes)")
        
        try:
            # 変換
            transformed = spe.apply_transform(test_data)
            
            # 逆変換
            restored = spe.reverse_transform(transformed)
            
            # 検証
            success = test_data == restored
            print(f"  Result: {'✅ PASS' if success else '❌ FAIL'}")
            
            if not success:
                print(f"  Expected length: {len(test_data)}")
                print(f"  Actual length:   {len(restored)}")
                
                # 最初の違いを見つける
                min_len = min(len(test_data), len(restored))
                first_diff = -1
                for j in range(min_len):
                    if test_data[j] != restored[j]:
                        first_diff = j
                        break
                
                if first_diff >= 0:
                    print(f"  First difference at byte {first_diff}")
                    print(f"    Expected: {test_data[first_diff] if first_diff < len(test_data) else 'N/A'}")
                    print(f"    Actual:   {restored[first_diff] if first_diff < len(restored) else 'N/A'}")
                
                all_passed = False
        
        except Exception as e:
            print(f"  Result: ❌ ERROR - {e}")
            all_passed = False
    
    print("=" * 50)
    print(f"Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 SPE Core v3.0 is fully functional!")
        print("✅ Complete reversibility confirmed")
        print("✅ Ready for production use")
    
    return all_passed


def performance_test():
    """パフォーマンステスト"""
    import time
    
    spe = SPECore()
    
    print("\n⚡ Performance Test")
    print("=" * 30)
    
    # 大きなデータでのテスト
    large_data = bytes(range(256)) * 100  # 25.6KB
    
    # 変換時間測定
    start = time.time()
    transformed = spe.apply_transform(large_data)
    transform_time = time.time() - start
    
    # 逆変換時間測定
    start = time.time()
    restored = spe.reverse_transform(transformed)
    reverse_time = time.time() - start
    
    # 検証
    success = large_data == restored
    
    print(f"Data size: {len(large_data):,} bytes")
    print(f"Transform time: {transform_time:.4f} sec")
    print(f"Reverse time: {reverse_time:.4f} sec")
    print(f"Total time: {transform_time + reverse_time:.4f} sec")
    print(f"Throughput: {len(large_data) / (transform_time + reverse_time) / 1024:.1f} KB/sec")
    print(f"Correctness: {'✅ PASS' if success else '❌ FAIL'}")
    
    return success


if __name__ == "__main__":
    print("🚀 NXZip SPE Core v3.0 - Complete Rewrite")
    print("=" * 60)
    
    # 包括的テスト
    basic_success = comprehensive_test()
    
    # パフォーマンステスト
    if basic_success:
        perf_success = performance_test()
    else:
        perf_success = False
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    print(f"Functionality: {'✅ PASSED' if basic_success else '❌ FAILED'}")
    print(f"Performance:   {'✅ PASSED' if perf_success else '❌ FAILED'}")
    
    if basic_success and perf_success:
        print("\n🎉 SUCCESS! SPE Core v3.0 is ready for integration!")
    else:
        print("\n⚠️  Further debugging required.")
