#!/usr/bin/env python3
"""
デバッグ用の簡単なSPEテスト
問題を特定して修正します
"""

import struct
from typing import List, Tuple


def debug_spe_transform(data: bytes) -> bytes:
    """デバッグ用のSPE変換"""
    print(f"\n🔍 Debug SPE Transform for {len(data)} bytes")
    result = bytearray(data)
    
    # 1. 構造保持パディング
    original_len = len(result)
    print(f"Step 1: Original length = {original_len}")
    
    padded_len = ((original_len + 15) // 16) * 16  # 16バイト境界
    result.extend(b'\x00' * (padded_len - original_len))
    print(f"Step 1: After padding = {len(result)} bytes")
    
    # 元の長さを末尾に記録（8バイト）
    result.extend(struct.pack('<Q', original_len))
    print(f"Step 1: After length info = {len(result)} bytes")
    
    # 2. ブロックシャッフル（16バイトブロック）
    if len(result) >= 32:
        print(f"Step 2: Applying block shuffle...")
        apply_block_shuffle_debug(result)
    else:
        print(f"Step 2: Skipping block shuffle (too small)")
        
    # 3. XOR難読化
    print(f"Step 3: Applying XOR obfuscation...")
    xor_key = b"NXZip_SPE_2024"
    for i in range(len(result)):
        result[i] ^= xor_key[i % len(xor_key)]
    
    print(f"Final result: {len(result)} bytes")
    return bytes(result)


def debug_spe_reverse(data: bytes) -> bytes:
    """デバッグ用のSPE逆変換"""
    print(f"\n🔍 Debug SPE Reverse for {len(data)} bytes")
    result = bytearray(data)
    
    # 1. XOR除去
    print(f"Step 1: Removing XOR obfuscation...")
    xor_key = b"NXZip_SPE_2024"
    for i in range(len(result)):
        result[i] ^= xor_key[i % len(xor_key)]
    
    # 2. ブロックシャッフル逆変換
    if len(result) >= 32:
        print(f"Step 2: Reversing block shuffle...")
        reverse_block_shuffle_debug(result)
    else:
        print(f"Step 2: Skipping block shuffle reverse (too small)")
        
    # 3. パディング除去
    print(f"Step 3: Removing padding...")
    if len(result) >= 8:
        # 末尾8バイトから元の長さを取得
        length_bytes = result[-8:]
        original_len = struct.unpack('<Q', length_bytes)[0]
        print(f"Step 3: Original length from data = {original_len}")
        
        # 長さ情報を除去
        result = result[:-8]
        print(f"Step 3: After removing length info = {len(result)} bytes")
        
        # 元のサイズに切り詰め
        if original_len <= len(result):
            result = result[:original_len]
            print(f"Step 3: After truncation = {len(result)} bytes")
        else:
            print(f"ERROR: original_len ({original_len}) > current length ({len(result)})")
    
    print(f"Final result: {len(result)} bytes")
    return bytes(result)


def apply_block_shuffle_debug(data: bytearray) -> None:
    """デバッグ版ブロックシャッフル"""
    block_size = 16
    num_blocks = len(data) // block_size
    print(f"  Block shuffle: {num_blocks} blocks of {block_size} bytes each")
    
    swaps_made = 0
    for i in range(num_blocks):
        swap_with = (i * 7 + 3) % num_blocks
        if i != swap_with:
            start1 = i * block_size
            start2 = swap_with * block_size
            
            print(f"  Swapping block {i} (pos {start1}) with block {swap_with} (pos {start2})")
            
            # ブロック単位でスワップ
            for j in range(block_size):
                if start1 + j < len(data) and start2 + j < len(data):
                    data[start1 + j], data[start2 + j] = data[start2 + j], data[start1 + j]
            
            swaps_made += 1
    
    print(f"  Total swaps made: {swaps_made}")


def reverse_block_shuffle_debug(data: bytearray) -> None:
    """デバッグ版ブロックシャッフル逆変換"""
    print(f"  Reversing block shuffle...")
    apply_block_shuffle_debug(data)  # 自己逆変換


def test_simple_case():
    """簡単なケースでテスト"""
    print("Testing simple case...")
    
    test_data = b"Hello"
    print(f"Original: {test_data}")
    
    transformed = debug_spe_transform(test_data)
    print(f"Transformed hex: {transformed.hex()}")
    
    restored = debug_spe_reverse(transformed)
    print(f"Restored: {restored}")
    
    success = test_data == restored
    print(f"Success: {success}")
    
    return success


def test_medium_case():
    """中程度のケースでテスト"""
    print("\n" + "="*50)
    print("Testing medium case...")
    
    test_data = bytes(range(100))  # 100バイトのデータ
    print(f"Original: {len(test_data)} bytes")
    
    transformed = debug_spe_transform(test_data)
    print(f"Transformed: {len(transformed)} bytes")
    
    restored = debug_spe_reverse(transformed)
    print(f"Restored: {len(restored)} bytes")
    
    success = test_data == restored
    print(f"Success: {success}")
    
    if not success:
        # 差分を詳細に調べる
        print("Analyzing differences...")
        min_len = min(len(test_data), len(restored))
        differences = 0
        for i in range(min_len):
            if test_data[i] != restored[i]:
                differences += 1
                if differences <= 10:  # 最初の10個の差分を表示
                    print(f"  Diff at {i}: original={test_data[i]}, restored={restored[i]}")
        
        if differences > 10:
            print(f"  ... and {differences - 10} more differences")
        
        print(f"Total differences: {differences}")
    
    return success


if __name__ == "__main__":
    print("🔧 NXZip SPE Debug Test")
    print("="*50)
    
    # 簡単なケースをテスト
    simple_success = test_simple_case()
    
    # 中程度のケースをテスト
    medium_success = test_medium_case()
    
    print("\n" + "="*50)
    print("Summary:")
    print(f"Simple case: {'✅ PASS' if simple_success else '❌ FAIL'}")
    print(f"Medium case: {'✅ PASS' if medium_success else '❌ FAIL'}")
