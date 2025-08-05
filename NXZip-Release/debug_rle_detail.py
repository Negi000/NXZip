#!/usr/bin/env python3
"""
RLE詳細デバッグ
"""

import sys
from nxzip_core import TMCEngine, CompressionMode, NXZipCore

def debug_rle_step_by_step():
    """RLEを1バイトずつデバッグ"""
    
    # 問題のあるデータを小さく再現
    test_data = bytearray()
    test_data.extend(b'MZ')  
    test_data.extend(b'\x00' * 10)  # 10個の0x00の繰り返し
    test_data.extend(b'PE')
    test_data.extend(b'\x90' * 8)   # 8個のNOPの繰り返し  
    test_data.extend(b'\xFE\xFE\xFE\xFE')  # 4個の0xFE
    test_data.extend(b'END')
    
    original = bytes(test_data)
    print(f"🔍 RLE詳細デバッグ")
    print(f"元データ ({len(original)} bytes): {original.hex()}")
    
    # TMCEngine初期化
    tmc_engine = TMCEngine(CompressionMode.FAST)
    core = NXZipCore()
    
    # 圧縮
    compressed = tmc_engine._reduce_redundancy(original)
    print(f"圧縮後 ({len(compressed)} bytes): {compressed.hex()}")
    
    # 手動解析
    print(f"\n📋 圧縮データ解析:")
    i = 0
    while i < len(compressed):
        byte_val = compressed[i]
        if byte_val == 0xFE and i + 1 < len(compressed):
            if i + 1 < len(compressed) and compressed[i + 1] == 0x00:
                print(f"  位置{i}: エスケープ 0xFE -> 単一の 0xFE")
                i += 2
            elif i + 2 < len(compressed):
                rle_byte = compressed[i + 1]
                rle_count = compressed[i + 2]
                print(f"  位置{i}: RLE 0x{rle_byte:02x} x {rle_count}")
                i += 3
            else:
                print(f"  位置{i}: 不完全シーケンス 0x{byte_val:02x}")
                i += 1
        else:
            print(f"  位置{i}: 通常バイト 0x{byte_val:02x}")
            i += 1
    
    # 復元
    restored = core._restore_redundancy(compressed)
    print(f"\n復元後 ({len(restored)} bytes): {restored.hex()}")
    
    # 比較
    print(f"\n🔍 比較結果:")
    print(f"元データ: {original.hex()}")
    print(f"復元後:   {restored.hex()}")
    print(f"一致: {'✅' if original == restored else '❌'}")
    
    if original != restored:
        print(f"\n❌ 相違点:")
        min_len = min(len(original), len(restored))
        for i in range(min_len):
            if original[i] != restored[i]:
                print(f"  位置{i}: 元=0x{original[i]:02x} 復元=0x{restored[i]:02x}")
        
        if len(original) != len(restored):
            print(f"  サイズ違い: 元={len(original)} 復元={len(restored)}")

def main():
    debug_rle_step_by_step()

if __name__ == "__main__":
    main()
