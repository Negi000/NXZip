#!/usr/bin/env python3
"""NEXUS理論完全版：統合処理も完全可逆化したテスト"""

from nexus_advanced_engine import NexusAdvancedCompressor
import hashlib

# テストデータ
test_text = "NEXUS INFECTED!"
test_data = test_text.encode('utf-16le')

print(f"🔥 NEXUS THEORY FULL INFECTION TEST")
print(f"===================================")
print(f"Original text: {test_text}")
print(f"Original data: {list(test_data)}")
print(f"Original hex: {test_data.hex()}")

# NEXUS感染エンジン
compressor = NexusAdvancedCompressor(use_ai=True)

print(f"\n=== NEXUS COMPRESSION (Infected) ===")
compressed = compressor.compress(test_data, silent=False)
print(f"Compressed length: {len(compressed)} bytes")

print(f"\n=== NEXUS DECOMPRESSION (Infected) ===")
decompressed = compressor.decompress(compressed, silent=False)

print(f"\n=== NEXUS RESULTS ===")
print(f"Original:    {list(test_data)}")
print(f"Decompressed: {list(decompressed)}")
print(f"🎯 Perfect Match: {test_data == decompressed}")
print(f"Original MD5: {hashlib.md5(test_data).hexdigest()}")
print(f"Decompressed MD5: {hashlib.md5(decompressed).hexdigest()}")

if test_data == decompressed:
    print(f"\n🚀 NEXUS THEORY FULLY INFECTED! 🚀")
    print(f"   完全可逆性を達成しました！")
    print(f"   圧縮の逆が解凍です！")
else:
    print(f"\n⚠️  NEXUS INFECTION INCOMPLETE")
    print(f"   Further infection required...")
    # 最初の10個の違いを表示
    diff_count = 0
    for i in range(min(len(test_data), len(decompressed))):
        if test_data[i] != decompressed[i]:
            print(f"   Byte {i}: original={test_data[i]} != decompressed={decompressed[i]}")
            diff_count += 1
            if diff_count >= 10:
                break
