#!/usr/bin/env python3
"""
シンプルなデバッグ版
"""

import os
import hashlib
from nexus_advanced_engine import NexusAdvancedCompressor

# 小さなファイルをテスト
filename = "test_small.txt"
compressor = NexusAdvancedCompressor()

# ファイル読み込み
with open(filename, 'rb') as f:
    original_data = f.read()

print(f"Original data ({len(original_data)} bytes):")
print(f"  Hex: {original_data.hex()}")
print(f"  Text: {original_data.decode('utf-8', errors='ignore')}")
print(f"  MD5: {hashlib.md5(original_data).hexdigest()}")

# 圧縮
compressed_data = compressor.compress(original_data, silent=True)
print(f"\nCompressed data ({len(compressed_data)} bytes)")

# 解凍
decompressed_data = compressor.decompress(compressed_data, silent=True)
print(f"\nDecompressed data ({len(decompressed_data)} bytes):")
print(f"  Hex: {decompressed_data.hex()}")
print(f"  Text: {decompressed_data.decode('utf-8', errors='ignore')}")
print(f"  MD5: {hashlib.md5(decompressed_data).hexdigest()}")

# 比較
print(f"\n--- COMPARISON ---")
print(f"Length match: {len(original_data) == len(decompressed_data)}")
print(f"Hash match: {hashlib.md5(original_data).hexdigest() == hashlib.md5(decompressed_data).hexdigest()}")
print(f"Data match: {original_data == decompressed_data}")

# バイト単位の差分チェック
if len(original_data) == len(decompressed_data):
    for i, (o, d) in enumerate(zip(original_data, decompressed_data)):
        if o != d:
            print(f"Byte {i}: original={o} (0x{o:02x}) != decompressed={d} (0x{d:02x})")
else:
    print(f"Length mismatch: {len(original_data)} vs {len(decompressed_data)}")
