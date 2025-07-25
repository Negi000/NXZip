#!/usr/bin/env python3
"""デバッグ用：文字列データでのブロック生成と復元の詳細プロセス表示"""

from nexus_advanced_engine import NexusAdvancedCompressor
import hashlib

# テストデータ
test_text = "Hello NEXUS Advanced Engine Test!"
test_data = test_text.encode('utf-16le')

print(f"Original text: {test_text}")
print(f"Original data length: {len(test_data)} bytes")
print(f"Original hex (first 32 bytes): {test_data[:32].hex()}")

# エンジンの初期化
compressor = NexusAdvancedCompressor()

# 圧縮（詳細ログ有効）
print("\n=== COMPRESSION ===")
compressed = compressor.compress(test_data, silent=False)
print(f"Compressed length: {len(compressed)} bytes")

# 展開（詳細ログ有効）
print("\n=== DECOMPRESSION ===")
decompressed = compressor.decompress(compressed, silent=False)
print(f"Decompressed length: {len(decompressed)} bytes")
print(f"Decompressed hex (first 32 bytes): {decompressed[:32].hex()}")

# 結果検証
print("\n=== VERIFICATION ===")
print(f"Length match: {len(test_data) == len(decompressed)}")
print(f"Data match: {test_data == decompressed}")
print(f"Original MD5: {hashlib.md5(test_data).hexdigest()}")
print(f"Decompressed MD5: {hashlib.md5(decompressed).hexdigest()}")

if test_data != decompressed:
    print("\n=== FIRST 10 BYTE DIFFERENCES ===")
    diff_count = 0
    for i in range(min(len(test_data), len(decompressed))):
        if test_data[i] != decompressed[i]:
            print(f"Byte {i}: original={test_data[i]} (0x{test_data[i]:02x}) != decompressed={decompressed[i]} (0x{decompressed[i]:02x})")
            diff_count += 1
            if diff_count >= 10:
                break
