#!/usr/bin/env python3
"""NEXUSの理論テスト：統合処理を無効にした純粋な形状変換のみ"""

from nexus_advanced_engine import NexusAdvancedCompressor
import hashlib

# テストデータ
test_text = "Hello NEXUS!"
test_data = test_text.encode('utf-16le')

print(f"Original text: {test_text}")
print(f"Original data: {list(test_data)}")
print(f"Original hex: {test_data.hex()}")

# 統合処理を無効にしたエンジン
compressor = NexusAdvancedCompressor(use_ai=False)  # AIも無効にして単純化

# エンジンの統合機能を一時的に無効化
original_consolidate = compressor._consolidate_by_elements

def no_consolidation(normalized_groups, show_progress=False):
    """統合処理をスキップして元のグループをそのまま返す"""
    print("   [DEBUG] Consolidation DISABLED for testing")
    # 統合マップは恒等変換
    consolidation_map = {}
    for normalized, group_id in normalized_groups.items():
        consolidation_map[group_id] = {
            'new_group_id': group_id,
            'consolidation_type': 'none',
            'canonical_form': normalized
        }
    return normalized_groups, consolidation_map

# 統合機能を無効化
compressor._consolidate_by_elements = no_consolidation

print("\n=== COMPRESSION (No Consolidation) ===")
compressed = compressor.compress(test_data, silent=False)
print(f"Compressed length: {len(compressed)} bytes")

print("\n=== DECOMPRESSION (No Consolidation) ===")
decompressed = compressor.decompress(compressed, silent=False)

print(f"\n=== RESULTS ===")
print(f"Original:    {list(test_data)}")
print(f"Decompressed: {list(decompressed)}")
print(f"Match: {test_data == decompressed}")
print(f"Original MD5: {hashlib.md5(test_data).hexdigest()}")
print(f"Decompressed MD5: {hashlib.md5(decompressed).hexdigest()}")

if test_data != decompressed:
    print("\n=== DIFFERENCES ===")
    for i in range(min(len(test_data), len(decompressed))):
        if test_data[i] != decompressed[i]:
            print(f"Byte {i}: original={test_data[i]} != decompressed={decompressed[i]}")
            if i >= 5:  # 最初の5つの違いのみ表示
                break
