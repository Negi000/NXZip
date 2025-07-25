#!/usr/bin/env python3
"""簡単なブロック生成テスト"""

from nexus_advanced_engine import NexusAdvancedCompressor, POLYOMINO_SHAPES

# H-7形状の確認
h7_coords = POLYOMINO_SHAPES['H-7']
print(f"H-7 coordinates: {h7_coords}")

# シンプルなテストデータ（1-21の連続値）
test_data = bytes(range(1, 22))  # 1, 2, 3, ..., 21 (21 bytes)
print(f"Test data: {list(test_data)}")
print(f"Test data length: {len(test_data)} bytes")

# グリッドサイズを設定（3x7 = 21）
grid_width = 3
grid_height = 7

print(f"Grid: {grid_width} x {grid_height}")

# H-7形状の場合
shape_width = max(c for r, c in h7_coords) + 1  # 3
shape_height = max(r for r, c in h7_coords) + 1  # 3

print(f"Shape size: {shape_width} x {shape_height}")

# ブロック生成可能位置を計算
blocks_per_row = grid_width - shape_width + 1  # 3 - 3 + 1 = 1
blocks_per_col = grid_height - shape_height + 1  # 7 - 3 + 1 = 5

print(f"Possible block positions: {blocks_per_row} x {blocks_per_col} = {blocks_per_row * blocks_per_col}")

# 手動でブロック生成をシミュレート
print("\nManual block generation:")
for r in range(blocks_per_col):
    for c in range(blocks_per_row):
        block = []
        base_idx = r * grid_width + c
        print(f"Block at position ({r}, {c}), base_idx={base_idx}")
        
        for dr, dc in h7_coords:
            idx = base_idx + dr * grid_width + dc
            if idx < len(test_data):
                value = test_data[idx]
                block.append(value)
                print(f"  coord({dr},{dc}) -> idx={idx} -> value={value}")
            else:
                print(f"  coord({dr},{dc}) -> idx={idx} -> OUT OF BOUNDS")
                
        print(f"  Block result: {block}")
        print()

# 実際のエンジンでテスト
print("Engine test:")
compressor = NexusAdvancedCompressor()
compressed = compressor.compress(test_data, silent=True)
decompressed = compressor.decompress(compressed, silent=True)

print(f"Original:    {list(test_data)}")
print(f"Decompressed: {list(decompressed)}")
print(f"Match: {test_data == decompressed}")
