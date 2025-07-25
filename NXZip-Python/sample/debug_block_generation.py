#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ブロック生成のデバッグ専用スクリプト"""

import math

# 形状定義
POLYOMINO_SHAPES = {
    "I-1": ((0, 0),),
    "I-2": ((0, 0), (0, 1)),
    "I-3": ((0, 0), (0, 1), (0, 2)),
    "I-4": ((0, 0), (0, 1), (0, 2), (0, 3)),
    "O-4": ((0, 0), (0, 1), (1, 0), (1, 1)),
}

def debug_block_generation(data_bytes, shape_name):
    """ブロック生成の詳細デバッグ"""
    data = list(data_bytes)
    shape_coords = POLYOMINO_SHAPES[shape_name]
    
    print(f"=== DEBUG: {data} with {shape_name} ===")
    print(f"Data: {data} (len={len(data)})")
    print(f"Shape: {shape_coords}")
    
    # グリッド計算
    data_len = len(data)
    grid_width = math.ceil(math.sqrt(data_len))
    rows = data_len // grid_width
    print(f"Grid: width={grid_width}, rows={rows}, total_cells={grid_width*rows}")
    
    # 形状サイズ
    shape_width = max(c for r, c in shape_coords) + 1
    shape_height = max(r for r, c in shape_coords) + 1
    print(f"Shape size: width={shape_width}, height={shape_height}")
    
    # ブロック生成可能範囲
    max_r = rows - shape_height + 1
    max_c = grid_width - shape_width + 1
    print(f"Block generation range: r=[0, {max_r}), c=[0, {max_c})")
    
    if max_r <= 0 or max_c <= 0:
        print("❌ NO BLOCKS CAN BE GENERATED!")
        print(f"   Reason: max_r={max_r}, max_c={max_c}")
        return []
    
    # 実際のブロック生成
    blocks = []
    print(f"Attempting to generate blocks...")
    
    for r in range(max_r):
        for c in range(max_c):
            block = []
            valid_block = True
            
            print(f"  Position r={r}, c={c}:")
            base_idx = r * grid_width + c
            for dr, dc in shape_coords:
                idx = base_idx + dr * grid_width + dc
                print(f"    ({dr},{dc}) -> idx={idx}", end="")
                if idx >= data_len:
                    print(" ❌ OUT OF BOUNDS")
                    valid_block = False
                    break
                else:
                    print(f" ✅ data[{idx}]={data[idx]}")
                    block.append(data[idx])
            
            if valid_block:
                print(f"    → Block: {block}")
                blocks.append(tuple(block))
            else:
                print(f"    → Invalid block")
    
    print(f"Generated {len(blocks)} blocks: {blocks}")
    return blocks

# テストケース
print("Testing various data sizes with different shapes:")
print()

test_cases = [
    (b'A', 'I-1'),
    (b'AB', 'I-2'), 
    (b'ABC', 'I-3'),
    (b'ABC', 'I-1'),
    (b'ABC', 'I-2'),
    (b'ABCD', 'I-4'),
    (b'ABCD', 'O-4'),
]

for data, shape in test_cases:
    debug_block_generation(data, shape)
    print()
