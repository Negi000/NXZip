#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEXUS理論 - 詳細デバッグ版
"""

import numpy as np
import struct
import hashlib
from collections import defaultdict

# Tetris shapes
TETRIS_SHAPES = {
    'I': np.array([[1,1,1,1]], dtype=bool),
    'O': np.array([[1,1],[1,1]], dtype=bool),
    'T': np.array([[0,1,0],[1,1,1]], dtype=bool),
    'J': np.array([[1,0,0],[1,1,1]], dtype=bool),
    'L': np.array([[0,0,1],[1,1,1]], dtype=bool),
    'S': np.array([[0,1,1],[1,1,0]], dtype=bool),
    'Z': np.array([[1,1,0],[0,1,1]], dtype=bool)
}

def generate_shape_variants(shape_mask):
    variants = []
    for rot in range(4):
        rotated = np.rot90(shape_mask, rot)
        variants.append(rotated)
        mirrored = np.fliplr(rotated)
        variants.append(mirrored)
    unique = {}
    for v in variants:
        positions = frozenset(tuple(pos) for pos in np.argwhere(v))
        unique[positions] = v
    return list(unique.values())

ALL_SHAPES = {name: generate_shape_variants(mask) for name, mask in TETRIS_SHAPES.items()}

class NexusDebugger:
    def __init__(self, block_size=4):
        self.max_block_size = block_size
        self.shape_types = list(TETRIS_SHAPES.keys())

    def debug_simple_case(self):
        """シンプルなケースで詳細デバッグ"""
        print("🔍 NEXUS理論 - 詳細デバッグ")
        print("=" * 50)
        
        # 非常にシンプルなテストケース
        test_data = np.array([[100, 200],
                             [300, 400]], dtype=np.int32)
        
        print(f"テストデータ:\n{test_data}")
        print(f"形状: {test_data.shape}")
        
        # ブロック分解を手動で実行
        print(f"\n1. ブロック分解詳細:")
        y, x = 0, 0
        shape_name = 'O'  # 2x2の正方形
        variant_idx = 0
        shape_mask = ALL_SHAPES[shape_name][variant_idx]
        
        print(f"   選択した形状: {shape_name}, バリアント: {variant_idx}")
        print(f"   形状マスク:\n{shape_mask}")
        
        # グループ抽出
        group_values = test_data[shape_mask]
        print(f"   抽出値: {group_values}")
        
        # 正規化
        sort_indices = np.argsort(group_values)
        sorted_values = group_values[sort_indices]
        print(f"   ソート後: {sorted_values}")
        print(f"   ソートインデックス: {sort_indices}")
        
        # ハッシュキー生成
        hash_key = hashlib.sha256(sorted_values.tobytes()).hexdigest()[:16]
        print(f"   ハッシュキー: {hash_key}")
        
        print(f"\n2. 圧縮データ構造:")
        # ユニークブロック
        unique_blocks = {hash_key: (shape_name, variant_idx, sorted_values)}
        design_map = [(y, x, hash_key, sort_indices, shape_name, variant_idx, self.max_block_size)]
        
        print(f"   ユニークブロック: {unique_blocks}")
        print(f"   設計マップ: {design_map}")
        
        print(f"\n3. 復元プロセス:")
        # 復元
        reconstructed = np.zeros_like(test_data)
        
        # マップエントリから復元
        for y, x, hash_key, perm, shape_name, v_idx, opt_size in design_map:
            print(f"   復元中: y={y}, x={x}, shape={shape_name}")
            
            # 値復元
            shape_name_stored, _, sorted_values = unique_blocks[hash_key]
            original_values = np.empty_like(sorted_values)
            original_values[perm] = sorted_values
            
            print(f"     ソート済み値: {sorted_values}")
            print(f"     順列: {perm}")
            print(f"     復元値: {original_values}")
            
            # 形状マスク適用
            shape_mask = ALL_SHAPES[shape_name][v_idx]
            print(f"     形状マスク:\n{shape_mask}")
            
            # 復元配置
            end_y = min(y + opt_size, reconstructed.shape[0])
            end_x = min(x + opt_size, reconstructed.shape[1])
            
            print(f"     配置範囲: [{y}:{end_y}, {x}:{end_x}]")
            
            # 実際の配置サイズ
            actual_h = end_y - y
            actual_w = end_x - x
            mask_h = min(shape_mask.shape[0], actual_h)
            mask_w = min(shape_mask.shape[1], actual_w)
            
            print(f"     実際サイズ: {actual_h}x{actual_w}")
            print(f"     マスクサイズ: {mask_h}x{mask_w}")
            
            # マスク領域
            mask_region = shape_mask[:mask_h, :mask_w]
            print(f"     使用マスク:\n{mask_region}")
            
            # 値の配置
            block_region = reconstructed[y:y+mask_h, x:x+mask_w]
            print(f"     配置前ブロック:\n{block_region}")
            
            # インデックス確認
            mask_indices = np.where(mask_region)
            num_mask_pixels = len(mask_indices[0])
            print(f"     マスクピクセル数: {num_mask_pixels}")
            print(f"     値数: {len(original_values)}")
            
            if num_mask_pixels <= len(original_values):
                block_region[mask_region] = original_values[:num_mask_pixels]
            
            print(f"     配置後ブロック:\n{block_region}")
        
        print(f"\n4. 最終結果:")
        print(f"   復元データ:\n{reconstructed}")
        print(f"   元データ:\n{test_data}")
        print(f"   一致: {np.array_equal(test_data, reconstructed)}")
        
        if not np.array_equal(test_data, reconstructed):
            diff = test_data - reconstructed
            print(f"   差分:\n{diff}")

if __name__ == "__main__":
    debugger = NexusDebugger()
    debugger.debug_simple_case()
