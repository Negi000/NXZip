#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEXUS理論 - オーバーラップ修正版
"""

import numpy as np
import struct
import hashlib
from collections import defaultdict
import torch
import torch.nn as nn
import os
from PIL import Image

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

class NexusCompressorFixed:
    def __init__(self, block_size=4, shape_types=list(TETRIS_SHAPES.keys()), overlap_step=2):
        self.max_block_size = block_size
        self.shape_types = shape_types
        self.overlap_step = overlap_step

    def _extract_group(self, data, y, x, shape_mask, local_size):
        h, w = shape_mask.shape
        if h > local_size or w > local_size:
            shape_mask = shape_mask[:local_size, :local_size]
        
        padded_mask = np.zeros((local_size, local_size), dtype=bool)
        padded_mask[:shape_mask.shape[0], :shape_mask.shape[1]] = shape_mask
        
        end_y = min(y + local_size, data.shape[0])
        end_x = min(x + local_size, data.shape[1])
        
        block = data[y:end_y, x:end_x]
        mask_region = padded_mask[:end_y-y, :end_x-x]
        
        return block[mask_region]

    def _normalize_group(self, group_values):
        if len(group_values) == 0: 
            return None, None, None
        sort_indices = np.argsort(group_values)
        sorted_values = group_values[sort_indices]
        hash_key = hashlib.sha256(sorted_values.tobytes()).hexdigest()[:16]
        return hash_key, sorted_values, sort_indices

    def _select_best_shape(self, data, y, x):
        max_size = min(self.max_block_size, data.shape[0]-y, data.shape[1]-x)
        if max_size < 2: 
            return None, None, None
        
        best_shape = None
        best_variant = None
        best_var = float('inf')
        
        for shape_name in self.shape_types:
            for v_idx, variant in enumerate(ALL_SHAPES[shape_name]):
                group = self._extract_group(data, y, x, variant, max_size)
                if len(group) < 2: 
                    continue
                var = np.var(group)
                if var < best_var:
                    best_var = var
                    best_shape = shape_name
                    best_variant = v_idx
        
        return best_shape, best_variant, max_size

    def decompose_and_group(self, data):
        height, width = data.shape
        self.unique_blocks = {}
        self.design_map = []
        
        # 非重複グリッドベースの分解に変更
        for y in range(0, height, self.max_block_size):
            for x in range(0, width, self.max_block_size):
                shape_name, variant_idx, opt_size = self._select_best_shape(data, y, x)
                if shape_name is None: 
                    continue
                
                shape_mask = ALL_SHAPES[shape_name][variant_idx]
                group_values = self._extract_group(data, y, x, shape_mask, opt_size)
                hash_key, norm_values, perm = self._normalize_group(group_values)
                
                if hash_key is None: 
                    continue
                    
                if hash_key not in self.unique_blocks:
                    self.unique_blocks[hash_key] = (shape_name, variant_idx, norm_values)
                
                self.design_map.append((y, x, hash_key, perm, shape_name, variant_idx, opt_size))

    def compress(self, data):
        self.decompose_and_group(data)
        
        header = struct.pack('3s B I I I I', b'NXZ', 1, self.max_block_size, len(self.unique_blocks), data.shape[0], data.shape[1])
        
        # 形状データ
        num_shapes = len(self.shape_types)
        shapes_data = b''
        for name in self.shape_types:
            name_b = name.encode('utf-8')
            shapes_data += struct.pack('B', len(name_b)) + name_b
        header += struct.pack('B 3x I', num_shapes, len(shapes_data)) + shapes_data
        
        # ユニークブロックデータ
        unique_data = b''
        block_index = {hash_key: idx for idx, hash_key in enumerate(self.unique_blocks)}
        shape_to_idx = {name: i for i, name in enumerate(self.shape_types)}
        
        for hash_key, (shape_name, v_idx, norm_values) in self.unique_blocks.items():
            s_idx = shape_to_idx[shape_name]
            num_elems = len(norm_values)
            unique_data += struct.pack('B B 2x I', s_idx, v_idx, num_elems) + norm_values.tobytes()
        
        # マップデータ
        map_data = b''
        for y, x, hash_key, perm, shape_name, v_idx, opt_size in self.design_map:
            u_idx = block_index[hash_key]
            s_idx = shape_to_idx[shape_name]
            num_perm = len(perm)
            map_data += struct.pack('I I I B B 2x I I', y, x, u_idx, s_idx, v_idx, num_perm, opt_size)
            map_data += b''.join(struct.pack('H', p) for p in perm)
        
        return header + unique_data + map_data

    def decompress(self, compressed_data):
        offset = 0
        
        # ヘッダー読み込み
        magic, version, max_block_size, num_unique, height, width = struct.unpack('3s B I I I I', compressed_data[offset:offset+20])
        offset += 20
        
        if magic != b'NXZ': 
            raise ValueError(f"不正なマジックナンバー: {magic}")
        
        # 形状データ読み込み
        num_shapes, shapes_len = struct.unpack('B 3x I', compressed_data[offset:offset+8])
        offset += 8
        
        shapes_data = compressed_data[offset:offset+shapes_len]
        offset += shapes_len
        
        # 形状名解析
        shape_types = []
        s_offset = 0
        while s_offset < shapes_len:
            if s_offset >= len(shapes_data):
                break
            name_len = shapes_data[s_offset]
            s_offset += 1
            if s_offset + name_len > len(shapes_data):
                break
            name = shapes_data[s_offset:s_offset+name_len].decode('utf-8')
            shape_types.append(name)
            s_offset += name_len
        
        used_shapes = {name: ALL_SHAPES[name] for name in shape_types}
        
        # ユニークブロック読み込み
        unique_blocks = []
        for i in range(num_unique):
            s_idx, v_idx, num_elems = struct.unpack('B B 2x I', compressed_data[offset:offset+8])
            offset += 8
            
            value_bytes = num_elems * 4
            values = np.frombuffer(compressed_data[offset:offset+value_bytes], dtype=np.int32)
            unique_blocks.append((shape_types[s_idx], v_idx, values))
            offset += value_bytes
        
        # 復元 - 重複なしの配置
        reconstructed = np.zeros((height, width), dtype=np.int32)
        
        # マップデータ処理
        while offset < len(compressed_data):
            if len(compressed_data) - offset < 24:
                break
            
            y, x, u_idx, s_idx, v_idx, num_perm, opt_size = struct.unpack('I I I B B 2x I I', compressed_data[offset:offset+24])
            offset += 24
            
            perm_bytes = num_perm * 2
            if len(compressed_data) - offset < perm_bytes: 
                break
                
            perm = np.array(struct.unpack(f'{num_perm}H', compressed_data[offset:offset+perm_bytes]))
            offset += perm_bytes
            
            # 値復元
            shape_name, _, sorted_values = unique_blocks[u_idx]
            original_values = np.empty_like(sorted_values)
            original_values[perm] = sorted_values
            
            # 形状マスク適用
            shape_mask = used_shapes[shape_name][v_idx]
            
            # ブロック配置（重複なし）
            end_y = min(y + opt_size, height)
            end_x = min(x + opt_size, width)
            
            block_height = end_y - y
            block_width = end_x - x
            
            # マスクを実際のブロックサイズに調整
            h, w = min(shape_mask.shape[0], block_height), min(shape_mask.shape[1], block_width)
            mask_region = shape_mask[:h, :w]
            
            # 値を直接配置
            block_region = reconstructed[y:y+h, x:x+w]
            block_region[mask_region] = original_values[:np.sum(mask_region)]
        
        return reconstructed

def test_fixed_compression():
    """修正版圧縮テスト"""
    print("🔍 NEXUS理論 - オーバーラップ修正版テスト")
    print("=" * 60)
    
    compressor = NexusCompressorFixed(block_size=8)
    
    test_files = [
        "red_simple.png",
        "small_test.png", 
        "medium_test.png"
    ]
    
    success_count = 0
    
    for filename in test_files:
        file_path = f"../../bin/{filename}"
        
        if not os.path.exists(file_path):
            print(f"❌ {filename}: ファイルが見つかりません")
            continue
            
        print(f"\n📁 テスト: {filename}")
        
        try:
            # 画像読み込み
            with Image.open(file_path) as img:
                img_array = np.array(img)
                
            if len(img_array.shape) == 3:
                img_array = img_array[:, :, 0]  # 赤チャンネル
                
            print(f"   画像サイズ: {img_array.shape}")
            print(f"   データ型: {img_array.dtype}")
            print(f"   値範囲: {img_array.min()} - {img_array.max()}")
            
            # 圧縮
            start_time = __import__('time').time()
            compressed = compressor.compress(img_array.astype(np.int32))
            compress_time = __import__('time').time() - start_time
            
            print(f"   圧縮時間: {compress_time:.3f}秒")
            print(f"   圧縮サイズ: {len(compressed)} bytes")
            
            # 圧縮率計算
            original_size = img_array.nbytes
            compression_ratio = original_size / len(compressed)
            print(f"   圧縮率: {compression_ratio:.2f}x")
            
            # 展開
            start_time = __import__('time').time()
            decompressed = compressor.decompress(compressed)
            decompress_time = __import__('time').time() - start_time
            
            print(f"   展開時間: {decompress_time:.3f}秒")
            print(f"   復元サイズ: {decompressed.shape}")
            
            # 一致確認
            original_int32 = img_array.astype(np.int32)
            is_identical = np.array_equal(original_int32, decompressed)
            
            if is_identical:
                print(f"   ✅ 完全一致!")
                success_count += 1
            else:
                diff = np.abs(original_int32 - decompressed)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                print(f"   ❌ 不一致 - 最大差分: {max_diff}, 平均差分: {mean_diff:.3f}")
                
                # 詳細分析
                zero_count = np.sum(decompressed == 0)
                total_pixels = decompressed.size
                print(f"   ゼロ値ピクセル: {zero_count}/{total_pixels} ({100*zero_count/total_pixels:.1f}%)")
                
                # 一致するピクセル数
                match_count = np.sum(original_int32 == decompressed)
                print(f"   一致ピクセル: {match_count}/{total_pixels} ({100*match_count/total_pixels:.1f}%)")
        
        except Exception as e:
            print(f"   ❌ エラー: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n📊 最終結果: {success_count}/{len(test_files)} 成功")
    
if __name__ == "__main__":
    test_fixed_compression()
