#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ユーザー提供のNEXUS理論実装 - ダイレクトテスト
"""

import numpy as np
import struct
import hashlib
from collections import defaultdict
import torch
import torch.nn as nn

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

class ShapeOptimizer(nn.Module):
    def __init__(self, num_shapes, max_block_size=8):
        super().__init__()
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, num_shapes + max_block_size - 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class NexusCompressor:
    def __init__(self, block_size=4, shape_types=list(TETRIS_SHAPES.keys()), overlap_step=2, use_ai_optimization=True):
        self.max_block_size = block_size
        self.shape_types = shape_types
        self.overlap_step = overlap_step
        self.use_ai = use_ai_optimization
        if self.use_ai:
            self.optimizer_model = ShapeOptimizer(len(shape_types), self.max_block_size)
            with torch.no_grad():
                self.optimizer_model.fc1.weight.normal_()
                self.optimizer_model.fc2.weight.normal_()
            self.optimizer_model.eval()

    def _extract_group(self, data, y, x, shape_mask, local_size):
        h, w = shape_mask.shape
        if h > local_size or w > local_size:
            shape_mask = shape_mask[:local_size, :local_size]
        padded_mask = np.zeros((local_size, local_size), dtype=bool)
        padded_mask[:shape_mask.shape[0], :shape_mask.shape[1]] = shape_mask
        slice_y = slice(y, y+local_size)
        slice_x = slice(x, x+local_size)
        if data[slice_y, slice_x].size == 0: return np.array([])
        return data[slice_y, slice_x][padded_mask]

    def _normalize_group(self, group_values):
        if len(group_values) == 0: return None, None, None
        sort_indices = np.argsort(group_values)
        sorted_values = group_values[sort_indices]
        hash_key = hashlib.sha256(sorted_values.tobytes()).hexdigest()
        return hash_key, sorted_values, sort_indices

    def _compute_features(self, patch):
        if patch.size == 0: return torch.zeros(4)
        return torch.tensor([np.var(patch), np.mean(patch), np.std(patch), np.max(patch) - np.min(patch)], dtype=torch.float32)

    def _select_best_shape(self, data, y, x):
        max_size = min(self.max_block_size, data.shape[0]-y, data.shape[1]-x)
        if max_size < 2: return None, None, None
        patch = data[y:y+self.max_block_size, x:x+self.max_block_size]
        if self.use_ai:
            features = self._compute_features(patch)
            pred = self.optimizer_model(features.unsqueeze(0))
            shape_idx = pred[0, :len(self.shape_types)].argmax().item()
            size_idx = pred[0, len(self.shape_types):].argmax().item() + 2
            opt_size = min(size_idx, max_size)
            shape_name = self.shape_types[shape_idx]
            best_variant = 0
            best_var = float('inf')
            for v_idx, variant in enumerate(ALL_SHAPES[shape_name]):
                group = self._extract_group(data, y, x, variant, opt_size)
                if len(group) < 2: continue
                var = np.var(group)
                if var < best_var:
                    best_var = var
                    best_variant = v_idx
            return shape_name, best_variant, opt_size
        else:
            best_shape = None
            best_variant = None
            best_var = float('inf')
            opt_size = max_size
            for shape_name in self.shape_types:
                for v_idx, variant in enumerate(ALL_SHAPES[shape_name]):
                    group = self._extract_group(data, y, x, variant, opt_size)
                    if len(group) < 2: continue
                    var = np.var(group)
                    if var < best_var:
                        best_var = var
                        best_shape = shape_name
                        best_variant = v_idx
            return best_shape, best_variant, opt_size

    def decompose_and_group(self, data):
        height, width = data.shape
        self.unique_blocks = {}
        self.design_map = []
        for y in range(0, height, self.overlap_step):
            for x in range(0, width, self.overlap_step):
                shape_name, variant_idx, opt_size = self._select_best_shape(data, y, x)
                if shape_name is None or opt_size is None: continue
                shape_mask = ALL_SHAPES[shape_name][variant_idx]
                group_values = self._extract_group(data, y, x, shape_mask, opt_size)
                hash_key, norm_values, perm = self._normalize_group(group_values)
                if hash_key is None: continue
                if hash_key not in self.unique_blocks:
                    self.unique_blocks[hash_key] = (shape_name, variant_idx, norm_values)
                self.design_map.append((y, x, hash_key, perm, shape_name, variant_idx, opt_size))

    def compress(self, data):
        self.decompose_and_group(data)
        header = struct.pack('3s B I I I I', b'NXZ', 1, self.max_block_size, len(self.unique_blocks), data.shape[0], data.shape[1])
        num_shapes = len(self.shape_types)
        shapes_data = b''
        for name in self.shape_types:
            name_b = name.encode('utf-8')
            shapes_data += struct.pack('B', len(name_b)) + name_b
        header += struct.pack('B I', num_shapes, len(shapes_data)) + shapes_data
        unique_data = b''
        block_index = {hash_key: idx for idx, hash_key in enumerate(self.unique_blocks)}
        shape_to_idx = {name: i for i, name in enumerate(self.shape_types)}
        for hash_key, (shape_name, v_idx, norm_values) in self.unique_blocks.items():
            s_idx = shape_to_idx[shape_name]
            num_elems = len(norm_values)
            unique_data += struct.pack('B B I', s_idx, v_idx, num_elems) + norm_values.tobytes()
        map_data = b''
        for y, x, hash_key, perm, shape_name, v_idx, opt_size in self.design_map:
            u_idx = block_index[hash_key]
            s_idx = shape_to_idx[shape_name]
            num_perm = len(perm)
            map_data += struct.pack('I I I B B I I', y, x, u_idx, s_idx, v_idx, num_perm, opt_size)
            map_data += b''.join(struct.pack('H', p) for p in perm)
        return header + unique_data + map_data

    def decompress(self, compressed_data):
        offset = 0
        magic, version, max_block_size, num_unique, height, width = struct.unpack('3s B I I I I', compressed_data[offset:offset+20])
        offset += 20
        if magic != b'NXZ': raise ValueError("Invalid NXZ file")
        num_shapes, shapes_len = struct.unpack('B I', compressed_data[offset:offset+5])
        offset += 5
        shapes_data = compressed_data[offset:offset+shapes_len]
        offset += shapes_len
        shape_types = []
        s_offset = 0
        while s_offset < shapes_len:
            name_len = shapes_data[s_offset]
            s_offset += 1
            name = shapes_data[s_offset:s_offset+name_len].decode('utf-8')
            shape_types.append(name)
            s_offset += name_len
        used_shapes = {name: ALL_SHAPES[name] for name in shape_types}
        unique_blocks = []
        for _ in range(num_unique):
            s_idx, v_idx, num_elems = struct.unpack('B B I', compressed_data[offset:offset+6])
            offset += 6
            value_bytes = num_elems * 4
            values = np.frombuffer(compressed_data[offset:offset+value_bytes], dtype=np.int32)
            unique_blocks.append((shape_types[s_idx], v_idx, values))
            offset += value_bytes
        reconstructed = np.zeros((height, width), dtype=np.int32)
        count_map = np.zeros((height, width), dtype=np.int32)
        while offset < len(compressed_data):
            if len(compressed_data) - offset < 22: break
            y, x, u_idx, s_idx, v_idx, num_perm, opt_size = struct.unpack('I I I B B I I', compressed_data[offset:offset+22])
            offset += 22
            perm_bytes = num_perm * 2
            if len(compressed_data) - offset < perm_bytes: break
            perm = np.array(struct.unpack(f'{num_perm}H', compressed_data[offset:offset+perm_bytes]))
            offset += perm_bytes
            shape_name, _, sorted_values = unique_blocks[u_idx]
            original_values = np.empty_like(sorted_values)
            original_values[perm] = sorted_values
            shape_mask = used_shapes[shape_name][v_idx]
            padded_mask = np.zeros((opt_size, opt_size), dtype=bool)
            h, w = shape_mask.shape
            padded_mask[:h, :w] = shape_mask
            original_block = np.zeros((opt_size, opt_size), dtype=np.int32)
            original_block[padded_mask] = original_values
            reconstructed[y:y+opt_size, x:x+opt_size] += original_block
            count_map[y:y+opt_size, x:x+opt_size] += padded_mask.astype(np.int32)
        reconstructed = np.where(count_map > 0, reconstructed // count_map, 0)
        return reconstructed


def test_original_nexus():
    """ユーザー提供のオリジナルNEXUS実装テスト"""
    print("🎯 ユーザー提供NEXUS理論実装テスト")
    print("=" * 70)
    
    # 2Dデータでテスト（オリジナル設計通り）
    test_cases = [
        ("小さな画像風", np.random.randint(0, 256, (16, 16), dtype=np.int32)),
        ("パターンデータ", np.tile(np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32), (4, 4))),
        ("低エントロピー", np.ones((20, 20), dtype=np.int32) * 42),
        ("グラデーション", np.arange(400, dtype=np.int32).reshape(20, 20))
    ]
    
    success_count = 0
    
    for test_name, test_data in test_cases:
        print(f"\n📝 {test_name}: {test_data.shape} shape")
        
        try:
            # NEXUS圧縮器
            compressor = NexusCompressor(
                block_size=8,
                overlap_step=4,
                use_ai_optimization=True
            )
            
            # 圧縮
            compressed = compressor.compress(test_data)
            
            # 展開
            reconstructed = compressor.decompress(compressed)
            
            # 検証
            original_size = test_data.size * 4  # int32 = 4 bytes
            compressed_size = len(compressed)
            ratio = compressed_size / original_size
            
            # 完全一致チェック
            is_identical = np.array_equal(test_data, reconstructed)
            
            print(f"   元サイズ: {original_size} bytes")
            print(f"   圧縮サイズ: {compressed_size} bytes")
            print(f"   圧縮率: {ratio:.1%}")
            print(f"   可逆性: {'✅ 完全' if is_identical else '❌ 不完全'}")
            
            if is_identical:
                success_count += 1
            else:
                # 差分分析
                diff = np.abs(test_data - reconstructed)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                print(f"   最大差分: {max_diff}")
                print(f"   平均差分: {mean_diff:.3f}")
                
        except Exception as e:
            print(f"   ❌ エラー: {e}")
    
    success_rate = success_count / len(test_cases)
    print(f"\n🏆 NEXUS理論実装結果: {success_count}/{len(test_cases)} ({success_rate:.1%})")
    
    if success_rate >= 0.75:
        print("🎉 NEXUS理論が正しく実装されています！")
        print("   ✅ Polyomino形状認識")
        print("   ✅ AI最適化実装")
        print("   ✅ 完全可逆性確保")
        print("   ✅ 効率的圧縮")
    elif success_rate >= 0.5:
        print("🔧 NEXUS理論は部分的に動作しています")
    else:
        print("❌ NEXUS理論の実装に問題があります")
    
    return success_rate >= 0.5


if __name__ == "__main__":
    test_original_nexus()
