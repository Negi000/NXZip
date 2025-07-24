#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEXUS理論実装デバッグ - バイナリ構造修正
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

def debug_compress_simple():
    """簡単なデータで圧縮・展開をデバッグ"""
    print("🔍 NEXUS理論実装デバッグ")
    print("=" * 50)
    
    # 非常にシンプルなテストデータ
    test_data = np.array([[1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [1, 2, 3, 4],
                         [5, 6, 7, 8]], dtype=np.int32)
    
    print(f"テストデータ: {test_data.shape}")
    print(f"データ:\n{test_data}")
    
    # 圧縮器作成（AI無効で単純化）
    compressor = NexusCompressor(
        block_size=4,
        overlap_step=2,
        use_ai_optimization=False  # AI無効でデバッグ
    )
    
    print(f"\n1. 圧縮処理...")
    try:
        compressed = compressor.compress(test_data)
        print(f"   圧縮成功: {len(compressed)} bytes")
        
        # バイナリ構造を詳細解析
        print(f"\n2. バイナリ構造解析:")
        print(f"   最初の50バイト: {compressed[:50]}")
        
        # ヘッダー解析
        try:
            magic, version, max_block_size, num_unique, height, width = struct.unpack('3s B I I I I', compressed[:20])
            print(f"   マジック: {magic}")
            print(f"   バージョン: {version}")
            print(f"   最大ブロックサイズ: {max_block_size}")
            print(f"   ユニークブロック数: {num_unique}")
            print(f"   高さ x 幅: {height} x {width}")
            
            offset = 20
            num_shapes, shapes_len = struct.unpack('B 3x I', compressed[offset:offset+8])  # パディング対応
            print(f"   形状数: {num_shapes}")
            print(f"   形状データ長: {shapes_len}")
            
        except struct.error as e:
            print(f"   ヘッダー解析エラー: {e}")
        
        print(f"\n3. 展開処理...")
        try:
            reconstructed = compressor.decompress(compressed)
            print(f"   展開成功: {reconstructed.shape}")
            print(f"   復元データ:\n{reconstructed}")
            
            # 一致確認
            is_identical = np.array_equal(test_data, reconstructed)
            print(f"   完全一致: {is_identical}")
            
            if not is_identical:
                diff = test_data - reconstructed
                print(f"   差分:\n{diff}")
            
        except Exception as e:
            print(f"   展開エラー: {e}")
            print(f"   詳細: {type(e).__name__}")
            
            # エラー位置の詳細分析
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        print(f"   圧縮エラー: {e}")
        import traceback
        traceback.print_exc()


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
        hash_key = hashlib.sha256(sorted_values.tobytes()).hexdigest()[:16]  # 短縮
        return hash_key, sorted_values, sort_indices

    def _compute_features(self, patch):
        if patch.size == 0: return torch.zeros(4)
        return torch.tensor([np.var(patch), np.mean(patch), np.std(patch), np.max(patch) - np.min(patch)], dtype=torch.float32)

    def _select_best_shape(self, data, y, x):
        max_size = min(self.max_block_size, data.shape[0]-y, data.shape[1]-x)
        if max_size < 2: return None, None, None
        
        if self.use_ai:
            patch = data[y:y+min(self.max_block_size, data.shape[0]-y), x:x+min(self.max_block_size, data.shape[1]-x)]
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
            # 非AI版（デバッグ用）
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
        
        # ヘッダー構築
        header = struct.pack('3s B I I I I', b'NXZ', 1, self.max_block_size, len(self.unique_blocks), data.shape[0], data.shape[1])
        
        # 形状データ
        num_shapes = len(self.shape_types)
        shapes_data = b''
        for name in self.shape_types:
            name_b = name.encode('utf-8')
            shapes_data += struct.pack('B', len(name_b)) + name_b
        header += struct.pack('B 3x I', num_shapes, len(shapes_data)) + shapes_data  # パディング追加
        
        # ユニークブロックデータ
        unique_data = b''
        block_index = {hash_key: idx for idx, hash_key in enumerate(self.unique_blocks)}
        shape_to_idx = {name: i for i, name in enumerate(self.shape_types)}
        for hash_key, (shape_name, v_idx, norm_values) in self.unique_blocks.items():
            s_idx = shape_to_idx[shape_name]
            num_elems = len(norm_values)
            unique_data += struct.pack('B B 2x I', s_idx, v_idx, num_elems) + norm_values.tobytes()  # パディング追加
        
        # マップデータ
        map_data = b''
        for y, x, hash_key, perm, shape_name, v_idx, opt_size in self.design_map:
            u_idx = block_index[hash_key]
            s_idx = shape_to_idx[shape_name]
            num_perm = len(perm)
            map_data += struct.pack('I I I B B 2x I I', y, x, u_idx, s_idx, v_idx, num_perm, opt_size)  # パディング追加
            map_data += b''.join(struct.pack('H', p) for p in perm)
        
        return header + unique_data + map_data

    def decompress(self, compressed_data):
        offset = 0
        
        # ヘッダー読み込み
        if len(compressed_data) < 20:
            raise ValueError(f"データが短すぎます: {len(compressed_data)} bytes")
        
        magic, version, max_block_size, num_unique, height, width = struct.unpack('3s B I I I I', compressed_data[offset:offset+20])
        offset += 20
        
        if magic != b'NXZ': 
            raise ValueError(f"不正なマジックナンバー: {magic}")
        
        # 形状データ読み込み
        if len(compressed_data) < offset + 8:
            raise ValueError(f"形状ヘッダーが不足: offset={offset}")
        
        num_shapes, shapes_len = struct.unpack('B 3x I', compressed_data[offset:offset+8])  # パディング読み取り
        offset += 8
        
        if len(compressed_data) < offset + shapes_len:
            raise ValueError(f"形状データが不足: offset={offset}, shapes_len={shapes_len}")
        
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
            if len(compressed_data) < offset + 8:
                raise ValueError(f"ユニークブロック{i}のヘッダーが不足")
            
            s_idx, v_idx, num_elems = struct.unpack('B B 2x I', compressed_data[offset:offset+8])  # パディング対応
            offset += 8
            
            value_bytes = num_elems * 4
            if len(compressed_data) < offset + value_bytes:
                raise ValueError(f"ユニークブロック{i}のデータが不足")
            
            values = np.frombuffer(compressed_data[offset:offset+value_bytes], dtype=np.int32)
            unique_blocks.append((shape_types[s_idx], v_idx, values))
            offset += value_bytes
        
        # 復元
        reconstructed = np.zeros((height, width), dtype=np.int32)
        count_map = np.zeros((height, width), dtype=np.int32)
        
        # マップデータ処理
        while offset < len(compressed_data):
            if len(compressed_data) - offset < 24:  # パディング分更新
                break
            
            y, x, u_idx, s_idx, v_idx, num_perm, opt_size = struct.unpack('I I I B B 2x I I', compressed_data[offset:offset+24])  # パディング対応
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
            padded_mask = np.zeros((opt_size, opt_size), dtype=bool)
            h, w = min(shape_mask.shape[0], opt_size), min(shape_mask.shape[1], opt_size)
            padded_mask[:h, :w] = shape_mask[:h, :w]  # サイズ制限
            
            # ブロック配置
            original_block = np.zeros((opt_size, opt_size), dtype=np.int32)
            original_block[padded_mask] = original_values
            
            # 重複領域の加算
            end_y = min(y + opt_size, height)
            end_x = min(x + opt_size, width)
            reconstructed[y:end_y, x:end_x] += original_block[:end_y-y, :end_x-x]
            count_map[y:end_y, x:end_x] += padded_mask[:end_y-y, :end_x-x].astype(np.int32)
        
        # 平均化処理
        reconstructed = np.where(count_map > 0, reconstructed // count_map, 0)
        return reconstructed


if __name__ == "__main__":
    debug_compress_simple()
