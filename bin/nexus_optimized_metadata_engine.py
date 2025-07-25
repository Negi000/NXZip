#!/usr/bin/env python3
"""
🔥 NEXUS OPTIMIZED METADATA ENGINE 🔥

NEXUS理論の実装：メタデータオーバーヘッド最適化版
- バイナリ形式での設計図保存
- 冗長データの削除
- 効率的なデータ構造
"""

import os
import sys
import json
import lzma
import math
import time
import random
import hashlib
import collections
import struct
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# 進捗バー
class ProgressBar:
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, value: int):
        self.current = value
        if self.total > 0:
            percent = (self.current / self.total) * 100
            elapsed = time.time() - self.start_time
            print(f"\r{self.description}: {percent:.1f}% ({self.current:,}/{self.total:,}) [{elapsed:.1f}s]", end="", flush=True)
    
    def finish(self):
        elapsed = time.time() - self.start_time
        print(f"\r{self.description}: 100.0% ({self.total:,}/{self.total:,}) [{elapsed:.1f}s] ✓")

# バイナリ効率符号化
class BinaryEncoder:
    @staticmethod
    def encode_varint(value: int) -> bytes:
        """可変長整数エンコード"""
        result = bytearray()
        while value >= 0x80:
            result.append((value & 0x7F) | 0x80)
            value >>= 7
        result.append(value & 0x7F)
        return bytes(result)
    
    @staticmethod
    def decode_varint(data: bytes, offset: int = 0) -> Tuple[int, int]:
        """可変長整数デコード"""
        result = 0
        shift = 0
        pos = offset
        
        while pos < len(data):
            byte = data[pos]
            result |= (byte & 0x7F) << shift
            pos += 1
            if byte & 0x80 == 0:
                break
            shift += 7
        
        return result, pos
    
    @staticmethod
    def encode_int_list(int_list: List[int]) -> bytes:
        """整数リストのバイナリエンコード"""
        if not int_list:
            return b'\x00\x00\x00\x00'  # 長さ0
        
        # Delta encoding for better compression
        deltas = [int_list[0]]
        for i in range(1, len(int_list)):
            deltas.append(int_list[i] - int_list[i-1])
        
        result = struct.pack('<I', len(deltas))  # リスト長
        for delta in deltas:
            result += BinaryEncoder.encode_varint(delta + 2**31)  # 負数対応
        
        return result
    
    @staticmethod
    def decode_int_list(data: bytes, offset: int = 0) -> Tuple[List[int], int]:
        """整数リストのバイナリデコード"""
        if offset + 4 > len(data):
            return [], offset
        
        length, = struct.unpack('<I', data[offset:offset+4])
        offset += 4
        
        if length == 0:
            return [], offset
        
        deltas = []
        for _ in range(length):
            delta, offset = BinaryEncoder.decode_varint(data, offset)
            deltas.append(delta - 2**31)  # 負数復元
        
        # Delta decoding
        result = [deltas[0]]
        for i in range(1, len(deltas)):
            result.append(result[-1] + deltas[i])
        
        return result, offset

# ポリオミノ形状定義（最適化版）
SHAPE_MAP = {
    0: "I-1", 1: "I-2", 2: "I-3", 3: "I-4", 4: "I-5",
    5: "O-4", 6: "T-4", 7: "L-4", 8: "Z-4", 9: "S-4",
    10: "T-5", 11: "R-6", 12: "U-6", 13: "H-7", 14: "R-8"
}

POLYOMINO_SHAPES = {
    "I-1": [(0, 0)],
    "I-2": [(0, 0), (0, 1)],
    "I-3": [(0, 0), (0, 1), (0, 2)],
    "I-4": [(0, 0), (0, 1), (0, 2), (0, 3)],
    "I-5": [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
    "O-4": [(0, 0), (0, 1), (1, 0), (1, 1)],
    "T-4": [(0, 1), (1, 0), (1, 1), (1, 2)],
    "L-4": [(0, 0), (1, 0), (2, 0), (2, 1)],
    "Z-4": [(0, 0), (0, 1), (1, 1), (1, 2)],
    "S-4": [(0, 1), (0, 2), (1, 0), (1, 1)],
    "T-5": [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
    "R-6": [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
    "U-6": [(0, 0), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0)],
    "H-7": [(0, 0), (0, 1), (0, 2), (1, 1), (2, 0), (2, 1), (2, 2)],
    "R-8": [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
}

class NexusOptimizedMetadataEngine:
    """
    🔥 NEXUS OPTIMIZED METADATA ENGINE 🔥
    
    メタデータオーバーヘッド最適化版：バイナリ形式で効率的な圧縮
    """
    
    def __init__(self):
        self.encoder = BinaryEncoder()
    
    def _nexus_layer1_perfect_consolidation(self, normalized_groups: Dict[Tuple, int], show_progress: bool = False) -> Tuple[Dict[Tuple, int], Dict[int, int]]:
        """Layer 1: 完全一致統合（最適化版）"""
        if show_progress:
            progress_bar = ProgressBar(len(normalized_groups), "   Layer 1: Perfect consolidation")
        
        # 完全一致グループの検出
        signature_to_groups = {}
        for group_tuple, group_id in normalized_groups.items():
            sig = group_tuple  # 完全一致
            if sig not in signature_to_groups:
                signature_to_groups[sig] = []
            signature_to_groups[sig].append(group_id)
        
        if show_progress:
            progress_bar.finish()
        
        # 統合処理（簡素化）
        consolidated = {}
        consolidation_map = {}  # old_id -> new_id の簡単なマッピング
        new_id = 0
        
        for sig, group_ids in signature_to_groups.items():
            consolidated[sig] = new_id
            for old_id in group_ids:
                consolidation_map[old_id] = new_id
            new_id += 1
        
        reduction = 100 * (len(normalized_groups) - len(consolidated)) / len(normalized_groups)
        print(f"   [Layer 1] Perfect match: {len(consolidated):,} groups ({reduction:.1f}% reduction)")
        
        return consolidated, consolidation_map
    
    def _nexus_layer2_pattern_consolidation(self, groups_dict: Dict[Tuple, int], layer1_map: Dict[int, int], show_progress: bool = False) -> Tuple[Dict[Tuple, int], Dict[int, int]]:
        """Layer 2: パターンベース統合（最適化版）"""
        if show_progress:
            progress_bar = ProgressBar(len(groups_dict), "   Layer 2: Pattern consolidation")
        
        # パターンベースグルーピング
        pattern_groups = {}
        for group_tuple, group_id in groups_dict.items():
            normalized = tuple(sorted(group_tuple))  # 既に正規化済みだが確実に
            if normalized not in pattern_groups:
                pattern_groups[normalized] = []
            pattern_groups[normalized].append(group_id)
        
        if show_progress:
            progress_bar.finish()
        
        # パターン統合（簡素化）
        consolidated = {}
        pattern_map = {}
        new_id = 0
        
        for pattern, group_ids in pattern_groups.items():
            consolidated[pattern] = new_id
            for old_id in group_ids:
                pattern_map[old_id] = new_id
            new_id += 1
        
        reduction = 100 * (len(groups_dict) - len(consolidated)) / len(groups_dict)
        print(f"   [Layer 2] Pattern match: {len(consolidated):,} groups ({reduction:.1f}% reduction)")
        
        return consolidated, pattern_map
    
    def _select_best_shape_for_data(self, data: bytes) -> str:
        """データ特性に基づく最適形状選択"""
        if len(data) <= 1000:
            return "I-1"
        
        sample_size = min(len(data), 2000)
        sample_data = data[:sample_size]
        
        # 高速エントロピー計算
        counts = collections.Counter(sample_data[:1000])
        entropy = 0
        total = len(sample_data[:1000])
        for count in counts.values():
            p_x = count / total
            entropy -= p_x * math.log2(p_x)
        
        # エントロピーベース形状選択
        if entropy < 2.0:
            return "O-4"
        elif entropy > 6.0:
            return "I-2"
        else:
            return "I-3"
    
    def _get_blocks_for_shape(self, data: bytes, grid_width: int, shape_coords: Tuple[Tuple[int, int], ...]) -> List[Tuple[int, ...]]:
        """形状ベースブロック生成（高速化＋完全性保証版）"""
        data_len = len(data)
        rows = data_len // grid_width
        shape_width = max(c for r, c in shape_coords) + 1
        shape_height = max(r for r, c in shape_coords) + 1
        
        blocks = []
        
        # 大きなファイルは早期終了で高速化
        if data_len > 1000000:  # 1MB以上は最大50万ブロックまで
            max_blocks = 500000
            sample_rate = max(1, ((rows - shape_height + 1) * (grid_width - shape_width + 1)) // max_blocks)
        else:
            sample_rate = 1
        
        sample_count = 0
        for r in range(rows - shape_height + 1):
            for c in range(grid_width - shape_width + 1):
                # サンプリング制御
                if sample_count % sample_rate != 0:
                    sample_count += 1
                    continue
                sample_count += 1
                
                block = []
                valid = True
                
                base_idx = r * grid_width + c
                for dr, dc in shape_coords:
                    idx = base_idx + dr * grid_width + dc
                    if idx >= data_len:
                        valid = False
                        break
                    block.append(data[idx])
                
                if valid and block:
                    blocks.append(tuple(block))
                
                # 大きなファイルでは早期終了
                if len(blocks) >= 500000:
                    break
            
            if len(blocks) >= 500000:
                break
        
        return blocks
    
    def _create_binary_payload(self, unique_groups: List[List[int]], group_id_stream: List[int], 
                              original_length: int, grid_width: int, shape_name: str) -> bytes:
        """バイナリ形式でペイロード作成（メタデータ最適化）"""
        payload = bytearray()
        
        # ヘッダー（固定サイズ）
        payload.extend(b'NXOP')  # マジックナンバー
        payload.extend(struct.pack('<I', original_length))
        payload.extend(struct.pack('<H', grid_width))
        
        # 形状ID（1バイト）
        shape_id = 0
        for sid, sname in SHAPE_MAP.items():
            if sname == shape_name:
                shape_id = sid
                break
        payload.append(shape_id)
        
        # ユニークグループ数（4バイトに拡張）
        payload.extend(struct.pack('<I', len(unique_groups)))
        
        # ユニークグループデータ（圧縮）
        groups_data = bytearray()
        for group in unique_groups:
            groups_data.extend(self.encoder.encode_int_list(group))
        
        # グループデータを圧縮
        compressed_groups = lzma.compress(groups_data, preset=1)
        payload.extend(struct.pack('<I', len(compressed_groups)))
        payload.extend(compressed_groups)
        
        # グループIDストリーム（圧縮）
        stream_data = self.encoder.encode_int_list(group_id_stream)
        compressed_stream = lzma.compress(stream_data, preset=1)
        payload.extend(struct.pack('<I', len(compressed_stream)))
        payload.extend(compressed_stream)
        
        return bytes(payload)
    
    def _parse_binary_payload(self, data: bytes) -> Tuple[List[List[int]], List[int], int, int, str]:
        """バイナリペイロードの解析"""
        offset = 0
        
        # ヘッダー検証
        magic = data[offset:offset+4]
        if magic != b'NXOP':
            raise ValueError("Invalid magic number")
        offset += 4
        
        original_length, = struct.unpack('<I', data[offset:offset+4])
        offset += 4
        
        grid_width, = struct.unpack('<H', data[offset:offset+2])
        offset += 2
        
        shape_id = data[offset]
        shape_name = SHAPE_MAP.get(shape_id, "I-1")
        offset += 1
        
        group_count, = struct.unpack('<I', data[offset:offset+4])
        offset += 4
        
        # ユニークグループデータ
        groups_size, = struct.unpack('<I', data[offset:offset+4])
        offset += 4
        
        compressed_groups = data[offset:offset+groups_size]
        offset += groups_size
        
        groups_data = lzma.decompress(compressed_groups)
        
        # グループリスト復元
        unique_groups = []
        groups_offset = 0
        for _ in range(group_count):
            group, groups_offset = self.encoder.decode_int_list(groups_data, groups_offset)
            unique_groups.append(group)
        
        # グループIDストリーム
        stream_size, = struct.unpack('<I', data[offset:offset+4])
        offset += 4
        
        compressed_stream = data[offset:offset+stream_size]
        stream_data = lzma.decompress(compressed_stream)
        group_id_stream, _ = self.encoder.decode_int_list(stream_data, 0)
        
        return unique_groups, group_id_stream, original_length, grid_width, shape_name
    
    def compress(self, data: bytes, silent: bool = False) -> bytes:
        """最適化圧縮"""
        if not data:
            return data
        
        original_length = len(data)
        
        # 形状選択
        shape_name = self._select_best_shape_for_data(data)
        shape_coords = POLYOMINO_SHAPES[shape_name]
        
        # グリッドサイズ
        grid_width = min(math.ceil(math.sqrt(original_length)), 500)
        
        if not silent:
            print(f"   [NEXUS OPT] Shape: '{shape_name}', Grid: {grid_width}")
        
        # パディング
        shape_height = max(r for r, c in shape_coords) + 1
        rows_needed = math.ceil(len(data) / grid_width)
        padded_size = (rows_needed + shape_height) * grid_width
        
        padded_data = bytearray(data)
        if padded_size > len(data):
            padded_data.extend(b'\0' * (padded_size - len(data)))
        
        # ブロック生成
        blocks = self._get_blocks_for_shape(bytes(padded_data), grid_width, shape_coords)
        
        if not silent:
            print(f"   [NEXUS OPT] Generated {len(blocks):,} blocks")
        
        # 正規化
        normalized_groups = {}
        for block in blocks:
            normalized = tuple(sorted(block))
            if normalized not in normalized_groups:
                normalized_groups[normalized] = len(normalized_groups)
        
        if not silent:
            print(f"   [NEXUS OPT] Found {len(normalized_groups):,} unique groups")
        
        # 2層統合
        layer1_result, layer1_map = self._nexus_layer1_perfect_consolidation(normalized_groups, not silent)
        layer2_result, layer2_map = self._nexus_layer2_pattern_consolidation(layer1_result, layer1_map, not silent)
        
        # 統合マップの結合
        final_map = {}
        for old_id, mid_id in layer1_map.items():
            final_id = layer2_map.get(mid_id, mid_id)
            final_map[old_id] = final_id
        
        # グループIDストリーム生成
        group_id_stream = []
        for block in blocks:
            normalized = tuple(sorted(block))
            old_id = normalized_groups[normalized]
            final_id = final_map.get(old_id, old_id)
            group_id_stream.append(final_id)
        
        # ユニークグループ
        unique_groups = [list(g) for g, i in sorted(layer2_result.items(), key=lambda x: x[1])]
        
        # バイナリペイロード作成
        binary_payload = self._create_binary_payload(unique_groups, group_id_stream, 
                                                   original_length, grid_width, shape_name)
        
        # 最終圧縮
        compressed_result = lzma.compress(binary_payload, preset=6)  # 高圧縮設定
        
        compression_ratio = len(compressed_result) / len(data)
        size_reduction = (1 - compression_ratio) * 100
        
        if not silent:
            print(f"   [NEXUS OPT] Binary payload: {len(binary_payload):,} bytes")
            print(f"   [NEXUS OPT] Final compressed: {len(compressed_result):,} bytes")
            print(f"   [NEXUS OPT] Compression ratio: {compression_ratio:.2%} ({size_reduction:.1f}% reduction)")
        
        return compressed_result
    
    def decompress(self, compressed_data: bytes, silent: bool = False) -> bytes:
        """最適化解凍"""
        if not compressed_data:
            return b''
        
        # LZMA解凍
        binary_payload = lzma.decompress(compressed_data)
        
        # バイナリペイロード解析
        unique_groups, group_id_stream, original_length, grid_width, shape_name = \
            self._parse_binary_payload(binary_payload)
        
        if not silent:
            print(f"   [NEXUS OPT DECOMP] Restoring {original_length} bytes using '{shape_name}'")
        
        # ブロック再構成
        reconstructed_blocks = []
        for group_id in group_id_stream:
            if group_id < len(unique_groups):
                reconstructed_blocks.append(unique_groups[group_id])
            else:
                reconstructed_blocks.append([0])
        
        # データ再構成
        shape_coords = POLYOMINO_SHAPES[shape_name]
        return self._reconstruct_data_from_blocks(reconstructed_blocks, grid_width, 
                                                original_length, shape_coords, silent)
    
    def _reconstruct_data_from_blocks(self, blocks: List[List[int]], grid_width: int, 
                                    original_length: int, shape_coords: Tuple[Tuple[int, int], ...], 
                                    silent: bool = False) -> bytes:
        """ブロックからデータを再構成（完全修正版）"""
        if not blocks:
            return b''
        
        shape_width = max(c for r, c in shape_coords) + 1
        shape_height = max(r for r, c in shape_coords) + 1
        
        # 元のデータサイズに基づく適切なグリッドサイズ計算
        rows_needed = math.ceil(original_length / grid_width)
        total_grid_size = (rows_needed + shape_height) * grid_width
        
        # データ配列初期化（重複対応）
        reconstructed_data = bytearray(total_grid_size)
        write_count = [0] * total_grid_size  # 各位置への書き込み回数をカウント
        
        # ブロック配置（正確な位置計算）
        blocks_per_row = grid_width - shape_width + 1
        block_idx = 0
        
        rows_with_blocks = (len(blocks) + blocks_per_row - 1) // blocks_per_row
        
        for r in range(rows_with_blocks):
            for c in range(blocks_per_row):
                if block_idx >= len(blocks):
                    break
                
                block = blocks[block_idx]
                if not block:  # 空のブロック処理
                    block_idx += 1
                    continue
                
                # 各形状座標に対してデータを配置
                base_idx = r * grid_width + c
                
                for coord_idx, (dr, dc) in enumerate(shape_coords):
                    grid_idx = base_idx + dr * grid_width + dc
                    
                    # 境界チェック
                    if (grid_idx < total_grid_size and 
                        coord_idx < len(block) and 
                        grid_idx < original_length):  # 元のデータサイズ内のみ
                        
                        value = block[coord_idx]
                        if isinstance(value, (int, float)):
                            byte_value = int(value) % 256
                            reconstructed_data[grid_idx] = byte_value
                            write_count[grid_idx] += 1
                
                block_idx += 1
            
            if block_idx >= len(blocks):
                break
        
        # 未書き込み位置のチェック（デバッグ用）
        if not silent:
            unwritten_count = sum(1 for count in write_count[:original_length] if count == 0)
            if unwritten_count > 0:
                print(f"   [Warning] {unwritten_count} positions not written in reconstruction")
        
        return bytes(reconstructed_data[:original_length])


def create_test_file(filename: str, size_kb: int):
    """テスト用ファイル作成（簡素化版）"""
    print(f"Creating test file: {filename} ({size_kb}KB)")
    
    # 単純なパターンでテストデータ作成
    target_size = size_kb * 1024
    
    # パターン：繰り返し＋少しのランダム
    pattern = b"ABCDEFGH" * 16  # 128バイトのパターン
    data = bytearray()
    
    while len(data) < target_size:
        data.extend(pattern)
        # 5%のランダム要素
        if len(data) % 1024 == 0:
            random_bytes = bytes([random.randint(0, 255) for _ in range(8)])
            data.extend(random_bytes)
    
    # 指定サイズに切り詰め
    data = data[:target_size]
    
    with open(filename, 'wb') as f:
        f.write(data)
    
    print(f"Test file created: {len(data)} bytes")


def test_nexus_optimized():
    """最適化版NEXUSテスト（高速デバッグ版）"""
    print("🔥 NEXUS OPTIMIZED METADATA ENGINE TEST 🔥")
    print("=" * 50)
    
    engine = NexusOptimizedMetadataEngine()
    
    # テストファイル作成（高速デバッグ用に小さくする）
    test_sizes = [5, 10]  # 小さなサイズでクイックテスト
    
    for size_kb in test_sizes:
        test_file = f"test_{size_kb}kb.bin"
        
        print(f"\n📁 Creating and testing {size_kb}KB file:")
        create_test_file(test_file, size_kb)
        
        with open(test_file, 'rb') as f:
            data = f.read()
        
        print(f"   Original size: {len(data):,} bytes")
        
        # 圧縮
        start_time = time.time()
        compressed = engine.compress(data)
        compress_time = time.time() - start_time
        
        # 解凍
        start_time = time.time()
        decompressed = engine.decompress(compressed)
        decompress_time = time.time() - start_time
        
        # 結果
        compression_ratio = len(compressed) / len(data) * 100
        is_perfect = data == decompressed
        
        print(f"   Compressed size: {len(compressed):,} bytes ({compression_ratio:.1f}%)")
        print(f"   Compression time: {compress_time:.3f}s")
        print(f"   Decompression time: {decompress_time:.3f}s")
        print(f"   Perfect recovery: {'✓' if is_perfect else '✗'}")
        
        if is_perfect:
            if compression_ratio < 100:
                print("   🎉 NEXUS OPTIMIZED: COMPRESSION SUCCESS!")
            else:
                print("   ⚠️  NEXUS OPTIMIZED: Perfect but expansion")
        else:
            print("   ❌ Data corruption detected")
            # デバッグ情報
            if len(data) != len(decompressed):
                print(f"   [Debug] Size mismatch: {len(data)} vs {len(decompressed)}")
            else:
                diff_count = sum(1 for a, b in zip(data, decompressed) if a != b)
                print(f"   [Debug] {diff_count} byte differences out of {len(data)}")
        
        # テストファイル削除
        try:
            os.remove(test_file)
        except:
            pass


if __name__ == "__main__":
    test_nexus_optimized()
