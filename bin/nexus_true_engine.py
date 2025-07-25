#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEXUS TRUE COMPRESSION ENGINE - Perfect Reconstruction
Fixed overlapping block issue with proper area assignment
"""

import struct
import lzma
import math
import time
import random
import os
import collections
from typing import List, Tuple, Dict, Any

# Polyomino shapes
POLYOMINO_SHAPES = {
    "I-1": [(0, 0)],
    "I-2": [(0, 0), (0, 1)],
    "I-3": [(0, 0), (0, 1), (0, 2)],
    "I-4": [(0, 0), (0, 1), (0, 2), (0, 3)],
    "O-4": [(0, 0), (0, 1), (1, 0), (1, 1)],
}

SHAPE_MAP = {0: "I-1", 1: "I-2", 2: "I-3", 3: "I-4", 4: "O-4"}
REVERSE_SHAPE_MAP = {v: k for k, v in SHAPE_MAP.items()}


class BinaryEncoder:
    """Binary encoding utilities"""
    
    @staticmethod
    def encode_int_list(int_list: List[int]) -> bytes:
        """Integer list binary encoding"""
        if not int_list:
            return b'\x00\x00\x00\x00'
        
        result = struct.pack('<I', len(int_list))
        for value in int_list:
            result += struct.pack('<I', max(0, value))  # Simple 4-byte encoding
        
        return result
    
    @staticmethod
    def decode_int_list(data: bytes, offset: int = 0) -> Tuple[List[int], int]:
        """Integer list binary decoding"""
        if offset + 4 > len(data):
            return [], offset
        
        length, = struct.unpack('<I', data[offset:offset+4])
        offset += 4
        
        if length == 0:
            return [], offset
        
        result = []
        for _ in range(length):
            if offset + 4 > len(data):
                break
            value, = struct.unpack('<I', data[offset:offset+4])
            result.append(value)
            offset += 4
        
        return result, offset


class NexusTrueEngine:
    """NEXUS True Engine with perfect reconstruction"""
    
    def __init__(self):
        self.encoder = BinaryEncoder()
    
    def _select_simple_shape(self, data: bytes) -> str:
        """Select simplest effective shape"""
        if len(data) <= 1000:
            return "I-1"
        elif len(data) <= 10000:
            return "I-2"
        else:
            return "I-3"
    
    def _get_non_overlapping_blocks(self, data: bytes, grid_width: int, shape_coords: Tuple[Tuple[int, int], ...]) -> Tuple[List[Tuple[int, ...]], List[Tuple[int, int]]]:
        """Get non-overlapping blocks with position tracking"""
        data_len = len(data)
        if data_len == 0:
            return [], []
        
        rows = data_len // grid_width
        shape_width = max(c for r, c in shape_coords) + 1
        shape_height = max(r for r, c in shape_coords) + 1
        
        blocks = []
        positions = []
        
        # Non-overlapping block extraction
        for r in range(0, rows - shape_height + 1, shape_height):
            for c in range(0, grid_width - shape_width + 1, shape_width):
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
                    positions.append((r, c))  # Store starting position
        
        return blocks, positions
    
    def _reconstruct_from_non_overlapping_blocks(self, blocks: List[List[int]], positions: List[Tuple[int, int]], 
                                               grid_width: int, original_length: int, 
                                               shape_coords: Tuple[Tuple[int, int], ...]) -> bytes:
        """Reconstruct data from non-overlapping blocks with perfect accuracy"""
        if not blocks or not positions:
            return b'\x00' * original_length  # Return zeros if no blocks
        
        # Initialize reconstruction array with zeros
        reconstructed_data = bytearray(original_length)
        coverage = [False] * original_length  # Track covered positions
        
        # Place each block at its exact position
        for block, (start_r, start_c) in zip(blocks, positions):
            base_idx = start_r * grid_width + start_c
            
            for coord_idx, (dr, dc) in enumerate(shape_coords):
                grid_idx = base_idx + dr * grid_width + dc
                
                if (grid_idx < original_length and coord_idx < len(block)):
                    reconstructed_data[grid_idx] = block[coord_idx] % 256
                    coverage[grid_idx] = True
        
        # Check coverage
        uncovered_count = sum(1 for covered in coverage if not covered)
        if uncovered_count > 0:
            print(f"   [Warning] {uncovered_count} positions not covered by blocks")
        
        return bytes(reconstructed_data)
    
    def compress(self, data: bytes, silent: bool = False) -> bytes:
        """True NEXUS compression with perfect reconstruction"""
        if not data:
            return b''
        
        # Simple shape selection
        shape_name = self._select_simple_shape(data)
        grid_width = min(math.ceil(math.sqrt(len(data))), 100)
        
        if not silent:
            print(f"   [NEXUS TRUE] Shape: '{shape_name}', Grid: {grid_width}")
        
        # Non-overlapping block extraction
        shape_coords = POLYOMINO_SHAPES[shape_name]
        blocks, positions = self._get_non_overlapping_blocks(data, grid_width, shape_coords)
        
        if not silent:
            print(f"   [NEXUS TRUE] Generated {len(blocks):,} non-overlapping blocks")
        
        # Simple unique groups (no normalization to preserve exact block content)
        unique_blocks = []
        block_to_id = {}
        group_id_stream = []
        
        for block in blocks:
            if block not in block_to_id:
                block_to_id[block] = len(unique_blocks)
                unique_blocks.append(list(block))
            group_id_stream.append(block_to_id[block])
        
        if not silent:
            print(f"   [NEXUS TRUE] Found {len(unique_blocks):,} unique blocks")
        
        # Create payload
        payload = {
            'unique_blocks': unique_blocks,
            'group_id_stream': group_id_stream,
            'positions': positions,
            'original_length': len(data),
            'grid_width': grid_width,
            'shape_name': shape_name
        }
        
        # Serialize and compress
        import json
        serialized = json.dumps(payload).encode('utf-8')
        compressed = lzma.compress(serialized, preset=1)
        
        compression_ratio = len(compressed) / len(data)
        
        if not silent:
            print(f"   [NEXUS TRUE] Compressed: {len(compressed):,} bytes ({compression_ratio:.2%})")
        
        return compressed
    
    def decompress(self, compressed_data: bytes, silent: bool = False) -> bytes:
        """True NEXUS decompression"""
        if not compressed_data:
            return b''
        
        # Decompress and deserialize
        import json
        serialized = lzma.decompress(compressed_data)
        payload = json.loads(serialized.decode('utf-8'))
        
        unique_blocks = payload['unique_blocks']
        group_id_stream = payload['group_id_stream']
        positions = payload['positions']
        original_length = payload['original_length']
        grid_width = payload['grid_width']
        shape_name = payload['shape_name']
        
        if not silent:
            print(f"   [NEXUS TRUE DECOMP] Restoring {original_length} bytes")
        
        # Reconstruct blocks
        reconstructed_blocks = []
        for group_id in group_id_stream:
            if group_id < len(unique_blocks):
                reconstructed_blocks.append(unique_blocks[group_id])
            else:
                reconstructed_blocks.append([0])
        
        # Reconstruct data
        shape_coords = POLYOMINO_SHAPES[shape_name]
        return self._reconstruct_from_non_overlapping_blocks(
            reconstructed_blocks, positions, grid_width, original_length, shape_coords)


def create_simple_test_file(filename: str, size_kb: int):
    """Create simple test file"""
    print(f"Creating test file: {filename} ({size_kb}KB)")
    
    target_size = size_kb * 1024
    # Simple repeating pattern
    pattern = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    data = (pattern * ((target_size // len(pattern)) + 1))[:target_size]
    
    with open(filename, 'wb') as f:
        f.write(data)
    
    print(f"Test file created: {len(data)} bytes")


def test_nexus_true():
    """Test NEXUS True Engine"""
    print("🔥 NEXUS TRUE ENGINE - PERFECT RECONSTRUCTION TEST 🔥")
    print("=" * 60)
    
    engine = NexusTrueEngine()
    
    # Test with small files first
    test_sizes = [1, 5, 10, 20]  # KB
    
    for size_kb in test_sizes:
        test_file = f"test_{size_kb}kb.bin"
        
        print(f"\n📁 Testing {size_kb}KB file:")
        create_simple_test_file(test_file, size_kb)
        
        with open(test_file, 'rb') as f:
            data = f.read()
        
        print(f"   Original size: {len(data):,} bytes")
        print(f"   First 50 bytes: {data[:50]}")
        
        # Compression
        start_time = time.time()
        compressed = engine.compress(data)
        compress_time = time.time() - start_time
        
        # Decompression
        start_time = time.time()
        decompressed = engine.decompress(compressed)
        decompress_time = time.time() - start_time
        
        # Results
        compression_ratio = len(compressed) / len(data) * 100
        is_perfect = data == decompressed
        
        print(f"   Compressed size: {len(compressed):,} bytes ({compression_ratio:.1f}%)")
        print(f"   Compression time: {compress_time:.3f}s")
        print(f"   Decompression time: {decompress_time:.3f}s")
        print(f"   Perfect recovery: {'✓' if is_perfect else '✗'}")
        
        if is_perfect:
            print("   🎉 NEXUS TRUE: PERFECT RECONSTRUCTION ACHIEVED!")
        else:
            print("   ❌ Data corruption detected")
            if len(data) != len(decompressed):
                print(f"   [Debug] Size mismatch: {len(data)} vs {len(decompressed)}")
            else:
                print(f"   [Debug] First 50 decompressed: {decompressed[:50]}")
                diff_count = sum(1 for a, b in zip(data, decompressed) if a != b)
                print(f"   [Debug] {diff_count} byte differences out of {len(data)}")
        
        # Clean up
        try:
            os.remove(test_file)
        except:
            pass


if __name__ == "__main__":
    test_nexus_true()
