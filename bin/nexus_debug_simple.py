#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEXUS OPTIMIZED DEBUG VERSION
Simple test to isolate the data reconstruction issue
"""

import struct
import lzma
import math
import time
import random
import os
from typing import List, Tuple, Dict, Any

# Polyomino shapes
POLYOMINO_SHAPES = {
    "I-1": [(0, 0)],
    "I-2": [(0, 0), (0, 1)],
    "I-3": [(0, 0), (0, 1), (0, 2)],
}

def simple_test():
    """Simple test to debug data reconstruction"""
    print("🔍 NEXUS DEBUG - Simple reconstruction test")
    print("=" * 50)
    
    # Create simple test data
    original_data = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 4  # 104 bytes
    print(f"Original data: {len(original_data)} bytes")
    print(f"First 26 bytes: {original_data[:26]}")
    
    # Simple grid parameters
    grid_width = 10
    shape_name = "I-3"
    shape_coords = POLYOMINO_SHAPES[shape_name]
    
    print(f"Grid width: {grid_width}")
    print(f"Shape: {shape_name} -> {shape_coords}")
    
    # Generate blocks manually for debugging
    blocks = []
    data_len = len(original_data)
    rows = data_len // grid_width
    shape_width = max(c for r, c in shape_coords) + 1
    shape_height = max(r for r, c in shape_coords) + 1
    
    print(f"Data length: {data_len}, Rows: {rows}")
    print(f"Shape size: {shape_width}x{shape_height}")
    
    for r in range(rows - shape_height + 1):
        for c in range(grid_width - shape_width + 1):
            block = []
            valid = True
            
            base_idx = r * grid_width + c
            for dr, dc in shape_coords:
                idx = base_idx + dr * grid_width + dc
                if idx >= data_len:
                    valid = False
                    break
                block.append(original_data[idx])
            
            if valid and block:
                blocks.append(block)
    
    print(f"Generated {len(blocks)} blocks")
    print(f"First block: {blocks[0]} -> {[chr(b) for b in blocks[0]]}")
    print(f"Last block: {blocks[-1]} -> {[chr(b) for b in blocks[-1]]}")
    
    # Now reconstruct
    print("\n🔄 Reconstructing data...")
    
    # Calculate reconstruction parameters
    blocks_per_row = grid_width - shape_width + 1
    rows_with_blocks = (len(blocks) + blocks_per_row - 1) // blocks_per_row
    
    print(f"Blocks per row: {blocks_per_row}")
    print(f"Rows with blocks: {rows_with_blocks}")
    
    # Initialize reconstruction array
    total_grid_size = (rows_with_blocks + shape_height) * grid_width
    reconstructed_data = bytearray(total_grid_size)
    
    print(f"Reconstruction grid size: {total_grid_size}")
    
    # Place blocks back
    block_idx = 0
    for r in range(rows_with_blocks):
        for c in range(blocks_per_row):
            if block_idx >= len(blocks):
                break
            
            block = blocks[block_idx]
            base_idx = r * grid_width + c
            
            print(f"Block {block_idx}: {[chr(b) for b in block]} at base_idx {base_idx}")
            
            for coord_idx, (dr, dc) in enumerate(shape_coords):
                grid_idx = base_idx + dr * grid_width + dc
                
                if (grid_idx < total_grid_size and 
                    coord_idx < len(block) and 
                    grid_idx < data_len):  # Only write within original data
                    
                    reconstructed_data[grid_idx] = block[coord_idx]
                    print(f"  Placed {chr(block[coord_idx])} at grid_idx {grid_idx}")
            
            block_idx += 1
            
            if block_idx >= 5:  # Debug first few blocks only
                break
        
        if block_idx >= 5:
            break
    
    # Compare results
    reconstructed = bytes(reconstructed_data[:data_len])
    print(f"\nReconstructed data length: {len(reconstructed)}")
    print(f"First 26 bytes: {reconstructed[:26]}")
    
    is_perfect = original_data == reconstructed
    print(f"Perfect match: {is_perfect}")
    
    if not is_perfect:
        print("\n❌ Differences found:")
        for i in range(min(len(original_data), len(reconstructed), 50)):
            if original_data[i] != reconstructed[i]:
                print(f"  Position {i}: expected {chr(original_data[i])}, got {chr(reconstructed[i])}")


if __name__ == "__main__":
    simple_test()
