#!/usr/bin/env python3
"""ブロック生成プロセスの詳細分析"""
from nexus_advanced_engine import NexusAdvancedCompressor

def trace_block_generation():
    """ブロック生成プロセスを詳細に追跡"""
    
    # 単純なテストデータ
    test_data = b"ABCDEFGH"  # [65, 66, 67, 68, 69, 70, 71, 72]
    print(f"Original data: {test_data}")
    print(f"Original bytes: {list(test_data)}")
    
    engine = NexusAdvancedCompressor()
    
    # 形状選択と前処理
    print("\n=== SHAPE SELECTION ===")
    # H-7 shape coordinates from the engine
    shape_coords = ((0, 0), (0, 1), (0, 2), (1, 1), (2, 0), (2, 1), (2, 2))
    print(f"Shape H-7 coordinates: {shape_coords}")
    
    grid_width = 3  # テストケースから
    padded_data = [65, 66, 67, 68, 69, 70, 71, 72, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 9バイトパディング
    print(f"Grid width: {grid_width}")
    print(f"Padded data: {padded_data}")
    
    # ブロック生成をシミュレート
    print("\n=== BLOCK GENERATION ===")
    blocks = engine._get_blocks_for_shape(bytes(padded_data), grid_width, shape_coords)
    print(f"Generated blocks: {blocks}")
    print(f"Number of blocks: {len(blocks)}")
    
    for i, block in enumerate(blocks):
        print(f"Block {i}: {block}")
        sorted_block = tuple(sorted(block))
        print(f"  Sorted: {sorted_block}")
        
        # 順列マップ計算
        perm_map = engine._calculate_permutation_map(block, sorted_block)
        print(f"  Permutation map: {perm_map}")
        
        # 検証：perm_mapを使って元に戻せるか
        reconstructed = [sorted_block[i] for i in perm_map]
        print(f"  Reconstruction test: {tuple(reconstructed)} == {block} ? {tuple(reconstructed) == block}")
        
        # 正しい逆変換も計算
        print(f"  Inverse test: sorted[{perm_map}] -> original")
        for j, sorted_pos in enumerate(perm_map):
            print(f"    original[{j}] = sorted[{sorted_pos}] = {sorted_block[sorted_pos]}")

if __name__ == "__main__":
    trace_block_generation()
