#!/usr/bin/env python3
"""詳細なペイロード解析とデバッグ"""
import json
import hashlib
from nexus_advanced_engine import NexusAdvancedCompressor

def debug_transform():
    engine = NexusAdvancedCompressor()
    
    # 明確に解析可能なテストデータ
    test_data = b"ABCDEFGHIJKLMNOP"  # 16バイト
    print(f"Original: {test_data}")
    print(f"Original hash: {hashlib.md5(test_data).hexdigest()}")
    
    # 圧縮
    compressed = engine.compress(test_data, level=0)
    
    # ペイロード詳細解析
    try:
        import lzma
        decompressed_payload = lzma.decompress(compressed)
        payload = json.loads(decompressed_payload.decode('utf-8'))
        
        print("\n=== DETAILED PAYLOAD ANALYSIS ===")
        
        # Unique groupsの分析
        print(f"Unique groups: {payload['unique_groups']}")
        
        # Perm map詳細
        print(f"Perm maps: {payload['perm_map_dict']}")
        
        # エンコードされたストリーム
        print(f"Group ID stream length: {len(payload['encoded_streams']['group_ids'])}")
        print(f"Perm ID stream length: {len(payload['encoded_streams']['perm_ids'])}")
        
        # Huffman復号化して実際のストリームを表示
        group_huff_tree = payload['huffman_trees']['group_ids']
        perm_huff_tree = payload['huffman_trees']['perm_ids']
        
        group_ids = engine.huffman_encoder.decode(payload['encoded_streams']['group_ids'], group_huff_tree)
        perm_ids = engine.huffman_encoder.decode(payload['encoded_streams']['perm_ids'], perm_huff_tree)
        
        print(f"Decoded group IDs: {group_ids}")
        print(f"Decoded perm IDs: {perm_ids}")
        
        print("\n=== MANUAL RECONSTRUCTION ===")
        # 手動でブロック再構成
        unique_groups = [tuple(g) for g in payload['unique_groups']]
        perm_map_dict = {int(k): tuple(v) for k, v in payload['perm_map_dict'].items()}
        
        reconstructed_blocks = []
        for i, (group_id, perm_id) in enumerate(zip(group_ids, perm_ids)):
            print(f"Block {i}: Group ID {group_id} -> {unique_groups[group_id]}, Perm ID {perm_id} -> {perm_map_dict[perm_id]}")
            
            group = unique_groups[group_id]
            perm_map = perm_map_dict[perm_id]
            
            # 逆変換
            result = [0] * len(group)
            for original_pos, shuffled_pos in enumerate(perm_map):
                if 0 <= shuffled_pos < len(result):
                    result[original_pos] = group[shuffled_pos]
            
            print(f"  -> Reconstructed: {tuple(result)}")
            reconstructed_blocks.append(tuple(result))
            
        # フラット化
        flat_data = []
        for block in reconstructed_blocks:
            flat_data.extend(block)
        
        result = bytes(flat_data[:len(test_data)])
        print(f"Manual result: {result}")
        print(f"Manual hash: {hashlib.md5(result).hexdigest()}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    # 公式の解凍
    try:
        decompressed = engine.decompress(compressed)
        print(f"\nOfficial decompressed: {decompressed}")
        print(f"Official hash: {hashlib.md5(decompressed).hexdigest()}")
        print(f"Match: {test_data == decompressed}")
    except Exception as e:
        print(f"Official decompress error: {e}")

if __name__ == "__main__":
    debug_transform()
