#!/usr/bin/env python3
"""NEXUS 可逆性の根本原因分析"""
import json
import hashlib
from nexus_advanced_engine import NexusAdvancedCompressor

def analyze_reversibility():
    """圧縮・解凍プロセスの詳細分析"""
    engine = NexusAdvancedCompressor()
    
    # 単純で明確なテストデータ
    test_data = b"ABCDEFGH"  # 8バイト、全て異なる値
    print(f"🔍 Original data: {test_data}")
    print(f"🔍 Original bytes: {list(test_data)}")
    print(f"🔍 Original hash: {hashlib.md5(test_data).hexdigest()}")
    
    # 圧縮プロセスの詳細監視
    print("\n" + "="*60)
    print("📊 COMPRESSION PROCESS ANALYSIS")
    print("="*60)
    
    # 一時的にエンジンをデバッグモードに変更
    compressed = engine.compress(test_data, level=0)
    
    # ペイロード解析
    print("\n" + "="*60)
    print("📦 PAYLOAD STRUCTURE ANALYSIS")
    print("="*60)
    
    try:
        import lzma
        decompressed_payload = lzma.decompress(compressed)
        payload = json.loads(decompressed_payload.decode('utf-8'))
        
        print(f"Header: {payload['header']}")
        print(f"Unique groups count: {len(payload['unique_groups'])}")
        print(f"Unique groups: {payload['unique_groups']}")
        print(f"Perm map dict: {payload['perm_map_dict']}")
        
        # Huffman復号
        group_huff_tree = payload['huffman_trees']['group_ids']
        group_ids = engine.huffman_encoder.decode(payload['encoded_streams']['group_ids'], group_huff_tree)
        
        if 'perm_ids' in payload['encoded_streams'] and payload['encoded_streams']['perm_ids']:
            perm_huff_tree = payload['huffman_trees']['perm_ids']
            perm_ids = engine.huffman_encoder.decode(payload['encoded_streams']['perm_ids'], perm_huff_tree)
        else:
            perm_ids = [0] * len(group_ids)
        
        print(f"Group ID stream: {group_ids}")
        print(f"Perm ID stream: {perm_ids}")
        
        print("\n" + "="*60)
        print("🔄 DECOMPRESSION STEP-BY-STEP")
        print("="*60)
        
        # 手動でステップごとに復元
        unique_groups = [tuple(g) for g in payload['unique_groups']]
        perm_map_dict = {int(k): tuple(v) for k, v in payload['perm_map_dict'].items()}
        
        reconstructed_blocks = []
        for i, (group_id, perm_id) in enumerate(zip(group_ids, perm_ids)):
            print(f"\nBlock {i}:")
            print(f"  Group ID: {group_id} -> {unique_groups[group_id] if group_id < len(unique_groups) else 'INVALID'}")
            print(f"  Perm ID: {perm_id} -> {perm_map_dict.get(perm_id, 'INVALID')}")
            
            if group_id < len(unique_groups) and perm_id in perm_map_dict:
                group = unique_groups[group_id]
                perm_map = perm_map_dict[perm_id]
                
                print(f"  Before inverse permutation: {group}")
                print(f"  Permutation map: {perm_map}")
                
                # 逆変換の詳細ログ
                if len(group) == len(perm_map):
                    result = [0] * len(group)
                    for original_pos, shuffled_pos in enumerate(perm_map):
                        if 0 <= shuffled_pos < len(result):
                            result[original_pos] = group[shuffled_pos]
                            print(f"    result[{original_pos}] = group[{shuffled_pos}] = {group[shuffled_pos]}")
                    
                    reconstructed_block = tuple(result)
                    print(f"  After inverse permutation: {reconstructed_block}")
                    reconstructed_blocks.append(reconstructed_block)
                else:
                    print(f"  ⚠️ Length mismatch: group={len(group)}, perm_map={len(perm_map)}")
                    reconstructed_blocks.append(group)
            else:
                print(f"  ❌ Invalid indices")
                reconstructed_blocks.append((0,))
        
        # フラット化
        print(f"\nReconstructed blocks: {reconstructed_blocks}")
        flat_data = []
        for block in reconstructed_blocks:
            flat_data.extend(block)
        
        original_length = payload['header']['original_length']
        result_bytes = bytes(flat_data[:original_length])
        
        print(f"Flat data: {flat_data}")
        print(f"Result bytes: {list(result_bytes)}")
        print(f"Result string: {result_bytes}")
        print(f"Result hash: {hashlib.md5(result_bytes).hexdigest()}")
        print(f"Match: {test_data == result_bytes}")
        
    except Exception as e:
        print(f"❌ Payload analysis error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("🏁 OFFICIAL DECOMPRESSION")
    print("="*60)
    
    try:
        official_result = engine.decompress(compressed)
        print(f"Official result: {official_result}")
        print(f"Official bytes: {list(official_result)}")
        print(f"Official hash: {hashlib.md5(official_result).hexdigest()}")
        print(f"Official match: {test_data == official_result}")
    except Exception as e:
        print(f"❌ Official decompression error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_reversibility()
