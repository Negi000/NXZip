#!/usr/bin/env python3
"""NEXUS 解凍デバッグスクリプト"""
import json
import hashlib
from nexus_advanced_engine import NexusAdvancedCompressor

def test_debug():
    engine = NexusAdvancedCompressor()
    
    # 簡単なテストデータ
    test_data = b"Hello World! This is a test message."
    print(f"Original: {test_data}")
    print(f"Original hash: {hashlib.md5(test_data).hexdigest()}")
    
    # 圧縮
    compressed = engine.compress(test_data, level=0)
    print(f"Compressed size: {len(compressed)}")
    
    # ペイロード構造を確認
    try:
        # LZMA解凍してJSONペイロード取得
        import lzma
        decompressed_payload = lzma.decompress(compressed)
        payload = json.loads(decompressed_payload.decode('utf-8'))
        
        print("\n=== PAYLOAD STRUCTURE ===")
        print(f"Keys: {list(payload.keys())}")
        
        if 'header' in payload:
            print(f"Header: {payload['header']}")
        
        if 'unique_groups' in payload:
            print(f"Unique groups count: {len(payload['unique_groups'])}")
            print(f"First few groups: {payload['unique_groups'][:3]}")
            
        if 'perm_map_dict' in payload:
            print(f"Perm map keys: {list(payload['perm_map_dict'].keys())[:5]}")
            
        if 'encoded_streams' in payload:
            print(f"Encoded streams: {list(payload['encoded_streams'].keys())}")
            
    except Exception as e:
        print(f"Payload parsing error: {e}")
    
    # 解凍
    try:
        decompressed = engine.decompress(compressed)
        print(f"\nDecompressed: {decompressed}")
        print(f"Decompressed hash: {hashlib.md5(decompressed).hexdigest()}")
        print(f"Match: {test_data == decompressed}")
    except Exception as e:
        print(f"Decompress error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_debug()
