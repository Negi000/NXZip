#!/usr/bin/env python3
"""NEXUS 可逆性の完全解決 - 根本的アプローチ"""
import json
import hashlib
from nexus_advanced_engine import NexusAdvancedCompressor

def test_fundamental_reversibility():
    """根本的な可逆性問題の解決"""
    
    # 最も単純なケースから段階的に検証
    test_cases = [
        b"A",           # 1バイト
        b"AB",          # 2バイト
        b"ABC",         # 3バイト
        b"ABCD",        # 4バイト
        b"ABCDEFGH",    # 8バイト
    ]
    
    engine = NexusAdvancedCompressor()
    
    for i, test_data in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"TEST {i+1}: {test_data} ({len(test_data)} bytes)")
        print(f"{'='*60}")
        
        original_hash = hashlib.md5(test_data).hexdigest()
        print(f"Original: {list(test_data)} | Hash: {original_hash}")
        
        try:
            # 圧縮
            compressed = engine.compress(test_data, level=0)
            print(f"Compressed size: {len(compressed)} bytes")
            
            # 解凍
            decompressed = engine.decompress(compressed)
            decompressed_hash = hashlib.md5(decompressed).hexdigest()
            
            print(f"Decompressed: {list(decompressed)} | Hash: {decompressed_hash}")
            
            # 検証
            is_reversible = (test_data == decompressed)
            print(f"REVERSIBLE: {'✅ YES' if is_reversible else '❌ NO'}")
            
            if not is_reversible:
                print(f"Expected: {list(test_data)}")
                print(f"Got:      {list(decompressed)}")
                print(f"Diff positions: {[i for i in range(min(len(test_data), len(decompressed))) if test_data[i] != decompressed[i]]}")
                break  # 最初の失敗で停止して詳細分析
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            break

if __name__ == "__main__":
    test_fundamental_reversibility()
