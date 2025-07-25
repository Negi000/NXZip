#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
個別ファイル圧縮・解凍テスト（安定動作版）
"""

import os
import hashlib
from nexus_advanced_engine import NexusAdvancedCompressor

def test_individual_file(filename):
    """個別ファイルの圧縮・解凍テスト"""
    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        return False
    
    # ファイル読み込み
    with open(filename, 'rb') as f:
        original_data = f.read()
    
    original_size = len(original_data)
    original_hash = hashlib.md5(original_data).hexdigest()
    
    print(f"\n📁 {filename}")
    print(f"Size: {original_size:,} bytes | Hash: {original_hash[:16]}...")
    
    # 圧縮
    compressor = NexusAdvancedCompressor()
    
    try:
        compressed = compressor.compress(original_data, level=0)
        compressed_size = len(compressed)
        
        # .nxzファイルに保存
        nxz_filename = filename + ".nxz"
        with open(nxz_filename, 'wb') as f:
            f.write(compressed)
        
        print(f"→ Compressed: {compressed_size:,} bytes ({compressed_size/original_size:.3f}x)")
        print(f"→ Saved: {nxz_filename}")
        
        # 解凍
        decompressed = compressor.decompress(compressed, level=0)
        decompressed_hash = hashlib.md5(decompressed).hexdigest()
        
        # 可逆性チェック
        is_reversible = (original_hash == decompressed_hash and len(decompressed) == original_size)
        
        if is_reversible:
            print("✅ REVERSIBLE")
        else:
            print("❌ NOT REVERSIBLE")
            if len(decompressed) != original_size:
                print(f"   Size: {original_size} → {len(decompressed)}")
            if original_hash != decompressed_hash:
                print(f"   Hash: {original_hash[:16]} → {decompressed_hash[:16]}")
        
        return is_reversible
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """個別テスト実行"""
    print("🔧 Individual File Compression Test")
    
    # テストファイル（小さいものから）
    test_files = [
        "test_small.txt",
        "element_test_small.bin", 
        "element_test_medium.bin",
    ]
    
    results = []
    
    for filename in test_files:
        try:
            result = test_individual_file(filename)
            results.append((filename, result))
        except KeyboardInterrupt:
            print("\n⏹️  Test interrupted by user")
            break
        except Exception as e:
            print(f"❌ Unexpected error with {filename}: {e}")
            results.append((filename, False))
    
    # 結果
    print(f"\n{'='*50}")
    print("📊 RESULTS")
    print(f"{'='*50}")
    
    for filename, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {filename}")

if __name__ == "__main__":
    main()
