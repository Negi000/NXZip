#!/usr/bin/env python3
"""
個別ファイル圧縮テスト - サイレント版
"""

import os
import sys
import hashlib
import time
from nexus_advanced_engine import NexusAdvancedCompressor

def calculate_md5(data):
    """MD5ハッシュを計算"""
    return hashlib.md5(data).hexdigest()

def test_single_file(filename, compressor):
    """単一ファイルの圧縮テスト"""
    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        return False
    
    print(f"📁 Testing: {filename}")
    
    # ファイル読み込み
    with open(filename, 'rb') as f:
        original_data = f.read()
    
    original_size = len(original_data)
    original_hash = calculate_md5(original_data)
    print(f"   Original: {original_size:,} bytes")
    
    # 圧縮
    start_time = time.time()
    try:
        compressed_data = compressor.compress(original_data, silent=True)
        compress_time = time.time() - start_time
    except Exception as e:
        print(f"   ❌ Compression failed: {e}")
        return False
    
    compressed_size = len(compressed_data)
    ratio = compressed_size / original_size
    print(f"   Compressed: {compressed_size:,} bytes (ratio: {ratio:.4f}) in {compress_time:.2f}s")
    
    # 解凍
    start_time = time.time()
    try:
        decompressed_data = compressor.decompress(compressed_data, silent=True)
        decompress_time = time.time() - start_time
    except Exception as e:
        print(f"   ❌ Decompression failed: {e}")
        return False
    
    # 検証
    decompressed_hash = calculate_md5(decompressed_data)
    reversible = (original_hash == decompressed_hash and len(original_data) == len(decompressed_data))
    
    print(f"   Decompressed: {len(decompressed_data):,} bytes in {decompress_time:.2f}s")
    
    if reversible:
        print(f"   🎉 REVERSIBLE: ✅ YES")
        
        # 圧縮ファイル保存
        compressed_filename = f"{filename}.nxz"
        with open(compressed_filename, 'wb') as f:
            f.write(compressed_data)
        print(f"   💾 Saved: {compressed_filename}")
        
        # 復元ファイル保存
        restored_filename = f"{filename}_restored"
        with open(restored_filename, 'wb') as f:
            f.write(decompressed_data)
        print(f"   📤 Restored: {restored_filename}")
        
        return True
    else:
        print(f"   ❌ REVERSIBLE: NO")
        print(f"      Size: {len(original_data)} vs {len(decompressed_data)}")
        print(f"      Hash: {original_hash} vs {decompressed_hash}")
        return False

def main():
    """メイン処理"""
    # 圧縮エンジン初期化
    compressor = NexusAdvancedCompressor(use_ai=True, max_recursion_level=0)
    
    # テストファイルリスト
    test_files = [
        "test_small.txt",
        "element_test_small.bin", 
        "element_test_medium.bin"
    ]
    
    success_count = 0
    total_count = len(test_files)
    
    print("🔥 NEXUS Individual File Compression Test (Silent Mode)")
    print("=" * 60)
    
    for filename in test_files:
        print()
        if test_single_file(filename, compressor):
            success_count += 1
        print("-" * 40)
    
    print()
    print(f"📊 Results: {success_count}/{total_count} files passed")
    
    if success_count == total_count:
        print("🎊 All tests PASSED!")
    else:
        print("⚠️  Some tests FAILED!")

if __name__ == "__main__":
    main()
