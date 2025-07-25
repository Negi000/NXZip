#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
実際のサンプルファイルでのNEXUS可逆性テスト
"""

import os
import hashlib
from nexus_advanced_engine import NexusAdvancedCompressor

def calculate_hash(data):
    """データのハッシュ値を計算"""
    return hashlib.md5(data).hexdigest()

def test_file_compression(file_path, description=""):
    """ファイル圧縮の可逆性テスト"""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    print(f"\n{'='*80}")
    print(f"📁 Testing: {os.path.basename(file_path)} {description}")
    print(f"{'='*80}")
    
    # ファイル読み込み
    with open(file_path, 'rb') as f:
        original_data = f.read()
    
    original_size = len(original_data)
    original_hash = calculate_hash(original_data)
    
    print(f"Original: {original_size:,} bytes | Hash: {original_hash}")
    
    # 圧縮テスト
    compressor = NexusAdvancedCompressor()
    
    try:
        # 圧縮
        compressed = compressor.compress(original_data, level=0)
        compressed_size = len(compressed)
        
        print(f"Compressed: {compressed_size:,} bytes")
        print(f"Compression ratio: {compressed_size/original_size:.4f}")
        
        # 解凍
        decompressed = compressor.decompress(compressed, level=0)
        decompressed_size = len(decompressed)
        decompressed_hash = calculate_hash(decompressed)
        
        print(f"Decompressed: {decompressed_size:,} bytes | Hash: {decompressed_hash}")
        
        # 可逆性チェック
        is_reversible = (original_hash == decompressed_hash and original_size == decompressed_size)
        
        if is_reversible:
            print("🎉 REVERSIBLE: ✅ YES")
        else:
            print("❌ REVERSIBLE: NO")
            if original_size != decompressed_size:
                print(f"   Size mismatch: {original_size} != {decompressed_size}")
            if original_hash != decompressed_hash:
                print(f"   Hash mismatch: {original_hash} != {decompressed_hash}")
        
        return is_reversible
        
    except Exception as e:
        print(f"❌ Error during compression/decompression: {e}")
        return False

def main():
    """メインテスト実行"""
    print("🚀 NEXUS Advanced Engine - Sample File Reversibility Test")
    print("Testing various file types with pure NEXUS theory (no fallbacks)")
    
    # テストファイルリスト
    test_files = [
        ("test_small.txt", "(Small text file)"),
        ("element_test_small.bin", "(Small binary)"),
        ("element_test_medium.bin", "(Medium binary)"),
        ("COT-001.png", "(PNG image)"),
        ("陰謀論.mp3", "(MP3 audio)"),
        ("Python基礎講座3_4月26日-3.mp4", "(MP4 video)"),
    ]
    
    results = []
    
    for filename, description in test_files:
        file_path = filename
        result = test_file_compression(file_path, description)
        results.append((filename, result))
    
    # 結果サマリー
    print(f"\n{'='*80}")
    print("📊 REVERSIBILITY TEST RESULTS SUMMARY")
    print(f"{'='*80}")
    
    total_tests = len(results)
    passed_tests = sum(1 for _, result in results if result)
    
    for filename, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} | {filename}")
    
    print(f"\n🎯 Success Rate: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Pure NEXUS theory achieves complete reversibility!")
    else:
        print("⚠️  Some tests failed. Need further optimization.")

if __name__ == "__main__":
    main()
