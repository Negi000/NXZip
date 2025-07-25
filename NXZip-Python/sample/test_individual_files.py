#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
簡潔ログ版：実際のサンプルファイルでのNEXUS可逆性テスト
"""

import os
import hashlib
from nexus_advanced_engine import NexusAdvancedCompressor

def calculate_hash(data):
    """データのハッシュ値を計算"""
    return hashlib.md5(data).hexdigest()

def test_single_file(file_path):
    """単一ファイルの圧縮・解凍テスト（簡潔ログ版）"""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    filename = os.path.basename(file_path)
    print(f"\n📁 Testing: {filename}")
    
    try:
        # ファイル読み込み
        with open(file_path, 'rb') as f:
            original_data = f.read()
        
        original_size = len(original_data)
        original_hash = calculate_hash(original_data)
        
        print(f"   Original: {original_size:,} bytes")
        
        # 圧縮テスト
        compressor = NexusAdvancedCompressor()
        
        # 圧縮実行
        compressed = compressor.compress(original_data, level=0)
        compressed_size = len(compressed)
        ratio = compressed_size / original_size
        
        print(f"   Compressed: {compressed_size:,} bytes (ratio: {ratio:.4f})")
        
        # 解凍実行
        decompressed = compressor.decompress(compressed, level=0)
        decompressed_size = len(decompressed)
        decompressed_hash = calculate_hash(decompressed)
        
        # 可逆性チェック
        is_reversible = (original_hash == decompressed_hash and original_size == decompressed_size)
        
        if is_reversible:
            print(f"   🎉 REVERSIBLE: ✅ YES")
            
            # .nxzファイルに保存
            nxz_path = file_path + ".nxz"
            with open(nxz_path, 'wb') as f:
                f.write(compressed)
            print(f"   💾 Saved: {os.path.basename(nxz_path)}")
            
            # .nxzから読み込み・解凍テスト
            with open(nxz_path, 'rb') as f:
                loaded_compressed = f.read()
            
            restored = compressor.decompress(loaded_compressed, level=0)
            restored_hash = calculate_hash(restored)
            
            if restored_hash == original_hash:
                print(f"   🔄 File round-trip: ✅ YES")
                
                # 復元ファイル保存
                restored_path = file_path + "_restored"
                if file_path.endswith('.txt'):
                    restored_path += '.txt'
                elif file_path.endswith('.png'):
                    restored_path += '.png'
                elif file_path.endswith('.mp3'):
                    restored_path += '.mp3'
                elif file_path.endswith('.mp4'):
                    restored_path += '.mp4'
                
                with open(restored_path, 'wb') as f:
                    f.write(restored)
                print(f"   📤 Restored: {os.path.basename(restored_path)}")
                
            else:
                print(f"   ❌ File round-trip: FAILED")
                return False
        else:
            print(f"   ❌ REVERSIBLE: NO")
            if original_size != decompressed_size:
                print(f"      Size: {original_size} != {decompressed_size}")
            if original_hash != decompressed_hash:
                print(f"      Hash: {original_hash} != {decompressed_hash}")
            return False
        
        return is_reversible
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """個別ファイルテスト実行"""
    print("🚀 NEXUS Advanced Engine - Individual File Test")
    print("Testing real sample files with minimal logging")
    
    # 利用可能なテストファイルをチェック
    test_files = [
        "test_small.txt",
        "element_test_small.bin", 
        "element_test_medium.bin",
        "COT-001.png",
        "陰謀論.mp3"
    ]
    
    print(f"\n📂 Available files:")
    for filename in test_files:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"   ✅ {filename} ({size:,} bytes)")
        else:
            print(f"   ❌ {filename} (not found)")
    
    print(f"\n" + "="*60)
    
    # 各ファイルを個別にテスト
    results = []
    for filename in test_files:
        if os.path.exists(filename):
            result = test_single_file(filename)
            results.append((filename, result))
    
    # 結果サマリー
    print(f"\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(1 for _, result in results if result)
    
    for filename, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} | {filename}")
    
    print(f"\n🎯 Success Rate: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! NEXUS achieves complete reversibility!")
    else:
        print("⚠️  Some tests failed. Check individual file results above.")

if __name__ == "__main__":
    main()
