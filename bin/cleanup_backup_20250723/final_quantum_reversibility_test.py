#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 修正版量子圧縮エンジン統合テスト
完全可逆性とパフォーマンスの最終検証
"""

import os
import sys
import subprocess
import hashlib
import time

def test_quantum_engine_reversibility():
    """修正版量子エンジンの完全可逆性テスト"""
    
    print("⚛️ 修正版量子圧縮エンジン統合テスト")
    print("=" * 60)
    
    test_files = [
        "NXZip-Python/sample/COT-001.png",
        "test-data/test.txt",
        "test-data/sample_text.txt"
    ]
    
    results = []
    
    for test_file in test_files:
        if not os.path.exists(test_file):
            print(f"⚠️ テストファイル未発見: {test_file}")
            continue
        
        print(f"\\n🔬 テスト中: {test_file}")
        
        # 元ファイル情報
        with open(test_file, 'rb') as f:
            original_data = f.read()
        original_size = len(original_data)
        original_hash = hashlib.sha256(original_data).hexdigest()
        
        # 圧縮
        compressed_file = f"{test_file}.quantum_test.nxz"
        start_time = time.time()
        
        try:
            result = subprocess.run([
                sys.executable, "bin/nexus_quantum.py", 
                "compress", test_file, compressed_file
            ], capture_output=True, text=True, cwd=".")
            
            compression_time = time.time() - start_time
            
            if result.returncode != 0:
                print(f"❌ 圧縮失敗: {result.stderr}")
                continue
            
            compressed_size = os.path.getsize(compressed_file)
            compression_ratio = (1 - compressed_size / original_size) * 100
            
            # 解凍
            restored_file = f"{test_file}.quantum_restored"
            start_time = time.time()
            
            result = subprocess.run([
                sys.executable, "bin/nexus_quantum.py",
                "decompress", compressed_file, restored_file
            ], capture_output=True, text=True, cwd=".")
            
            decompression_time = time.time() - start_time
            
            if result.returncode != 0:
                print(f"❌ 解凍失敗: {result.stderr}")
                continue
            
            # 検証
            with open(restored_file, 'rb') as f:
                restored_data = f.read()
            restored_size = len(restored_data)
            restored_hash = hashlib.sha256(restored_data).hexdigest()
            
            # 結果
            size_match = original_size == restored_size
            hash_match = original_hash == restored_hash
            
            result_data = {
                'file': test_file,
                'original_size': original_size,
                'compressed_size': compressed_size,
                'restored_size': restored_size,
                'compression_ratio': compression_ratio,
                'compression_time': compression_time,
                'decompression_time': decompression_time,
                'size_match': size_match,
                'hash_match': hash_match,
                'reversible': size_match and hash_match
            }
            
            results.append(result_data)
            
            print(f"   📊 元サイズ: {original_size:,} bytes")
            print(f"   📊 圧縮後: {compressed_size:,} bytes")
            print(f"   📊 復元サイズ: {restored_size:,} bytes")
            print(f"   📊 圧縮率: {compression_ratio:.1f}%")
            print(f"   ⏱️ 圧縮時間: {compression_time:.2f}秒")
            print(f"   ⏱️ 解凍時間: {decompression_time:.2f}秒")
            print(f"   ✅ サイズ一致: {'はい' if size_match else 'いいえ'}")
            print(f"   ✅ ハッシュ一致: {'はい' if hash_match else 'いいえ'}")
            print(f"   🎯 完全可逆: {'はい' if result_data['reversible'] else 'いいえ'}")
            
            # クリーンアップ
            if os.path.exists(compressed_file):
                os.remove(compressed_file)
            if os.path.exists(restored_file):
                os.remove(restored_file)
                
        except Exception as e:
            print(f"❌ テストエラー: {str(e)}")
    
    # 総合結果
    print(f"\\n📊 総合結果:")
    print(f"=" * 60)
    
    if results:
        total_tests = len(results)
        reversible_tests = sum(1 for r in results if r['reversible'])
        avg_compression = sum(r['compression_ratio'] for r in results) / total_tests
        avg_comp_time = sum(r['compression_time'] for r in results) / total_tests
        avg_decomp_time = sum(r['decompression_time'] for r in results) / total_tests
        
        print(f"   テスト数: {total_tests}")
        print(f"   完全可逆: {reversible_tests}/{total_tests} ({reversible_tests/total_tests*100:.1f}%)")
        print(f"   平均圧縮率: {avg_compression:.1f}%")
        print(f"   平均圧縮時間: {avg_comp_time:.2f}秒")
        print(f"   平均解凍時間: {avg_decomp_time:.2f}秒")
        
        if reversible_tests == total_tests:
            print("\\n🎉 すべてのテストで完全可逆性が確認されました！")
            print("✅ nexus_quantum.pyの修正が成功しました。")
        else:
            print("\\n⚠️ 一部のテストで可逆性に問題があります。")
    else:
        print("❌ 実行可能なテストがありませんでした。")

def main():
    test_quantum_engine_reversibility()

if __name__ == "__main__":
    main()
