#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔥 NEXUS COMPREHENSIVE SAMPLE TEST 🔥
サンプルファイルでの圧縮率テストと詳細分析
"""

import os
import time
import hashlib
from pathlib import Path
from nexus_advanced_engine import NexusAdvancedCompressor

def format_bytes(bytes_val):
    """バイト数を読みやすい形式にフォーマット"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"

def calculate_md5(data):
    """MD5ハッシュを計算"""
    return hashlib.md5(data).hexdigest()

def test_file_compression(file_path, compressor):
    """ファイル圧縮テストを実行"""
    print(f"\n🔥 TESTING: {file_path}")
    print("=" * 60)
    
    # ファイル読み込み
    try:
        with open(file_path, 'rb') as f:
            original_data = f.read()
    except Exception as e:
        print(f"❌ Failed to read file: {e}")
        return None
    
    original_size = len(original_data)
    original_md5 = calculate_md5(original_data)
    
    print(f"📄 File size: {format_bytes(original_size)} ({original_size:,} bytes)")
    print(f"🔍 Original MD5: {original_md5}")
    
    # 圧縮実行
    print("\n🔥 NEXUS COMPRESSION STARTING...")
    start_time = time.time()
    
    try:
        compressed_data = compressor.compress(original_data, silent=False)
        compression_time = time.time() - start_time
        
        compressed_size = len(compressed_data)
        compression_ratio = compressed_size / original_size
        space_saved = original_size - compressed_size
        space_saved_percent = (space_saved / original_size) * 100
        
        print(f"\n✅ COMPRESSION COMPLETE!")
        print(f"⏱️  Compression time: {compression_time:.2f}s")
        print(f"📦 Compressed size: {format_bytes(compressed_size)} ({compressed_size:,} bytes)")
        print(f"📊 Compression ratio: {compression_ratio:.4f} ({compression_ratio:.2%})")
        print(f"💾 Space saved: {format_bytes(space_saved)} ({space_saved_percent:.2f}%)")
        
    except Exception as e:
        print(f"❌ COMPRESSION FAILED: {e}")
        return None
    
    # 解凍実行
    print("\n🔥 NEXUS DECOMPRESSION STARTING...")
    start_time = time.time()
    
    try:
        decompressed_data = compressor.decompress(compressed_data, silent=False)
        decompression_time = time.time() - start_time
        
        decompressed_md5 = calculate_md5(decompressed_data)
        
        print(f"\n✅ DECOMPRESSION COMPLETE!")
        print(f"⏱️  Decompression time: {decompression_time:.2f}s")
        print(f"📄 Decompressed size: {format_bytes(len(decompressed_data))} ({len(decompressed_data):,} bytes)")
        print(f"🔍 Decompressed MD5: {decompressed_md5}")
        
        # 整合性確認
        if original_md5 == decompressed_md5:
            print("🎯 ✅ PERFECT MATCH - NEXUS INFECTION SUCCESSFUL!")
            integrity_status = "✅ SUCCESS"
        else:
            print("⚠️  ❌ MD5 MISMATCH - NEXUS INFECTION INCOMPLETE")
            integrity_status = "❌ FAILED"
            
            # バイト単位比較
            mismatch_count = 0
            for i, (orig, decomp) in enumerate(zip(original_data, decompressed_data)):
                if orig != decomp:
                    mismatch_count += 1
                    if mismatch_count <= 10:  # 最初の10個のみ表示
                        print(f"   Byte {i}: original={orig} != decompressed={decomp}")
            
            if len(original_data) != len(decompressed_data):
                print(f"   Length mismatch: original={len(original_data)} vs decompressed={len(decompressed_data)}")
        
    except Exception as e:
        print(f"❌ DECOMPRESSION FAILED: {e}")
        integrity_status = "❌ ERROR"
        compression_ratio = float('inf')
        space_saved_percent = 0
        compression_time = 0
        decompression_time = 0
    
    return {
        'file_path': file_path,
        'original_size': original_size,
        'compressed_size': compressed_size if 'compressed_size' in locals() else 0,
        'compression_ratio': compression_ratio if 'compression_ratio' in locals() else float('inf'),
        'space_saved_percent': space_saved_percent if 'space_saved_percent' in locals() else 0,
        'compression_time': compression_time if 'compression_time' in locals() else 0,
        'decompression_time': decompression_time if 'decompression_time' in locals() else 0,
        'integrity_status': integrity_status
    }

def main():
    """メイン実行関数"""
    print("🔥" * 30)
    print("🔥 NEXUS COMPREHENSIVE SAMPLE TEST 🔥")
    print("🔥" * 30)
    
    # NEXUS圧縮エンジン初期化
    print("\n🚀 Initializing NEXUS Advanced Compressor...")
    compressor = NexusAdvancedCompressor(use_ai=True, max_recursion_level=0)
    print("✅ NEXUS Engine ready for infection!")
    
    # サンプルファイル検索
    sample_dir = Path("c:/Users/241822/Desktop/新しいフォルダー (2)/NXZip/sample")
    
    # テスト対象ファイル
    test_files = []
    
    # sample ディレクトリのファイルを検索
    if sample_dir.exists():
        for file_path in sample_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in ['.txt', '.bin', '.dat', '.log']:
                test_files.append(str(file_path))
    
    # 追加のテストファイル（存在する場合）
    additional_files = [
        "c:/Users/241822/Desktop/新しいフォルダー (2)/NXZip/test-data/COT-001_final_restored.png",
        "c:/Users/241822/Desktop/新しいフォルダー (2)/NXZip/bin/medium_test.png",
        "c:/Users/241822/Desktop/新しいフォルダー (2)/NXZip/README.md"
    ]
    
    for file_path in additional_files:
        if os.path.exists(file_path):
            test_files.append(file_path)
    
    if not test_files:
        print("❌ No test files found!")
        return
    
    print(f"\n📁 Found {len(test_files)} test files:")
    for file_path in test_files:
        print(f"   📄 {file_path}")
    
    # 各ファイルでテスト実行
    results = []
    
    for file_path in test_files:
        result = test_file_compression(file_path, compressor)
        if result:
            results.append(result)
    
    # 総合結果レポート
    print("\n" + "🔥" * 60)
    print("🔥 NEXUS COMPREHENSIVE TEST RESULTS 🔥")
    print("🔥" * 60)
    
    if results:
        total_original_size = sum(r['original_size'] for r in results)
        total_compressed_size = sum(r['compressed_size'] for r in results)
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   📁 Files tested: {len(results)}")
        print(f"   📄 Total original size: {format_bytes(total_original_size)} ({total_original_size:,} bytes)")
        print(f"   📦 Total compressed size: {format_bytes(total_compressed_size)} ({total_compressed_size:,} bytes)")
        
        if total_original_size > 0:
            overall_ratio = total_compressed_size / total_original_size
            overall_saved = ((total_original_size - total_compressed_size) / total_original_size) * 100
            print(f"   📊 Overall compression ratio: {overall_ratio:.4f} ({overall_ratio:.2%})")
            print(f"   💾 Overall space saved: {overall_saved:.2f}%")
        
        # 個別ファイル結果
        print(f"\n📋 INDIVIDUAL FILE RESULTS:")
        print("-" * 120)
        print(f"{'File':<40} {'Size':<12} {'Ratio':<8} {'Saved':<8} {'Status':<10} {'Time':<10}")
        print("-" * 120)
        
        for result in results:
            filename = os.path.basename(result['file_path'])
            size_str = format_bytes(result['original_size'])
            ratio_str = f"{result['compression_ratio']:.3f}"
            saved_str = f"{result['space_saved_percent']:.1f}%"
            time_str = f"{result['compression_time']:.2f}s"
            
            print(f"{filename:<40} {size_str:<12} {ratio_str:<8} {saved_str:<8} {result['integrity_status']:<10} {time_str:<10}")
        
        # 成功率統計
        success_count = sum(1 for r in results if 'SUCCESS' in r['integrity_status'])
        success_rate = (success_count / len(results)) * 100
        
        print(f"\n🎯 SUCCESS RATE: {success_count}/{len(results)} ({success_rate:.1f}%)")
        
        if success_rate == 100.0:
            print("🚀 🎉 PERFECT NEXUS INFECTION ACHIEVED! 🎉 🚀")
        elif success_rate >= 80.0:
            print("✅ Good NEXUS infection rate - minor improvements needed")
        else:
            print("⚠️  NEXUS infection needs significant improvement")
    
    else:
        print("❌ No successful tests completed")

if __name__ == "__main__":
    main()
