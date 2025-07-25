#!/usr/bin/env python3
"""
NEXUS理論の可逆性テスト
既存サンプルデータを使用してNEXUSマルチレイヤー統合の完全性を検証
"""

import sys
import os
import time
import hashlib

# パスの追加
sys.path.append(os.path.dirname(__file__))

from nexus_advanced_engine import NexusAdvancedCompressor

def calculate_hash(data: bytes) -> str:
    """データのSHA256ハッシュを計算"""
    return hashlib.sha256(data).hexdigest()

def test_reversibility_with_samples():
    """既存サンプルファイルで可逆性テスト"""
    print("🔍 NEXUS Theory Reversibility Test with Sample Data")
    print("=" * 70)
    
    compressor = NexusAdvancedCompressor(use_ai=True)
    
    # テスト対象ファイル
    test_files = [
        "test_small.txt",
        "COT-001.png", 
        "COT-012.png",
        "element_test_small.bin",
        "element_test_medium.bin",
        "Python基礎講座3_4月26日-3.7z",
        "出庫実績明細_202412.txt"
    ]
    
    results = []
    
    for i, filename in enumerate(test_files, 1):
        print(f"\n📁 Test {i}: {filename}")
        print("-" * 50)
        
        if not os.path.exists(filename):
            print(f"   ⚠️ File not found: {filename}")
            continue
        
        try:
            # 元データ読み込み
            with open(filename, "rb") as f:
                original_data = f.read()
            
            original_size = len(original_data)
            original_hash = calculate_hash(original_data)
            
            print(f"   Original size: {original_size:,} bytes")
            print(f"   Original hash: {original_hash[:16]}...")
            
            if original_size == 0:
                print("   ⚠️ Empty file, skipping")
                continue
            
            # 圧縮テスト
            start_time = time.time()
            compressed_data = compressor.compress(original_data)
            compress_time = time.time() - start_time
            
            compressed_size = len(compressed_data)
            compression_ratio = compressed_size / original_size * 100
            
            print(f"   Compressed size: {compressed_size:,} bytes")
            print(f"   Compression ratio: {compression_ratio:.2f}%")
            print(f"   Compression time: {compress_time:.2f}s")
            
            # 展開テスト（可逆性検証）
            start_time = time.time()
            try:
                decompressed_data = compressor.decompress(compressed_data)
                decompress_time = time.time() - start_time
                
                decompressed_hash = calculate_hash(decompressed_data)
                
                print(f"   Decompressed size: {len(decompressed_data):,} bytes")
                print(f"   Decompressed hash: {decompressed_hash[:16]}...")
                print(f"   Decompression time: {decompress_time:.2f}s")
                
                # 可逆性検証
                if original_hash == decompressed_hash:
                    print(f"   ✅ PERFECT REVERSIBILITY: Hash match!")
                    if len(original_data) == len(decompressed_data):
                        print(f"   ✅ Size match: {original_size:,} bytes")
                        reversible = True
                    else:
                        print(f"   ❌ Size mismatch: {original_size:,} ≠ {len(decompressed_data):,}")
                        reversible = False
                else:
                    print(f"   ❌ HASH MISMATCH: Data corruption detected!")
                    reversible = False
                
            except Exception as e:
                print(f"   ❌ Decompression failed: {e}")
                reversible = False
                decompress_time = 0
            
            # 結果保存
            result = {
                'filename': filename,
                'original_size': original_size,
                'compressed_size': compressed_size,
                'compression_ratio': compression_ratio,
                'reversible': reversible,
                'compress_time': compress_time,
                'decompress_time': decompress_time
            }
            results.append(result)
            
            # 圧縮効果の評価
            if compression_ratio < 50:
                print(f"   🏆 Excellent compression ({100-compression_ratio:.1f}% reduction)")
            elif compression_ratio < 80:
                print(f"   🥉 Good compression ({100-compression_ratio:.1f}% reduction)")
            elif compression_ratio < 100:
                print(f"   🔶 Minor compression ({100-compression_ratio:.1f}% reduction)")
            else:
                print(f"   📈 Expansion ({compression_ratio-100:.1f}% increase)")
                
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            continue
    
    return results

def test_extreme_cases():
    """極端なケースでの可逆性テスト"""
    print(f"\n\n🧪 Extreme Cases Reversibility Test")
    print("=" * 70)
    
    compressor = NexusAdvancedCompressor(use_ai=True)
    
    test_cases = [
        {
            "name": "Empty Data",
            "data": b""
        },
        {
            "name": "Single Byte",
            "data": b"A"
        },
        {
            "name": "All Zeros (1KB)",
            "data": b"\x00" * 1024
        },
        {
            "name": "All 255s (1KB)",
            "data": b"\xFF" * 1024
        },
        {
            "name": "Sequential Pattern",
            "data": bytes(range(256)) * 4  # 0-255 repeated 4 times
        },
        {
            "name": "Random Binary (2KB)",
            "data": os.urandom(2048)
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔬 Extreme Test {i}: {test_case['name']}")
        print("-" * 40)
        
        original_data = test_case['data']
        original_size = len(original_data)
        
        if original_size == 0:
            print("   Empty data - skipping compression test")
            continue
        
        original_hash = calculate_hash(original_data)
        print(f"   Size: {original_size:,} bytes")
        print(f"   Hash: {original_hash[:16]}...")
        
        try:
            # 圧縮・展開サイクル
            compressed = compressor.compress(original_data)
            decompressed = compressor.decompress(compressed)
            
            decompressed_hash = calculate_hash(decompressed)
            
            print(f"   Compressed: {len(compressed):,} bytes ({len(compressed)/original_size*100:.1f}%)")
            
            if original_hash == decompressed_hash and len(original_data) == len(decompressed):
                print(f"   ✅ PERFECT REVERSIBILITY")
            else:
                print(f"   ❌ REVERSIBILITY FAILED")
                if original_hash != decompressed_hash:
                    print(f"      Hash mismatch: {original_hash[:8]} ≠ {decompressed_hash[:8]}")
                if len(original_data) != len(decompressed):
                    print(f"      Size mismatch: {len(original_data)} ≠ {len(decompressed)}")
                    
        except Exception as e:
            print(f"   ❌ Error: {e}")

def summarize_results(results):
    """結果サマリー"""
    print(f"\n\n📊 Test Results Summary")
    print("=" * 70)
    
    if not results:
        print("No valid test results")
        return
    
    total_tests = len(results)
    reversible_count = sum(1 for r in results if r['reversible'])
    
    print(f"Total tests: {total_tests}")
    print(f"Reversible: {reversible_count}/{total_tests} ({reversible_count/total_tests*100:.1f}%)")
    
    if reversible_count == total_tests:
        print("🎉 ALL TESTS PASSED - NEXUS THEORY PERFECTLY IMPLEMENTED!")
    else:
        print(f"⚠️ {total_tests - reversible_count} test(s) failed reversibility")
    
    # 圧縮効率統計
    compression_ratios = [r['compression_ratio'] for r in results]
    avg_ratio = sum(compression_ratios) / len(compression_ratios)
    
    print(f"\nCompression Statistics:")
    print(f"Average compression ratio: {avg_ratio:.2f}%")
    print(f"Best compression: {min(compression_ratios):.2f}%")
    print(f"Worst case: {max(compression_ratios):.2f}%")
    
    # 処理時間統計
    total_compress_time = sum(r['compress_time'] for r in results)
    total_decompress_time = sum(r['decompress_time'] for r in results)
    
    print(f"\nPerformance Statistics:")
    print(f"Total compression time: {total_compress_time:.2f}s")
    print(f"Total decompression time: {total_decompress_time:.2f}s")
    print(f"Average compression speed: {sum(r['original_size'] for r in results)/total_compress_time/1024:.1f} KB/s")

def main():
    """メインテスト実行"""
    print("🎯 NEXUS Multi-Layer Theory Reversibility Validation")
    print("Testing NEXUS implementation with existing sample data")
    print("Verifying complete data integrity and compression effectiveness")
    print()
    
    # 既存サンプルファイルでのテスト
    results = test_reversibility_with_samples()
    
    # 極端ケースでのテスト
    test_extreme_cases()
    
    # 結果サマリー
    summarize_results(results)
    
    print("\n" + "=" * 70)
    print("✅ NEXUS Reversibility Test Complete")
    print("NEXUS Multi-Layer Consolidation System validation finished")

if __name__ == "__main__":
    main()
