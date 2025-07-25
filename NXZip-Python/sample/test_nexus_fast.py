#!/usr/bin/env python3
"""
NEXUS理論の高速可逆性テスト（小さなファイル優先）
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

def test_small_files_first():
    """小さなファイルを優先してテスト"""
    print("🚀 NEXUS Fast Reversibility Test (Small Files Priority)")
    print("=" * 70)
    
    compressor = NexusAdvancedCompressor(use_ai=True)
    
    # ファイルサイズ順でソート
    test_files = [
        "test_small.txt",
        "element_test_small.bin",
        "element_test_medium.bin"
        # 大きなファイルは除外して高速テスト
    ]
    
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
            
            if original_size == 0:
                print("   ⚠️ Empty file")
                continue
            
            if original_size > 1000000:  # 1MB以上は処理時間表示
                print(f"   ⚠️ Large file ({original_size/1024/1024:.1f}MB) - may take time")
            
            # 圧縮テスト
            start_time = time.time()
            compressed_data = compressor.compress(original_data)
            compress_time = time.time() - start_time
            
            compressed_size = len(compressed_data)
            compression_ratio = compressed_size / original_size * 100
            
            print(f"   Compressed: {compressed_size:,} bytes ({compression_ratio:.1f}%)")
            print(f"   Compress time: {compress_time:.2f}s")
            
            # 展開テスト
            start_time = time.time()
            try:
                decompressed_data = compressor.decompress(compressed_data)
                decompress_time = time.time() - start_time
                
                decompressed_hash = calculate_hash(decompressed_data)
                
                print(f"   Decompressed: {len(decompressed_data):,} bytes")
                print(f"   Decompress time: {decompress_time:.2f}s")
                
                # 可逆性検証
                if original_hash == decompressed_hash and len(original_data) == len(decompressed_data):
                    print(f"   ✅ PERFECT REVERSIBILITY")
                    
                    if compression_ratio < 80:
                        print(f"   🏆 Good compression ({100-compression_ratio:.1f}% reduction)")
                    elif compression_ratio < 100:
                        print(f"   🔶 Minor compression ({100-compression_ratio:.1f}% reduction)")
                    else:
                        print(f"   📈 Expansion ({compression_ratio-100:.1f}% increase)")
                else:
                    print(f"   ❌ REVERSIBILITY FAILED")
                    if original_hash != decompressed_hash:
                        print(f"      Hash mismatch!")
                    if len(original_data) != len(decompressed_data):
                        print(f"      Size mismatch: {len(original_data)} ≠ {len(decompressed_data)}")
                
            except Exception as e:
                print(f"   ❌ Decompression failed: {e}")
                
        except Exception as e:
            print(f"   ❌ Test failed: {e}")

def test_edge_cases():
    """エッジケースのテスト"""
    print(f"\n\n🧪 Edge Cases Test")
    print("=" * 70)
    
    compressor = NexusAdvancedCompressor(use_ai=True)
    
    test_cases = [
        ("Empty", b""),
        ("Single byte", b"A"),
        ("Zeros (512B)", b"\x00" * 512),
        ("Sequential", bytes(range(100))),
        ("Random (1KB)", os.urandom(1024))
    ]
    
    for name, data in test_cases:
        print(f"\n🔬 {name}: {len(data)} bytes")
        
        if len(data) == 0:
            print("   Skip empty data")
            continue
        
        try:
            original_hash = calculate_hash(data)
            compressed = compressor.compress(data)
            decompressed = compressor.decompress(compressed)
            decompressed_hash = calculate_hash(decompressed)
            
            ratio = len(compressed) / len(data) * 100 if len(data) > 0 else 0
            print(f"   Ratio: {ratio:.1f}%")
            
            if original_hash == decompressed_hash and len(data) == len(decompressed):
                print(f"   ✅ Reversible")
            else:
                print(f"   ❌ Failed")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

def main():
    """メインテスト"""
    print("🎯 NEXUS Multi-Layer Theory - Fast Validation")
    print("Priority: Small files → Edge cases → Performance verification")
    print()
    
    # 小さなファイル優先テスト
    test_small_files_first()
    
    # エッジケーステスト
    test_edge_cases()
    
    print("\n" + "=" * 70)
    print("✅ Fast NEXUS Validation Complete")

if __name__ == "__main__":
    main()
