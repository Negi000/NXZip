#!/usr/bin/env python3
"""
NEXUSマルチレイヤー統合システムのテスト
理論的限界を突破する4段階統合の効果検証
"""

import sys
import os
import time

# パスの追加
sys.path.append(os.path.dirname(__file__))

from nexus_advanced_engine import NexusAdvancedCompressor

def test_multilayer_on_different_data():
    """異なるタイプのデータでマルチレイヤー統合をテスト"""
    print("🔥 NEXUS Multi-Layer Consolidation Test")
    print("=" * 60)
    
    compressor = NexusAdvancedCompressor(use_ai=True)
    
    # テストデータセット
    test_datasets = [
        {
            "name": "Compressed 7z Data (High Entropy)",
            "file": "../test-data/small_test.7z"
        },
        {
            "name": "PNG Image Data (Structured)",
            "file": "../test-data/small_test.png"
        },
        {
            "name": "Random Binary Data",
            "data": os.urandom(5000)  # 5KB random
        },
        {
            "name": "Repetitive Pattern Data",
            "data": b"ABCDEFGH" * 625  # 5KB pattern
        }
    ]
    
    for i, dataset in enumerate(test_datasets, 1):
        print(f"\n📊 Test {i}: {dataset['name']}")
        print("-" * 50)
        
        # データ読み込み
        if "file" in dataset:
            try:
                with open(dataset["file"], "rb") as f:
                    data = f.read()
                print(f"   File size: {len(data):,} bytes")
            except FileNotFoundError:
                print(f"   ⚠️ File not found: {dataset['file']}")
                continue
        else:
            data = dataset["data"]
            print(f"   Data size: {len(data):,} bytes")
        
        if len(data) == 0:
            print("   ⚠️ Empty data, skipping")
            continue
        
        # マルチレイヤー統合テスト
        start_time = time.time()
        
        try:
            result = compressor.compress(data)
            
            processing_time = time.time() - start_time
            
            # 結果分析
            original_size = len(data)
            compressed_size = len(result)
            ratio = compressed_size / original_size * 100
            
            print(f"   Original size: {original_size:,} bytes")
            print(f"   Result size: {compressed_size:,} bytes")
            print(f"   Compression ratio: {ratio:.2f}%")
            print(f"   Processing time: {processing_time:.2f}s")
            
            if ratio < 100:
                print(f"   ✅ COMPRESSION ACHIEVED! ({100-ratio:.1f}% reduction)")
            elif ratio < 150:
                print(f"   🔶 Slight expansion ({ratio-100:.1f}% increase)")
            else:
                print(f"   ❌ Significant expansion ({ratio-100:.1f}% increase)")
                
        except Exception as e:
            print(f"   ❌ Error during compression: {e}")
            continue

def test_theoretical_limits():
    """理論的限界のテスト"""
    print("\n\n🧠 Theoretical Limits Analysis")
    print("=" * 60)
    
    compressor = NexusAdvancedCompressor(use_ai=True)
    
    # 理論的に圧縮不可能なデータ
    print("\n📈 Testing Incompressible Data (Random)")
    random_data = os.urandom(2048)  # 2KB pure random
    
    try:
        start_time = time.time()
        result = compressor.compress(random_data)
        processing_time = time.time() - start_time
        
        ratio = len(result) / len(random_data) * 100
        print(f"   Random data: {len(random_data):,} → {len(result):,} bytes ({ratio:.1f}%)")
        print(f"   Processing time: {processing_time:.2f}s")
        
        if ratio > 100:
            print(f"   ✅ Expected expansion for random data ({ratio-100:.1f}% increase)")
        else:
            print(f"   🤔 Unexpected compression of random data")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 理論的に圧縮可能なデータ
    print("\n📉 Testing Highly Compressible Data (Repetitive)")
    repetitive_data = b"A" * 2048  # 2KB all same byte
    
    try:
        start_time = time.time()
        result = compressor.compress(repetitive_data)
        processing_time = time.time() - start_time
        
        ratio = len(result) / len(repetitive_data) * 100
        print(f"   Repetitive data: {len(repetitive_data):,} → {len(result):,} bytes ({ratio:.1f}%)")
        print(f"   Processing time: {processing_time:.2f}s")
        
        if ratio < 50:
            print(f"   ✅ Excellent compression ({100-ratio:.1f}% reduction)")
        elif ratio < 100:
            print(f"   🔶 Good compression ({100-ratio:.1f}% reduction)")
        else:
            print(f"   ❌ Failed to compress repetitive data")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")

def main():
    """メインテスト実行"""
    print("🎯 NEXUS Multi-Layer Consolidation System Test")
    print("Testing 4-layer consolidation algorithm:")
    print("  Layer 1: Exact Match Consolidation")
    print("  Layer 2: Pattern-Based Consolidation") 
    print("  Layer 3: Approximate Consolidation (Compressed Data Optimized)")
    print("  Layer 4: Structural Consolidation")
    print()
    
    # マルチレイヤーテスト
    test_multilayer_on_different_data()
    
    # 理論限界テスト
    test_theoretical_limits()
    
    print("\n" + "=" * 60)
    print("✅ Multi-Layer Consolidation Test Complete")
    print("Check results above to validate NEXUS theory improvements")

if __name__ == "__main__":
    main()
