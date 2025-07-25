#!/usr/bin/env python3
"""
構成要素ベースの統合機能をテストするための専用スクリプト
様々なパターンのデータを生成して統合効果を検証
"""

import os
import random
import time

def create_element_test_file(filename: str, size: int = 50000):
    """構成要素統合のテストに最適化されたデータファイルを生成"""
    print(f"Creating element consolidation test file: {filename}")
    
    with open(filename, 'wb') as f:
        # パターン1: 同じ要素を持つが順序が異なるブロックを意図的に作成
        # 例: [1,2,3], [3,1,2], [2,3,1] - 全て同じ要素だが順序が違う
        
        # 基本パターンセット
        base_patterns = [
            [1, 2, 3, 4],
            [5, 6, 7, 8], 
            [9, 10, 11, 12],
            [13, 14, 15, 16],
            [17, 18, 19, 20]
        ]
        
        # 各基本パターンを複数の順序で配置
        for _ in range(size // 100):
            for pattern in base_patterns:
                # 元の順序
                f.write(bytes(pattern))
                
                # シャッフルされた順序（複数バリエーション）
                shuffled1 = pattern.copy()
                random.shuffle(shuffled1)
                f.write(bytes(shuffled1))
                
                shuffled2 = pattern.copy()
                random.shuffle(shuffled2)
                f.write(bytes(shuffled2))
                
                # 逆順
                reversed_pattern = list(reversed(pattern))
                f.write(bytes(reversed_pattern))
                
                # 回転パターン
                rotated = pattern[1:] + [pattern[0]]
                f.write(bytes(rotated))
        
        # パターン2: ランダムなノイズを少量追加
        for _ in range(size // 1000):
            noise = [random.randint(21, 255) for _ in range(4)]
            f.write(bytes(noise))
    
    print(f"Generated {filename} ({os.path.getsize(filename):,} bytes)")

def run_compression_test(input_file: str, output_file: str):
    """圧縮テストを実行"""
    print(f"\n=== Testing: {input_file} ===")
    
    # 構成要素統合版で圧縮
    start_time = time.time()
    os.system(f'python nexus_advanced_engine.py compress "{input_file}" "{output_file}"')
    end_time = time.time()
    
    # 結果表示
    original_size = os.path.getsize(input_file)
    compressed_size = os.path.getsize(output_file)
    compression_ratio = compressed_size / original_size
    processing_time = end_time - start_time
    
    print(f"\n📊 Test Results:")
    print(f"   Input file: {input_file}")
    print(f"   Original size: {original_size:,} bytes")
    print(f"   Compressed size: {compressed_size:,} bytes")
    print(f"   Compression ratio: {compression_ratio:.4f}")
    print(f"   Size reduction: {(1-compression_ratio)*100:.2f}%")
    print(f"   Processing time: {processing_time:.2f} seconds")
    print(f"   Speed: {original_size/(1024*1024*processing_time):.2f} MB/sec")

def main():
    print("🧪 Element-Based Consolidation Test Suite")
    print("=" * 50)
    
    # テストファイル1: 小サイズ（詳細観察用）
    test_file_1 = "element_test_small.bin"
    create_element_test_file(test_file_1, 10000)
    run_compression_test(test_file_1, "element_test_small.nxz")
    
    # テストファイル2: 中サイズ（効果確認用）
    test_file_2 = "element_test_medium.bin"
    create_element_test_file(test_file_2, 50000)
    run_compression_test(test_file_2, "element_test_medium.nxz")
    
    # テストファイル3: 大サイズ（性能確認用）
    test_file_3 = "element_test_large.bin"
    create_element_test_file(test_file_3, 200000)
    run_compression_test(test_file_3, "element_test_large.nxz")
    
    print("\n🎯 Element Consolidation Test Complete!")
    print("Check the consolidation rates in the output above.")

if __name__ == "__main__":
    main()
