#!/usr/bin/env python3
"""
🧪 Real File Format Testing Suite
実際のファイルでUniversal Ultra Compression Engine v8.0をテスト
"""

import os
import sys
from universal_compressor_v8_fixed import UniversalUltraCompressor

def test_real_files():
    """実際のファイルでテスト"""
    base_path = r"c:\Users\241822\Desktop\新しいフォルダー (2)\NXZip"
    
    # テスト対象ファイル
    test_files = [
        r"test-data\test.txt",
        r"test-data\large_test.txt", 
        r"package.json",
        r"gui\tsconfig.json",
        r"test-data\repetitive_test.txt"
    ]
    
    compressor = UniversalUltraCompressor()
    
    print("🧪 Real File Testing - Universal Ultra Compression Engine v8.0")
    print("=" * 70)
    
    for rel_path in test_files:
        full_path = os.path.join(base_path, rel_path)
        
        if not os.path.exists(full_path):
            print(f"❌ ファイルが見つかりません: {rel_path}")
            continue
            
        try:
            with open(full_path, 'rb') as f:
                data = f.read()
            
            filename = os.path.basename(full_path)
            file_size = len(data)
            
            print(f"\n📁 実ファイル: {rel_path}")
            print(f"📊 サイズ: {file_size:,} bytes")
            
            if file_size == 0:
                print("⚠️  空ファイルです")
                continue
                
            # 圧縮実行
            compressed, stats = compressor.compress(data, filename)
            
            print(f"🔍 検出形式: {stats['detected_format']}")
            print(f"📈 圧縮率: {stats['compression_ratio']:.3f}%")
            print(f"⚡ 速度: {stats['speed_mbps']:.2f} MB/s")
            print(f"⏱️  時間: {stats['processing_time']:.3f}秒")
            
            # 7Zipとの比較評価
            if stats['compression_ratio'] > 99.0:
                print("🏆 優秀: 99%超の圧縮率!")
            elif stats['compression_ratio'] > 95.0:
                print("✅ 良好: 95%超の圧縮率")
            elif stats['compression_ratio'] > 90.0:
                print("📈 普通: 90%超の圧縮率")
            else:
                print("⚠️  改善余地あり")
            
        except Exception as e:
            print(f"❌ エラー: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    test_real_files()
