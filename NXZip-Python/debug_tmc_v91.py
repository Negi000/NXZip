#!/usr/bin/env python3
"""
TMC v9.1 詳細デバッグテスト
解凍プロセスの詳細分析
"""

import sys
import os

# パスの追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nxzip'))

from nxzip.engine.nexus_tmc_v91_modular import NEXUSTMCEngineV91

def debug_compression_decompression():
    """圧縮・解凍プロセスの詳細デバッグ"""
    print("🐛 TMC v9.1 詳細デバッグテスト")
    print("=" * 60)
    
    # テストデータの作成（シンプル）
    test_data = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 100
    print(f"📊 テストデータ: {len(test_data)} bytes")
    print(f"📊 データ内容: {test_data[:50]}...")
    
    # エンジン初期化
    engine = NEXUSTMCEngineV91(lightweight_mode=True)
    
    print("\n🗜️ 圧縮フェーズ")
    print("-" * 40)
    
    # 圧縮実行
    compressed_data, info = engine.compress(test_data)
    
    print(f"✅ 圧縮完了:")
    print(f"  元サイズ: {len(test_data)} bytes")
    print(f"  圧縮後: {len(compressed_data)} bytes")
    print(f"  圧縮率: {info.get('compression_ratio', 0):.2f}%")
    
    # 圧縮情報の詳細表示
    print(f"\n📋 圧縮情報:")
    print(f"  Method: {info.get('method', 'unknown')}")
    print(f"  Chunks: {len(info.get('chunks', []))}")
    
    for i, chunk_info in enumerate(info.get('chunks', [])):
        print(f"  Chunk {i+1}:")
        print(f"    Start: {chunk_info.get('start_pos', 0)}")
        print(f"    Size: {chunk_info.get('compressed_size', 0)}")
        print(f"    Original: {chunk_info.get('original_size', 0)}")
        print(f"    Transforms: {len(chunk_info.get('transforms', []))}")
        
        for j, transform in enumerate(chunk_info.get('transforms', [])):
            print(f"      Transform {j+1}: {transform.get('type', 'unknown')}")
    
    print("\n📤 解凍フェーズ")
    print("-" * 40)
    
    # 解凍実行
    try:
        decompressed_data = engine.decompress(compressed_data, info)
        
        print(f"✅ 解凍完了:")
        print(f"  圧縮データ: {len(compressed_data)} bytes")
        print(f"  解凍データ: {len(decompressed_data)} bytes")
        
        # 可逆性チェック
        if test_data == decompressed_data:
            print("🎉 可逆性テスト: 成功!")
        else:
            print("❌ 可逆性テスト: 失敗")
            print(f"  元データ長: {len(test_data)}")
            print(f"  解凍データ長: {len(decompressed_data)}")
            
            # 最初の100バイトを比較
            print(f"  元データ先頭: {test_data[:100]}")
            print(f"  解凍データ先頭: {decompressed_data[:100]}")
            
            # バイト単位で差分を確認
            min_len = min(len(test_data), len(decompressed_data))
            differences = 0
            for i in range(min_len):
                if test_data[i] != decompressed_data[i]:
                    differences += 1
                    if differences <= 10:  # 最初の10個の差分を表示
                        print(f"  差分 {i}: {test_data[i]} != {decompressed_data[i]}")
            
            print(f"  総差分数: {differences}/{min_len}")
    
    except Exception as e:
        print(f"❌ 解凍エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_compression_decompression()
