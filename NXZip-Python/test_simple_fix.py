#!/usr/bin/env python3
"""
TMC v9.1 修正テスト - 簡易解凍処理
BWTの解凍処理を実際のBWTTransformerを参照して修正
"""

import sys
import os

# パスの追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nxzip'))

from nxzip.engine.nexus_tmc_v91_modular import NEXUSTMCEngineV91

def test_simple_compression():
    """シンプルなテスト"""
    print("🔧 TMC v9.1 簡易解凍テスト")
    print("=" * 50)
    
    # 非常にシンプルなデータ
    test_data = b"AAABBBCCCDDDEEEFFFGGGHHHIIIJJJKKKLLLMMM"
    print(f"📊 テストデータ: {len(test_data)} bytes")
    print(f"📊 内容: {test_data}")
    
    # エンジン初期化
    engine = NEXUSTMCEngineV91(lightweight_mode=True)
    
    # 圧縮
    print("\n🗜️ 圧縮中...")
    compressed_data, info = engine.compress(test_data)
    print(f"圧縮結果: {len(test_data)} -> {len(compressed_data)} bytes")
    
    # 解凍
    print("\n📤 解凍中...")
    try:
        decompressed_data = engine.decompress(compressed_data, info)
        print(f"解凍結果: {len(compressed_data)} -> {len(decompressed_data)} bytes")
        
        # 比較
        if test_data == decompressed_data:
            print("🎉 成功！完全可逆")
        else:
            print("❌ 失敗")
            print(f"元データ: {test_data}")
            print(f"解凍データ: {decompressed_data}")
            
    except Exception as e:
        print(f"❌ 解凍エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_compression()
