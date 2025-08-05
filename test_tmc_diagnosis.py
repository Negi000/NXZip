#!/usr/bin/env python3
"""
TMC圧縮・解凍テスト - 問題の特定と修正検証
"""

import os
import sys
import hashlib

# パス追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'NXZip-Release'))

try:
    from engine.nexus_tmc_v91_modular import NEXUSTMCEngineV91
    print("✅ TMCエンジンインポート成功")
except ImportError as e:
    print(f"❌ TMCエンジンインポート失敗: {e}")
    sys.exit(1)

def test_tmc_round_trip():
    """TMC圧縮・解凍のラウンドトリップテスト"""
    print("🧪 TMC v9.1 ラウンドトリップテスト")
    print("=" * 50)
    
    # テストデータ（実データに近い形式）
    test_data = """code,name,quantity,price
A001,Product A,100,1500
A002,Product B,200,2500
A003,Product C,300,3500""" * 1000  # 大量繰り返し
    
    test_bytes = test_data.encode('utf-8')
    
    print(f"📊 テストデータ:")
    print(f"  サイズ: {len(test_bytes):,} bytes")
    print(f"  内容: {test_data[:100]}...")
    
    # 元データハッシュ
    original_hash = hashlib.sha256(test_bytes).hexdigest()
    print(f"🔐 元ハッシュ: {original_hash}")
    
    # TMCエンジン初期化
    engine = NEXUSTMCEngineV91(lightweight_mode=False)
    
    print(f"\n🗜️ 圧縮フェーズ")
    print("-" * 30)
    
    try:
        # 圧縮実行
        compressed_data, compression_info = engine.compress(test_bytes)
        
        print(f"✅ 圧縮成功:")
        print(f"  圧縮後サイズ: {len(compressed_data):,} bytes")
        print(f"  圧縮率: {compression_info.get('compression_ratio', 0):.2f}%")
        print(f"  エンジン: {compression_info.get('engine', 'unknown')}")
        print(f"  メソッド: {compression_info.get('method', 'unknown')}")
        
        # 詳細情報
        if 'tmc_info' in compression_info:
            tmc_info = compression_info['tmc_info']
            print(f"📋 TMC情報:")
            print(f"  チャンク数: {len(tmc_info.get('chunks', []))}")
            print(f"  データタイプ: {tmc_info.get('data_type', 'unknown')}")
            print(f"  変換適用: {tmc_info.get('transforms_applied', False)}")
    
    except Exception as e:
        print(f"❌ 圧縮エラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n📂 解凍フェーズ")
    print("-" * 30)
    
    try:
        # 解凍実行
        decompressed_data = engine.decompress(compressed_data, compression_info)
        
        print(f"✅ 解凍成功:")
        print(f"  解凍後サイズ: {len(decompressed_data):,} bytes")
        
        # 可逆性検証
        decompressed_hash = hashlib.sha256(decompressed_data).hexdigest()
        print(f"🔐 解凍ハッシュ: {decompressed_hash}")
        
        if original_hash == decompressed_hash:
            print("🎉 ✅ 完全可逆性確認 - データ整合性100%")
            return True
        else:
            print("❌ データ不整合 - ハッシュ不一致")
            print(f"  元サイズ: {len(test_bytes):,}")
            print(f"  解凍サイズ: {len(decompressed_data):,}")
            
            # サイズが同じ場合は部分的な差分を表示
            if len(test_bytes) == len(decompressed_data):
                differences = sum(1 for a, b in zip(test_bytes, decompressed_data) if a != b)
                print(f"  バイト差分: {differences:,}/{len(test_bytes):,} ({differences/len(test_bytes)*100:.2f}%)")
            
            return False
    
    except Exception as e:
        print(f"❌ 解凍エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_problem_identification():
    """問題の特定テスト"""
    print("\n🔍 問題特定テスト")
    print("=" * 30)
    
    # 小さなデータでのテスト
    simple_data = b"ABCD" * 10000
    original_hash = hashlib.sha256(simple_data).hexdigest()
    
    engine = NEXUSTMCEngineV91(lightweight_mode=False)
    
    compressed_data, info = engine.compress(simple_data)
    print(f"圧縮: {len(simple_data)} -> {len(compressed_data)} bytes")
    print(f"圧縮情報: {info}")
    
    # 解凍処理の詳細監視
    print("\n🔬 解凍処理詳細:")
    decompressed = engine.decompress(compressed_data, info)
    
    decompressed_hash = hashlib.sha256(decompressed).hexdigest()
    
    print(f"元ハッシュ   : {original_hash}")
    print(f"解凍ハッシュ : {decompressed_hash}")
    print(f"サイズ比較: {len(simple_data)} vs {len(decompressed)}")
    
    return original_hash == decompressed_hash

if __name__ == "__main__":
    print("🎯 TMC v9.1 問題診断・修正検証システム")
    print("=" * 60)
    
    # ラウンドトリップテスト
    success1 = test_tmc_round_trip()
    
    # 問題特定テスト
    success2 = test_problem_identification()
    
    print(f"\n📊 テスト結果サマリー")
    print("=" * 30)
    print(f"ラウンドトリップテスト: {'✅ 成功' if success1 else '❌ 失敗'}")
    print(f"問題特定テスト: {'✅ 成功' if success2 else '❌ 失敗'}")
    
    if success1 and success2:
        print("\n🎉 TMC v9.1 は正常に動作しています")
        print("📝 GUIでの問題は別の要因の可能性があります")
    else:
        print("\n⚠️ TMC v9.1 に問題が確認されました")
        print("🔧 追加修正が必要です")
