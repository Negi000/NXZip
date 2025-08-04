#!/usr/bin/env python3
"""
シンプル可逆性テスト - 基本機能のみ
"""

import sys
import os
import hashlib

# パスの追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nxzip'))

from nxzip.engine.nexus_tmc_v91_modular import NEXUSTMCEngineV91

def test_simple_reversibility():
    """シンプルな可逆性テスト"""
    print("🎯 シンプル可逆性テスト")
    print("=" * 40)
    
    # 軽量モードエンジン（並列処理無効）
    engine = NEXUSTMCEngineV91(lightweight_mode=True)
    
    # テストデータ
    test_cases = [
        ("基本テキスト", b"Hello, World!"),
        ("繰り返し", b"ABC" * 100),
        ("数値", bytes(range(256))),
        ("空データ", b""),
    ]
    
    all_passed = True
    
    for name, original_data in test_cases:
        print(f"\n📋 テスト: {name}")
        print(f"📊 元データ: {len(original_data)} bytes")
        
        try:
            # ハッシュ計算
            original_hash = hashlib.sha256(original_data).hexdigest()
            print(f"🔐 元ハッシュ: {original_hash[:16]}...")
            
            # 圧縮
            compressed_data, info = engine.compress(original_data)
            print(f"🗜️ 圧縮: {len(compressed_data)} bytes")
            
            # 解凍
            restored_data = engine.decompress(compressed_data, info)
            print(f"📤 解凍: {len(restored_data)} bytes")
            
            # 検証
            restored_hash = hashlib.sha256(restored_data).hexdigest()
            print(f"🔐 復元ハッシュ: {restored_hash[:16]}...")
            
            if original_data == restored_data and original_hash == restored_hash:
                print(f"✅ {name}: 完全可逆")
            else:
                print(f"❌ {name}: 可逆性失敗")
                all_passed = False
                
        except Exception as e:
            print(f"❌ {name}: エラー - {e}")
            all_passed = False
    
    print(f"\n🏆 結果: {'全テスト成功' if all_passed else '一部失敗'}")
    return all_passed

if __name__ == "__main__":
    success = test_simple_reversibility()
    exit(0 if success else 1)
