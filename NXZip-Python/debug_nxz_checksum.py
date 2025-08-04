#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NXZデバッグテスト - チェックサム問題の詳細分析
"""

import hashlib
import sys
import os
sys.path.append(os.path.dirname(__file__))

# NXZ形式をインポート
try:
    from nxzip.formats.enhanced_nxz import SuperNXZipFile
    print("✅ SuperNXZipFile形式をインポート")
except ImportError as e:
    print(f"❌ SuperNXZipFile形式のインポートエラー: {e}")
    sys.exit(1)

def debug_nxz_checksum():
    """NXZ形式でのチェックサム問題をデバッグ"""
    print("🎯 NXZ チェックサム詳細デバッグ")
    print("=" * 50)
    
    # テストデータ
    test_data = b"Hello, World!"
    print(f"📊 テストデータ: {test_data}")
    print(f"📊 データ長: {len(test_data)} bytes")
    
    # 元データのハッシュ
    original_hash = hashlib.sha256(test_data).digest()
    print(f"🔐 元ハッシュ: {original_hash.hex()[:16]}...")
    
    # NXZ形式でアーカイブ作成
    nxz = SuperNXZipFile(
        compression_algo='TMC',
        encryption_algo=None,
        lightweight_mode=True
    )
    
    print("\n🗜️ アーカイブ作成フェーズ")
    print("-" * 30)
    archive = nxz.create_archive(test_data, password=None, show_progress=True)
    print(f"📦 アーカイブサイズ: {len(archive)} bytes")
    
    print("\n📤 アーカイブ解凍フェーズ")
    print("-" * 30)
    try:
        restored_data = nxz.extract_archive(archive, password=None, show_progress=True)
        
        # 復元データのハッシュ
        restored_hash = hashlib.sha256(restored_data).digest()
        print(f"🔐 復元ハッシュ: {restored_hash.hex()[:16]}...")
        
        # 詳細比較
        print(f"\n🔍 詳細比較")
        print(f"   元データ: {test_data}")
        print(f"   復元データ: {restored_data}")
        print(f"   データ一致: {test_data == restored_data}")
        print(f"   ハッシュ一致: {original_hash == restored_hash}")
        
        if test_data == restored_data:
            print("✅ データは100%正確に復元されました")
        else:
            print("❌ データの復元に問題があります")
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_nxz_checksum()
