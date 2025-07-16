#!/usr/bin/env python3
"""
NXZip デバッグ用簡易テスト
"""

import traceback
import sys
import os

# nxzip_complete モジュールをインポート
sys.path.append('.')
from nxzip_complete import NXZipFile

def test_basic_functionality():
    """基本的な機能をテスト"""
    print("🧪 NXZip 基本機能テスト")
    print("=" * 40)
    
    nxzip = NXZipFile()
    
    # テストデータ
    test_data = b"Hello, NXZip! This is a test."
    print(f"テストデータ: {len(test_data)} bytes")
    
    try:
        # 1. 圧縮なし、暗号化なし
        print("\n1. 基本テスト（圧縮なし、暗号化なし）")
        archive1 = nxzip.create_archive(test_data)
        restored1 = nxzip.extract_archive(archive1)
        print(f"   結果: {'✅ 成功' if test_data == restored1 else '❌ 失敗'}")
        
        # 2. 圧縮あり、暗号化なし
        print("\n2. 圧縮テスト")
        large_data = b"x" * 2000  # 大きなデータで圧縮をテスト
        archive2 = nxzip.create_archive(large_data)
        restored2 = nxzip.extract_archive(archive2)
        print(f"   結果: {'✅ 成功' if large_data == restored2 else '❌ 失敗'}")
        print(f"   圧縮率: {len(archive2) / len(large_data) * 100:.1f}%")
        
        # 3. 暗号化テスト
        print("\n3. 暗号化テスト")
        password = "TestPassword123"
        archive3 = nxzip.create_archive(test_data, password)
        restored3 = nxzip.extract_archive(archive3, password)
        print(f"   結果: {'✅ 成功' if test_data == restored3 else '❌ 失敗'}")
        
        # 4. 間違ったパスワードでのテスト
        print("\n4. 間違ったパスワードテスト")
        try:
            nxzip.extract_archive(archive3, "WrongPassword")
            print("   結果: ❌ 失敗（間違ったパスワードで成功してしまった）")
        except Exception as e:
            print(f"   結果: ✅ 成功（正しくエラーが発生: {type(e).__name__}）")
        
    except Exception as e:
        print(f"❌ エラーが発生しました:")
        print(f"   {type(e).__name__}: {e}")
        print("\nトレースバック:")
        traceback.print_exc()
        return False
    
    return True

def test_file_operations():
    """ファイル操作のテスト"""
    print("\n🗂️ ファイル操作テスト")
    print("=" * 40)
    
    nxzip = NXZipFile()
    
    try:
        # テストファイルを読み込み
        if not os.path.exists('test_input.txt'):
            print("❌ test_input.txt が見つかりません")
            return False
        
        with open('test_input.txt', 'rb') as f:
            test_data = f.read()
        
        print(f"テストファイル: {len(test_data)} bytes")
        
        # アーカイブ作成（パスワードなし）
        print("\n1. ファイルアーカイブ作成（パスワードなし）")
        archive = nxzip.create_archive(test_data)
        
        # 保存
        with open('debug_test1.nxz', 'wb') as f:
            f.write(archive)
        print(f"   アーカイブサイズ: {len(archive)} bytes")
        
        # 読み込みと展開
        with open('debug_test1.nxz', 'rb') as f:
            loaded_archive = f.read()
        
        restored = nxzip.extract_archive(loaded_archive)
        
        # 検証
        success = test_data == restored
        print(f"   結果: {'✅ 成功' if success else '❌ 失敗'}")
        
        if success:
            with open('debug_extracted1.txt', 'wb') as f:
                f.write(restored)
            print("   展開結果を debug_extracted1.txt に保存")
        
        # アーカイブ作成（パスワードあり）
        print("\n2. ファイルアーカイブ作成（パスワードあり）")
        password = "TestPassword123"
        encrypted_archive = nxzip.create_archive(test_data, password)
        
        # 保存
        with open('debug_test2.nxz', 'wb') as f:
            f.write(encrypted_archive)
        print(f"   暗号化アーカイブサイズ: {len(encrypted_archive)} bytes")
        
        # 読み込みと展開
        with open('debug_test2.nxz', 'rb') as f:
            loaded_encrypted = f.read()
        
        decrypted = nxzip.extract_archive(loaded_encrypted, password)
        
        # 検証
        success = test_data == decrypted
        print(f"   結果: {'✅ 成功' if success else '❌ 失敗'}")
        
        if success:
            with open('debug_extracted2.txt', 'wb') as f:
                f.write(decrypted)
            print("   展開結果を debug_extracted2.txt に保存")
        else:
            print(f"   元サイズ: {len(test_data)}")
            print(f"   復元サイズ: {len(decrypted)}")
            # 最初の違いを探す
            for i in range(min(len(test_data), len(decrypted))):
                if test_data[i] != decrypted[i]:
                    print(f"   最初の違い: index {i}, 元={test_data[i]}, 復元={decrypted[i]}")
                    break
        
        return success
        
    except Exception as e:
        print(f"❌ エラーが発生しました:")
        print(f"   {type(e).__name__}: {e}")
        print("\nトレースバック:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 NXZip デバッグテスト")
    print("=" * 50)
    
    # 基本機能テスト
    basic_success = test_basic_functionality()
    
    # ファイル操作テスト
    file_success = test_file_operations()
    
    print("\n" + "=" * 50)
    print("テスト結果:")
    print(f"基本機能: {'✅ 成功' if basic_success else '❌ 失敗'}")
    print(f"ファイル操作: {'✅ 成功' if file_success else '❌ 失敗'}")
    
    if basic_success and file_success:
        print("\n🎉 全てのテストが成功しました！")
    else:
        print("\n⚠️ 一部のテストが失敗しました。")
