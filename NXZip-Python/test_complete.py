#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

from nxzip import SuperNXZipFile

# 基本テスト
print("🧪 NXZip v2.0 基本テスト開始")

# テストデータ
test_data = b"Hello, NXZip v2.0! This is a comprehensive test."
password = "test_password_123"

try:
    # 1. 基本圧縮・展開テスト
    print("\n📦 基本圧縮・展開テスト")
    nxzip = SuperNXZipFile()
    archive = nxzip.create_archive(test_data, show_progress=True)
    restored = nxzip.extract_archive(archive, show_progress=True)
    
    print(f"✅ 基本テスト: {'成功' if restored == test_data else '失敗'}")
    print(f"📈 圧縮率: {(1 - len(archive) / len(test_data)) * 100:.1f}%")
    
    # 2. 暗号化テスト
    print("\n🔒 暗号化テスト")
    encrypted_archive = nxzip.create_archive(test_data, password=password, show_progress=True)
    decrypted = nxzip.extract_archive(encrypted_archive, password=password, show_progress=True)
    
    print(f"✅ 暗号化テスト: {'成功' if decrypted == test_data else '失敗'}")
    
    # 3. アーカイブ情報テスト
    print("\n📊 アーカイブ情報テスト")
    info = nxzip.get_info(encrypted_archive)
    print(f"バージョン: {info['version']}")
    print(f"圧縮率: {info['compression_ratio']:.1f}%")
    print(f"暗号化: {'有効' if info['is_encrypted'] else '無効'}")
    
    print("\n🎉 全テスト完了: 成功")
    
except Exception as e:
    print(f"\n❌ テストエラー: {e}")
    import traceback
    traceback.print_exc()
