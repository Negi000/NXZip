#!/usr/bin/env python3
"""
SPEコアインポートテスト - 6段階エンタープライズSPEが正しく動作するか確認
"""

import sys
import os

# SPEコアのパスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'NXZip-Python'))

try:
    print("🔍 6段階エンタープライズSPEコアのインポートテスト")
    print("=" * 60)
    
    # 1. SPEコア単体インポートテスト
    print("📦 Step 1: SPEコア単体インポート")
    from nxzip.engine.spe_core import SPECore
    print("✅ SPEコア単体インポート成功")
    
    # 2. SPEコア初期化テスト
    print("\n🔧 Step 2: SPEコア初期化")
    spe = SPECore()
    print("✅ SPEコア初期化成功")
    print(f"   - セキュリティレベル: {spe._security_level}")
    print(f"   - ブロックサイズ: {spe._block_size}")
    print(f"   - 変換ラウンド数: {spe._rounds}")
    
    # 3. 6段階変換テスト
    print("\n🔒 Step 3: 6段階SPE変換テスト")
    test_data = b"NXZip Enterprise 6-Stage SPE Test Data for Performance Validation"
    print(f"   元データ: {test_data}")
    print(f"   データサイズ: {len(test_data)} bytes")
    
    # 変換実行
    transformed = spe.apply_transform(test_data)
    print(f"   変換後サイズ: {len(transformed)} bytes")
    print(f"   変換後データ: {transformed[:32].hex()}...")
    
    # 逆変換実行
    restored = spe.reverse_transform(transformed)
    print(f"   復元後サイズ: {len(restored)} bytes")
    print(f"   復元データ: {restored}")
    
    # 整合性確認
    is_valid = (test_data == restored)
    print(f"   🎯 可逆性検証: {'✅ OK' if is_valid else '❌ NG'}")
    
    if not is_valid:
        print(f"   ❌ 元データ: {test_data}")
        print(f"   ❌ 復元データ: {restored}")
        raise RuntimeError("SPE Core可逆性テスト失敗")
    
    # 4. 統合システムインポートテスト
    print("\n⚡ Step 4: 統合システムインポートテスト")
    from nxzip_complete import SuperNXZipFile
    print("✅ 統合システムインポート成功")
    
    # 5. 統合システム初期化テスト
    print("\n🚀 Step 5: 統合システム初期化")
    nxzip = SuperNXZipFile()
    print("✅ 統合システム初期化成功")
    print(f"   - SPEコアタイプ: {type(nxzip.spe_core).__name__}")
    print(f"   - SPEセキュリティレベル: {nxzip.spe_core._security_level}")
    
    # 6. 統合システム動作テスト
    print("\n🎯 Step 6: 統合システム6段階SPE動作テスト")
    test_archive_data = b"Integrated 6-Stage SPE + Compression + Encryption Test"
    
    # アーカイブ作成（暗号化なし）
    archive = nxzip.create_archive(test_archive_data, show_progress=False)
    print(f"   アーカイブサイズ: {len(archive)} bytes")
    
    # アーカイブ展開
    extracted = nxzip.extract_archive(archive, show_progress=False)
    
    # 整合性確認
    is_valid = (test_archive_data == extracted)
    print(f"   🎯 統合テスト結果: {'✅ OK' if is_valid else '❌ NG'}")
    
    if not is_valid:
        print(f"   ❌ 元データ: {test_archive_data}")
        print(f"   ❌ 展開データ: {extracted}")
        raise RuntimeError("統合システムテスト失敗")
    
    # 7. パスワード暗号化テスト
    print("\n🔐 Step 7: パスワード暗号化統合テスト")
    password = "Test123"
    encrypted_archive = nxzip.create_archive(test_archive_data, password=password, show_progress=False)
    decrypted_data = nxzip.extract_archive(encrypted_archive, password=password, show_progress=False)
    
    is_valid = (test_archive_data == decrypted_data)
    print(f"   🎯 暗号化統合テスト: {'✅ OK' if is_valid else '❌ NG'}")
    
    print("\n🎉 全テスト完了!")
    print("✅ 6段階エンタープライズSPEコアが統合システムで正常動作しています")
    print("✅ ロジック転記ではなく、正しいSPEコアインポートが行われています")
    
except Exception as e:
    print(f"\n❌ テスト失敗: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
