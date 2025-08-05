#!/usr/bin/env python3
"""
SPE暗号化完全テスト - 圧縮・暗号化・復号化・展開の全パイプライン
"""

import os
import sys
import hashlib

# NXZip-Releaseディレクトリを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'NXZip-Release'))

from nxzip_core import NXZipCore, NXZipContainer

def test_complete_pipeline():
    """完全なパイプライン（圧縮→暗号化→復号化→展開）テスト"""
    print("🔥 SPE暗号化完全パイプラインテスト")
    print("=" * 60)
    
    # テストデータ準備
    test_message = "SPE暗号化テスト用データ！これが正しく復号化されれば成功です。" * 100
    test_data = test_message.encode('utf-8')
    
    print(f"📊 テストデータ: {len(test_data)} bytes")
    print(f"   内容: {test_message[:50]}...")
    print(f"   元のハッシュ: {hashlib.sha256(test_data).hexdigest()[:16]}")
    
    # NXZip Core初期化
    nxzip = NXZipCore()
    
    # 暗号化キー（将来の追加暗号化用）
    user_key = b"user_secret_key_123456789012"
    
    print(f"\n🔥 Step 1: 圧縮+SPE暗号化")
    print(f"   パイプライン: TMC変換 → 圧縮 → SPE暗号化")
    
    # 圧縮＋暗号化
    compress_result = nxzip.compress(test_data, mode="balanced", encryption_key=user_key)
    
    if not compress_result.success:
        print(f"❌ 圧縮失敗: {compress_result.error_message}")
        return
    
    print(f"✅ 圧縮成功!")
    print(f"   原始サイズ: {compress_result.original_size} bytes")
    print(f"   暗号化後サイズ: {compress_result.compressed_size} bytes")
    print(f"   圧縮率: {compress_result.compression_ratio:.2f}%")
    
    # 暗号化データの確認
    encrypted_data = compress_result.compressed_data
    print(f"   暗号化データ: {encrypted_data[:20].hex()}...")
    
    # コンテナにパック
    print(f"\n🔥 Step 2: NXZipコンテナにパック")
    container_data = NXZipContainer.pack(encrypted_data, compress_result.metadata, "test.txt")
    print(f"✅ コンテナ作成: {len(container_data)} bytes")
    
    # コンテナから展開
    print(f"\n🔥 Step 3: コンテナから展開")
    unpacked_data, unpacked_info = NXZipContainer.unpack(container_data)
    print(f"✅ コンテナ展開: {len(unpacked_data)} bytes")
    
    # 復号化＋展開
    print(f"\n🔥 Step 4: SPE復号化+展開")
    print(f"   パイプライン: SPE復号化 → 展開 → TMC逆変換")
    
    decompress_result = nxzip.decompress(unpacked_data, unpacked_info)
    
    if not decompress_result.success:
        print(f"❌ 展開失敗: {decompress_result.error_message}")
        return
    
    print(f"✅ 展開成功!")
    print(f"   展開サイズ: {decompress_result.original_size} bytes")
    print(f"   展開時間: {decompress_result.decompression_time:.3f}秒")
    
    # データ整合性確認
    restored_data = decompress_result.decompressed_data
    print(f"\n🔍 データ整合性確認:")
    print(f"   元のサイズ: {len(test_data)} bytes")
    print(f"   復元サイズ: {len(restored_data)} bytes")
    print(f"   元のハッシュ: {hashlib.sha256(test_data).hexdigest()[:16]}")
    print(f"   復元ハッシュ: {hashlib.sha256(restored_data).hexdigest()[:16]}")
    
    # 厳密比較
    is_identical = test_data == restored_data
    print(f"   データ同一性: {'✅ 完全一致' if is_identical else '❌ 不一致'}")
    
    # 内容確認
    if is_identical:
        try:
            restored_message = restored_data.decode('utf-8')
            print(f"   復元内容: {restored_message[:50]}...")
            print(f"🎉 SPE暗号化パイプライン完全成功！")
        except UnicodeDecodeError:
            print(f"⚠️ データは一致しているが、文字列デコードに失敗")
    else:
        print(f"❌ データが一致しません")
        # デバッグ用に差分を調べる
        if len(test_data) == len(restored_data):
            diff_count = sum(1 for i in range(len(test_data)) if test_data[i] != restored_data[i])
            print(f"   差分バイト数: {diff_count}")
        else:
            print(f"   サイズが異なります")
    
    # SPE効果分析
    print(f"\n📊 SPE暗号化効果分析:")
    
    # 暗号化なしと比較
    print(f"   SPE暗号化あり: {compress_result.compressed_size} bytes")
    
    # 暗号化なしで同じ圧縮
    no_encryption_result = nxzip.compress(test_data, mode="balanced", encryption_key=None)
    print(f"   SPE暗号化なし: {no_encryption_result.compressed_size} bytes")
    
    overhead = compress_result.compressed_size - no_encryption_result.compressed_size
    print(f"   SPE暗号化オーバーヘッド: {overhead} bytes")
    
    if overhead <= 16:  # 適正範囲（16バイト以下）
        print(f"   ✅ オーバーヘッドは適正範囲内")
    else:
        print(f"   ⚠️ オーバーヘッドがやや大きめ")

def test_spe_reversibility():
    """SPE暗号化・復号化の可逆性テスト"""
    print(f"\n🧪 SPE可逆性テスト")
    print("=" * 40)
    
    from engine.spe_core_jit import SPECoreJIT
    
    spe = SPECoreJIT()
    
    test_cases = [
        b"Hello World!",
        b"A" * 1000,
        b"\x00\x01\x02\x03" * 250,
        os.urandom(500),
    ]
    
    for i, data in enumerate(test_cases, 1):
        print(f"   テスト{i}: {len(data)} bytes")
        
        # 暗号化→復号化
        encrypted = spe.apply_transform(data)
        decrypted = spe.reverse_transform(encrypted)
        
        is_correct = data == decrypted
        print(f"     可逆性: {'✅' if is_correct else '❌'}")
        
        if not is_correct:
            print(f"     元: {data[:20].hex()}")
            print(f"     復: {decrypted[:20].hex()}")

if __name__ == "__main__":
    test_complete_pipeline()
    test_spe_reversibility()
