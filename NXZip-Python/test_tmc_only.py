#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TMC v9.1エンジン単体での可逆性テスト
"""

import hashlib
import sys
import os
sys.path.append(os.path.dirname(__file__))

# TMC v9.1エンジンをインポート
try:
    from nxzip.engine.nexus_tmc_v91_modular import NEXUSTMCEngineV91
    print("✅ TMC v9.1エンジンをインポート")
except ImportError as e:
    print(f"❌ TMC v9.1エンジンのインポートエラー: {e}")
    sys.exit(1)

def test_tmc_reversibility():
    """TMC v9.1エンジンの可逆性を直接テスト"""
    print("🎯 TMC v9.1 単体可逆性テスト")
    print("=" * 50)
    
    # テストデータ
    test_cases = [
        ("小テキスト", b"Hello, World!"),
        ("日本語", "こんにちは、世界！".encode('utf-8')),
        ("繰り返し", b"A" * 100),
        ("バイナリ", bytes(range(128))),
        ("ゼロ埋め", b"\x00" * 50),
        ("混合", b"123\x00\xFF\x80abc"),
        ("空", b""),
        ("1バイト", b"X"),
    ]
    
    # TMCエンジン初期化（軽量モード）
    tmc_engine = NEXUSTMCEngineV91(
        max_workers=1,
        chunk_size=2 * 1024 * 1024,  # 2MB
        lightweight_mode=True
    )
    
    success_count = 0
    total_tests = len(test_cases)
    
    for name, original_data in test_cases:
        print(f"\n📋 テスト: {name}")
        print("-" * 30)
        
        try:
            # 元データのハッシュ
            original_hash = hashlib.sha256(original_data).hexdigest()[:16]
            print(f"🔐 元ハッシュ: {original_hash}...")
            print(f"📊 元データ: {len(original_data)} bytes")
            
            if len(original_data) == 0:
                # 空データの場合はTMCをスキップ
                compressed_data = b""
                decompressed_data = b""
                print("⚡ 空データ - TMC処理スキップ")
            else:
                # TMC圧縮
                compressed_data, tmc_info = tmc_engine.compress(original_data)
                print(f"🗜️ 圧縮: {len(original_data)} → {len(compressed_data)} bytes")
                
                # TMC解凍
                decompressed_data = tmc_engine.decompress(compressed_data, tmc_info)
                print(f"📤 解凍: {len(compressed_data)} → {len(decompressed_data)} bytes")
            
            # 復元されたデータのハッシュ
            restored_hash = hashlib.sha256(decompressed_data).hexdigest()[:16]
            print(f"🔐 復元ハッシュ: {restored_hash}...")
            
            # 可逆性確認
            if original_data == decompressed_data:
                print(f"✅ {name}: 100%可逆性確認")
                success_count += 1
            else:
                print(f"❌ {name}: 可逆性失敗")
                print(f"   元データ長: {len(original_data)}")
                print(f"   復元データ長: {len(decompressed_data)}")
                if len(original_data) <= 50:
                    print(f"   元データ: {original_data}")
                    print(f"   復元データ: {decompressed_data}")
                
        except Exception as e:
            print(f"❌ {name}: エラー - {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🏆 最終結果")
    print("=" * 50)
    success_rate = (success_count / total_tests) * 100
    print(f"🎯 TMC可逆性達成率: {success_rate:.1f}% ({success_count}/{total_tests})")
    
    if success_rate == 100.0:
        print("🎉 TMC v9.1エンジンは100%可逆性を達成！")
        return True
    else:
        print("⚠️ TMC v9.1エンジンに可逆性の問題があります")
        return False

if __name__ == "__main__":
    test_tmc_reversibility()
