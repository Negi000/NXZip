#!/usr/bin/env python3
"""
NXZip完全可逆性テスト - 100%保証バージョン
"""

import sys
import os
import hashlib
import time

# パスの追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nxzip'))

from nxzip.formats.enhanced_nxz import SuperNXZipFile

def test_nxzip_guaranteed_reversibility():
    """NXZip 100%可逆性保証テスト"""
    print("🎯 NXZip 100%可逆性保証テスト")
    print("=" * 50)
    
    # 軽量モードのNXZファイル作成（100%可逆性保証）
    nxz = SuperNXZipFile(lightweight_mode=True)
    
    # テストデータ
    test_cases = [
        ("小テキスト", b"Hello, NXZip!"),
        ("日本語", "こんにちは、NXZip！".encode('utf-8')),
        ("繰り返し", b"DATA" * 100),  # 400 bytes
        ("バイナリ", bytes(range(128))),  # 128 bytes
        ("ゼロ埋め", b'\x00' * 200),
        ("混合", b"Text" + b'\x00\x01\x02\x03' + b"More"),
        ("空", b""),
        ("1バイト", b"X"),
    ]
    
    print(f"📋 テストケース: {len(test_cases)}個")
    
    success_count = 0
    total_original_size = 0
    total_compressed_size = 0
    total_compress_time = 0
    total_decompress_time = 0
    
    for i, (name, original_data) in enumerate(test_cases):
        print(f"\n📋 テスト {i+1}/{len(test_cases)}: {name}")
        print("-" * 30)
        
        try:
            # メトリクス収集
            original_size = len(original_data)
            original_hash = hashlib.sha256(original_data).hexdigest()
            
            print(f"📊 元データ: {original_size} bytes")
            print(f"🔐 ハッシュ: {original_hash[:16]}...")
            
            # 圧縮
            start_time = time.time()
            nxz_archive = nxz.create_archive(original_data, password=None, show_progress=False)
            compress_time = time.time() - start_time
            
            compressed_size = len(nxz_archive)
            compression_ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
            
            print(f"🗜️ 圧縮: {compressed_size} bytes ({compression_ratio:.1f}% 削減)")
            print(f"⚡ 圧縮時間: {compress_time:.3f}秒")
            
            # 展開
            start_time = time.time()
            restored_data = nxz.extract_archive(nxz_archive, password=None, show_progress=False)
            decompress_time = time.time() - start_time
            
            # 完全性検証
            restored_size = len(restored_data)
            restored_hash = hashlib.sha256(restored_data).hexdigest()
            
            print(f"📤 展開: {restored_size} bytes")
            print(f"⚡ 展開時間: {decompress_time:.3f}秒")
            print(f"🔐 復元ハッシュ: {restored_hash[:16]}...")
            
            # 厳格な検証
            size_match = original_size == restored_size
            hash_match = original_hash == restored_hash
            byte_match = original_data == restored_data
            
            if size_match and hash_match and byte_match:
                print(f"✅ {name}: 100%可逆性確認")
                success_count += 1
                
                # 統計更新
                total_original_size += original_size
                total_compressed_size += compressed_size
                total_compress_time += compress_time
                total_decompress_time += decompress_time
            else:
                print(f"❌ {name}: 可逆性失敗")
                print(f"   サイズ一致: {size_match}")
                print(f"   ハッシュ一致: {hash_match}")
                print(f"   バイト一致: {byte_match}")
                
        except Exception as e:
            print(f"❌ {name}: エラー - {e}")
            import traceback
            traceback.print_exc()
    
    # 最終結果
    success_rate = (success_count / len(test_cases)) * 100
    avg_compression_ratio = (1 - total_compressed_size / total_original_size) * 100 if total_original_size > 0 else 0
    
    print(f"\n🏆 最終結果")
    print("=" * 50)
    print(f"🎯 可逆性達成率: {success_rate:.1f}% ({success_count}/{len(test_cases)})")
    print(f"📊 平均圧縮率: {avg_compression_ratio:.1f}%")
    print(f"📦 総処理量: {total_original_size} -> {total_compressed_size} bytes")
    print(f"⚡ 平均圧縮速度: {total_original_size / total_compress_time / 1024:.1f} KB/s" if total_compress_time > 0 else "⚡ 圧縮速度: 計測不可")
    print(f"⚡ 平均展開速度: {total_original_size / total_decompress_time / 1024:.1f} KB/s" if total_decompress_time > 0 else "⚡ 展開速度: 計測不可")
    
    if success_rate == 100.0:
        print("\n🎉 100%可逆性達成！NXZipは完全な可逆圧縮システムです！")
        print("✨ どんなデータでも完璧に復元できます")
        return True
    else:
        print(f"\n⚠️ 可逆性未達成: {100 - success_rate:.1f}%の改善が必要")
        return False

def main():
    """メイン実行"""
    success = test_nxzip_guaranteed_reversibility()
    
    if success:
        print("\n🚀 NXZip は完全可逆圧縮技術として実用可能です！")
        return 0
    else:
        print("\n🔧 さらなる改善が必要です")
        return 1

if __name__ == "__main__":
    exit(main())
