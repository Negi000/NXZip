#!/usr/bin/env python3
"""
🧪 NXZip Basic Usage Examples

NXZipの基本的な使用例とサンプルコード
"""

import os
import sys
import time

# パッケージパスを追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nxzip import NXZipArchive, NEXUSCompressor


def create_test_files():
    """テスト用ファイル作成"""
    test_dir = "test_files"
    os.makedirs(test_dir, exist_ok=True)
    
    # テキストファイル
    with open(f"{test_dir}/sample.txt", "w", encoding="utf-8") as f:
        f.write("これはNXZipのテストファイルです。\n" * 1000)
    
    # JSONファイル
    import json
    test_data = {
        "name": "NXZip",
        "version": "1.0.0",
        "features": ["compression", "encryption", "archiving"],
        "performance": {"compression_ratio": 99.93, "speed_mbps": 11.37}
    }
    with open(f"{test_dir}/config.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    # バイナリファイル
    with open(f"{test_dir}/binary.dat", "wb") as f:
        f.write(b"NXZIP" * 10000)
    
    print(f"✅ テストファイル作成完了: {test_dir}/")
    return test_dir


def example_compression():
    """圧縮のみの例"""
    print("\n🚀 === NEXUS圧縮テスト ===")
    
    compressor = NEXUSCompressor()
    
    # ファイル圧縮
    test_file = "test_files/sample.txt"
    if os.path.exists(test_file):
        with open(test_file, "rb") as f:
            data = f.read()
        
        print(f"📄 対象ファイル: {test_file}")
        print(f"📊 元サイズ: {len(data):,} bytes")
        
        start_time = time.time()
        compressed_data, metadata = compressor.compress(data, test_file)
        compression_time = time.time() - start_time
        
        print(f"📦 圧縮サイズ: {len(compressed_data):,} bytes")
        print(f"⚡ 圧縮率: {metadata['ratio']:.2f}%")
        print(f"🔍 検出フォーマット: {metadata['format']}")
        print(f"⏱️ 処理時間: {compression_time:.3f}秒")
        
        # 圧縮ファイル保存
        with open("sample_compressed.nexus", "wb") as f:
            f.write(compressed_data)
        print(f"💾 圧縮ファイル保存: sample_compressed.nexus")


def example_archive_create():
    """アーカイブ作成の例"""
    print("\n📦 === NXZアーカイブ作成テスト ===")
    
    archive_path = "test_archive.nxz"
    archive = NXZipArchive(archive_path)
    
    # ファイル追加
    test_files = [
        "test_files/sample.txt",
        "test_files/config.json", 
        "test_files/binary.dat"
    ]
    
    added_count = 0
    for file_path in test_files:
        if os.path.exists(file_path):
            if archive.add_file(file_path):
                print(f"  ✅ 追加: {file_path}")
                added_count += 1
            else:
                print(f"  ❌ 失敗: {file_path}")
    
    # アーカイブ保存
    if archive.save():
        stats = archive.get_stats()
        print(f"\n🎉 アーカイブ作成完了!")
        print(f"📄 ファイル数: {stats['total_files']}")
        print(f"📊 元サイズ: {stats['total_original_size']:,} bytes")
        print(f"📦 圧縮サイズ: {stats['total_compressed_size']:,} bytes") 
        print(f"⚡ 総合圧縮率: {stats['overall_compression_ratio']:.2f}%")
        print(f"💾 保存先: {archive_path}")
    else:
        print("❌ アーカイブ保存失敗")


def example_archive_secure():
    """暗号化アーカイブの例"""
    print("\n🔒 === セキュアアーカイブ作成テスト ===")
    
    password = "test123"
    secure_archive_path = "secure_archive.nxz"
    
    archive = NXZipArchive(secure_archive_path, password)
    
    # ファイル追加
    if os.path.exists("test_files/sample.txt"):
        if archive.add_file("test_files/sample.txt"):
            print(f"  🔐 暗号化追加: test_files/sample.txt")
    
    if archive.save():
        stats = archive.get_stats()
        print(f"\n🎉 セキュアアーカイブ作成完了!")
        print(f"🔒 暗号化: 有効")
        print(f"📄 ファイル数: {stats['total_files']}")
        print(f"⚡ 圧縮率: {stats['overall_compression_ratio']:.2f}%")
        print(f"💾 保存先: {secure_archive_path}")


def example_archive_extract():
    """アーカイブ展開の例"""
    print("\n📂 === アーカイブ展開テスト ===")
    
    archive_path = "test_archive.nxz"
    output_dir = "extracted_files"
    
    if os.path.exists(archive_path):
        archive = NXZipArchive(archive_path)
        
        if archive.load():
            print(f"📦 アーカイブ読み込み成功: {archive_path}")
            
            # 内容一覧
            entries = archive.list_entries()
            print(f"📄 含まれるファイル:")
            for entry in entries:
                print(f"  - {entry['filepath']} ({entry['original_size']:,} bytes)")
            
            # 全展開
            extracted_count = archive.extract_all(output_dir)
            print(f"\n🎉 展開完了!")
            print(f"📄 展開ファイル数: {extracted_count}")
            print(f"📁 出力先: {output_dir}/")
        else:
            print("❌ アーカイブ読み込み失敗")
    else:
        print(f"❌ アーカイブファイルが見つかりません: {archive_path}")


def example_stats():
    """統計情報の例"""
    print("\n📊 === 統計情報テスト ===")
    
    compressor = NEXUSCompressor()
    
    # 複数ファイルで統計取得
    test_files = ["test_files/sample.txt", "test_files/config.json", "test_files/binary.dat"]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            with open(test_file, "rb") as f:
                data = f.read()
            compressed_data, metadata = compressor.compress(data, test_file)
    
    # 統計表示
    stats = compressor.get_stats()
    print(f"📄 処理ファイル数: {stats['files_processed']}")
    print(f"📊 平均圧縮率: {stats['average_ratio']:.2f}%")
    print(f"🏆 最高圧縮率: {stats['best_ratio']:.2f}%")
    print(f"📉 最低圧縮率: {stats['worst_ratio']:.2f}%")
    print(f"⚡ 総合圧縮率: {stats['total_ratio']:.2f}%")


def cleanup():
    """テストファイル削除"""
    import shutil
    
    cleanup_items = [
        "test_files/",
        "extracted_files/",
        "test_archive.nxz",
        "secure_archive.nxz", 
        "sample_compressed.nexus"
    ]
    
    print("\n🧹 === クリーンアップ ===")
    for item in cleanup_items:
        try:
            if os.path.isfile(item):
                os.remove(item)
                print(f"  🗑️ ファイル削除: {item}")
            elif os.path.isdir(item):
                shutil.rmtree(item)
                print(f"  🗑️ フォルダ削除: {item}")
        except:
            pass


def main():
    """メイン実行"""
    print("🚀 NXZip サンプル実行")
    print("=" * 50)
    
    try:
        # テストファイル作成
        create_test_files()
        
        # 各種テスト実行
        example_compression()
        example_archive_create()
        example_archive_secure()
        example_archive_extract()
        example_stats()
        
        print("\n🎉 全てのサンプル実行完了!")
        
        # クリーンアップ確認
        cleanup_choice = input("\n🧹 テストファイルを削除しますか? (y/N): ")
        if cleanup_choice.lower() == 'y':
            cleanup()
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
