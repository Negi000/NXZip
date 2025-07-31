#!/usr/bin/env python3
"""
NEXUS TMC v9.0 エンジンテスト
"""
import sys
import os
import time
from pathlib import Path

# NXZipモジュールのパスを追加
sys.path.insert(0, str(Path(__file__).parent / "NXZip-Python"))

try:
    from nxzip.engine.nexus_tmc import NEXUSTMCEngineV9
    print("✅ NEXUS TMC v9.0 エンジンのインポート成功")
except ImportError as e:
    print(f"❌ インポートエラー: {e}")
    sys.exit(1)

def test_basic_functionality():
    """基本機能テスト"""
    print("\n🧪 基本機能テスト開始")
    print("=" * 50)
    
    # エンジン初期化
    try:
        engine = NEXUSTMCEngineV9(max_workers=2)
        print("✅ エンジン初期化成功")
    except Exception as e:
        print(f"❌ エンジン初期化失敗: {e}")
        return False
    
    # テストデータ
    test_vectors = [
        b"Hello, NEXUS TMC v9.0!",
        b"A" * 100,  # 繰り返しデータ
        b"NEXUS TMC Engine Test " * 50,  # 中サイズデータ
        bytes(range(256)) * 10,  # バイナリデータ
    ]
    
    success_count = 0
    total_tests = len(test_vectors)
    
    for i, test_data in enumerate(test_vectors):
        print(f"\n📄 テストケース {i+1}: {len(test_data)} bytes")
        
        try:
            # 圧縮テスト
            start_time = time.time()
            compressed, meta = engine.compress_tmc(test_data)
            compress_time = time.time() - start_time
            
            # 展開テスト
            start_time = time.time()
            decompressed = engine.decompress_tmc(compressed, meta)
            decompress_time = time.time() - start_time
            
            # 可逆性検証
            is_identical = test_data == decompressed
            compression_ratio = len(compressed) / len(test_data) * 100
            
            # 結果表示
            print(f"  📊 圧縮率: {compression_ratio:.1f}%")
            print(f"  ⏱️ 圧縮時間: {compress_time*1000:.1f}ms")
            print(f"  ⏱️ 展開時間: {decompress_time*1000:.1f}ms")
            print(f"  🔄 可逆性: {'✅ OK' if is_identical else '❌ NG'}")
            
            if is_identical:
                success_count += 1
            else:
                print(f"    ⚠️ データ不一致: 元={len(test_data)}, 復元={len(decompressed)}")
                
        except Exception as e:
            print(f"  ❌ エラー: {e}")
    
    print(f"\n📊 テスト結果: {success_count}/{total_tests} 成功")
    return success_count == total_tests

def test_real_files():
    """実ファイルテスト"""
    print("\n🗂️ 実ファイルテスト開始")
    print("=" * 50)
    
    # サンプルファイルディレクトリ
    sample_dir = Path(__file__).parent / "NXZip-Python" / "sample"
    
    if not sample_dir.exists():
        print("⚠️ サンプルディレクトリが見つかりません")
        return False
    
    # テスト対象ファイル
    test_files = [
        "出庫実績明細_202412.txt",
        "COT-001.jpg",
        "COT-001.png"
    ]
    
    engine = NEXUSTMCEngineV9(max_workers=2)
    success_count = 0
    
    for filename in test_files:
        file_path = sample_dir / filename
        if not file_path.exists():
            print(f"⚠️ スキップ: {filename} (ファイルが見つかりません)")
            continue
            
        try:
            print(f"\n📄 テスト: {filename}")
            
            # ファイル読み込み
            with open(file_path, 'rb') as f:
                original_data = f.read()
            
            print(f"  📊 ファイルサイズ: {len(original_data):,} bytes")
            
            # 圧縮・展開テスト
            start_time = time.time()
            compressed, meta = engine.compress_tmc(original_data)
            compress_time = time.time() - start_time
            
            start_time = time.time()
            decompressed = engine.decompress_tmc(compressed, meta)
            decompress_time = time.time() - start_time
            
            # 結果検証
            is_identical = original_data == decompressed
            compression_ratio = len(compressed) / len(original_data) * 100
            
            print(f"  📊 圧縮率: {compression_ratio:.1f}%")
            print(f"  ⏱️ 圧縮時間: {compress_time:.2f}s")
            print(f"  ⏱️ 展開時間: {decompress_time:.2f}s")
            print(f"  🔄 可逆性: {'✅ OK' if is_identical else '❌ NG'}")
            
            if 'data_type' in meta:
                print(f"  🔍 データ型: {meta['data_type']}")
            
            if is_identical:
                success_count += 1
                
        except Exception as e:
            print(f"  ❌ エラー: {e}")
    
    print(f"\n📊 実ファイルテスト結果: {success_count} ファイル成功")
    return success_count > 0

def main():
    """メインテスト関数"""
    print("🚀 NEXUS TMC v9.0 エンジンテスト")
    print("=" * 60)
    
    # 基本機能テスト
    basic_success = test_basic_functionality()
    
    # 実ファイルテスト
    file_success = test_real_files()
    
    # 総合結果
    print("\n" + "=" * 60)
    print("🎯 総合テスト結果")
    print("=" * 60)
    print(f"基本機能テスト: {'✅ PASS' if basic_success else '❌ FAIL'}")
    print(f"実ファイルテスト: {'✅ PASS' if file_success else '❌ FAIL'}")
    
    if basic_success and file_success:
        print("\n🎉 NEXUS TMC v9.0 エンジン - 全テスト成功！")
        return True
    else:
        print("\n⚠️ 一部テストに問題があります")
        return False

if __name__ == "__main__":
    main()
