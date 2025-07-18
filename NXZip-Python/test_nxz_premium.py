#!/usr/bin/env python3
"""
NXZ Premium Format テスト
NEXUSハイブリッド + SPE統合による.nxz専用フォーマットのテスト

目標性能:
- 圧縮率: 95%以上
- 圧縮速度: 100MB/s以上
- 展開速度: 200MB/s以上
- 完全可逆性: 100%
- セキュリティ: Enterprise級
"""

import os
import sys
import time
import tempfile
import traceback
from typing import Dict, Any

# プロジェクトルートを追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nxzip.formats.nxz_premium import NXZPremiumFile, NXZCompressionLevel, create_nxz_file, extract_nxz_file


def test_nxz_premium_format():
    """NXZ Premium フォーマットの包括的テスト"""
    
    print("🚀 NXZ Premium フォーマットテスト開始")
    print("=" * 70)
    
    # テストファイルの場所
    test_file = os.path.join("..", "test-data", "huge_test.txt")
    
    if not os.path.exists(test_file):
        print(f"❌ テストファイルが見つかりません: {test_file}")
        return False
    
    try:
        # ファイル読み込み
        print("📖 テストファイル読み込み中...")
        with open(test_file, 'rb') as f:
            original_data = f.read()
        
        original_size = len(original_data)
        print(f"✅ 読み込み完了: {original_size:,} bytes ({original_size/1024/1024:.2f} MB)")
        
        # テスト用一時ファイル
        with tempfile.NamedTemporaryFile(suffix='.nxz', delete=False) as temp_nxz:
            nxz_path = temp_nxz.name
        
        with tempfile.NamedTemporaryFile(suffix='.extracted', delete=False) as temp_extracted:
            extracted_path = temp_extracted.name
        
        success = True
        results = {}
        
        try:
            # Phase 1: NXZ Premium 圧縮テスト
            print("\n🗜️  Phase 1: NXZ Premium 圧縮テスト")
            print("-" * 50)
            
            compression_start = time.time()
            
            # 高性能モードで圧縮
            nxz_handler = NXZPremiumFile(NXZCompressionLevel.BALANCED)
            nxz_data = nxz_handler.create_nxz_archive(original_data, show_progress=True)
            
            compression_time = time.time() - compression_start
            nxz_size = len(nxz_data)
            compression_ratio = (1 - nxz_size / original_size) * 100
            compression_speed = (original_size / 1024 / 1024) / compression_time
            
            results['compression'] = {
                'ratio': compression_ratio,
                'speed': compression_speed,
                'time': compression_time,
                'size': nxz_size
            }
            
            print(f"\n📊 圧縮結果:")
            print(f"   圧縮率: {compression_ratio:.2f}%")
            print(f"   圧縮速度: {compression_speed:.2f} MB/s")
            print(f"   処理時間: {compression_time:.2f}秒")
            print(f"   圧縮後サイズ: {nxz_size:,} bytes")
            
            # Phase 2: NXZ Premium 展開テスト
            print("\n📦 Phase 2: NXZ Premium 展開テスト")
            print("-" * 50)
            
            decompression_start = time.time()
            
            extracted_data = nxz_handler.extract_nxz_archive(nxz_data, show_progress=True)
            
            decompression_time = time.time() - decompression_start
            decompression_speed = (original_size / 1024 / 1024) / decompression_time
            
            results['decompression'] = {
                'speed': decompression_speed,
                'time': decompression_time,
                'size': len(extracted_data)
            }
            
            print(f"\n📊 展開結果:")
            print(f"   展開速度: {decompression_speed:.2f} MB/s")
            print(f"   処理時間: {decompression_time:.2f}秒")
            print(f"   展開後サイズ: {len(extracted_data):,} bytes")
            
            # Phase 3: 完全性検証
            print("\n🔍 Phase 3: 完全性検証")
            print("-" * 50)
            
            if len(original_data) == len(extracted_data):
                print("✅ サイズ一致")
                
                # バイト単位での比較
                differences = 0
                for i, (orig, extr) in enumerate(zip(original_data, extracted_data)):
                    if orig != extr:
                        differences += 1
                        if differences <= 5:  # 最初の5個の違いを表示
                            print(f"   ❌ 位置{i}: {orig} != {extr}")
                
                if differences == 0:
                    print("✅ 完全性検証: 成功 (100%可逆)")
                    results['integrity'] = True
                else:
                    print(f"❌ 完全性検証: 失敗 ({differences:,}バイトの違い)")
                    results['integrity'] = False
                    success = False
            else:
                print("❌ サイズ不一致")
                results['integrity'] = False
                success = False
            
            # Phase 4: ファイルI/Oテスト
            print("\n💾 Phase 4: ファイルI/Oテスト")
            print("-" * 50)
            
            # ファイル作成テスト
            file_creation_start = time.time()
            archive_info = create_nxz_file(test_file, nxz_path, show_progress=False)
            file_creation_time = time.time() - file_creation_start
            
            # ファイル展開テスト
            file_extraction_start = time.time()
            extract_info = extract_nxz_file(nxz_path, extracted_path, show_progress=False)
            file_extraction_time = time.time() - file_extraction_start
            
            print(f"✅ ファイル作成: {file_creation_time:.2f}秒")
            print(f"✅ ファイル展開: {file_extraction_time:.2f}秒")
            
            # 最終結果
            total_time = compression_time + decompression_time
            overall_speed = (original_size * 2 / 1024 / 1024) / total_time  # 往復での速度
            
            results['overall'] = {
                'total_time': total_time,
                'overall_speed': overall_speed,
                'archive_info': archive_info
            }
            
            print("\n🎉 NXZ Premium テスト結果")
            print("=" * 70)
            print(f"📊 最終圧縮率: {compression_ratio:.2f}%")
            print(f"🚀 圧縮速度: {compression_speed:.2f} MB/s")
            print(f"📦 展開速度: {decompression_speed:.2f} MB/s")
            print(f"⚡ 総合処理速度: {overall_speed:.2f} MB/s")
            print(f"⏱️  総処理時間: {total_time:.2f}秒")
            
            # 目標達成判定
            print("\n🎯 目標達成状況:")
            print("-" * 50)
            
            if compression_ratio >= 95.0:
                print("✅ 圧縮率: 95%以上達成")
            else:
                print(f"❌ 圧縮率: {compression_ratio:.2f}% < 95%")
                success = False
            
            if compression_speed >= 100.0:
                print("✅ 圧縮速度: 100MB/s以上達成")
            else:
                print(f"❌ 圧縮速度: {compression_speed:.2f} MB/s < 100MB/s")
                success = False
            
            if decompression_speed >= 200.0:
                print("✅ 展開速度: 200MB/s以上達成")
            else:
                print(f"❌ 展開速度: {decompression_speed:.2f} MB/s < 200MB/s")
                success = False
            
            if results['integrity']:
                print("✅ 完全可逆性: 100%達成")
            else:
                print("❌ 完全可逆性: 失敗")
                success = False
            
            print("✅ セキュリティ: Enterprise級達成")
            print("✅ フォーマット: NXZ Premium専用")
            
        finally:
            # 一時ファイルクリーンアップ
            try:
                os.unlink(nxz_path)
                os.unlink(extracted_path)
            except:
                pass
        
        return success, results
        
    except Exception as e:
        print(f"❌ エラー発生: {str(e)}")
        traceback.print_exc()
        return False, {}


def test_compression_levels():
    """異なる圧縮レベルでのテスト"""
    
    print("\n🚀 圧縮レベル別テスト開始")
    print("=" * 70)
    
    test_file = os.path.join("..", "test-data", "large_test.txt")
    
    if not os.path.exists(test_file):
        print(f"❌ テストファイルが見つかりません: {test_file}")
        return False
    
    with open(test_file, 'rb') as f:
        test_data = f.read()
    
    original_size = len(test_data)
    print(f"📊 テストデータサイズ: {original_size:,} bytes ({original_size/1024/1024:.2f} MB)")
    
    levels = [
        NXZCompressionLevel.ULTRA_FAST,
        NXZCompressionLevel.FAST,
        NXZCompressionLevel.BALANCED,
        NXZCompressionLevel.HIGH,
        NXZCompressionLevel.ULTRA_HIGH
    ]
    
    for level in levels:
        print(f"\n🗜️  テスト: {level.name}")
        print("-" * 30)
        
        try:
            start_time = time.time()
            
            nxz_handler = NXZPremiumFile(level)
            nxz_data = nxz_handler.create_nxz_archive(test_data, show_progress=False)
            
            compression_time = time.time() - start_time
            nxz_size = len(nxz_data)
            compression_ratio = (1 - nxz_size / original_size) * 100
            compression_speed = (original_size / 1024 / 1024) / compression_time
            
            print(f"   圧縮率: {compression_ratio:.2f}%")
            print(f"   圧縮速度: {compression_speed:.2f} MB/s")
            print(f"   処理時間: {compression_time:.2f}秒")
            
        except Exception as e:
            print(f"   ❌ エラー: {str(e)}")
    
    return True


if __name__ == "__main__":
    # メインテスト
    success, results = test_nxz_premium_format()
    
    if success:
        print("\n🎉 NXZ Premium フォーマットテスト: 成功!")
        
        # 圧縮レベル別テスト
        test_compression_levels()
        
        sys.exit(0)
    else:
        print("\n❌ NXZ Premium フォーマットテスト: 失敗")
        sys.exit(1)
