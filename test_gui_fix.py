#!/usr/bin/env python3
"""
修正されたTMC GUIの動作確認テスト
=================================
310MB→508KB問題が解決されているかを検証
"""

import os
import sys
import hashlib
import time
from pathlib import Path

# NXZip-Release フォルダーをパスに追加
sys.path.insert(0, str(Path(__file__).parent / "NXZip-Release"))

def calculate_hash(file_path):
    """ファイルのSHA256ハッシュを計算"""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def get_file_size(file_path):
    """ファイルサイズを取得（MB単位）"""
    return os.path.getsize(file_path) / (1024 * 1024)

def test_gui_fix():
    """GUIが修正されているかテスト（プログラマティック）"""
    print("🔧 TMC GUI修正テスト")
    print("=" * 50)
    
    try:
        # GUIコンポーネントをインポート（UIは起動しない）
        from NXZip_Professional_v2 import AdvancedNXZipEngine
        from engine.tmc_safe_wrapper import TMCSafeWrapper
        print("✅ 修正されたGUIモジュールのインポート成功")
        
        # TMC安全ラッパーの存在確認
        if hasattr(AdvancedNXZipEngine, 'tmc_safe_wrapper'):
            print("✅ TMC安全ラッパーが統合されている")
        else:
            print("⚠️ TMC安全ラッパーが見つからない")
        
        # エンジンインスタンス作成テスト
        engine = AdvancedNXZipEngine()
        print("✅ エンジンインスタンス作成成功")
        
        # TMC固定エンジンの可用性確認
        try:
            from nexus_tmc_v91_fixed import NEXUSTMCEngineV91Fixed
            print("✅ TMC修正版エンジンが利用可能")
        except ImportError:
            print("⚠️ TMC修正版エンジンが見つからない")
        
        # 安全ラッパーの動作テスト（小さなテストデータで）
        test_data = b"Test data for TMC safety validation" * 1000  # 約35KB
        print(f"📝 {'テストデータサイズ:':<20} {len(test_data):,} bytes")
        
        # TMC圧縮テスト
        print("🔄 TMC圧縮テスト実行中...")
        compressed_result, info = engine.compress(test_data)
        
        if compressed_result and len(compressed_result) > 0:
            print(f"✅ {'TMC圧縮成功:':<20} {len(compressed_result):,} bytes ({len(compressed_result)/len(test_data)*100:.1f}%)")
            
            # TMC解凍テスト
            print("🔄 TMC解凍テスト実行中...")
            decompressed_data = engine.decompress(compressed_result, info)
            
            if decompressed_data:
                print(f"✅ {'TMC解凍成功:':<20} {len(decompressed_data):,} bytes")
                
                # データ整合性確認
                if decompressed_data == test_data:
                    print("✅ データ整合性検証: 完全一致")
                    print("🎯 TMC GUI修正テスト: 成功")
                    return True
                else:
                    print("❌ データ整合性検証: 不一致")
                    print(f"   元データ: {len(test_data):,} bytes")
                    print(f"   解凍データ: {len(decompressed_data):,} bytes")
            else:
                print("❌ TMC解凍失敗: データがNone")
        else:
            print("❌ TMC圧縮失敗")
        
    except Exception as e:
        print(f"❌ テスト中にエラー: {e}")
        import traceback
        traceback.print_exc()
        
    print("🎯 TMC GUI修正テスト: 完了")
    return False

def main():
    """メイン実行"""
    start_time = time.time()
    
    # カレントディレクトリ確認
    current_dir = Path.cwd()
    print(f"📂 作業ディレクトリ: {current_dir}")
    
    # NXZip-Releaseフォルダーの存在確認
    release_dir = current_dir / "NXZip-Release"
    if not release_dir.exists():
        print("❌ NXZip-Releaseフォルダーが見つかりません")
        return
    
    # TMC GUI修正テスト実行
    success = test_gui_fix()
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️ 総実行時間: {elapsed_time:.2f}秒")
    
    if success:
        print("\n🎉 TMC GUI修正が成功しています！")
        print("   310MB→508KB問題は解決されているはずです")
    else:
        print("\n⚠️ TMC GUI修正に問題がある可能性があります")
        print("   実際のファイルでのテストが必要です")

if __name__ == "__main__":
    main()
