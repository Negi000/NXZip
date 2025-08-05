#!/usr/bin/env python3
"""
NXZip Core v2.0 テストスクリプト
新しい統括モジュールの動作確認
"""

import os
import sys
import time
from pathlib import Path

# NXZip Core インポート
try:
    from nxzip_core import NXZipCore, NXZipContainer
    print("✅ NXZip Core v2.0 インポート成功")
except ImportError as e:
    print(f"❌ NXZip Core インポート失敗: {e}")
    sys.exit(1)

def progress_callback(info):
    """進捗表示コールバック"""
    progress = info['progress']
    message = info['message']
    speed = info.get('speed', 0)
    
    # 速度を適切な単位で表示
    if speed > 1024 * 1024:
        speed_str = f"{speed / (1024 * 1024):.1f} MB/s"
    elif speed > 1024:
        speed_str = f"{speed / 1024:.1f} KB/s"
    else:
        speed_str = f"{speed:.0f} B/s"
    
    print(f"\r🔄 {progress:5.1f}% | {message} | {speed_str}", end="", flush=True)

def test_text_compression():
    """テキストデータ圧縮テスト"""
    print("\n" + "="*60)
    print("📝 テキストデータ圧縮テスト")
    print("="*60)
    
    # テストデータ作成（日本語テキスト）
    test_text = """
    NXZip Core v2.0 - 次世代統括圧縮プラットフォーム
    これは新しいアーキテクチャでの圧縮テストです。
    
    コンセプト:
    - 標準モード: 7Zレベル圧縮率 + 7Z×2以上の速度
    - 高速モード: Zstdレベル速度 + Zstdを超える圧縮率
    - TMC (Transform-Model-Code) + SPE (Structure-Preserving Encryption) 統合
    
    このテキストは冗長性があり、圧縮効果が期待できます。
    このテキストは冗長性があり、圧縮効果が期待できます。
    このテキストは冗長性があり、圧縮効果が期待できます。
    """ * 100  # 繰り返しで冗長性を追加
    
    test_data = test_text.encode('utf-8')
    original_size = len(test_data)
    
    print(f"📊 テストデータサイズ: {original_size:,} bytes ({original_size/1024:.1f} KB)")
    
    # NXZip Core初期化
    core = NXZipCore()
    core.set_progress_callback(progress_callback)
    
    # 各モードでテスト
    modes = ["fast", "balanced", "maximum"]
    
    for mode in modes:
        print(f"\n🚀 {mode.upper()}モード テスト:")
        
        # 圧縮
        start_time = time.time()
        result = core.compress(test_data, mode=mode, filename="test.txt")
        
        if result.success:
            print(f"\n✅ 圧縮成功!")
            print(f"   圧縮率: {result.compression_ratio:.2f}%")
            print(f"   圧縮時間: {result.compression_time:.3f}秒")
            
            # ゼロ除算対策
            if result.compression_time > 0:
                speed_mbps = (original_size / (1024 * 1024)) / result.compression_time
                print(f"   速度: {speed_mbps:.1f} MB/s")
            else:
                print(f"   速度: 非常に高速（測定不可）")
                
            print(f"   エンジン: {result.engine}")
            print(f"   メソッド: {result.method}")
            
            # 目標達成度確認
            target_eval = result.metadata.get('target_evaluation', {})
            if target_eval:
                print(f"   目標達成: {'✅' if target_eval.get('target_achieved') else '❌'}")
                print(f"   コンセプト: {target_eval.get('concept', 'N/A')}")
            
            # コンテナ作成テスト
            container_data = NXZipContainer.pack(
                result.compressed_data, 
                result.metadata,
                "test.txt"
            )
            print(f"   コンテナサイズ: {len(container_data):,} bytes")
            
            # 展開テスト
            print("   🔓 展開テスト中...")
            decomp_result = core.decompress(result.compressed_data, result.metadata)
            
            if decomp_result.success:
                # 整合性確認
                integrity = core.validate_integrity(test_data, decomp_result.decompressed_data)
                print(f"   整合性: {'✅' if integrity['integrity_ok'] else '❌'}")
                print(f"   展開時間: {decomp_result.decompression_time:.3f}秒")
            else:
                print(f"   ❌ 展開失敗: {decomp_result.error_message}")
                
        else:
            print(f"\n❌ 圧縮失敗: {result.error_message}")

def test_binary_data():
    """バイナリデータ圧縮テスト"""
    print("\n" + "="*60)
    print("🗂️ バイナリデータ圧縮テスト")
    print("="*60)
    
    # バイナリテストデータ作成（構造化された数値データ）
    import numpy as np
    
    # 浮動小数点配列（科学計算データを模擬）
    test_array = np.random.normal(0, 1, 10000).astype(np.float32)
    test_data = test_array.tobytes()
    original_size = len(test_data)
    
    print(f"📊 テストデータサイズ: {original_size:,} bytes ({original_size/1024:.1f} KB)")
    print(f"📊 データタイプ: 浮動小数点配列")
    
    # NXZip Core初期化
    core = NXZipCore()
    core.set_progress_callback(progress_callback)
    
    # バランスモードでテスト
    print(f"\n🚀 BALANCEDモード テスト:")
    
    start_time = time.time()
    result = core.compress(test_data, mode="balanced", filename="data.bin")
    
    if result.success:
        print(f"\n✅ 圧縮成功!")
        print(f"   圧縮率: {result.compression_ratio:.2f}%")
        print(f"   圧縮時間: {result.compression_time:.3f}秒")
        
        # ゼロ除算対策
        if result.compression_time > 0:
            speed_mbps = (original_size / (1024 * 1024)) / result.compression_time
            print(f"   速度: {speed_mbps:.1f} MB/s")
        else:
            print(f"   速度: 非常に高速（測定不可）")
        
        # パイプライン詳細確認
        pipeline_info = result.metadata
        print(f"   データタイプ: {pipeline_info.get('data_type', 'N/A')}")
        
        stages = pipeline_info.get('stages', [])
        for stage_name, stage_info in stages:
            if stage_name == 'tmc_transform':
                transforms = stage_info.get('transforms_applied', [])
                if transforms:
                    print(f"   TMC変換: {', '.join(transforms)}")
            elif stage_name == 'spe_integration':
                if stage_info.get('spe_applied'):
                    print(f"   SPE適用: ✅")
            elif stage_name == 'final_compression':
                method = stage_info.get('method', 'N/A')
                stage_ratio = stage_info.get('stage_ratio', 0)
                print(f"   最終圧縮: {method} ({stage_ratio:.1f}%)")
        
        # 展開テスト
        print("   🔓 展開テスト中...")
        decomp_result = core.decompress(result.compressed_data, result.metadata)
        
        if decomp_result.success:
            integrity = core.validate_integrity(test_data, decomp_result.decompressed_data)
            print(f"   整合性: {'✅' if integrity['integrity_ok'] else '❌'}")
            
            # 数値データの精度確認
            original_array = np.frombuffer(test_data, dtype=np.float32)
            recovered_array = np.frombuffer(decomp_result.decompressed_data, dtype=np.float32)
            
            if np.array_equal(original_array, recovered_array):
                print(f"   数値精度: ✅ 完全一致")
            else:
                max_diff = np.max(np.abs(original_array - recovered_array))
                print(f"   数値精度: ⚠️ 最大差分: {max_diff}")
        else:
            print(f"   ❌ 展開失敗: {decomp_result.error_message}")
    else:
        print(f"\n❌ 圧縮失敗: {result.error_message}")

def test_small_file():
    """小さなファイル圧縮テスト"""
    print("\n" + "="*60)
    print("📄 小ファイル圧縮テスト")
    print("="*60)
    
    test_data = b"Hello, NXZip Core v2.0! This is a small test file."
    original_size = len(test_data)
    
    print(f"📊 テストデータサイズ: {original_size} bytes")
    
    core = NXZipCore()
    core.set_progress_callback(progress_callback)
    
    result = core.compress(test_data, mode="fast", filename="small.txt")
    
    if result.success:
        print(f"\n✅ 圧縮成功!")
        print(f"   圧縮後サイズ: {result.compressed_size} bytes")
        print(f"   圧縮率: {result.compression_ratio:.2f}%")
        print(f"   圧縮時間: {result.compression_time:.6f}秒")
        
        # 展開確認
        decomp_result = core.decompress(result.compressed_data, result.metadata)
        if decomp_result.success:
            integrity = core.validate_integrity(test_data, decomp_result.decompressed_data)
            print(f"   整合性: {'✅' if integrity['integrity_ok'] else '❌'}")
        else:
            print(f"   ❌ 展開失敗: {decomp_result.error_message}")
    else:
        print(f"\n❌ 圧縮失敗: {result.error_message}")

def main():
    """メインテスト実行"""
    print("🚀 NXZip Core v2.0 総合テスト開始")
    print("="*60)
    
    try:
        # 各テストを実行
        test_text_compression()
        test_binary_data()
        test_small_file()
        
        print("\n" + "="*60)
        print("🎉 全テスト完了!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ テスト中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
