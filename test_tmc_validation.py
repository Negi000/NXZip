#!/usr/bin/env python3
"""
🔍 NEXUS TMC v9.1 真のテスト - 圧縮と展開の整合性検証
"""

import os
import sys
import hashlib
import time
from pathlib import Path

# NXZip-Python パスを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'NXZip-Python'))

try:
    from nxzip.engine.nexus_tmc_v91_modular import NEXUSTMCEngineV91
    print('✅ NEXUS TMC v9.1 エンジンインポート成功')
except ImportError as e:
    print(f'❌ インポートエラー: {e}')
    sys.exit(1)

def create_test_data(size_mb: float = 1.0) -> bytes:
    """テスト用データ（反復テキスト）生成"""
    base_text = "これはNEXUS TMC v9.1の圧縮テストデータです。" * 100
    target_bytes = int(size_mb * 1024 * 1024)
    base_bytes = base_text.encode('utf-8')
    repeat_count = target_bytes // len(base_bytes)
    return (base_text * repeat_count).encode('utf-8')

def validate_compression_cycle(engine, test_data: bytes, test_name: str):
    """圧縮→展開→検証のサイクルテスト"""
    print(f"\n🧪 {test_name} テスト開始")
    print(f"📊 元データサイズ: {len(test_data):,} bytes")
    
    # 元データのハッシュ計算
    original_hash = hashlib.sha256(test_data).hexdigest()
    print(f"🔍 元データハッシュ: {original_hash[:16]}...")
    
    try:
        # 圧縮フェーズ
        print("🗜️ 圧縮中...")
        start_time = time.time()
        compressed_data, compress_info = engine.compress(test_data)
        compress_time = time.time() - start_time
        
        compression_ratio = compress_info.get('compression_ratio', 0)
        print(f"✅ 圧縮完了: {len(compressed_data):,} bytes ({compression_ratio:.2f}% 圧縮)")
        print(f"⏱️ 圧縮時間: {compress_time:.2f}秒")
        print(f"🔥 TMC変換: {'適用' if compress_info.get('transform_applied') else 'バイパス'}")
        
        # 展開フェーズ
        print("📂 展開中...")
        start_time = time.time()
        
        # IMPORTANT: TMC v9.1では正しい展開メソッドを使用
        decompressed_data = engine.decompress(compressed_data, compress_info)
        decompress_time = time.time() - start_time
        
        print(f"✅ 展開完了: {len(decompressed_data):,} bytes")
        print(f"⏱️ 展開時間: {decompress_time:.2f}秒")
        
        # 整合性検証
        decompressed_hash = hashlib.sha256(decompressed_data).hexdigest()
        print(f"🔍 展開データハッシュ: {decompressed_hash[:16]}...")
        
        # 結果判定
        if original_hash == decompressed_hash:
            print("🎉 ✅ 整合性検証成功 - TMC v9.1は正常に動作しています！")
            size_match = len(test_data) == len(decompressed_data)
            print(f"📏 サイズ一致: {'✅' if size_match else '❌'} ({len(test_data)} vs {len(decompressed_data)})")
            return True
        else:
            print("❌ 整合性検証失敗 - データが破損しています")
            print(f"🔍 サイズ比較: 元={len(test_data)} vs 展開={len(decompressed_data)}")
            
            # 部分比較（デバッグ用）
            if len(decompressed_data) > 0:
                match_length = 0
                min_len = min(len(test_data), len(decompressed_data))
                for i in range(min_len):
                    if test_data[i] == decompressed_data[i]:
                        match_length += 1
                    else:
                        break
                print(f"🔍 一致する先頭バイト数: {match_length}/{min_len}")
            
            return False
            
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン実行関数"""
    print("🎯 NEXUS TMC v9.1 真のテスト開始")
    print("=" * 60)
    
    # エンジン初期化
    print("🚀 TMC v9.1 エンジン初期化中...")
    engine = NEXUSTMCEngineV91(
        max_workers=4,
        chunk_size=2*1024*1024,  # 2MB チャンク
        lightweight_mode=False   # 通常モード（最大圧縮）
    )
    print(f"✅ エンジン初期化完了: {engine.max_workers}ワーカー, {engine.chunk_size//1024//1024}MBチャンク")
    
    # テストケース1: 小さなデータ（16KB）
    print("\n" + "="*60)
    small_data = create_test_data(0.016)  # 16KB
    success1 = validate_compression_cycle(engine, small_data, "小サイズデータ（16KB）")
    
    # テストケース2: 中サイズデータ（1MB）
    print("\n" + "="*60)
    medium_data = create_test_data(1)  # 1MB
    success2 = validate_compression_cycle(engine, medium_data, "中サイズデータ（1MB）")
    
    # テストケース3: 大サイズデータ（10MB）
    print("\n" + "="*60)
    large_data = create_test_data(10)  # 10MB
    success3 = validate_compression_cycle(engine, large_data, "大サイズデータ（10MB）")
    
    # 総合結果
    print("\n" + "="*60)
    print("🏆 テスト結果サマリー")
    print("=" * 60)
    print(f"小サイズ（16KB）: {'✅ 成功' if success1 else '❌ 失敗'}")
    print(f"中サイズ（1MB） : {'✅ 成功' if success2 else '❌ 失敗'}")
    print(f"大サイズ（10MB）: {'✅ 成功' if success3 else '❌ 失敗'}")
    
    all_success = success1 and success2 and success3
    if all_success:
        print("\n🎉 🔥 NEXUS TMC v9.1 - 完全成功！真のTMC圧縮が実現されています！")
        print("🚀 7-Zip + Zstandard超越の準備完了！")
    else:
        print("\n⚠️ 一部のテストで問題が検出されました。")
        print("🔧 TMC変換または展開ロジックの調整が必要です。")
    
    # エンジン統計表示
    print("\n📊 エンジン統計:")
    stats = engine.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
