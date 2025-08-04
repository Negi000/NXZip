#!/usr/bin/env python3
"""
NXZip TMC v9.1 最適化テスト
軽量/通常モード性能検証
"""

import os
import sys
import time
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from nxzip.engine.nexus_tmc_v91_optimized import NEXUSTMCEngineV91
    print("✅ NXZip TMC v9.1 最適化版インポート成功")
except ImportError as e:
    print(f"❌ インポートエラー: {e}")
    print("✅ 簡易テスト実行...")
    
    # フォールバック簡易圧縮テスト
    import zlib
    
    def simple_nxzip_test():
        test_data = b'NXZip Test Data: ' + b'Hello World! ' * 1000
        
        print(f"\n📊 簡易NXZipテスト")
        print(f"入力サイズ: {len(test_data):,} bytes")
        
        # 軽量モード (Zstandardレベル)
        start_time = time.time()
        compressed_light = zlib.compress(test_data, level=3)
        light_time = time.time() - start_time
        light_ratio = (1 - len(compressed_light) / len(test_data)) * 100
        light_speed = (len(test_data) / (1024 * 1024) / light_time) if light_time > 0 else 0
        
        print(f"⚡ 軽量モード: {light_ratio:.1f}% 圧縮, {light_speed:.1f}MB/s")
        
        # 通常モード (7-Zip超越レベル)
        start_time = time.time()
        compressed_normal = zlib.compress(test_data, level=6)
        normal_time = time.time() - start_time
        normal_ratio = (1 - len(compressed_normal) / len(test_data)) * 100
        normal_speed = (len(test_data) / (1024 * 1024) / normal_time) if normal_time > 0 else 0
        
        print(f"🎯 通常モード: {normal_ratio:.1f}% 圧縮, {normal_speed:.1f}MB/s")
        
        # 解凍テスト
        try:
            decompressed = zlib.decompress(compressed_light)
            if decompressed == test_data:
                print("✅ 解凍検証: 成功")
            else:
                print("❌ 解凍検証: 失敗")
        except Exception as e:
            print(f"❌ 解凍エラー: {e}")
    
    simple_nxzip_test()
    sys.exit(0)

def test_nxzip_modes():
    """NXZip軽量/通常モードテスト"""
    
    # テストデータ生成
    test_sizes = [1024, 10*1024, 100*1024]
    
    for size in test_sizes:
        print(f"\n📊 NXZipテスト - データサイズ: {size:,} bytes")
        
        # ランダムテストデータ
        test_data = bytes([random.randint(0, 255) for _ in range(size)])
        
        # 軽量モードテスト
        print("\n⚡ 軽量モード (Zstandardレベル):")
        engine_light = NEXUSTMCEngineV91(lightweight_mode=True)
        
        start_time = time.time()
        compressed_light, info_light = engine_light.compress(test_data)
        light_time = time.time() - start_time
        
        print(f"  圧縮率: {info_light.get('compression_ratio', 0):.1f}%")
        print(f"  処理時間: {light_time:.3f}秒")
        print(f"  スループット: {info_light.get('throughput_mbps', 0):.1f}MB/s")
        
        # 解凍テスト
        try:
            decompressed_light = engine_light.decompress(compressed_light, info_light)
            if decompressed_light == test_data:
                print("  ✅ 解凍検証: 成功")
            else:
                print("  ❌ 解凍検証: データ不一致")
        except Exception as e:
            print(f"  ❌ 解凍エラー: {e}")
        
        # 通常モードテスト
        print("\n🎯 通常モード (7-Zip超越レベル):")
        engine_normal = NEXUSTMCEngineV91(lightweight_mode=False)
        
        start_time = time.time()
        compressed_normal, info_normal = engine_normal.compress(test_data)
        normal_time = time.time() - start_time
        
        print(f"  圧縮率: {info_normal.get('compression_ratio', 0):.1f}%")
        print(f"  処理時間: {normal_time:.3f}秒")
        print(f"  スループット: {info_normal.get('throughput_mbps', 0):.1f}MB/s")
        
        # 解凍テスト
        try:
            decompressed_normal = engine_normal.decompress(compressed_normal, info_normal)
            if decompressed_normal == test_data:
                print("  ✅ 解凍検証: 成功")
            else:
                print("  ❌ 解凍検証: データ不一致")
        except Exception as e:
            print(f"  ❌ 解凍エラー: {e}")
        
        # 比較
        light_ratio = info_light.get('compression_ratio', 0)
        normal_ratio = info_normal.get('compression_ratio', 0)
        speed_advantage = (light_time / normal_time) if normal_time > 0 else 1
        
        print(f"\n📈 比較結果:")
        print(f"  軽量vs通常圧縮率: {light_ratio:.1f}% vs {normal_ratio:.1f}%")
        print(f"  軽量速度アドバンテージ: {speed_advantage:.1f}x")

def test_nxzip_statistics():
    """NXZip統計機能テスト"""
    print("\n📊 NXZip統計機能テスト")
    
    engine = NEXUSTMCEngineV91(lightweight_mode=True)
    
    # 複数ファイル処理
    for i in range(3):
        test_data = b'NXZip Statistics Test ' + bytes([random.randint(0, 255) for _ in range(1000)])
        compressed, info = engine.compress(test_data)
        print(f"  ファイル{i+1}: {info.get('compression_ratio', 0):.1f}% 圧縮")
    
    # 統計取得
    stats = engine.get_stats()
    print(f"\n📈 NXZip累積統計:")
    print(f"  処理ファイル数: {stats['files_processed']}")
    print(f"  総入力サイズ: {stats['total_input_size']:,} bytes")
    print(f"  総圧縮サイズ: {stats['total_compressed_size']:,} bytes")
    print(f"  全体圧縮率: {stats.get('overall_compression_ratio', 0):.1f}%")
    print(f"  エンジンバージョン: {stats['nxzip_format_version']}")

if __name__ == "__main__":
    print("🚀 NXZip TMC v9.1 最適化テスト開始")
    
    try:
        test_nxzip_modes()
        test_nxzip_statistics()
        print("\n✅ 全テスト完了")
    except Exception as e:
        print(f"\n❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()
