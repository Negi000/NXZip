#!/usr/bin/env python3
"""
NXZip TMC v9.1 統括モジュール最適化テスト
オリジナル圧縮アーキテクチャの性能検証
"""

import time
from nxzip.engine.nexus_tmc_v91_modular import NEXUSTMCEngineV91

def test_nxzip_tmc_optimization():
    """最適化されたNXZip TMC v9.1のテスト"""
    print("=== NXZip TMC v9.1 最適化テスト ===")
    
    # テストデータ
    test_data = b"NXZip TMC v9.1 optimization test data " * 100
    print(f"📊 テストデータ: {len(test_data)} bytes")
    
    try:
        # 軽量モードテスト
        print("\n⚡ 軽量モード (Zstandardレベル目標)")
        engine_light = NEXUSTMCEngineV91(lightweight_mode=True)
        
        start = time.time()
        compressed_light, info_light = engine_light.compress(test_data)
        light_time = time.time() - start
        
        print(f"結果: {info_light.get('compression_ratio', 0):.1f}% 圧縮, {light_time:.3f}秒")
        print(f"エンジン: {info_light.get('engine_version', 'unknown')}")
        
    except Exception as e:
        print(f"❌ 軽量モードエラー: {e}")
    
    try:
        # 通常モードテスト
        print("\n🎯 通常モード (7-Zip超越目標)")
        engine_normal = NEXUSTMCEngineV91(lightweight_mode=False)
        
        start = time.time()
        compressed_normal, info_normal = engine_normal.compress(test_data)
        normal_time = time.time() - start
        
        print(f"結果: {info_normal.get('compression_ratio', 0):.1f}% 圧縮, {normal_time:.3f}秒")
        print(f"エンジン: {info_normal.get('engine_version', 'unknown')}")
        
    except Exception as e:
        print(f"❌ 通常モードエラー: {e}")
    
    print("\n✅ NXZip TMC v9.1 最適化テスト完了")

if __name__ == "__main__":
    test_nxzip_tmc_optimization()
