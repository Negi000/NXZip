#!/usr/bin/env python3
"""
修正版NXZip緊急テスト
修正点:
1. データタイプ判定ロジック修正
2. 遅延初期化による高速化
3. 圧縮レベル最適化
"""

import os
import sys
import time
import hashlib
from pathlib import Path

# NXZip-Pythonパスを追加
sys.path.insert(0, str(Path(__file__).parent / "NXZip-Python"))

try:
    from nxzip.engine.nexus_tmc_v91_modular import NEXUSTMCEngineV91
    print("✅ NEXUSTMCEngineV91 インポート成功")
except ImportError as e:
    print(f"❌ NEXUSTMCEngineV91 インポートエラー: {e}")
    sys.exit(1)

def test_data_type_detection():
    """データタイプ判定の修正確認"""
    print("🔍 データタイプ判定テスト開始")
    
    test_cases = {
        'text': "Hello World! " * 1000,
        'numeric': bytes([i % 100 for i in range(2000)]),
        'mixed': ("Header: " + "A" * 100).encode() + bytes([i % 50 for i in range(1000)][:500])
    }
    
    engine = NEXUSTMCEngineV91(lightweight_mode=True)
    
    for name, data in test_cases.items():
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        data_type = engine.dispatcher.dispatch_data_type(data)
        print(f"  {name}: {data_type.value}")
    
    return True

def speed_test():
    """速度改善確認"""
    print("⚡ 速度改善テスト開始")
    
    # 小さなテストデータ
    test_data = "Speed test data! " * 100  # 約1.7KB
    data_bytes = test_data.encode('utf-8')
    
    # 軽量モード（改善版）
    start_time = time.time()
    engine = NEXUSTMCEngineV91(lightweight_mode=True)
    init_time = time.time() - start_time
    
    start_time = time.time()
    compressed, info = engine.compress(data_bytes)
    compress_time = time.time() - start_time
    
    start_time = time.time()
    decompressed = engine.decompress(compressed, info)
    decompress_time = time.time() - start_time
    
    # 検証
    original_hash = hashlib.sha256(data_bytes).hexdigest()
    decompressed_hash = hashlib.sha256(decompressed).hexdigest()
    valid = original_hash == decompressed_hash
    
    compression_ratio = (1 - len(compressed) / len(data_bytes)) * 100
    
    print(f"  初期化時間: {init_time:.3f}秒")
    print(f"  圧縮時間: {compress_time:.3f}秒")
    print(f"  解凍時間: {decompress_time:.3f}秒")
    print(f"  圧縮率: {compression_ratio:.1f}%")
    print(f"  可逆性: {'✅' if valid else '❌'}")
    
    return valid and init_time < 0.5  # 500ms以下の初期化時間

def compression_ratio_test():
    """圧縮率改善確認"""
    print("🗜️ 圧縮率改善テスト開始")
    
    # 高圧縮率期待データ
    repetitive_data = ("Compression test pattern! " * 500).encode('utf-8')
    
    engine = NEXUSTMCEngineV91(lightweight_mode=True)
    compressed, info = engine.compress(repetitive_data)
    
    compression_ratio = (1 - len(compressed) / len(repetitive_data)) * 100
    print(f"  元サイズ: {len(repetitive_data):,} bytes")
    print(f"  圧縮サイズ: {len(compressed):,} bytes")
    print(f"  圧縮率: {compression_ratio:.1f}%")
    
    # 目標: 99%以上
    return compression_ratio >= 99.0

def main():
    """修正効果確認"""
    print("🚀 NXZip修正効果確認テスト開始")
    
    tests = [
        ("データタイプ判定", test_data_type_detection),
        ("速度改善", speed_test),
        ("圧縮率改善", compression_ratio_test)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"✅ {test_name}: {'合格' if result else '要改善'}")
        except Exception as e:
            print(f"❌ {test_name}: エラー - {e}")
            results.append((test_name, False))
    
    print(f"\n{'='*50}")
    print(f"📊 修正効果サマリー")
    print(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 合格" if result else "❌ 要改善"
        print(f"  {test_name}: {status}")
    
    print(f"\n総合結果: {passed}/{total} 合格")
    
    if passed == total:
        print("🎉 全テスト合格！修正効果確認")
    else:
        print("⚠️ さらなる修正が必要")

if __name__ == "__main__":
    main()
