#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TMC v9.1 モジュラーエンジンの可逆性テスト（修正版）
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nxzip.engine.nexus_tmc_v91_modular import NEXUSTMCEngineV91
import asyncio

async def test_tmc_v91_fix():
    """修正されたTMC v9.1エンジンの可逆性をテスト"""
    print("🔧 TMC v9.1 修正版テスト開始...")
    
    engine = NEXUSTMCEngineV91(lightweight_mode=True)
    
    # テストデータ
    test_strings = [
        "Hello, World! This is a simple test.",
        "あいうえおかきくけこ" * 10,  # 日本語繰り返し
        b"Binary data with \x00\xff\xaa\x55" * 20,  # バイナリデータ
        "A" * 1000,  # 単純な繰り返し
    ]
    
    all_passed = True
    
    for i, test_data in enumerate(test_strings):
        print(f"\n--- テスト {i+1}: {type(test_data).__name__} ---")
        
        if isinstance(test_data, str):
            test_data = test_data.encode('utf-8')
        
        try:
            # 圧縮
            print("🔄 圧縮中...")
            compressed, info = await engine.compress_tmc_v91_async(test_data)
            compression_ratio = (1 - len(compressed) / len(test_data)) * 100
            print(f"圧縮率: {compression_ratio:.1f}% ({len(test_data)} → {len(compressed)} bytes)")
            
            # 解凍
            print("🔄 解凍中...")
            decompressed = engine.decompress(compressed, info)
            
            # 検証
            if decompressed == test_data:
                print("✅ 可逆性OK！")
            else:
                print("❌ データが一致しません")
                print(f"元サイズ: {len(test_data)}, 復元サイズ: {len(decompressed)}")
                if len(test_data) < 100 and len(decompressed) < 100:
                    print(f"元データ: {test_data}")
                    print(f"復元データ: {decompressed}")
                all_passed = False
                
        except Exception as e:
            print(f"❌ エラー発生: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print(f"\n{'🎉 全テスト成功！' if all_passed else '❌ 一部テスト失敗'}")
    return all_passed

if __name__ == "__main__":
    asyncio.run(test_tmc_v91_fix())
