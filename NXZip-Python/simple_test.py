#!/usr/bin/env python3
"""
シンプルなTMCテスト
"""
import sys
import os

print("🔍 TMC v9.0 エンジンテスト開始")

try:
    # 段階的にインポートしてどこで失敗するかを確認
    print("Step 1: 基本ライブラリ...")
    import numpy as np
    print("✅ NumPy OK")
    
    import psutil
    print("✅ psutil OK")
    
    print("Step 2: NXZip エンジン...")
    sys.path.insert(0, '.')
    
    # まず個別に確認
    try:
        from nxzip.engine import nexus_tmc
        print("✅ nexus_tmc モジュール OK")
    except Exception as e:
        print(f"❌ nexus_tmc モジュール: {e}")
    
    # クラスをインポート
    try:
        from nxzip.engine.nexus_tmc import NEXUSTMCEngineV9
        print("✅ NEXUSTMCEngineV9 クラス OK")
        
        # エンジン作成
        engine = NEXUSTMCEngineV9(max_workers=1)
        print("✅ エンジン初期化 OK")
        
        # 簡単なテスト
        test_data = b"Hello TMC"
        compressed, meta = engine.compress_tmc(test_data)
        decompressed = engine.decompress_tmc(compressed, meta)
        
        print(f"📊 圧縮: {len(test_data)} -> {len(compressed)} bytes")
        print(f"🔄 可逆性: {'OK' if test_data == decompressed else 'NG'}")
        
        if test_data == decompressed:
            print("🎉 TMC v9.0 テスト成功！")
        else:
            print("⚠️ 可逆性に問題があります")
            
    except Exception as e:
        print(f"❌ NEXUSTMCEngineV9: {e}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"❌ 基本ライブラリエラー: {e}")
    import traceback
    traceback.print_exc()
