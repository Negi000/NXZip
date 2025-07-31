#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEXUS TMC v9.0 エンジン簡単テスト
"""
import sys
import os

# 現在の状況を確認
print(f"🔍 現在ディレクトリ: {os.getcwd()}")
print(f"🐍 Python version: {sys.version}")

try:
    # NEXUS TMC エンジンをインポート
    from nxzip.engine.nexus_tmc import NEXUSTMCEngineV9
    print("✅ NEXUS TMC v9.0 エンジンのインポート成功")
    
    # エンジン初期化
    engine = NEXUSTMCEngineV9(max_workers=1)
    print("✅ エンジン初期化成功")
    
    # 基本テスト
    test_data = b"NEXUS TMC v9.0 Engine Test Data"
    print(f"📄 テストデータ: {len(test_data)} bytes")
    
    # 圧縮テスト
    compressed, meta = engine.compress_tmc(test_data)
    print(f"📦 圧縮完了: {len(compressed)} bytes")
    
    # 展開テスト
    decompressed = engine.decompress_tmc(compressed, meta)
    print(f"📂 展開完了: {len(decompressed)} bytes")
    
    # 可逆性確認
    is_identical = test_data == decompressed
    compression_ratio = len(compressed) / len(test_data) * 100
    
    print(f"📊 圧縮率: {compression_ratio:.1f}%")
    print(f"🔄 可逆性: {'✅ OK' if is_identical else '❌ NG'}")
    
    if 'data_type' in meta:
        print(f"🔍 検出データ型: {meta['data_type']}")
    
    if is_identical:
        print("🎉 NEXUS TMC v9.0 基本テスト - 完全成功！")
    else:
        print("⚠️ 可逆性に問題があります")
        print(f"   元データ: {test_data}")
        print(f"   復元データ: {decompressed}")
        
except ImportError as e:
    print(f"❌ インポートエラー: {e}")
except Exception as e:
    print(f"❌ 実行エラー: {e}")
    import traceback
    traceback.print_exc()
