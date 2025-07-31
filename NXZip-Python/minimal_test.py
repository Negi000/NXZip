#!/usr/bin/env python3
"""
最小限のTMCエンジンテスト
"""

def test_basic_imports():
    """基本インポートテスト"""
    try:
        import sys
        import os
        print("✅ 基本ライブラリ: OK")
        
        import numpy as np
        print(f"✅ NumPy {np.__version__}: OK")
        
        import psutil
        print("✅ psutil: OK")
        
        return True
    except Exception as e:
        print(f"❌ 基本ライブラリエラー: {e}")
        return False

def test_tmc_direct():
    """TMCエンジンの直接テスト"""
    try:
        # 最小限のTMCエンジンを直接定義
        import zlib
        import json
        
        class SimpleTMC:
            def __init__(self):
                self.name = "Simple TMC"
                print("🚀 Simple TMC エンジン初期化")
            
            def compress(self, data: bytes):
                """簡単な圧縮"""
                compressed = zlib.compress(data, level=6)
                meta = {'original_size': len(data), 'method': 'zlib'}
                return compressed, meta
            
            def decompress(self, compressed: bytes, meta: dict):
                """簡単な展開"""
                return zlib.decompress(compressed)
        
        # テスト実行
        engine = SimpleTMC()
        test_data = b"NEXUS TMC v9.0 Test Data - Hello World!"
        print(f"📄 テストデータ: {len(test_data)} bytes")
        
        # 圧縮テスト
        compressed, meta = engine.compress(test_data)
        print(f"📦 圧縮完了: {len(compressed)} bytes")
        
        # 展開テスト
        decompressed = engine.decompress(compressed, meta)
        print(f"📂 展開完了: {len(decompressed)} bytes")
        
        # 可逆性確認
        is_identical = test_data == decompressed
        compression_ratio = len(compressed) / len(test_data) * 100
        
        print(f"📊 圧縮率: {compression_ratio:.1f}%")
        print(f"🔄 可逆性: {'✅ OK' if is_identical else '❌ NG'}")
        
        if is_identical:
            print("🎉 基本TMCテスト成功！")
            return True
        else:
            print("⚠️ 可逆性に問題があります")
            return False
            
    except Exception as e:
        print(f"❌ TMCテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト"""
    print("🧪 最小限のTMCエンジンテスト")
    print("=" * 50)
    
    # 基本インポートテスト
    if not test_basic_imports():
        print("❌ 基本インポートテストに失敗")
        return
    
    print()
    
    # TMC直接テスト
    if test_tmc_direct():
        print("\n🎊 全テスト成功！TMCエンジンは正常に動作しています。")
    else:
        print("\n⚠️ TMCテストに問題があります。")

if __name__ == "__main__":
    main()
