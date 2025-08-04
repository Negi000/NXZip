#!/usr/bin/env python3
"""
NXZip TMC v9.1 簡単可逆性テスト
"""

import sys
from pathlib import Path
import hashlib
import time

# NXZip-Pythonパスを追加
sys.path.insert(0, str(Path(__file__).parent / "NXZip-Python"))

try:
    from nxzip.engine.nexus_tmc_v91_modular import NEXUSTMCEngineV91
    print("✅ NEXUSTMCEngineV91 インポート成功")
except ImportError as e:
    print(f"❌ NEXUSTMCEngineV91 インポートエラー: {e}")
    sys.exit(1)

def test_simple_reversibility():
    """シンプルな可逆性テスト"""
    print("🚀 NXZip TMC v9.1 シンプル可逆性テスト")
    
    # テストデータ
    test_data = "Hello World! " * 1000 + "TMC Test " * 500
    original_data = test_data.encode('utf-8')
    original_hash = hashlib.sha256(original_data).hexdigest()
    
    print(f"📊 元データ: {len(original_data)} bytes")
    print(f"📊 元Hash: {original_hash[:16]}...")
    
    # エンジン初期化
    engine = NEXUSTMCEngineV91(lightweight_mode=True)
    
    # 圧縮
    print("\n🗜️ 圧縮実行中...")
    start_time = time.time()
    compressed_data, compression_info = engine.compress(original_data)
    compression_time = time.time() - start_time
    
    print(f"📦 圧縮完了: {len(compressed_data)} bytes")
    print(f"⏱️ 圧縮時間: {compression_time:.3f}秒")
    print(f"🔄 変換適用: {compression_info.get('transform_applied', False)}")
    
    # 解凍
    print("\n🔄 解凍実行中...")
    start_time = time.time()
    decompressed_data = engine.decompress(compressed_data, compression_info)
    decompression_time = time.time() - start_time
    
    print(f"📦 解凍完了: {len(decompressed_data)} bytes")
    print(f"⏱️ 解凍時間: {decompression_time:.3f}秒")
    
    # 可逆性確認
    decompressed_hash = hashlib.sha256(decompressed_data).hexdigest()
    print(f"📊 解凍Hash: {decompressed_hash[:16]}...")
    
    if original_hash == decompressed_hash:
        print("✅ 可逆性テスト成功!")
        return True
    else:
        print("❌ 可逆性テスト失敗!")
        print(f"   サイズ差: {len(original_data)} -> {len(decompressed_data)}")
        return False

if __name__ == "__main__":
    test_simple_reversibility()
