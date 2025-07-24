#!/usr/bin/env python3
"""
NEXUS無限ループ修正後のテスト
"""

import time
import os
from nexus_compression_engine import NEXUSCompressor, MLCompressionConfig

def test_nexus_performance():
    """NEXUS性能テスト"""
    print("🧠 NEXUS無限ループ修正テスト")
    print("=" * 50)
    
    # テストデータ作成
    test_cases = [
        (b"Short text", "超短データ"),
        (b"Medium length test data " * 50, "中程度データ"),
        (b"Longer test data for NEXUS compression analysis " * 200, "長データ"),
    ]
    
    config = MLCompressionConfig(verbose=True)
    compressor = NEXUSCompressor(config)
    
    for test_data, description in test_cases:
        print(f"\n📊 {description} テスト ({len(test_data)} bytes)")
        print("-" * 30)
        
        # タイムアウト監視
        start_time = time.time()
        timeout_limit = 30.0  # 30秒でタイムアウト
        
        try:
            compressed = compressor.compress(test_data)
            compress_time = time.time() - start_time
            
            print(f"✅ 圧縮成功: {len(compressed)} bytes")
            print(f"⏱️ 処理時間: {compress_time:.3f}秒")
            print(f"📈 圧縮率: {len(compressed)/len(test_data)*100:.1f}%")
            
            if compress_time < 5.0:
                print("🎉 高速処理！無限ループなし！")
            elif compress_time < 15.0:
                print("✅ 正常な処理速度")
            else:
                print("⚠️ 処理が重い（要最適化）")
                
            # 展開テスト
            try:
                start_time = time.time()
                decompressed = compressor.decompress(compressed)
                decompress_time = time.time() - start_time
                
                print(f"🔄 展開時間: {decompress_time:.3f}秒")
                print(f"🔍 データ一致: {test_data == decompressed}")
            except Exception as e:
                print(f"⚠️ 展開エラー: {e}")
                
        except Exception as e:
            compress_time = time.time() - start_time
            print(f"❌ 圧縮エラー: {e}")
            print(f"⏱️ エラー発生時間: {compress_time:.3f}秒")
            
            if compress_time > timeout_limit:
                print("🚨 無限ループの可能性！")
            
    print("\n" + "=" * 50)
    print("🎯 NEXUS無限ループ修正テスト完了")

if __name__ == "__main__":
    test_nexus_performance()
