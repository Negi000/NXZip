#!/usr/bin/env python3
# 🔬 zlib_ultra_compress 専用テスト

import time
from nxzip.engine.nexus_experimental import NEXUSExperimentalEngine

print("🔬 zlib_ultra_compress 専用テスト")
print("=" * 50)

engine = NEXUSExperimentalEngine()

# テストデータ準備（中サイズで zlib_ultra_compress を誘発）
test_data = """これはTSVファイルのテストデータです。
カラム1	カラム2	カラム3	カラム4
データ1	値1	100	テスト
データ2	値2	200	テスト
データ3	値3	300	テスト
""" * 5000  # 約8MB以上になるように

data = test_data.encode('utf-8')
print(f"📊 テストデータ: {len(data):,} bytes ({len(data)/(1024*1024):.1f} MB)")

# zlib_ultra_compress を強制実行
print("🗜️  zlib_ultra_compress 実行...")
start_time = time.time()
try:
    compressed = engine._zlib_ultra_compress(data)
    compression_time = time.time() - start_time
    
    print(f"✅ 圧縮完了!")
    print(f"📦 圧縮サイズ: {len(compressed):,} bytes")
    print(f"📊 圧縮率: {(1 - len(compressed)/len(data)) * 100:.4f}%")
    print(f"⏱️  圧縮時間: {compression_time:.3f}秒")
    print(f"🏷️  ヘッダー: {compressed[:8]}")
    
    # 展開テスト
    print("\n⚡ 展開実行...")
    start_time = time.time()
    try:
        decompressed = engine._zlib_ultra_decompress_optimized(compressed)
        decompression_time = time.time() - start_time
        
        print(f"✅ 展開完了!")
        print(f"📤 展開サイズ: {len(decompressed):,} bytes")
        print(f"⏱️  展開時間: {decompression_time:.3f}秒")
        
        # データ検証
        if data == decompressed:
            print(f"🔍 データ検証: ✅ 完全一致!")
        else:
            print(f"🔍 データ検証: ❌ データ不一致!")
            print(f"   元データ: {len(data)} bytes")
            print(f"   展開データ: {len(decompressed)} bytes")
            
    except Exception as e:
        print(f"❌ 展開エラー: {e}")

except Exception as e:
    print(f"❌ 圧縮エラー: {e}")

print(f"\n🔚 テスト完了")
