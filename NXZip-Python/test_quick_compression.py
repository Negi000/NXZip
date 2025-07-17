#!/usr/bin/env python3
# 🔬 圧縮展開テスト - 小容量版

import time
from nxzip.engine.nexus_experimental import NEXUSExperimentalEngine

print("🔬 圧縮展開テスト開始")
print("=" * 50)

engine = NEXUSExperimentalEngine()

# テストデータ準備
test_data = """これはテストデータです。
圧縮率99.9%を目指します。
TSVファイルのようなテキストデータを想定。
同じパターンが繰り返される。
同じパターンが繰り返される。
同じパターンが繰り返される。
""" * 1000

data = test_data.encode('utf-8')
print(f"📊 テストデータ: {len(data):,} bytes")

# 圧縮テスト
print("🗜️  圧縮実行...")
start_time = time.time()
compressed, stats = engine.compress(data)
compression_time = time.time() - start_time

print(f"✅ 圧縮完了!")
print(f"📊 圧縮率: {stats['compression_ratio']:.4f}%")
print(f"🚀 圧縮速度: {stats['speed_mbps']:.2f} MB/s")
print(f"⏱️  圧縮時間: {compression_time:.3f}秒")
print(f"🏷️  手法: {stats['method']}")
print(f"📦 圧縮サイズ: {len(compressed):,} bytes")

# 展開テスト
print("\n⚡ 展開実行...")
try:
    start_time = time.time()
    decompressed, decomp_stats = engine.decompress(compressed)
    decompression_time = time.time() - start_time
    
    print(f"✅ 展開完了!")
    print(f"📤 展開サイズ: {len(decompressed):,} bytes")
    print(f"⚡ 展開速度: {decomp_stats['speed_mbps']:.2f} MB/s")
    print(f"⏱️  展開時間: {decompression_time:.3f}秒")
    
    # データ検証
    if data == decompressed:
        print(f"🔍 データ検証: ✅ 完全一致!")
        
        # 圧縮率評価
        ratio = stats['compression_ratio']
        if ratio >= 99.9:
            print(f"🎉🏆💎 99.9%圧縮率達成! ({ratio:.4f}%)")
        elif ratio >= 95.0:
            print(f"🎯💎 95%超達成! ({ratio:.4f}%)")
        elif ratio >= 90.0:
            print(f"🔶 90%超達成! ({ratio:.4f}%)")
        else:
            print(f"📊 圧縮率要改善: {ratio:.4f}%")
            
    else:
        print(f"🔍 データ検証: ❌ データ不一致!")
        print(f"   元データ: {len(data)} bytes")
        print(f"   展開データ: {len(decompressed)} bytes")
        
except Exception as e:
    print(f"❌ 展開エラー: {e}")

print(f"\n🔚 テスト完了")
