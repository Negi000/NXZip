#!/usr/bin/env python3
# ⚡ NEXUS Experimental v8.1 - 展開速度最適化テスト

from nxzip.engine.nexus_experimental import NEXUSExperimentalEngine
import time

print("⚡ NEXUS Experimental v8.1 - 展開速度最適化テスト")
print("=" * 60)

# テストファイル読み込み
with open("../test-data/large_test.txt", "rb") as f:
    test_data = f.read()

print(f"📊 元データ: {len(test_data):,} bytes ({len(test_data)/1024/1024:.2f} MB)")

# エンジン初期化
engine = NEXUSExperimentalEngine()
print(f"🎯 版本: {engine.version}")

# 圧縮テスト
print("\n🗜️  圧縮実行中...")
compressed_data, comp_stats = engine.compress(test_data)
print(f"✅ 圧縮完了: {comp_stats['compression_ratio']:.2f}%")
print(f"🚀 圧縮速度: {comp_stats['speed_mbps']:.2f} MB/s")

# 🚀 実験版展開テスト
print("\n⚡ 実験版展開テスト実行...")
decompressed_data, decomp_stats = engine.decompress(compressed_data)
is_valid = decompressed_data == test_data

print(f"✅ 展開完了: {len(decompressed_data):,} bytes")
print(f"⚡ 展開速度: {decomp_stats['speed_mbps']:.2f} MB/s ← 📊")
print(f"🔍 データ検証: {'OK' if is_valid else 'NG'}")
print(f"🏷️  手法: {decomp_stats['method']}")
print(f"🔬 時間精度: {decomp_stats.get('timing_precision', 'standard')}")

# 実験版性能サマリ
print("\n📈 実験版性能サマリ:")
print(f"🗜️  圧縮率: {comp_stats['compression_ratio']:.2f}%")
print(f"🚀 圧縮速度: {comp_stats['speed_mbps']:.2f} MB/s")
print(f"⚡ 展開速度: {decomp_stats['speed_mbps']:.2f} MB/s")
print(f"🎯 版本: {engine.version}")
print()
