#!/usr/bin/env python3
# ⚡ NEXUS Experimental v8.1 - 大容量展開速度テスト

from nxzip.engine.nexus_experimental import NEXUSExperimentalEngine
from nxzip.engine.nexus import NEXUSEngine
import time

print("⚡ NEXUS Experimental v8.1 vs v8.0 - 展開速度比較テスト")
print("=" * 70)

# テストファイル読み込み
with open("../test-data/large_test.txt", "rb") as f:
    test_data = f.read()

print(f"📊 テストデータ: {len(test_data):,} bytes ({len(test_data)/1024/1024:.2f} MB)")

# 🏷️ v8.0 安定版テスト
print("\n📦 NEXUS v8.0 安定版テスト")
stable_engine = NEXUSEngine()
compressed_stable, comp_stats_stable = stable_engine.compress(test_data)
decompressed_stable, decomp_stats_stable = stable_engine.decompress(compressed_stable)
stable_valid = decompressed_stable == test_data

print(f"🗜️  圧縮率: {comp_stats_stable['compression_ratio']:.2f}%")
print(f"🚀 圧縮速度: {comp_stats_stable['speed_mbps']:.2f} MB/s")
print(f"⚡ 展開速度: {decomp_stats_stable['speed_mbps']:.2f} MB/s")
print(f"🔍 データ検証: {'OK' if stable_valid else 'NG'}")

# 🧪 v8.1 実験版テスト
print("\n🧪 NEXUS Experimental v8.1 実験版テスト")
experimental_engine = NEXUSExperimentalEngine()
compressed_exp, comp_stats_exp = experimental_engine.compress(test_data)
decompressed_exp, decomp_stats_exp = experimental_engine.decompress(compressed_exp)
exp_valid = decompressed_exp == test_data

print(f"🗜️  圧縮率: {comp_stats_exp['compression_ratio']:.2f}%")
print(f"🚀 圧縮速度: {comp_stats_exp['speed_mbps']:.2f} MB/s")
print(f"⚡ 展開速度: {decomp_stats_exp['speed_mbps']:.2f} MB/s")
print(f"🔍 データ検証: {'OK' if exp_valid else 'NG'}")
print(f"🔬 時間精度: {decomp_stats_exp.get('timing_precision', 'standard')}")

# 📈 性能比較
print("\n📈 性能比較サマリ:")
print("─" * 50)
speed_improvement = decomp_stats_exp['speed_mbps'] / decomp_stats_stable['speed_mbps']
print(f"⚡ 展開速度改善倍率: {speed_improvement:.2f}x")
print(f"📊 v8.0 展開速度: {decomp_stats_stable['speed_mbps']:.2f} MB/s")
print(f"🚀 v8.1 展開速度: {decomp_stats_exp['speed_mbps']:.2f} MB/s")

if speed_improvement > 1.0:
    print(f"🎉 実験版が {speed_improvement:.1f}倍高速化に成功！")
else:
    print(f"⚠️  実験版の改善が必要（{speed_improvement:.2f}x）")

# 目標速度チェック
target_speed = 200.0  # MB/s
if decomp_stats_exp['speed_mbps'] >= target_speed:
    print(f"🎯 目標速度 {target_speed} MB/s 達成！")
else:
    remaining = target_speed - decomp_stats_exp['speed_mbps']
    print(f"🎯 目標まで残り {remaining:.1f} MB/s")

print()
