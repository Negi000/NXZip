#!/usr/bin/env python3
# 🚀 大容量ファイル極限性能テスト

from nxzip.engine.nexus_experimental import NEXUSExperimentalEngine
from nxzip.engine.nexus import NEXUSEngine
import time
import os

print("🚀 NEXUS Experimental - 大容量ファイル極限性能テスト")
print("=" * 70)

# 📊 大容量テストデータ生成
print("📊 大容量テストデータ生成中...")

# 高圧縮率テストデータ（繰り返しパターン）
repetitive_data = b"NEXUS High Performance Test Data Pattern 12345. " * 50000  # ~2.4MB
print(f"✅ 繰り返しデータ: {len(repetitive_data):,} bytes ({len(repetitive_data)/1024/1024:.2f} MB)")

# 中圧縮率テストデータ（混合パターン）
mixed_data = b"".join([
    f"NEXUS-{i:06d}-Mixed-Data-Pattern-Test-{i*123%997}-Performance".encode() 
    for i in range(50000)
])  # ~3.2MB
print(f"✅ 混合データ: {len(mixed_data):,} bytes ({len(mixed_data)/1024/1024:.2f} MB)")

# 低圧縮率テストデータ（ランダム風）
import random
random.seed(42)  # 再現性のため
random_like_data = bytes([random.randint(0, 255) for _ in range(1024*1024*2)])  # 2MB
print(f"✅ ランダム風データ: {len(random_like_data):,} bytes ({len(random_like_data)/1024/1024:.2f} MB)")

test_datasets = [
    ("繰り返しパターン", repetitive_data),
    ("混合パターン", mixed_data),
    ("ランダム風", random_like_data)
]

# エンジン初期化
stable_engine = NEXUSEngine()
experimental_engine = NEXUSExperimentalEngine()

print("\n🔬 大容量性能比較テスト開始")
print("=" * 50)

for name, data in test_datasets:
    print(f"\n📋 テストデータ: {name}")
    print(f"📊 サイズ: {len(data):,} bytes ({len(data)/1024/1024:.2f} MB)")
    
    # v8.0 安定版テスト
    print("\n📦 v8.0 安定版テスト")
    start_time = time.perf_counter()
    compressed_stable, comp_stats_stable = stable_engine.compress(data)
    comp_time_stable = time.perf_counter() - start_time
    
    start_time = time.perf_counter()
    decompressed_stable, decomp_stats_stable = stable_engine.decompress(compressed_stable)
    decomp_time_stable = time.perf_counter() - start_time
    
    stable_valid = decompressed_stable == data
    print(f"🗜️  圧縮率: {comp_stats_stable['compression_ratio']:.2f}%")
    print(f"🚀 圧縮速度: {comp_stats_stable['speed_mbps']:.2f} MB/s")
    print(f"⚡ 展開速度: {decomp_stats_stable['speed_mbps']:.2f} MB/s")
    print(f"🔍 データ検証: {'OK' if stable_valid else 'NG'}")
    
    # v8.1 実験版テスト
    print("\n🧪 v8.1 実験版テスト")
    start_time = time.perf_counter()
    compressed_exp, comp_stats_exp = experimental_engine.compress(data)
    comp_time_exp = time.perf_counter() - start_time
    
    start_time = time.perf_counter()
    decompressed_exp, decomp_stats_exp = experimental_engine.decompress(compressed_exp)
    decomp_time_exp = time.perf_counter() - start_time
    
    exp_valid = decompressed_exp == data
    print(f"🗜️  圧縮率: {comp_stats_exp['compression_ratio']:.2f}%")
    print(f"🚀 圧縮速度: {comp_stats_exp['speed_mbps']:.2f} MB/s")
    print(f"⚡ 展開速度: {decomp_stats_exp['speed_mbps']:.2f} MB/s")
    print(f"🔍 データ検証: {'OK' if exp_valid else 'NG'}")
    
    # 性能改善分析（ゼロ除算対策）
    comp_speed_stable = max(comp_stats_stable['speed_mbps'], 0.01)  # 最小値設定
    comp_speed_exp = comp_stats_exp['speed_mbps']
    decomp_speed_stable = max(decomp_stats_stable['speed_mbps'], 0.01)  # 最小値設定
    decomp_speed_exp = decomp_stats_exp['speed_mbps']
    
    comp_speed_improvement = comp_speed_exp / comp_speed_stable
    decomp_speed_improvement = decomp_speed_exp / decomp_speed_stable
    
    print(f"\n📈 性能改善:")
    print(f"🚀 圧縮速度改善: {comp_speed_improvement:.2f}x ({comp_speed_stable:.2f} → {comp_speed_exp:.2f} MB/s)")
    print(f"⚡ 展開速度改善: {decomp_speed_improvement:.2f}x ({decomp_speed_stable:.2f} → {decomp_speed_exp:.2f} MB/s)")
    
    if decomp_speed_improvement > 1.05:
        print(f"🎉 展開速度 {decomp_speed_improvement:.1f}倍向上!")
    if comp_speed_improvement > 1.05:
        print(f"🚀 圧縮速度も {comp_speed_improvement:.1f}倍向上!")
    
    # 極限性能評価
    if decomp_speed_exp > 500:
        print(f"⚡🚀 極限展開速度達成! {decomp_speed_exp:.1f} MB/s")
    if comp_speed_exp > 300:
        print(f"🗜️🚀 極限圧縮速度達成! {comp_speed_exp:.1f} MB/s")

print("\n🏆 大容量極限性能テスト完了")
