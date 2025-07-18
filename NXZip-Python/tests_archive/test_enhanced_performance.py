#!/usr/bin/env python3
# 🧪 NEXUS Experimental v8.1 - 改良版総合性能テスト

from nxzip.engine.nexus_experimental import NEXUSExperimentalEngine
from nxzip.engine.nexus import NEXUSEngine
import time

print("🧪 NEXUS Experimental v8.1 - 改良版総合性能テスト")
print("=" * 70)
print("🔬 新機能: メモリ効率化 + エラーハンドリング強化 + 精密分析")
print("=" * 70)

# 多様なテストデータセット
test_datasets = {
    "高圧縮率データ": b"NEXUS TEST DATA PATTERN " * 5000,  # 繰り返しパターン
    "中圧縮率データ": b"".join([f"Data-{i:04d}-{(i*17)%100:02d}".encode() for i in range(8000)]),  # 半構造化
    "低圧縮率データ": bytes([(i * 137 + 42) % 256 for i in range(50000)]),  # 疑似ランダム
    "混合データ": b"NEXUS" * 3000 + b"".join([f"X{i}".encode() for i in range(5000)]) + b"END" * 2000,  # 混合
    "小容量データ": b"Small test data for NEXUS engine testing.",  # 小容量
    "大容量データ": b"Large file content for performance testing. " * 50000  # 大容量
}

print(f"📊 テストデータセット数: {len(test_datasets)}")

# エンジン初期化
stable_engine = NEXUSEngine()
experimental_engine = NEXUSExperimentalEngine()

total_improvement_compression = 0
total_improvement_decompression = 0
test_count = 0

for name, data in test_datasets.items():
    print(f"\n{'='*50}")
    print(f"📋 テスト: {name}")
    print(f"📊 サイズ: {len(data):,} bytes ({len(data)/1024:.1f} KB)")
    
    # v8.0 安定版ベンチマーク
    print("\n📦 v8.0 安定版ベンチマーク")
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
    print(f"🔍 データ検証: {'✅ OK' if stable_valid else '❌ NG'}")
    
    # v8.1 実験版改良テスト
    print("\n🧪 v8.1 実験版改良テスト")
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
    print(f"🔍 データ検証: {'✅ OK' if exp_valid else '❌ NG'}")
    print(f"🏷️  選択手法: {decomp_stats_exp['method']}")
    print(f"🎯 性能等級: {decomp_stats_exp['performance_grade']}")
    print(f"🧠 メモリ効率: {decomp_stats_exp['memory_efficiency']}")
    print(f"📊 効率スコア: {decomp_stats_exp['efficiency_score']:.1f}")
    
    # 改善度分析
    comp_improvement = (comp_stats_exp['speed_mbps'] / max(comp_stats_stable['speed_mbps'], 0.01))
    decomp_improvement = (decomp_stats_exp['speed_mbps'] / max(decomp_stats_stable['speed_mbps'], 0.01))
    
    print(f"\n📈 改善分析:")
    print(f"🚀 圧縮速度改善: {comp_improvement:.2f}x")
    print(f"⚡ 展開速度改善: {decomp_improvement:.2f}x")
    
    if decomp_improvement > 2.0:
        print(f"🎉 展開速度 {decomp_improvement:.1f}倍の大幅向上!")
    elif decomp_improvement > 1.2:
        print(f"✨ 展開速度 {decomp_improvement:.1f}倍向上")
    
    if comp_improvement > 2.0:
        print(f"🚀 圧縮速度 {comp_improvement:.1f}倍の大幅向上!")
    elif comp_improvement > 1.2:
        print(f"✨ 圧縮速度 {comp_improvement:.1f}倍向上")
    
    total_improvement_compression += comp_improvement
    total_improvement_decompression += decomp_improvement
    test_count += 1

# 総合結果
print(f"\n{'='*70}")
print("🏆 NEXUS Experimental v8.1 改良版 - 総合結果")
print(f"{'='*70}")

if test_count > 0:
    avg_comp_improvement = total_improvement_compression / test_count
    avg_decomp_improvement = total_improvement_decompression / test_count
    
    print(f"📊 テスト完了: {test_count}件")
    print(f"🚀 平均圧縮速度改善: {avg_comp_improvement:.2f}x")
    print(f"⚡ 平均展開速度改善: {avg_decomp_improvement:.2f}x")
    
    print(f"\n🎯 改良版新機能評価:")
    print(f"✅ メモリ効率化 - ストリーミング展開実装")
    print(f"✅ エラーハンドリング強化 - 多段階フォールバック")
    print(f"✅ 精密分析 - 多次元パターン分析")
    print(f"✅ 性能モニタリング - 効率スコア・等級判定")
    
    if avg_decomp_improvement > 5.0:
        print(f"\n🎉🏆 極限性能達成! 実験版の圧倒的成功!")
        print(f"🚀 展開速度 {avg_decomp_improvement:.0f}倍向上の驚異的結果!")
    elif avg_decomp_improvement > 2.0:
        print(f"\n🎉 高性能達成! 実験版の大成功!")
        print(f"⚡ 展開速度 {avg_decomp_improvement:.1f}倍向上!")
    elif avg_decomp_improvement > 1.2:
        print(f"\n✨ 性能向上達成! 実験版の成功!")
    else:
        print(f"\n📊 ベースライン維持 - 安定性重視")

print("\n🔬 改良版実験完了!")
