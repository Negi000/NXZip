#!/usr/bin/env python3
# 🚀 NEXUS Experimental v8.1 - 実戦大容量TSVファイルテスト
# 真の性能測定 - 1.6GB出庫実績明細データ

from nxzip.engine.nexus_experimental import NEXUSExperimentalEngine
from nxzip.engine.nexus import NEXUSEngine
import time
import os
import gc

print("🚀 NEXUS Experimental v8.1 - 実戦大容量TSVファイルテスト")
print("=" * 80)
print("🎯 実戦ファイル: 出庫実績明細_202410.tsv (1.6GB)")
print("📊 真の性能測定開始 - 実際のビジネスデータでの極限テスト")
print("=" * 80)

# 実戦ファイルパス
real_file_path = r"C:\Users\241822\Desktop\新しいフォルダー (2)\出庫実績明細_202410.tsv"

# ファイル存在確認
if not os.path.exists(real_file_path):
    print(f"❌ ファイルが見つかりません: {real_file_path}")
    exit(1)

file_size = os.path.getsize(real_file_path)
print(f"📊 ファイルサイズ: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")

# メモリ使用量監視（psutil無しでも動作）
try:
    import psutil
    process = psutil.Process(os.getpid())
    def get_memory_info():
        mem_info = process.memory_info()
        return mem_info.rss / 1024 / 1024  # MB
    memory_available = True
except ImportError:
    def get_memory_info():
        return 0.0  # メモリ監視無効
    memory_available = False
    print("⚠️  psutil未インストール - メモリ監視は無効")

print(f"🧠 初期メモリ使用量: {get_memory_info():.1f} MB")

try:
    # 📖 大容量ファイル読み込み
    print("\n📖 大容量TSVファイル読み込み開始...")
    start_time = time.perf_counter()
    
    # チャンク読み込みでメモリ効率化
    chunk_size = 64 * 1024 * 1024  # 64MB chunks
    file_data = bytearray()
    
    with open(real_file_path, 'rb') as f:
        chunk_count = 0
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            file_data.extend(chunk)
            chunk_count += 1
            if chunk_count % 10 == 0:  # 10チャンクごとに進捗表示
                current_size = len(file_data)
                print(f"   📥 読み込み進捗: {current_size:,} bytes ({current_size/1024/1024:.1f} MB)")
    
    load_time = time.perf_counter() - start_time
    test_data = bytes(file_data)
    print(f"✅ ファイル読み込み完了: {len(test_data):,} bytes")
    print(f"⏱️  読み込み時間: {load_time:.3f}秒")
    print(f"🧠 読み込み後メモリ: {get_memory_info():.1f} MB")
    
    # エンジン初期化
    print("\n🔧 エンジン初期化...")
    stable_engine = NEXUSEngine()
    experimental_engine = NEXUSExperimentalEngine()
    
    # 🏷️ v8.0 安定版実戦テスト
    print("\n" + "="*60)
    print("📦 NEXUS v8.0 安定版 - 実戦大容量テスト")
    print("="*60)
    
    print("🗜️  圧縮実行中...")
    gc.collect()  # メモリクリーンアップ
    mem_before_stable_comp = get_memory_info()
    
    start_time = time.perf_counter()
    compressed_stable, comp_stats_stable = stable_engine.compress(test_data)
    comp_time_stable = time.perf_counter() - start_time
    
    mem_after_stable_comp = get_memory_info()
    
    print(f"✅ v8.0圧縮完了!")
    print(f"📊 圧縮率: {comp_stats_stable['compression_ratio']:.2f}%")
    print(f"🚀 圧縮速度: {comp_stats_stable['speed_mbps']:.2f} MB/s")
    print(f"⏱️  圧縮時間: {comp_time_stable:.3f}秒")
    print(f"📦 圧縮サイズ: {len(compressed_stable):,} bytes")
    print(f"🧠 メモリ使用量: {mem_before_stable_comp:.1f} → {mem_after_stable_comp:.1f} MB")
    
    print("\n⚡ 展開実行中...")
    gc.collect()
    mem_before_stable_decomp = get_memory_info()
    
    start_time = time.perf_counter()
    decompressed_stable, decomp_stats_stable = stable_engine.decompress(compressed_stable)
    decomp_time_stable = time.perf_counter() - start_time
    
    mem_after_stable_decomp = get_memory_info()
    
    stable_valid = decompressed_stable == test_data
    print(f"✅ v8.0展開完了!")
    print(f"⚡ 展開速度: {decomp_stats_stable['speed_mbps']:.2f} MB/s")
    print(f"⏱️  展開時間: {decomp_time_stable:.3f}秒")
    print(f"🔍 データ検証: {'✅ OK' if stable_valid else '❌ NG'}")
    print(f"🧠 メモリ使用量: {mem_before_stable_decomp:.1f} → {mem_after_stable_decomp:.1f} MB")
    
    # メモリクリーンアップ
    del decompressed_stable, compressed_stable
    gc.collect()
    
    # 🧪 v8.1 実験版実戦テスト
    print("\n" + "="*60)
    print("🧪 NEXUS Experimental v8.1 - 実戦大容量テスト")
    print("="*60)
    
    print("🗜️  圧縮実行中...")
    mem_before_exp_comp = get_memory_info()
    
    start_time = time.perf_counter()
    compressed_exp, comp_stats_exp = experimental_engine.compress(test_data)
    comp_time_exp = time.perf_counter() - start_time
    
    mem_after_exp_comp = get_memory_info()
    
    print(f"✅ v8.1圧縮完了!")
    print(f"📊 圧縮率: {comp_stats_exp['compression_ratio']:.2f}%")
    print(f"🚀 圧縮速度: {comp_stats_exp['speed_mbps']:.2f} MB/s")
    print(f"⏱️  圧縮時間: {comp_time_exp:.3f}秒")
    print(f"📦 圧縮サイズ: {len(compressed_exp):,} bytes")
    print(f"🏷️  選択手法: {comp_stats_exp['method']}")
    print(f"🧠 メモリ使用量: {mem_before_exp_comp:.1f} → {mem_after_exp_comp:.1f} MB")
    
    print("\n⚡ 展開実行中...")
    gc.collect()
    mem_before_exp_decomp = get_memory_info()
    
    start_time = time.perf_counter()
    decompressed_exp, decomp_stats_exp = experimental_engine.decompress(compressed_exp)
    decomp_time_exp = time.perf_counter() - start_time
    
    mem_after_exp_decomp = get_memory_info()
    
    exp_valid = decompressed_exp == test_data
    print(f"✅ v8.1展開完了!")
    print(f"⚡ 展開速度: {decomp_stats_exp['speed_mbps']:.2f} MB/s")
    print(f"⏱️  展開時間: {decomp_time_exp:.3f}秒")
    print(f"🔍 データ検証: {'✅ OK' if exp_valid else '❌ NG'}")
    print(f"🎯 性能等級: {decomp_stats_exp['performance_grade']}")
    print(f"🧠 メモリ効率: {decomp_stats_exp['memory_efficiency']}")
    print(f"📊 効率スコア: {decomp_stats_exp['efficiency_score']:.1f}")
    print(f"🧠 メモリ使用量: {mem_before_exp_decomp:.1f} → {mem_after_exp_decomp:.1f} MB")
    
    # 🏆 実戦結果比較
    print("\n" + "="*80)
    print("🏆 実戦大容量TSVファイル - 最終結果比較")
    print("="*80)
    
    # 性能改善分析
    comp_speed_stable = max(comp_stats_stable['speed_mbps'], 0.01)
    comp_speed_exp = comp_stats_exp['speed_mbps']
    decomp_speed_stable = max(decomp_stats_stable['speed_mbps'], 0.01)
    decomp_speed_exp = decomp_stats_exp['speed_mbps']
    
    comp_improvement = comp_speed_exp / comp_speed_stable
    decomp_improvement = decomp_speed_exp / decomp_speed_stable
    
    print(f"📊 ファイル: 出庫実績明細_202410.tsv ({file_size/1024/1024:.1f} MB)")
    print(f"\n🔄 圧縮性能比較:")
    print(f"   📦 v8.0: {comp_stats_stable['compression_ratio']:.2f}% | {comp_speed_stable:.2f} MB/s")
    print(f"   🧪 v8.1: {comp_stats_exp['compression_ratio']:.2f}% | {comp_speed_exp:.2f} MB/s")
    print(f"   📈 改善: {comp_improvement:.2f}x ({comp_speed_stable:.1f} → {comp_speed_exp:.1f} MB/s)")
    
    print(f"\n⚡ 展開性能比較:")
    print(f"   📦 v8.0: {decomp_speed_stable:.2f} MB/s")
    print(f"   🧪 v8.1: {decomp_speed_exp:.2f} MB/s")
    print(f"   📈 改善: {decomp_improvement:.2f}x ({decomp_speed_stable:.1f} → {decomp_speed_exp:.1f} MB/s)")
    
    print(f"\n🎯 実戦目標達成評価:")
    print(f"   🚀 圧縮速度目標(100+ MB/s): {'✅' if comp_speed_exp >= 100 else '🔶' if comp_speed_exp >= 75 else '🟡' if comp_speed_exp >= 50 else '❌'} {comp_speed_exp:.1f} MB/s")
    print(f"   ⚡ 展開速度目標(200+ MB/s): {'✅' if decomp_speed_exp >= 200 else '🔶' if decomp_speed_exp >= 150 else '🟡' if decomp_speed_exp >= 100 else '❌'} {decomp_speed_exp:.1f} MB/s")
    print(f"   📊 圧縮率目標(90%+): {'✅' if comp_stats_exp['compression_ratio'] >= 90 else '🔶' if comp_stats_exp['compression_ratio'] >= 70 else '❌'} {comp_stats_exp['compression_ratio']:.2f}%")
    print(f"   🔐 データ整合性: {'✅ 完璧' if exp_valid and stable_valid else '❌ 問題'}")
    
    # 総合評価
    print(f"\n🏆 実戦総合評価:")
    if decomp_speed_exp >= 200 and comp_speed_exp >= 100 and exp_valid:
        print(f"🎉🏆🚀 実戦完全成功! 1.6GB TSVファイルで目標完全達成!")
        print(f"⚡ 展開速度 {decomp_speed_exp:.0f} MB/s - 極限性能実証!")
        print(f"🚀 圧縮速度 {comp_speed_exp:.0f} MB/s - 高性能実証!")
    elif decomp_speed_exp >= 100 and exp_valid:
        print(f"🎉🚀 実戦成功! 大容量ファイルで高性能達成!")
        print(f"⚡ 展開速度 {decomp_speed_exp:.0f} MB/s")
    elif exp_valid:
        print(f"✅ 実戦基本成功! データ整合性100%")
    else:
        print(f"⚠️ 実戦課題あり - 要検証")
    
    if decomp_improvement > 2.0:
        print(f"🎊 実験版が安定版より {decomp_improvement:.1f}倍の展開速度向上を実現!")
    
    print(f"\n🌟 NEXUS Experimental v8.1 - 実戦大容量ファイル対応完了!")

except Exception as e:
    print(f"❌ 実戦テストエラー: {e}")
    import traceback
    traceback.print_exc()

finally:
    print(f"\n🧠 最終メモリ使用量: {get_memory_info():.1f} MB")
    print("🔚 実戦テスト完了")
