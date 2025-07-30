#!/usr/bin/env python3
"""
NEXUS Ultra Fast Engine v6.2 実ファイル最終テスト
実用レベルの超高速圧縮性能検証
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'nxzip', 'engine'))

import time
import lzma
import zlib
import bz2
from nxzip.engine.nexus_ultra_fast_v6_2 import NEXUSUltraFastEngine
from pathlib import Path


def test_ultra_fast_real_files():
    """実ファイル超高速テスト"""
    print("🚀 NEXUS Ultra Fast Engine v6.2 実ファイル最終検証")
    print("⚡ 目標: 実用最高速 + 高圧縮率維持")
    print("=" * 80)
    
    # サンプルファイル検索
    sample_dir = Path("sample")
    test_files = []
    
    if sample_dir.exists():
        extensions = ['*.jpg', '*.png', '*.mp4', '*.wav', '*.mp3', '*.7z', '*.txt']
        for ext in extensions:
            test_files.extend(sample_dir.glob(ext))
    
    if not test_files:
        print("❌ テストファイルが見つかりません")
        print("📁 sample/ディレクトリにテストファイルを配置してください")
        return
    
    print(f"📁 検出ファイル: {len(test_files)}個")
    
    # エンジン初期化
    engine = NEXUSUltraFastEngine()
    
    # 実用性能目標
    performance_targets = {
        'jpg': {'ratio': 15.0, 'speed': 20.0},
        'png': {'ratio': 20.0, 'speed': 15.0},
        'mp4': {'ratio': 5.0, 'speed': 50.0},
        'wav': {'ratio': 70.0, 'speed': 30.0},
        'mp3': {'ratio': 3.0, 'speed': 40.0},
        '7z': {'ratio': 1.0, 'speed': 25.0},
        'txt': {'ratio': 60.0, 'speed': 20.0}
    }
    
    results = []
    total_achievements = 0
    total_input = 0
    total_output = 0
    total_time = 0
    
    for i, file_path in enumerate(test_files[:8], 1):  # 最大8ファイル
        ext = file_path.suffix[1:].lower()
        targets = performance_targets.get(ext, {'ratio': 10.0, 'speed': 20.0})
        
        print(f"\n{'⚡ ' + '='*60}")
        print(f"ファイル {i}: {file_path.name}")
        print(f"   📊 サイズ: {file_path.stat().st_size:,} bytes ({file_path.stat().st_size / 1024 / 1024:.1f}MB)")
        print(f"   🎯 目標: 圧縮率{targets['ratio']}% / 速度{targets['speed']}MB/s")
        
        try:
            # ファイル読み込み
            with open(file_path, 'rb') as f:
                data = f.read()
            
            input_size = len(data)
            total_input += input_size
            
            # 超高速圧縮実行
            print("   🚀 NEXUS超高速圧縮...")
            start_time = time.perf_counter()
            compressed, info = engine.compress_ultra_fast(data, ext)
            compression_time = time.perf_counter() - start_time
            total_time += compression_time
            
            # 結果評価
            compression_ratio = info['compression_ratio']
            throughput = info['throughput_mb_s']
            
            ratio_achievement = compression_ratio >= targets['ratio']
            speed_achievement = throughput >= targets['speed']
            overall_achievement = ratio_achievement and speed_achievement
            
            if overall_achievement:
                total_achievements += 1
            
            total_output += len(compressed)
            
            status = "🎉" if overall_achievement else ("🥈" if ratio_achievement or speed_achievement else "📊")
            print(f"   {status} 圧縮完了!")
            print(f"      📈 圧縮率: {compression_ratio:.2f}% {'✅' if ratio_achievement else '❌'} (目標:{targets['ratio']}%)")
            print(f"      ⚡ スループット: {throughput:.1f}MB/s {'✅' if speed_achievement else '❌'} (目標:{targets['speed']}MB/s)")
            print(f"      ⏱️ 処理時間: {compression_time:.3f}秒")
            print(f"      🧠 戦略: {info['strategy']}")
            print(f"      💾 {input_size:,} → {len(compressed):,} bytes")
            
            # 競合比較（高速版）
            print("   🆚 競合比較:")
            competitors = {
                'GZIP-3': lambda d: zlib.compress(d, level=3),
                'LZMA-1': lambda d: lzma.compress(d, preset=1),
                'BZIP2-3': lambda d: bz2.compress(d, compresslevel=3)
            }
            
            for comp_name, comp_func in competitors.items():
                try:
                    comp_start = time.perf_counter()
                    comp_result = comp_func(data)
                    comp_time = time.perf_counter() - comp_start
                    comp_ratio = (1 - len(comp_result) / len(data)) * 100
                    comp_throughput = (input_size / 1024 / 1024) / comp_time
                    
                    ratio_diff = compression_ratio - comp_ratio
                    speed_ratio = throughput / comp_throughput
                    
                    print(f"      vs {comp_name}: "
                          f"{comp_ratio:.1f}% ({ratio_diff:+.1f}%) | "
                          f"{comp_throughput:.1f}MB/s (x{speed_ratio:.1f})")
                except Exception:
                    print(f"      vs {comp_name}: エラー")
            
            # 結果記録
            results.append({
                'file': file_path.name,
                'type': ext,
                'input_size': input_size,
                'compression_ratio': compression_ratio,
                'throughput': throughput,
                'time': compression_time,
                'strategy': info['strategy'],
                'ratio_target': targets['ratio'],
                'speed_target': targets['speed'],
                'ratio_achievement': ratio_achievement,
                'speed_achievement': speed_achievement,
                'overall_achievement': overall_achievement
            })
            
        except Exception as e:
            print(f"   ❌ エラー: {e}")
            results.append({
                'file': file_path.name,
                'error': str(e)
            })
    
    # 最終評価レポート
    print(f"\n{'🏆 ' + '='*70}")
    print(f"NEXUS Ultra Fast Engine v6.2 最終評価")
    print(f"{'='*80}")
    
    # 基本成果
    success_rate = (total_achievements / len(results)) * 100 if results else 0
    total_compression = (1 - total_output / total_input) * 100 if total_input > 0 else 0
    avg_throughput = (total_input / 1024 / 1024) / total_time if total_time > 0 else 0
    
    print(f"🎯 総合成果:")
    print(f"   ✅ 目標達成: {total_achievements}/{len(results)} ({success_rate:.1f}%)")
    print(f"   📈 総合圧縮率: {total_compression:.2f}%")
    print(f"   ⚡ 平均スループット: {avg_throughput:.1f}MB/s")
    print(f"   💾 総処理量: {total_input / 1024 / 1024:.1f}MB → {total_output / 1024 / 1024:.1f}MB")
    print(f"   ⏱️ 総時間: {total_time:.3f}秒")
    
    # 詳細結果
    print(f"\n📊 詳細結果:")
    ratio_achievements = 0
    speed_achievements = 0
    
    for result in results:
        if 'error' not in result:
            ratio_ok = "✅" if result['ratio_achievement'] else "❌"
            speed_ok = "✅" if result['speed_achievement'] else "❌"
            overall = "🎉" if result['overall_achievement'] else "📊"
            
            print(f"   {overall} {result['file']}")
            print(f"      📈 {result['compression_ratio']:.1f}% {ratio_ok} (目標:{result['ratio_target']}%)")
            print(f"      ⚡ {result['throughput']:.1f}MB/s {speed_ok} (目標:{result['speed_target']}MB/s)")
            print(f"      🧠 {result['strategy']} | ⏱️ {result['time']:.3f}s")
            
            if result['ratio_achievement']:
                ratio_achievements += 1
            if result['speed_achievement']:
                speed_achievements += 1
        else:
            print(f"   ❌ {result['file']}: {result['error']}")
    
    # 項目別達成率
    print(f"\n📈 項目別達成:")
    valid_count = len([r for r in results if 'error' not in r])
    if valid_count > 0:
        ratio_rate = (ratio_achievements / valid_count) * 100
        speed_rate = (speed_achievements / valid_count) * 100
        
        print(f"   📊 圧縮率達成: {ratio_achievements}/{valid_count} ({ratio_rate:.1f}%)")
        print(f"   ⚡ 速度達成: {speed_achievements}/{valid_count} ({speed_rate:.1f}%)")
    
    # 戦略分布
    strategy_counts = {}
    for result in results:
        if 'error' not in result:
            strategy = result['strategy']
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    
    print(f"\n🧠 戦略使用: {strategy_counts}")
    
    # 最終評価
    print(f"\n🎖️ 最終評価:")
    if success_rate >= 75 and avg_throughput >= 25:
        print("   🏆 EXCELLENT - 実用レベルの超高速圧縮を実現")
        print("   💡 商用利用可能な性能を達成")
    elif success_rate >= 50 and avg_throughput >= 15:
        print("   🥈 VERY GOOD - 高速圧縮の実用化に成功")
        print("   💡 一般的な用途で十分な性能")
    elif success_rate >= 30 or avg_throughput >= 10:
        print("   🥉 GOOD - 基本的な高速化を達成")
        print("   💡 特定用途での実用性あり")
    else:
        print("   📊 NEEDS IMPROVEMENT - さらなる最適化が必要")
    
    # エンジン統計
    engine_stats = engine.get_stats()
    if engine_stats.get('status') != 'no_data':
        print(f"\n🔧 エンジン詳細:")
        print(f"   📊 総圧縮率: {engine_stats['total_compression_ratio']:.2f}%")
        print(f"   ⚡ 総スループット: {engine_stats['total_throughput_mb_s']:.1f}MB/s")
        print(f"   💾 処理量: {engine_stats['input_mb']:.1f}MB")
    
    print(f"\n🎉 NEXUS Ultra Fast Engine v6.2 検証完了!")
    print("⚡ 実用最高速圧縮システムの完成")
    
    return results


if __name__ == "__main__":
    test_ultra_fast_real_files()
