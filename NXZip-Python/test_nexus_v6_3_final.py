#!/usr/bin/env python3
"""
NEXUS Ultimate Engine v6.3 最終検証テスト
画像・動画で40%以上の圧縮率 + 50MB/s以上の高速性能達成確認
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'nxzip', 'engine'))

import time
import lzma
import zlib
import bz2
from nxzip.engine.nexus_ultimate_v6_3 import NEXUSUltimateEngine
from pathlib import Path


def test_nexus_ultimate_final():
    """NEXUS Ultimate Engine v6.3 最終検証"""
    print("🚀 NEXUS Ultimate Engine v6.3 - 最終性能検証")
    print("🎯 目標: 画像・動画40%圧縮 + 全体50MB/s平均速度")
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
        return
    
    print(f"📁 検出ファイル: {len(test_files)}個")
    
    # エンジン初期化
    engine = NEXUSUltimateEngine()
    
    # 最終目標値
    ultimate_targets = {
        'jpg': {'ratio': 40.0, 'speed': 30.0},  # 画像40%圧縮
        'png': {'ratio': 45.0, 'speed': 25.0},  # PNG 45%圧縮
        'mp4': {'ratio': 40.0, 'speed': 50.0},  # 動画40%圧縮
        'wav': {'ratio': 90.0, 'speed': 70.0},  # 音声90%圧縮
        'mp3': {'ratio': 20.0, 'speed': 60.0},  # MP3 20%圧縮
        '7z': {'ratio': 5.0, 'speed': 40.0},    # アーカイブ5%圧縮
        'txt': {'ratio': 80.0, 'speed': 50.0}   # テキスト80%圧縮
    }
    
    results = []
    total_ultimate_achievements = 0
    total_input = 0
    total_output = 0
    total_time = 0
    
    for i, file_path in enumerate(test_files[:6], 1):  # 最大6ファイル
        ext = file_path.suffix[1:].lower()
        targets = ultimate_targets.get(ext, {'ratio': 30.0, 'speed': 40.0})
        
        print(f"\n{'🎯 ' + '='*60}")
        print(f"最終検証 {i}: {file_path.name}")
        print(f"   📊 サイズ: {file_path.stat().st_size:,} bytes ({file_path.stat().st_size / 1024 / 1024:.1f}MB)")
        print(f"   🎯 究極目標: 圧縮率{targets['ratio']}% / 速度{targets['speed']}MB/s")
        
        try:
            # ファイル読み込み
            with open(file_path, 'rb') as f:
                data = f.read()
            
            input_size = len(data)
            total_input += input_size
            
            # NEXUS究極圧縮実行
            print("   🚀 NEXUS Ultimate v6.3 実行...")
            start_time = time.perf_counter()
            compressed, info = engine.compress_nexus_ultimate(data, ext)
            compression_time = time.perf_counter() - start_time
            total_time += compression_time
            
            # 結果評価
            compression_ratio = info['compression_ratio']
            throughput = info['throughput_mb_s']
            
            ratio_ultimate = compression_ratio >= targets['ratio']
            speed_ultimate = throughput >= targets['speed']
            ultimate_achievement = ratio_ultimate and speed_ultimate
            
            if ultimate_achievement:
                total_ultimate_achievements += 1
            
            total_output += len(compressed)
            
            # 結果表示
            if ultimate_achievement:
                status = "🏆"
                message = "究極目標達成!"
            elif ratio_ultimate or speed_ultimate:
                status = "🥈"
                message = "部分達成"
            else:
                status = "📊"
                message = "要改善"
            
            print(f"   {status} {message}")
            print(f"      📈 圧縮率: {compression_ratio:.2f}% {'🎉' if ratio_ultimate else '❌'} (目標:{targets['ratio']}%)")
            print(f"      ⚡ スループット: {throughput:.1f}MB/s {'🎉' if speed_ultimate else '❌'} (目標:{targets['speed']}MB/s)")
            print(f"      ⏱️ 処理時間: {compression_time:.3f}秒")
            print(f"      🧠 NEXUS戦略: {info['strategy']}")
            print(f"      💾 {input_size:,} → {len(compressed):,} bytes")
            
            # NEXUS解析詳細
            if 'nexus_analysis' in info:
                na = info['nexus_analysis']
                print(f"      🔬 NEXUS解析:")
                print(f"         🎯 圧縮倍率: {na['compression_multiplier']:.1f}x")
                print(f"         ⚡ 速度ブースト: {na['speed_boost']:.1f}x")
                print(f"         🛠️ 処理モード: {na['processing_mode']}")
            
            # 競合アルゴリズム比較
            print("   🆚 競合との性能比較:")
            competitors = {
                'LZMA-6': lambda d: lzma.compress(d, preset=6),
                'GZIP-6': lambda d: zlib.compress(d, level=6),
                'BZIP2-6': lambda d: bz2.compress(d, compresslevel=6)
            }
            
            nexus_wins = 0
            nexus_speed_wins = 0
            
            for comp_name, comp_func in competitors.items():
                try:
                    comp_start = time.perf_counter()
                    comp_result = comp_func(data)
                    comp_time = time.perf_counter() - comp_start
                    comp_ratio = (1 - len(comp_result) / len(data)) * 100
                    comp_throughput = (input_size / 1024 / 1024) / comp_time
                    
                    ratio_advantage = compression_ratio - comp_ratio
                    speed_advantage = throughput / comp_throughput
                    
                    if ratio_advantage > 0:
                        nexus_wins += 1
                    if speed_advantage > 1.0:
                        nexus_speed_wins += 1
                    
                    print(f"      🥊 vs {comp_name}: "
                          f"圧縮{comp_ratio:.1f}% ({'🏆' if ratio_advantage > 0 else '📊'}{ratio_advantage:+.1f}%) | "
                          f"速度{comp_throughput:.1f}MB/s ({'🏆' if speed_advantage > 1 else '📊'}x{speed_advantage:.1f})")
                    
                except Exception:
                    print(f"      🥊 vs {comp_name}: エラー")
            
            print(f"      🏆 NEXUS優位: 圧縮{nexus_wins}/3 | 速度{nexus_speed_wins}/3")
            
            # 結果記録
            results.append({
                'file': file_path.name,
                'type': ext,
                'input_size': input_size,
                'compression_ratio': compression_ratio,
                'throughput': throughput,
                'time': compression_time,
                'strategy': info['strategy'],
                'targets': targets,
                'ultimate_achievement': ultimate_achievement,
                'ratio_ultimate': ratio_ultimate,
                'speed_ultimate': speed_ultimate,
                'nexus_wins': nexus_wins,
                'nexus_speed_wins': nexus_speed_wins
            })
            
        except Exception as e:
            print(f"   ❌ エラー: {e}")
            results.append({
                'file': file_path.name,
                'error': str(e)
            })
    
    # 最終評価レポート
    print(f"\n{'🏆 ' + '='*70}")
    print(f"NEXUS Ultimate Engine v6.3 最終判定")
    print(f"{'='*80}")
    
    # 総合成果
    success_rate = (total_ultimate_achievements / len(results)) * 100 if results else 0
    total_compression = (1 - total_output / total_input) * 100 if total_input > 0 else 0
    avg_throughput = (total_input / 1024 / 1024) / total_time if total_time > 0 else 0
    
    print(f"🎯 究極目標達成状況:")
    print(f"   🏆 完全達成: {total_ultimate_achievements}/{len(results)} ({success_rate:.1f}%)")
    print(f"   📈 総合圧縮率: {total_compression:.2f}%")
    print(f"   ⚡ 平均スループット: {avg_throughput:.1f}MB/s")
    print(f"   💾 総処理: {total_input / 1024 / 1024:.1f}MB → {total_output / 1024 / 1024:.1f}MB")
    print(f"   ⏱️ 総時間: {total_time:.3f}秒")
    
    # 目標別達成分析
    ratio_ultimates = sum(1 for r in results if 'error' not in r and r.get('ratio_ultimate', False))
    speed_ultimates = sum(1 for r in results if 'error' not in r and r.get('speed_ultimate', False))
    valid_count = len([r for r in results if 'error' not in r])
    
    if valid_count > 0:
        print(f"\n🎯 項目別達成:")
        print(f"   📊 圧縮率究極: {ratio_ultimates}/{valid_count} ({ratio_ultimates/valid_count*100:.1f}%)")
        print(f"   ⚡ 速度究極: {speed_ultimates}/{valid_count} ({speed_ultimates/valid_count*100:.1f}%)")
    
    # 詳細結果一覧
    print(f"\n📊 詳細達成状況:")
    total_nexus_wins = 0
    total_speed_wins = 0
    
    for result in results:
        if 'error' not in result:
            status = "🏆" if result['ultimate_achievement'] else ("🥈" if result.get('ratio_ultimate') or result.get('speed_ultimate') else "📊")
            
            print(f"   {status} {result['file']}")
            print(f"      📈 {result['compression_ratio']:.1f}% / {result['targets']['ratio']}% | "
                  f"⚡ {result['throughput']:.1f}MB/s / {result['targets']['speed']}MB/s")
            print(f"      🏆 競合優位: {result.get('nexus_wins', 0)}/3圧縮 | {result.get('nexus_speed_wins', 0)}/3速度")
            
            total_nexus_wins += result.get('nexus_wins', 0)
            total_speed_wins += result.get('nexus_speed_wins', 0)
        else:
            print(f"   ❌ {result['file']}: {result['error']}")
    
    # 競合比較総合
    max_wins = valid_count * 3
    if max_wins > 0:
        print(f"\n🥊 競合比較総合:")
        print(f"   📊 圧縮優位率: {total_nexus_wins}/{max_wins} ({total_nexus_wins/max_wins*100:.1f}%)")
        print(f"   ⚡ 速度優位率: {total_speed_wins}/{max_wins} ({total_speed_wins/max_wins*100:.1f}%)")
    
    # 戦略効果分析
    strategy_stats = {}
    for result in results:
        if 'error' not in result:
            strategy = result['strategy']
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {'count': 0, 'avg_ratio': 0, 'avg_speed': 0}
            
            stats = strategy_stats[strategy]
            stats['count'] += 1
            stats['avg_ratio'] += result['compression_ratio']
            stats['avg_speed'] += result['throughput']
    
    print(f"\n🧠 NEXUS戦略効果:")
    for strategy, stats in strategy_stats.items():
        avg_ratio = stats['avg_ratio'] / stats['count']
        avg_speed = stats['avg_speed'] / stats['count']
        print(f"   {strategy}: {avg_ratio:.1f}%圧縮 | {avg_speed:.1f}MB/s (使用{stats['count']}回)")
    
    # 最終判定
    print(f"\n🎖️ NEXUS Ultimate Engine v6.3 最終判定:")
    
    if success_rate >= 80 and avg_throughput >= 50 and total_compression >= 40:
        verdict = "🏆 PERFECT - 理論目標を完全達成"
        detail = "画像・動画40%圧縮 + 50MB/s高速処理の同時実現"
    elif success_rate >= 60 and avg_throughput >= 30 and total_compression >= 30:
        verdict = "🥇 EXCELLENT - 優秀な性能を実現"
        detail = "実用レベルの高圧縮・高速処理を達成"
    elif success_rate >= 40 and avg_throughput >= 20:
        verdict = "🥈 VERY GOOD - 良好な改善を達成"
        detail = "従来手法を大きく上回る性能"
    elif avg_throughput >= 15 or total_compression >= 20:
        verdict = "🥉 GOOD - 基本性能を確保"
        detail = "実用可能なレベルに到達"
    else:
        verdict = "📊 NEEDS IMPROVEMENT - 更なる最適化要"
        detail = "目標達成にはさらなる改良が必要"
    
    print(f"   {verdict}")
    print(f"   💡 {detail}")
    
    # エンジン統計
    engine_stats = engine.get_nexus_stats()
    if engine_stats.get('status') != 'no_data':
        print(f"\n🔧 NEXUS Engine統計:")
        print(f"   📊 総圧縮率: {engine_stats['total_compression_ratio']:.2f}%")
        print(f"   ⚡ 総スループット: {engine_stats['average_throughput_mb_s']:.1f}MB/s")
        print(f"   🧠 戦略分布: {engine_stats['strategy_distribution']}")
    
    print(f"\n🎉 NEXUS Ultimate Engine v6.3 最終検証完了!")
    print("🚀 次世代圧縮技術の実用化検証終了")
    
    return results


if __name__ == "__main__":
    test_nexus_ultimate_final()
