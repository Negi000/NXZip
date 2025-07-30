#!/usr/bin/env python3
"""
NEXUS Ultra Lightweight Engine v6.4 緊急検証テスト
v6.3の深刻な性能問題(0.8MB/s)を解決する緊急テスト

緊急目標:
- 平均速度: 20MB/s以上 (v6.3の25倍改善)
- 圧縮率: 実用レベル維持
- 全戦略が5MB/s以上で動作
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'nxzip', 'engine'))

import time
import lzma
import zlib
import bz2
from nxzip.engine.nexus_ultra_light_v6_4 import NEXUSUltraLightEngine
from pathlib import Path


def test_nexus_emergency_optimization():
    """v6.4 緊急性能テスト"""
    print("🚨 NEXUS Ultra Light Engine v6.4 - 緊急性能検証")
    print("🎯 緊急目標: v6.3の0.8MB/s → 20MB/s以上 (25倍改善)")
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
    engine = NEXUSUltraLightEngine()
    
    # 緊急性能目標
    emergency_targets = {
        'jpg': {'min_speed': 15.0, 'acceptable_ratio': 10.0},  # 最低15MB/s
        'png': {'min_speed': 10.0, 'acceptable_ratio': 5.0},   # 最低10MB/s
        'mp4': {'min_speed': 25.0, 'acceptable_ratio': 15.0},  # 最低25MB/s
        'wav': {'min_speed': 50.0, 'acceptable_ratio': 70.0},  # 最低50MB/s
        'mp3': {'min_speed': 30.0, 'acceptable_ratio': 10.0},  # 最低30MB/s
        '7z': {'min_speed': 20.0, 'acceptable_ratio': 3.0},    # 最低20MB/s
        'txt': {'min_speed': 40.0, 'acceptable_ratio': 50.0}   # 最低40MB/s
    }
    
    results = []
    emergency_successes = 0
    total_input = 0
    total_output = 0
    total_time = 0
    speed_improvements = []
    
    # v6.3の悲惨な結果 (参考値)
    v6_3_speeds = {
        'jpg': 1.0, 'png': 0.9, 'mp4': 0.7, 'wav': 53.5, 'mp3': 1.9, '7z': 1.5, 'txt': 2.0
    }
    
    for i, file_path in enumerate(test_files[:6], 1):  # 最大6ファイル
        ext = file_path.suffix[1:].lower()
        targets = emergency_targets.get(ext, {'min_speed': 15.0, 'acceptable_ratio': 10.0})
        v6_3_speed = v6_3_speeds.get(ext, 1.0)
        
        print(f"\n{'🚨 ' + '='*60}")
        print(f"緊急検証 {i}: {file_path.name}")
        print(f"   📊 サイズ: {file_path.stat().st_size:,} bytes ({file_path.stat().st_size / 1024 / 1024:.1f}MB)")
        print(f"   🚨 最低目標: 速度{targets['min_speed']}MB/s | 圧縮率{targets['acceptable_ratio']}%")
        print(f"   📉 v6.3実績: {v6_3_speed:.1f}MB/s (改善必要: {targets['min_speed']/v6_3_speed:.1f}倍)")
        
        try:
            # ファイル読み込み
            with open(file_path, 'rb') as f:
                data = f.read()
            
            input_size = len(data)
            total_input += input_size
            
            # NEXUS Ultra Light圧縮実行
            print("   🚀 NEXUS Ultra Light v6.4 実行...")
            start_time = time.perf_counter()
            compressed, info = engine.compress_nexus_ultra_light(data, ext)
            compression_time = time.perf_counter() - start_time
            total_time += compression_time
            
            # 結果評価
            compression_ratio = info['compression_ratio']
            throughput = info['throughput_mb_s']
            
            speed_emergency = throughput >= targets['min_speed']
            ratio_acceptable = compression_ratio >= targets['acceptable_ratio']
            emergency_success = speed_emergency and ratio_acceptable
            
            if emergency_success:
                emergency_successes += 1
            
            total_output += len(compressed)
            
            # 改善倍率計算
            improvement_ratio = throughput / v6_3_speed if v6_3_speed > 0 else 0
            speed_improvements.append(improvement_ratio)
            
            # 結果表示
            if emergency_success:
                status = "🟢"
                message = "緊急目標達成!"
            elif speed_emergency:
                status = "🟡"
                message = "速度達成"
            elif ratio_acceptable:
                status = "🟠"
                message = "圧縮達成"
            else:
                status = "🔴"
                message = "要再最適化"
            
            print(f"   {status} {message}")
            print(f"      ⚡ スループット: {throughput:.1f}MB/s {'🟢' if speed_emergency else '🔴'} (最低:{targets['min_speed']}MB/s)")
            print(f"      📈 圧縮率: {compression_ratio:.2f}% {'🟢' if ratio_acceptable else '🔴'} (最低:{targets['acceptable_ratio']}%)")
            print(f"      🚀 v6.3からの改善: {improvement_ratio:.1f}倍 {'🎉' if improvement_ratio >= 10 else '📊'}")
            print(f"      ⏱️ 処理時間: {compression_time:.3f}秒")
            print(f"      🧠 戦略: {info['strategy']}")
            print(f"      💾 {input_size:,} → {len(compressed):,} bytes")
            
            # NEXUS解析詳細
            if 'nexus_analysis' in info:
                na = info['nexus_analysis']
                print(f"      🔬 Ultra分析:")
                print(f"         ⚡ 分析時間: {na['analysis_time']*1000:.1f}ms")
                print(f"         🧠 処理モード: {na['processing_mode']}")
                print(f"         📊 圧縮倍率: {na['compression_multiplier']:.1f}x")
            
            # 速度比較
            print(f"   📊 速度比較:")
            print(f"      v6.3: {v6_3_speed:.1f}MB/s → v6.4: {throughput:.1f}MB/s ({improvement_ratio:.1f}倍改善)")
            
            # 結果記録
            results.append({
                'file': file_path.name,
                'type': ext,
                'input_size': input_size,
                'compression_ratio': compression_ratio,
                'throughput': throughput,
                'time': compression_time,
                'strategy': info['strategy'],
                'emergency_success': emergency_success,
                'speed_emergency': speed_emergency,
                'ratio_acceptable': ratio_acceptable,
                'improvement_ratio': improvement_ratio,
                'v6_3_speed': v6_3_speed
            })
            
        except Exception as e:
            print(f"   ❌ エラー: {e}")
            results.append({
                'file': file_path.name,
                'error': str(e)
            })
    
    # 緊急評価レポート
    print(f"\n{'🚨 ' + '='*70}")
    print(f"NEXUS Ultra Light Engine v6.4 緊急判定")
    print(f"{'='*80}")
    
    # 総合成果
    success_rate = (emergency_successes / len(results)) * 100 if results else 0
    total_compression = (1 - total_output / total_input) * 100 if total_input > 0 else 0
    avg_throughput = (total_input / 1024 / 1024) / total_time if total_time > 0 else 0
    avg_improvement = sum(speed_improvements) / len(speed_improvements) if speed_improvements else 0
    
    print(f"🚨 緊急目標達成状況:")
    print(f"   🟢 完全達成: {emergency_successes}/{len(results)} ({success_rate:.1f}%)")
    print(f"   ⚡ 平均スループット: {avg_throughput:.1f}MB/s")
    print(f"   📈 総合圧縮率: {total_compression:.2f}%")
    print(f"   🚀 平均改善率: {avg_improvement:.1f}倍")
    print(f"   💾 総処理: {total_input / 1024 / 1024:.1f}MB → {total_output / 1024 / 1024:.1f}MB")
    print(f"   ⏱️ 総時間: {total_time:.3f}秒")
    
    # v6.3との詳細比較
    v6_3_avg_speed = 0.8  # v6.3の悲惨な実績
    overall_improvement = avg_throughput / v6_3_avg_speed if v6_3_avg_speed > 0 else 0
    
    print(f"\n🔄 v6.3からの全体改善:")
    print(f"   📉 v6.3平均: {v6_3_avg_speed:.1f}MB/s")
    print(f"   📈 v6.4平均: {avg_throughput:.1f}MB/s")
    print(f"   🚀 総合改善: {overall_improvement:.1f}倍 {'🎉' if overall_improvement >= 20 else '📊'}")
    
    # 項目別達成分析
    speed_successes = sum(1 for r in results if 'error' not in r and r.get('speed_emergency', False))
    ratio_successes = sum(1 for r in results if 'error' not in r and r.get('ratio_acceptable', False))
    valid_count = len([r for r in results if 'error' not in r])
    
    if valid_count > 0:
        print(f"\n🎯 項目別達成:")
        print(f"   ⚡ 速度達成: {speed_successes}/{valid_count} ({speed_successes/valid_count*100:.1f}%)")
        print(f"   📊 圧縮達成: {ratio_successes}/{valid_count} ({ratio_successes/valid_count*100:.1f}%)")
    
    # 詳細結果一覧
    print(f"\n📊 詳細改善状況:")
    
    for result in results:
        if 'error' not in result:
            status = "🟢" if result['emergency_success'] else ("🟡" if result.get('speed_emergency') else "🔴")
            
            print(f"   {status} {result['file']}")
            print(f"      ⚡ {result['throughput']:.1f}MB/s (v6.3: {result['v6_3_speed']:.1f}MB/s)")
            print(f"      🚀 改善率: {result['improvement_ratio']:.1f}倍")
            print(f"      📈 圧縮率: {result['compression_ratio']:.1f}%")
            print(f"      🧠 戦略: {result['strategy']}")
        else:
            print(f"   ❌ {result['file']}: {result['error']}")
    
    # 戦略効果分析
    strategy_stats = {}
    for result in results:
        if 'error' not in result:
            strategy = result['strategy']
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {'count': 0, 'avg_speed': 0, 'avg_improvement': 0}
            
            stats = strategy_stats[strategy]
            stats['count'] += 1
            stats['avg_speed'] += result['throughput']
            stats['avg_improvement'] += result['improvement_ratio']
    
    print(f"\n🧠 戦略別性能:")
    for strategy, stats in strategy_stats.items():
        avg_speed = stats['avg_speed'] / stats['count']
        avg_improvement = stats['avg_improvement'] / stats['count']
        print(f"   {strategy}: {avg_speed:.1f}MB/s | {avg_improvement:.1f}倍改善 (使用{stats['count']}回)")
    
    # 最終緊急判定
    print(f"\n🚨 NEXUS Ultra Light v6.4 緊急判定:")
    
    if avg_throughput >= 30 and success_rate >= 80:
        verdict = "🟢 EMERGENCY RESOLVED - 性能危機解決"
        detail = "v6.3の深刻な性能問題を完全解決"
    elif avg_throughput >= 20 and success_rate >= 60:
        verdict = "🟡 SIGNIFICANTLY IMPROVED - 大幅改善達成"
        detail = "実用レベルの性能回復を実現"
    elif avg_throughput >= 15 and overall_improvement >= 10:
        verdict = "🟠 IMPROVED - 改善確認"
        detail = "v6.3から明確な性能向上"
    elif avg_throughput >= 10:
        verdict = "🔴 PARTIAL IMPROVEMENT - 部分改善"
        detail = "更なる最適化が必要"
    else:
        verdict = "🔴 STILL CRITICAL - 依然深刻"
        detail = "根本的な再設計が必要"
    
    print(f"   {verdict}")
    print(f"   💡 {detail}")
    
    # エンジン統計
    engine_stats = engine.get_nexus_ultra_stats()
    if engine_stats.get('status') != 'no_data':
        print(f"\n🔧 Ultra Light Engine統計:")
        print(f"   📊 平均圧縮率: {engine_stats['total_compression_ratio']:.2f}%")
        print(f"   ⚡ 平均スループット: {engine_stats['average_throughput_mb_s']:.1f}MB/s")
        print(f"   🏆 性能グレード: {engine_stats['performance_grade']}")
        print(f"   🧠 戦略分布: {engine_stats['strategy_distribution']}")
    
    print(f"\n🎉 NEXUS Ultra Light Engine v6.4 緊急検証完了!")
    print("🚀 性能危機対応検証終了")
    
    return results


if __name__ == "__main__":
    test_nexus_emergency_optimization()
