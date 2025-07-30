#!/usr/bin/env python3
"""
NEXUS v6.1改良版 具体的問題修正検証テスト
v6.1の良好な性能を基盤とし、v6.4で発生した問題を修正

検証項目:
1. データ膨張問題の解決確認
2. v6.1の良好なパフォーマンス維持
3. 具体的な改善効果測定
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'nxzip', 'engine'))

import time
import lzma
import zlib
import bz2
from nxzip.engine.nexus_v6_1_improved import NEXUSUltimateEngineImproved
from pathlib import Path


def test_nexus_v6_1_improvements():
    """v6.1改良版の問題修正検証"""
    print("🔧 NEXUS v6.1改良版 - 問題修正検証テスト")
    print("🎯 v6.1の良好な性能 + v6.4問題修正")
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
    engine = NEXUSUltimateEngineImproved()
    
    # v6.1の良好な実績（参考値）
    v6_1_targets = {
        'jpg': {'ratio': 15.0, 'speed': 8.0},   # v6.1実績ベース
        'png': {'ratio': 5.0, 'speed': 6.0},
        'mp4': {'ratio': 25.0, 'speed': 10.0},
        'wav': {'ratio': 80.0, 'speed': 40.0},
        'mp3': {'ratio': 15.0, 'speed': 12.0},
        '7z': {'ratio': 3.0, 'speed': 8.0},
        'txt': {'ratio': 60.0, 'speed': 20.0}
    }
    
    results = []
    improvements_confirmed = 0
    no_expansion_count = 0  # 膨張回避成功数
    total_input = 0
    total_output = 0
    total_time = 0
    
    # v6.4の問題実績（参考）
    v6_4_problems = {
        'jpg': {'ratio': -45.51, 'speed': 6.4},  # 膨張問題
        'png': {'ratio': -50.17, 'speed': 6.0},  # 膨張問題
        'mp4': {'ratio': 5.0, 'speed': 5.0}      # 想定
    }
    
    for i, file_path in enumerate(test_files[:6], 1):  # 最大6ファイル
        ext = file_path.suffix[1:].lower()
        targets = v6_1_targets.get(ext, {'ratio': 10.0, 'speed': 8.0})
        v6_4_issue = v6_4_problems.get(ext, {'ratio': 0, 'speed': 5.0})
        
        print(f"\n{'🔧 ' + '='*60}")
        print(f"改善検証 {i}: {file_path.name}")
        print(f"   📊 サイズ: {file_path.stat().st_size:,} bytes ({file_path.stat().st_size / 1024 / 1024:.1f}MB)")
        print(f"   🎯 v6.1目標: 圧縮率{targets['ratio']}% / 速度{targets['speed']}MB/s")
        if ext in v6_4_problems:
            print(f"   ⚠️ v6.4問題: 圧縮率{v6_4_issue['ratio']}% (膨張)")
        
        try:
            # ファイル読み込み
            with open(file_path, 'rb') as f:
                data = f.read()
            
            input_size = len(data)
            total_input += input_size
            
            # NEXUS v6.1改良版実行
            print("   🔧 NEXUS v6.1改良版 実行...")
            start_time = time.perf_counter()
            compressed, info = engine.compress_nexus_improved(data, ext)
            compression_time = time.perf_counter() - start_time
            total_time += compression_time
            
            # 結果評価
            compression_ratio = info['compression_ratio']
            throughput = info['throughput_mb_s']
            
            # 重要: 膨張チェック
            no_expansion = len(compressed) < len(data)
            if no_expansion:
                no_expansion_count += 1
            
            # v6.1基準達成チェック
            ratio_ok = compression_ratio >= targets['ratio']
            speed_ok = throughput >= targets['speed']
            v6_1_standard = ratio_ok and speed_ok and no_expansion
            
            if v6_1_standard:
                improvements_confirmed += 1
            
            total_output += len(compressed)
            
            # v6.4からの改善チェック
            if ext in v6_4_problems:
                expansion_fixed = compression_ratio > v6_4_issue['ratio']
                speed_maintained = throughput >= v6_4_issue['speed']
                v6_4_improvement = expansion_fixed and speed_maintained
            else:
                v6_4_improvement = True  # 該当なし
            
            # 結果表示
            if v6_1_standard and v6_4_improvement:
                status = "🟢"
                message = "改善成功!"
            elif no_expansion and (ratio_ok or speed_ok):
                status = "🟡"
                message = "部分改善"
            elif no_expansion:
                status = "🟠"
                message = "膨張回避成功"
            else:
                status = "🔴"
                message = "要再修正"
            
            print(f"   {status} {message}")
            print(f"      📈 圧縮率: {compression_ratio:.2f}% {'🟢' if ratio_ok else '🔴'} (目標:{targets['ratio']}%)")
            print(f"      ⚡ スループット: {throughput:.1f}MB/s {'🟢' if speed_ok else '🔴'} (目標:{targets['speed']}MB/s)")
            print(f"      🛡️ 膨張回避: {'🟢 成功' if no_expansion else '🔴 失敗'}")
            print(f"      ⏱️ 処理時間: {compression_time:.3f}秒")
            print(f"      🧠 戦略: {info['strategy']}")
            print(f"      💾 {input_size:,} → {len(compressed):,} bytes")
            
            # v6.4問題修正確認
            if ext in v6_4_problems:
                print(f"      🔧 v6.4問題修正:")
                print(f"         膨張解決: {compression_ratio:.1f}% vs v6.4:{v6_4_issue['ratio']:.1f}% {'🟢' if expansion_fixed else '🔴'}")
                print(f"         速度維持: {throughput:.1f}MB/s vs v6.4:{v6_4_issue['speed']:.1f}MB/s {'🟢' if speed_maintained else '🔴'}")
            
            # NEXUS解析詳細
            if 'nexus_analysis' in info:
                na = info['nexus_analysis']
                print(f"      🔬 改良版解析:")
                print(f"         🧠 処理モード: {na['processing_mode']}")
                print(f"         📊 エントロピー: {na['entropy_score']:.3f}")
                print(f"         🔗 パターン結合: {na['pattern_coherence']:.3f}")
            
            # 競合比較（簡易）
            print("   🆚 標準圧縮との比較:")
            try:
                lzma_result = lzma.compress(data, preset=6)
                lzma_ratio = (1 - len(lzma_result) / len(data)) * 100
                nexus_advantage = compression_ratio - lzma_ratio
                print(f"      🥊 vs LZMA: {lzma_ratio:.1f}% ({'🏆' if nexus_advantage > 0 else '📊'}{nexus_advantage:+.1f}%)")
            except:
                print(f"      🥊 vs LZMA: エラー")
            
            # 結果記録
            results.append({
                'file': file_path.name,
                'type': ext,
                'input_size': input_size,
                'compression_ratio': compression_ratio,
                'throughput': throughput,
                'time': compression_time,
                'strategy': info['strategy'],
                'no_expansion': no_expansion,
                'v6_1_standard': v6_1_standard,
                'v6_4_improvement': v6_4_improvement,
                'targets': targets
            })
            
        except Exception as e:
            print(f"   ❌ エラー: {e}")
            results.append({
                'file': file_path.name,
                'error': str(e)
            })
    
    # 改善評価レポート
    print(f"\n{'🔧 ' + '='*70}")
    print(f"NEXUS v6.1改良版 改善評価レポート")
    print(f"{'='*80}")
    
    # 総合成果
    success_rate = (improvements_confirmed / len(results)) * 100 if results else 0
    expansion_prevention = (no_expansion_count / len(results)) * 100 if results else 0
    total_compression = (1 - total_output / total_input) * 100 if total_input > 0 else 0
    avg_throughput = (total_input / 1024 / 1024) / total_time if total_time > 0 else 0
    
    print(f"🎯 改善達成状況:")
    print(f"   🟢 v6.1基準達成: {improvements_confirmed}/{len(results)} ({success_rate:.1f}%)")
    print(f"   🛡️ 膨張回避成功: {no_expansion_count}/{len(results)} ({expansion_prevention:.1f}%)")
    print(f"   📈 総合圧縮率: {total_compression:.2f}%")
    print(f"   ⚡ 平均スループット: {avg_throughput:.1f}MB/s")
    print(f"   💾 総処理: {total_input / 1024 / 1024:.1f}MB → {total_output / 1024 / 1024:.1f}MB")
    print(f"   ⏱️ 総時間: {total_time:.3f}秒")
    
    # 重要な改善確認
    print(f"\n🔧 重要な改善確認:")
    if expansion_prevention >= 90:
        print(f"   ✅ データ膨張問題: 解決済み ({expansion_prevention:.1f}%成功)")
    elif expansion_prevention >= 70:
        print(f"   🟡 データ膨張問題: 大幅改善 ({expansion_prevention:.1f}%成功)")
    else:
        print(f"   🔴 データ膨張問題: 要継続対応 ({expansion_prevention:.1f}%成功)")
    
    if avg_throughput >= 10:
        print(f"   ✅ 速度性能: 良好維持 ({avg_throughput:.1f}MB/s)")
    elif avg_throughput >= 6:
        print(f"   🟡 速度性能: 許容範囲 ({avg_throughput:.1f}MB/s)")
    else:
        print(f"   🔴 速度性能: 要改善 ({avg_throughput:.1f}MB/s)")
    
    # 詳細結果一覧
    print(f"\n📊 詳細改善結果:")
    
    for result in results:
        if 'error' not in result:
            status = "🟢" if result['v6_1_standard'] else ("🟡" if result['no_expansion'] else "🔴")
            
            print(f"   {status} {result['file']}")
            print(f"      📈 {result['compression_ratio']:.1f}% / {result['targets']['ratio']}% | "
                  f"⚡ {result['throughput']:.1f}MB/s / {result['targets']['speed']}MB/s")
            print(f"      🛡️ 膨張回避: {'✅' if result['no_expansion'] else '❌'}")
            print(f"      🧠 戦略: {result['strategy']}")
        else:
            print(f"   ❌ {result['file']}: {result['error']}")
    
    # 戦略効果分析
    strategy_stats = {}
    for result in results:
        if 'error' not in result:
            strategy = result['strategy']
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {'count': 0, 'avg_ratio': 0, 'avg_speed': 0, 'no_expansion': 0}
            
            stats = strategy_stats[strategy]
            stats['count'] += 1
            stats['avg_ratio'] += result['compression_ratio']
            stats['avg_speed'] += result['throughput']
            if result['no_expansion']:
                stats['no_expansion'] += 1
    
    print(f"\n🧠 戦略別改善効果:")
    for strategy, stats in strategy_stats.items():
        avg_ratio = stats['avg_ratio'] / stats['count']
        avg_speed = stats['avg_speed'] / stats['count']
        expansion_prevention_rate = (stats['no_expansion'] / stats['count']) * 100
        print(f"   {strategy}: {avg_ratio:.1f}%圧縮 | {avg_speed:.1f}MB/s | 膨張回避{expansion_prevention_rate:.1f}% (使用{stats['count']}回)")
    
    # エンジン統計
    engine_stats = engine.get_nexus_stats()
    if engine_stats.get('status') != 'no_data':
        print(f"\n🔧 改良版Engine統計:")
        print(f"   📊 平均圧縮率: {engine_stats['total_compression_ratio']:.2f}%")
        print(f"   ⚡ 平均スループット: {engine_stats['average_throughput_mb_s']:.1f}MB/s")
        print(f"   🏆 性能グレード: {engine_stats['performance_grade']}")
        print(f"   🛡️ フォールバック使用: {engine_stats['fallback_usage']}回")
        print(f"   🧠 戦略分布: {engine_stats['strategy_distribution']}")
    
    # 最終判定
    print(f"\n🏆 NEXUS v6.1改良版 最終判定:")
    
    if success_rate >= 70 and expansion_prevention >= 90:
        verdict = "🟢 IMPROVEMENT SUCCESS - 改善成功"
        detail = "v6.1性能維持 + v6.4問題解決"
    elif expansion_prevention >= 80 and avg_throughput >= 8:
        verdict = "🟡 SIGNIFICANT IMPROVEMENT - 大幅改善"
        detail = "主要問題解決、継続改良で完成"
    elif expansion_prevention >= 60:
        verdict = "🟠 PARTIAL IMPROVEMENT - 部分改善"
        detail = "膨張問題は改善、速度要調整"
    else:
        verdict = "🔴 NEEDS FURTHER WORK - 継続改良要"
        detail = "基本的な問題修正から再開"
    
    print(f"   {verdict}")
    print(f"   💡 {detail}")
    
    print(f"\n🎉 NEXUS v6.1改良版 検証完了!")
    print("🔧 具体的問題修正アプローチの検証終了")
    
    return results


if __name__ == "__main__":
    test_nexus_v6_1_improvements()
