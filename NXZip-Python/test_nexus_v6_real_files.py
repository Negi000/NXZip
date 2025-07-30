#!/usr/bin/env python3
"""
NEXUS Ultimate Engine v6.0 実ファイルテスト
実際のファイルで40%以上の圧縮率検証
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'nxzip', 'engine'))

import time
import lzma
import zlib
import bz2
from nxzip.engine.nexus_ultimate_v6 import NEXUSUltimateEngine
from pathlib import Path


def test_real_files_ultimate():
    """実ファイルでのNEXUS Ultimate v6.0テスト"""
    print("🚀 NEXUS Ultimate Engine v6.0 実ファイル検証")
    print("🎯 目標: 画像・動画・音声ファイルで大幅圧縮率向上")
    print("=" * 100)
    
    # サンプルファイルディレクトリ
    sample_dir = Path("sample")
    
    # ファイル検索
    test_files = []
    if sample_dir.exists():
        for ext in ['*.jpg', '*.png', '*.mp4', '*.wav', '*.mp3', '*.7z', '*.txt']:
            test_files.extend(sample_dir.glob(ext))
    
    if not test_files:
        print("❌ テストファイルが見つかりません")
        return
    
    print(f"📁 検出ファイル数: {len(test_files)}")
    for f in test_files[:10]:  # 最大10ファイル表示
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"   📄 {f.name} ({size_mb:.1f}MB)")
    
    if len(test_files) > 10:
        print(f"   ... 他 {len(test_files) - 10} ファイル")
    
    # エンジン初期化
    engine = NEXUSUltimateEngine()
    
    # ファイル種別による期待値
    compression_targets = {
        'jpg': 25.0,  # JPEG画像 25%目標
        'png': 30.0,  # PNG画像 30%目標  
        'mp4': 15.0,  # MP4動画 15%目標
        'wav': 80.0,  # WAV音声 80%目標
        'mp3': 5.0,   # MP3音声 5%目標
        '7z': 2.0,    # 圧縮ファイル 2%目標
        'txt': 70.0   # テキスト 70%目標
    }
    
    results = []
    total_achievements = 0
    total_input_size = 0
    total_output_size = 0
    
    for file_path in test_files:
        ext = file_path.suffix[1:].lower()
        target_ratio = compression_targets.get(ext, 10.0)
        
        print(f"\n{'🔬 ' + '='*80}")
        print(f"📄 ファイル: {file_path.name}")
        print(f"   📊 サイズ: {file_path.stat().st_size:,} bytes ({file_path.stat().st_size / 1024 / 1024:.1f}MB)")
        print(f"   🎯 目標圧縮率: {target_ratio}%")
        print(f"   📁 タイプ: {ext}")
        
        try:
            # ファイル読み込み
            print("   📖 ファイル読み込み中...")
            with open(file_path, 'rb') as f:
                data = f.read()
            
            input_size = len(data)
            total_input_size += input_size
            
            # NEXUS Ultimate 圧縮
            print("   🚀 NEXUS Ultimate v6.0 圧縮実行...")
            start_time = time.perf_counter()
            compressed, info = engine.compress_ultimate(data, ext)
            compression_time = time.perf_counter() - start_time
            
            # 結果分析
            compression_ratio = info['compression_ratio']
            throughput = (input_size / 1024 / 1024) / compression_time
            achievement = compression_ratio >= target_ratio
            
            if achievement:
                total_achievements += 1
            
            total_output_size += len(compressed)
            
            print(f"   {'✅' if achievement else '📊'} 圧縮完了!")
            print(f"      📈 達成圧縮率: {compression_ratio:.2f}% {'🎉' if achievement else '📊'}")
            print(f"      🎯 目標達成: {'YES' if achievement else 'NO'} ({compression_ratio:.1f}% / {target_ratio:.1f}%)")
            print(f"      ⚡ スループット: {throughput:.2f}MB/s")
            print(f"      ⏱️ 処理時間: {compression_time:.3f}秒")
            print(f"      🧠 最適戦略: {info['strategy']}")
            print(f"      💾 サイズ変化: {input_size:,} → {len(compressed):,} bytes")
            
            # 量子解析結果
            if 'quantum_analysis' in info:
                qa = info['quantum_analysis']
                print(f"      🔬 量子解析:")
                print(f"         📊 パターンコヒーレンス: {qa['pattern_coherence']:.3f}")
                print(f"         🎯 圧縮ポテンシャル: {qa['compression_potential']:.3f}")
                print(f"         🧮 次元複雑度: {qa['dimensional_complexity']:.3f}")
            
            # 競合比較テスト
            print("   📊 競合アルゴリズム比較...")
            competitors = {
                'LZMA-9': lambda d: lzma.compress(d, preset=9),
                'GZIP-9': lambda d: zlib.compress(d, level=9),
                'BZIP2-9': lambda d: bz2.compress(d, compresslevel=9)
            }
            
            comparison_results = {}
            for comp_name, comp_func in competitors.items():
                try:
                    comp_start = time.perf_counter()
                    comp_result = comp_func(data)
                    comp_time = time.perf_counter() - comp_start
                    comp_ratio = (1 - len(comp_result) / len(data)) * 100
                    comp_throughput = (input_size / 1024 / 1024) / comp_time
                    
                    improvement = compression_ratio - comp_ratio
                    speed_ratio = throughput / comp_throughput
                    
                    comparison_results[comp_name] = {
                        'ratio': comp_ratio,
                        'improvement': improvement,
                        'speed_ratio': speed_ratio
                    }
                    
                    print(f"      🆚 {comp_name}: {comp_ratio:.1f}% | "
                          f"NEXUS{'+' if improvement >= 0 else ''}{improvement:.1f}% | "
                          f"速度x{speed_ratio:.1f}")
                    
                except Exception as e:
                    print(f"      ❌ {comp_name}: エラー")
            
            # 結果記録
            results.append({
                'file': file_path.name,
                'type': ext,
                'input_size': input_size,
                'output_size': len(compressed),
                'target_ratio': target_ratio,
                'achieved_ratio': compression_ratio,
                'achievement': achievement,
                'throughput': throughput,
                'time': compression_time,
                'strategy': info['strategy'],
                'comparisons': comparison_results
            })
            
        except Exception as e:
            print(f"   ❌ エラー: {str(e)}")
            import traceback
            traceback.print_exc()
            
            results.append({
                'file': file_path.name,
                'type': ext,
                'target_ratio': target_ratio,
                'achieved_ratio': 0.0,
                'achievement': False,
                'error': str(e)
            })
    
    # 最終評価レポート
    print(f"\n{'🏆 ' + '='*90}")
    print(f"NEXUS Ultimate Engine v6.0 実ファイル最終評価")
    print(f"{'='*100}")
    
    # 基本統計
    success_rate = (total_achievements / len(results)) * 100 if results else 0
    total_compression_ratio = (1 - total_output_size / total_input_size) * 100 if total_input_size > 0 else 0
    
    print(f"🎯 目標達成統計:")
    print(f"   ✅ 達成ファイル: {total_achievements}/{len(results)}")
    print(f"   📊 達成率: {success_rate:.1f}%")
    print(f"   📈 総合圧縮率: {total_compression_ratio:.2f}%")
    print(f"   💾 総処理量: {total_input_size / 1024 / 1024:.1f}MB → {total_output_size / 1024 / 1024:.1f}MB")
    
    # ファイル種別分析
    print(f"\n📁 ファイル種別別成果:")
    type_stats = {}
    for result in results:
        if 'error' not in result:
            file_type = result['type']
            if file_type not in type_stats:
                type_stats[file_type] = {
                    'count': 0,
                    'achievements': 0,
                    'total_ratio': 0.0,
                    'best_ratio': 0.0
                }
            
            stats = type_stats[file_type]
            stats['count'] += 1
            if result['achievement']:
                stats['achievements'] += 1
            stats['total_ratio'] += result['achieved_ratio']
            stats['best_ratio'] = max(stats['best_ratio'], result['achieved_ratio'])
    
    for file_type, stats in type_stats.items():
        avg_ratio = stats['total_ratio'] / stats['count']
        achievement_rate = (stats['achievements'] / stats['count']) * 100
        
        print(f"   📄 {file_type.upper()}:")
        print(f"      📊 平均圧縮率: {avg_ratio:.1f}%")
        print(f"      🏆 最高圧縮率: {stats['best_ratio']:.1f}%")
        print(f"      ✅ 達成率: {achievement_rate:.1f}% ({stats['achievements']}/{stats['count']})")
    
    # 最高性能ファイル
    print(f"\n🏆 最高性能記録:")
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        best_compression = max(valid_results, key=lambda x: x['achieved_ratio'])
        best_speed = max(valid_results, key=lambda x: x['throughput'])
        
        print(f"   📈 最高圧縮率: {best_compression['achieved_ratio']:.2f}%")
        print(f"      📄 ファイル: {best_compression['file']}")
        print(f"      🧠 戦略: {best_compression['strategy']}")
        
        print(f"   ⚡ 最高速度: {best_speed['throughput']:.2f}MB/s")
        print(f"      📄 ファイル: {best_speed['file']}")
    
    # 競合比較統計
    print(f"\n🆚 競合アルゴリズム総合比較:")
    all_comparisons = {}
    for result in valid_results:
        if 'comparisons' in result:
            for comp_name, comp_data in result['comparisons'].items():
                if comp_name not in all_comparisons:
                    all_comparisons[comp_name] = {
                        'improvements': [],
                        'speed_ratios': []
                    }
                
                all_comparisons[comp_name]['improvements'].append(comp_data['improvement'])
                all_comparisons[comp_name]['speed_ratios'].append(comp_data['speed_ratio'])
    
    for comp_name, comp_stats in all_comparisons.items():
        avg_improvement = sum(comp_stats['improvements']) / len(comp_stats['improvements'])
        avg_speed_ratio = sum(comp_stats['speed_ratios']) / len(comp_stats['speed_ratios'])
        wins = sum(1 for imp in comp_stats['improvements'] if imp > 0)
        
        print(f"   🥊 vs {comp_name}:")
        print(f"      📈 平均改善: {avg_improvement:+.1f}%")
        print(f"      ⚡ 平均速度比: x{avg_speed_ratio:.1f}")
        print(f"      🏆 勝利率: {wins}/{len(comp_stats['improvements'])} ({wins/len(comp_stats['improvements'])*100:.1f}%)")
    
    # 総合評価
    print(f"\n🎖️ 総合評価判定:")
    if success_rate >= 70 and total_compression_ratio >= 30:
        print("   🏆 EXCELLENT - NEXUS理論の極めて優秀な実装")
        print("   💡 画像・動画での大幅圧縮率向上を達成")
    elif success_rate >= 50 and total_compression_ratio >= 20:
        print("   🥈 VERY GOOD - 理論的潜在能力の高い実現")
        print("   💡 既存手法を大きく上回る性能")
    elif success_rate >= 30 and total_compression_ratio >= 15:
        print("   🥉 GOOD - 基本的な改善目標を達成")
        print("   💡 競合手法との差別化に成功")
    else:
        print("   📊 NEEDS IMPROVEMENT - さらなる最適化が必要")
        print("   💡 理論実装の深化が要求される")
    
    # エンジン統計
    engine_report = engine.get_performance_report()
    if engine_report.get('status') != 'no_data':
        print(f"\n🔧 エンジン詳細統計:")
        print(f"   📊 総圧縮率: {engine_report['total_compression_ratio']:.2f}%")
        print(f"   ⚡ 平均スループット: {engine_report['average_throughput_mb_s']:.2f}MB/s")
        print(f"   ⏱️ 総処理時間: {engine_report['total_time']:.3f}秒")
        print(f"   🧠 戦略使用分布: {engine_report['strategy_distribution']}")
    
    print(f"\n🎉 NEXUS Ultimate Engine v6.0 実ファイルテスト完了!")
    print("🚀 次世代圧縮システムの実力検証終了")
    
    return results


if __name__ == "__main__":
    test_real_files_ultimate()
