#!/usr/bin/env python3
"""
NXZip最適化版 vs 標準圧縮ベンチマーク
改良されたアルゴリズム性能検証
"""

import os
import sys
import time
import random
import zlib
import gzip
import bz2
from typing import Dict, Any, List, Tuple

sys.path.insert(0, os.path.dirname(__file__))
from nxzip_optimized_v2 import OptimizedNXZipEngine

def benchmark_comprehensive():
    """包括的ベンチマーク"""
    print("🚀 NXZip最適化版 vs 標準圧縮ベンチマーク")
    print("🎯 目標: Zstandardレベル速度 + 7-Zip超越圧縮率")
    print("=" * 80)
    
    # テストデータセット
    test_datasets = [
        (b'A' * 5000, "完全繰り返し (5KB)"),
        (b'Hello World! This is a test. ' * 200, "英文繰り返し (6KB)"),
        (bytes([random.randint(0, 255) for _ in range(5000)]), "ランダムデータ (5KB)"),
        (b'ABCDEFGH' * 1000, "短パターン (8KB)"),
        (b''.join([f'Line {i:04d}: NXZip test data with variation {i%100}\n'.encode() 
                   for i in range(200)]), "構造化テキスト (10KB)"),
        (b'0123456789' * 2000, "数字パターン (20KB)"),
    ]
    
    engines = {
        'nxzip_light': OptimizedNXZipEngine(lightweight_mode=True),
        'nxzip_normal': OptimizedNXZipEngine(lightweight_mode=False)
    }
    
    results_summary = []
    
    for test_data, description in test_datasets:
        print(f"\n📊 データセット: {description}")
        print("=" * 60)
        
        dataset_results = {}
        
        # NXZip最適化版
        for engine_name, engine in engines.items():
            mode_name = "軽量" if "light" in engine_name else "通常"
            print(f"\n🔧 NXZip {mode_name}モード:")
            
            start_time = time.time()
            compressed, info = engine.compress(test_data)
            compress_time = time.time() - start_time
            
            result = {
                'compressed_size': len(compressed),
                'compression_ratio': info['compression_ratio'],
                'compression_time': compress_time,
                'throughput_mbps': (len(test_data) / (1024 * 1024) / compress_time) if compress_time > 0 else 0,
                'algorithm': info.get('algorithm_used', 'unknown')
            }
            
            dataset_results[engine_name] = result
            
            print(f"  アルゴリズム: {result['algorithm']}")
            print(f"  圧縮率: {result['compression_ratio']:.1f}%")
            print(f"  処理時間: {compress_time:.3f}秒")
            print(f"  スループット: {result['throughput_mbps']:.1f}MB/s")
        
        # 標準ライブラリ比較
        standard_methods = [
            ('zlib_1', lambda d: zlib.compress(d, level=1), "Zlib高速"),
            ('zlib_3', lambda d: zlib.compress(d, level=3), "Zlib標準"),
            ('zlib_6', lambda d: zlib.compress(d, level=6), "Zlib高圧縮"),
            ('gzip_6', lambda d: gzip.compress(d, compresslevel=6), "Gzip標準"),
            ('bz2_9', lambda d: bz2.compress(d, compresslevel=9), "Bzip2最高")
        ]
        
        for method_name, compress_func, display_name in standard_methods:
            print(f"\n📋 {display_name}:")
            
            try:
                start_time = time.time()
                compressed = compress_func(test_data)
                compress_time = time.time() - start_time
                
                result = {
                    'compressed_size': len(compressed),
                    'compression_ratio': (1 - len(compressed) / len(test_data)) * 100 if len(test_data) > 0 else 0,
                    'compression_time': compress_time,
                    'throughput_mbps': (len(test_data) / (1024 * 1024) / compress_time) if compress_time > 0 else 0,
                    'algorithm': method_name
                }
                
                dataset_results[method_name] = result
                
                print(f"  圧縮率: {result['compression_ratio']:.1f}%")
                print(f"  処理時間: {compress_time:.3f}秒")
                print(f"  スループット: {result['throughput_mbps']:.1f}MB/s")
                
            except Exception as e:
                print(f"  エラー: {e}")
        
        # データセット比較分析
        analyze_dataset_results(dataset_results, description, len(test_data))
        results_summary.append((description, dataset_results))
    
    # 総合分析
    print("\n" + "=" * 80)
    print("📈 総合ベンチマーク分析")
    print("=" * 80)
    
    analyze_overall_results(results_summary)

def analyze_dataset_results(results: Dict[str, Dict], description: str, data_size: int):
    """データセット結果分析"""
    print(f"\n📈 {description} 比較分析:")
    print("-" * 50)
    
    # 圧縮率ランキング
    print("🏆 圧縮率ランキング (上位5位):")
    sorted_by_ratio = sorted(results.items(), key=lambda x: x[1]['compression_ratio'], reverse=True)
    for i, (method, result) in enumerate(sorted_by_ratio[:5], 1):
        marker = "🔥" if "nxzip" in method else "📋"
        print(f"  {i}位: {marker} {method} ({result['compression_ratio']:.1f}%)")
    
    # 速度ランキング
    print("\n⚡ 速度ランキング (上位5位):")
    sorted_by_speed = sorted(results.items(), key=lambda x: x[1]['throughput_mbps'], reverse=True)
    for i, (method, result) in enumerate(sorted_by_speed[:5], 1):
        marker = "🔥" if "nxzip" in method else "📋"
        print(f"  {i}位: {marker} {method} ({result['throughput_mbps']:.1f}MB/s)")
    
    # NXZip vs 標準比較
    if 'nxzip_light' in results and 'zlib_3' in results:
        nxzip_light = results['nxzip_light']
        zlib_3 = results['zlib_3']
        
        ratio_adv = nxzip_light['compression_ratio'] - zlib_3['compression_ratio']
        speed_ratio = nxzip_light['throughput_mbps'] / zlib_3['throughput_mbps'] if zlib_3['throughput_mbps'] > 0 else float('inf')
        
        print(f"\n🔥 NXZip軽量 vs Zlib標準:")
        print(f"  圧縮率アドバンテージ: {ratio_adv:+.1f}%")
        print(f"  速度比: {speed_ratio:.1f}x")
        
        if ratio_adv >= 0 and speed_ratio >= 1.0:
            print("  ✅ Zstandardレベル目標達成")
        elif ratio_adv >= 0:
            print("  ⚠️ 圧縮率優位、速度改善余地")
        else:
            print("  🔧 改善が必要")
    
    if 'nxzip_normal' in results and 'zlib_6' in results:
        nxzip_normal = results['nxzip_normal']
        zlib_6 = results['zlib_6']
        
        ratio_adv = nxzip_normal['compression_ratio'] - zlib_6['compression_ratio']
        speed_ratio = nxzip_normal['throughput_mbps'] / zlib_6['throughput_mbps'] if zlib_6['throughput_mbps'] > 0 else float('inf')
        
        print(f"\n🔥 NXZip通常 vs Zlib高圧縮:")
        print(f"  圧縮率アドバンテージ: {ratio_adv:+.1f}%")
        print(f"  速度比: {speed_ratio:.1f}x")
        
        if ratio_adv >= 5 and speed_ratio >= 2.0:
            print("  ✅ 7-Zip超越目標達成")
        elif ratio_adv >= 0:
            print("  🔄 圧縮率優位、速度目標未達")
        else:
            print("  🔧 改善が必要")

def analyze_overall_results(results_summary: List[Tuple[str, Dict]]):
    """総合結果分析"""
    
    # 達成度評価
    light_achievements = 0
    normal_achievements = 0
    total_datasets = len(results_summary)
    
    algorithm_usage = {}
    
    print("📊 データセット別達成度:")
    print("-" * 60)
    
    for description, results in results_summary:
        light_success = False
        normal_success = False
        
        # 軽量モード評価
        if 'nxzip_light' in results and 'zlib_3' in results:
            light_ratio_adv = results['nxzip_light']['compression_ratio'] - results['zlib_3']['compression_ratio']
            light_speed_ratio = results['nxzip_light']['throughput_mbps'] / results['zlib_3']['throughput_mbps'] if results['zlib_3']['throughput_mbps'] > 0 else float('inf')
            
            if light_ratio_adv >= 0 and light_speed_ratio >= 1.0:
                light_success = True
                light_achievements += 1
        
        # 通常モード評価
        if 'nxzip_normal' in results and 'zlib_6' in results:
            normal_ratio_adv = results['nxzip_normal']['compression_ratio'] - results['zlib_6']['compression_ratio']
            normal_speed_ratio = results['nxzip_normal']['throughput_mbps'] / results['zlib_6']['throughput_mbps'] if results['zlib_6']['throughput_mbps'] > 0 else float('inf')
            
            if normal_ratio_adv >= 5 and normal_speed_ratio >= 2.0:
                normal_success = True
                normal_achievements += 1
        
        # アルゴリズム使用統計
        for engine in ['nxzip_light', 'nxzip_normal']:
            if engine in results:
                algo = results[engine]['algorithm']
                if algo not in algorithm_usage:
                    algorithm_usage[algo] = 0
                algorithm_usage[algo] += 1
        
        # データセット結果表示
        light_icon = "✅" if light_success else "❌"
        normal_icon = "✅" if normal_success else "❌"
        print(f"  {light_icon} {normal_icon} {description}")
    
    # 総合達成率
    light_success_rate = (light_achievements / total_datasets) * 100
    normal_success_rate = (normal_achievements / total_datasets) * 100
    
    print(f"\n🎯 総合達成率:")
    print(f"  ⚡ 軽量モード (Zstandardレベル): {light_success_rate:.1f}% ({light_achievements}/{total_datasets})")
    print(f"  🎯 通常モード (7-Zip超越): {normal_success_rate:.1f}% ({normal_achievements}/{total_datasets})")
    
    # アルゴリズム使用統計
    print(f"\n🔧 アルゴリズム使用統計:")
    for algo, count in sorted(algorithm_usage.items(), key=lambda x: x[1], reverse=True):
        print(f"  {algo}: {count}回使用")
    
    # 総合評価
    overall_success = (light_success_rate + normal_success_rate) / 2
    
    if overall_success >= 75:
        print(f"\n🏆 NXZip最適化版: 優秀 ({overall_success:.1f}%)")
        print("   目標をほぼ達成し、実用レベルに到達")
    elif overall_success >= 50:
        print(f"\n🔄 NXZip最適化版: 良好 ({overall_success:.1f}%)")
        print("   基本目標達成、さらなる最適化で向上可能")
    elif overall_success >= 25:
        print(f"\n🔧 NXZip最適化版: 改善要 ({overall_success:.1f}%)")
        print("   アルゴリズム調整が必要")
    else:
        print(f"\n❌ NXZip最適化版: 見直し要 ({overall_success:.1f}%)")
        print("   基本アーキテクチャの再検討が必要")

if __name__ == "__main__":
    try:
        benchmark_comprehensive()
        print("\n🏁 包括的ベンチマーク完了")
    except Exception as e:
        print(f"\n❌ ベンチマークエラー: {e}")
        import traceback
        traceback.print_exc()
