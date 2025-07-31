#!/usr/bin/env python3
"""
軽量モード vs Zstandard 簡易圧縮率比較
"""

import time
import zstandard as zstd
import sys
import os

# 軽量モードをインポート
sys.path.append('.')
from lightweight_mode import NEXUSTMCLightweight

def create_test_samples():
    """テストサンプル作成"""
    samples = {}
    
    # 1. 繰り返しテキスト（高圧縮期待）
    text = "The quick brown fox jumps over the lazy dog. " * 1000
    samples['繰り返しテキスト'] = text.encode('utf-8')
    
    # 2. JSON様データ
    json_pattern = '{"id": %d, "name": "user_%d", "status": "active"},'
    json_data = '[' + ''.join([json_pattern % (i, i) for i in range(500)])[:-1] + ']'
    samples['JSON構造データ'] = json_data.encode('utf-8')
    
    # 3. ランダムバイナリ（低圧縮期待）
    import random
    random.seed(42)
    binary = bytes([random.randint(0, 255) for _ in range(20000)])
    samples['ランダムバイナリ'] = binary
    
    # 4. パターンデータ
    pattern = b'ABCDEFGHIJ' * 2000
    samples['パターンデータ'] = pattern
    
    return samples

def simple_benchmark(name, data):
    """シンプルな圧縮率ベンチマーク"""
    print(f"\n{'='*50}")
    print(f"テスト: {name}")
    print(f"原始サイズ: {len(data):,} bytes")
    print(f"{'='*50}")
    
    results = {}
    
    # Zstandardレベル1
    try:
        start = time.perf_counter()
        zstd_compressed_1 = zstd.compress(data, level=1)
        zstd_time_1 = time.perf_counter() - start
        
        ratio_1 = len(zstd_compressed_1) / len(data)
        speed_1 = len(data) / (1024 * 1024 * zstd_time_1) if zstd_time_1 > 0 else 0
        
        print(f"Zstd レベル1: {len(zstd_compressed_1):,} bytes (圧縮率: {ratio_1:.3f}, 削減: {(1-ratio_1)*100:.1f}%, 速度: {speed_1:.1f} MB/s)")
        results['zstd_1'] = {'size': len(zstd_compressed_1), 'ratio': ratio_1, 'speed': speed_1}
    except Exception as e:
        print(f"Zstd レベル1: エラー - {e}")
    
    # Zstandardレベル3 (デフォルト)
    try:
        start = time.perf_counter()
        zstd_compressed_3 = zstd.compress(data, level=3)
        zstd_time_3 = time.perf_counter() - start
        
        ratio_3 = len(zstd_compressed_3) / len(data)
        speed_3 = len(data) / (1024 * 1024 * zstd_time_3) if zstd_time_3 > 0 else 0
        
        print(f"Zstd レベル3: {len(zstd_compressed_3):,} bytes (圧縮率: {ratio_3:.3f}, 削減: {(1-ratio_3)*100:.1f}%, 速度: {speed_3:.1f} MB/s)")
        results['zstd_3'] = {'size': len(zstd_compressed_3), 'ratio': ratio_3, 'speed': speed_3}
    except Exception as e:
        print(f"Zstd レベル3: エラー - {e}")
    
    # Zstandardレベル6
    try:
        start = time.perf_counter()
        zstd_compressed_6 = zstd.compress(data, level=6)
        zstd_time_6 = time.perf_counter() - start
        
        ratio_6 = len(zstd_compressed_6) / len(data)
        speed_6 = len(data) / (1024 * 1024 * zstd_time_6) if zstd_time_6 > 0 else 0
        
        print(f"Zstd レベル6: {len(zstd_compressed_6):,} bytes (圧縮率: {ratio_6:.3f}, 削減: {(1-ratio_6)*100:.1f}%, 速度: {speed_6:.1f} MB/s)")
        results['zstd_6'] = {'size': len(zstd_compressed_6), 'ratio': ratio_6, 'speed': speed_6}
    except Exception as e:
        print(f"Zstd レベル6: エラー - {e}")
    
    # NEXUS TMC 軽量モード
    try:
        nexus = NEXUSTMCLightweight()
        start = time.perf_counter()
        nexus_compressed, meta = nexus.compress_fast(data)
        nexus_time = time.perf_counter() - start
        
        # 展開テスト
        nexus_decompressed = nexus.decompress_fast(nexus_compressed, meta)
        if nexus_decompressed == data:
            ratio_nexus = len(nexus_compressed) / len(data)
            speed_nexus = len(data) / (1024 * 1024 * nexus_time) if nexus_time > 0 else 0
            
            print(f"NEXUS軽量: {len(nexus_compressed):,} bytes (圧縮率: {ratio_nexus:.3f}, 削減: {(1-ratio_nexus)*100:.1f}%, 速度: {speed_nexus:.1f} MB/s)")
            results['nexus'] = {'size': len(nexus_compressed), 'ratio': ratio_nexus, 'speed': speed_nexus}
        else:
            print("NEXUS軽量: データ整合性エラー")
    except Exception as e:
        print(f"NEXUS軽量: エラー - {e}")
    
    return results

def compare_all():
    """全体比較"""
    print("NEXUS TMC 軽量モード vs Zstandard 圧縮率・速度比較")
    print("="*60)
    
    samples = create_test_samples()
    all_results = {}
    
    for name, data in samples.items():
        all_results[name] = simple_benchmark(name, data)
    
    # 総合分析
    print(f"\n{'='*60}")
    print("総合分析")
    print(f"{'='*60}")
    
    # 平均値計算
    engine_stats = {}
    for test_name, results in all_results.items():
        for engine, stats in results.items():
            if engine not in engine_stats:
                engine_stats[engine] = {'ratios': [], 'speeds': [], 'sizes': []}
            engine_stats[engine]['ratios'].append(stats['ratio'])
            engine_stats[engine]['speeds'].append(stats['speed'])
            engine_stats[engine]['sizes'].append(stats['size'])
    
    print("\n平均性能:")
    for engine, stats in engine_stats.items():
        avg_ratio = sum(stats['ratios']) / len(stats['ratios'])
        avg_speed = sum(stats['speeds']) / len(stats['speeds'])
        avg_reduction = (1 - avg_ratio) * 100
        
        engine_names = {
            'zstd_1': 'Zstandard レベル1',
            'zstd_3': 'Zstandard レベル3', 
            'zstd_6': 'Zstandard レベル6',
            'nexus': 'NEXUS TMC 軽量'
        }
        
        print(f"{engine_names.get(engine, engine)}:")
        print(f"  平均圧縮率: {avg_ratio:.3f}")
        print(f"  平均容量削減: {avg_reduction:.1f}%")
        print(f"  平均速度: {avg_speed:.1f} MB/s")
    
    # 軽量モード vs Zstandardの詳細比較
    if 'nexus' in engine_stats and 'zstd_3' in engine_stats:
        print(f"\n{'='*40}")
        print("軽量モード vs Zstandard詳細比較")
        print(f"{'='*40}")
        
        nexus_avg_ratio = sum(engine_stats['nexus']['ratios']) / len(engine_stats['nexus']['ratios'])
        zstd3_avg_ratio = sum(engine_stats['zstd_3']['ratios']) / len(engine_stats['zstd_3']['ratios'])
        
        nexus_avg_speed = sum(engine_stats['nexus']['speeds']) / len(engine_stats['nexus']['speeds'])
        zstd3_avg_speed = sum(engine_stats['zstd_3']['speeds']) / len(engine_stats['zstd_3']['speeds'])
        
        ratio_diff_percent = ((nexus_avg_ratio - zstd3_avg_ratio) / zstd3_avg_ratio) * 100
        speed_diff_percent = ((nexus_avg_speed - zstd3_avg_speed) / zstd3_avg_speed) * 100
        
        print(f"圧縮率差: {ratio_diff_percent:+.1f}% (+ = 軽量モードの方が低圧縮)")
        print(f"速度差: {speed_diff_percent:+.1f}% (+ = 軽量モードの方が高速)")
        
        if ratio_diff_percent > 0:
            print(f"\n結論: 軽量モードは圧縮率で{ratio_diff_percent:.1f}%劣るが、速度で{speed_diff_percent:.1f}%優位")
        else:
            print(f"\n結論: 軽量モードは圧縮率で{-ratio_diff_percent:.1f}%優位、速度でも{speed_diff_percent:.1f}%優位")
        
        print("\nトレードオフ評価:")
        if abs(ratio_diff_percent) < 5 and speed_diff_percent > 0:
            print("✅ 軽量モードは圧縮率をほぼ保持しながら高速化を実現")
        elif ratio_diff_percent > 5 and speed_diff_percent > 20:
            print("⚖️ 軽量モードは圧縮率を犠牲にして大幅な高速化を実現")
        elif ratio_diff_percent < 0:
            print("🎯 軽量モードは圧縮率・速度ともに優位！")

if __name__ == "__main__":
    compare_all()
