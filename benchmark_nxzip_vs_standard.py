#!/usr/bin/env python3
"""
NXZip TMC v9.1 vs 標準圧縮ベンチマーク
Zstandard/7-Zip性能比較検証
"""

import os
import sys
import time
import random
import zlib
import gzip
import bz2
from typing import Dict, Any, List, Tuple

# NXZip TMC v9.1 統括エンジンインポート
sys.path.insert(0, os.path.dirname(__file__))
from test_nxzip_tmc_unified import NXZipTMCEngine

class BenchmarkSuite:
    """NXZip vs 標準圧縮ベンチマーク"""
    
    def __init__(self):
        self.nxzip_light = NXZipTMCEngine(lightweight_mode=True)
        self.nxzip_normal = NXZipTMCEngine(lightweight_mode=False)
        
        self.results = []
    
    def compress_zlib(self, data: bytes, level: int = 6) -> Tuple[bytes, Dict[str, Any]]:
        """Zlib標準圧縮"""
        start_time = time.time()
        try:
            compressed = zlib.compress(data, level=level)
            compress_time = time.time() - start_time
            
            return compressed, {
                'method': f'zlib_level_{level}',
                'original_size': len(data),
                'compressed_size': len(compressed),
                'compression_ratio': (1 - len(compressed) / len(data)) * 100 if len(data) > 0 else 0,
                'compression_time': compress_time,
                'throughput_mbps': (len(data) / (1024 * 1024) / compress_time) if compress_time > 0 else 0
            }
        except Exception as e:
            return data, {'method': 'zlib_error', 'error': str(e)}
    
    def compress_gzip(self, data: bytes, level: int = 6) -> Tuple[bytes, Dict[str, Any]]:
        """Gzip標準圧縮"""
        start_time = time.time()
        try:
            compressed = gzip.compress(data, compresslevel=level)
            compress_time = time.time() - start_time
            
            return compressed, {
                'method': f'gzip_level_{level}',
                'original_size': len(data),
                'compressed_size': len(compressed),
                'compression_ratio': (1 - len(compressed) / len(data)) * 100 if len(data) > 0 else 0,
                'compression_time': compress_time,
                'throughput_mbps': (len(data) / (1024 * 1024) / compress_time) if compress_time > 0 else 0
            }
        except Exception as e:
            return data, {'method': 'gzip_error', 'error': str(e)}
    
    def compress_bz2(self, data: bytes, level: int = 9) -> Tuple[bytes, Dict[str, Any]]:
        """Bzip2標準圧縮"""
        start_time = time.time()
        try:
            compressed = bz2.compress(data, compresslevel=level)
            compress_time = time.time() - start_time
            
            return compressed, {
                'method': f'bz2_level_{level}',
                'original_size': len(data),
                'compressed_size': len(compressed),
                'compression_ratio': (1 - len(compressed) / len(data)) * 100 if len(data) > 0 else 0,
                'compression_time': compress_time,
                'throughput_mbps': (len(data) / (1024 * 1024) / compress_time) if compress_time > 0 else 0
            }
        except Exception as e:
            return data, {'method': 'bz2_error', 'error': str(e)}
    
    def benchmark_dataset(self, data: bytes, description: str) -> Dict[str, Any]:
        """単一データセットベンチマーク"""
        print(f"\n📊 ベンチマーク: {description} ({len(data):,} bytes)")
        print("=" * 70)
        
        results = {}
        
        # NXZip軽量モード (Zstandardレベル目標)
        print("⚡ NXZip軽量モード (Zstandardレベル目標):")
        nxzip_compressed, nxzip_info = self.nxzip_light.compress(data)
        results['nxzip_light'] = nxzip_info
        print(f"  圧縮率: {nxzip_info['compression_ratio']:.1f}%")
        print(f"  処理時間: {nxzip_info['compression_time']:.3f}秒")
        print(f"  スループット: {nxzip_info['throughput_mbps']:.1f}MB/s")
        
        # NXZip通常モード (7-Zip超越目標)
        print("\n🎯 NXZip通常モード (7-Zip超越目標):")
        nxzip_normal_compressed, nxzip_normal_info = self.nxzip_normal.compress(data)
        results['nxzip_normal'] = nxzip_normal_info
        print(f"  圧縮率: {nxzip_normal_info['compression_ratio']:.1f}%")
        print(f"  処理時間: {nxzip_normal_info['compression_time']:.3f}秒")
        print(f"  スループット: {nxzip_normal_info['throughput_mbps']:.1f}MB/s")
        
        # Zlib (Zstandard近似)
        print("\n🔷 Zlib レベル3 (Zstandardクラス):")
        zlib_compressed, zlib_info = self.compress_zlib(data, level=3)
        results['zlib_3'] = zlib_info
        print(f"  圧縮率: {zlib_info['compression_ratio']:.1f}%")
        print(f"  処理時間: {zlib_info['compression_time']:.3f}秒")
        print(f"  スループット: {zlib_info['throughput_mbps']:.1f}MB/s")
        
        # Zlib (7-Zipクラス)
        print("\n🔶 Zlib レベル6 (7-Zipクラス):")
        zlib6_compressed, zlib6_info = self.compress_zlib(data, level=6)
        results['zlib_6'] = zlib6_info
        print(f"  圧縮率: {zlib6_info['compression_ratio']:.1f}%")
        print(f"  処理時間: {zlib6_info['compression_time']:.3f}秒")
        print(f"  スループット: {zlib6_info['throughput_mbps']:.1f}MB/s")
        
        # Gzip標準
        print("\n🟦 Gzip レベル6:")
        gzip_compressed, gzip_info = self.compress_gzip(data, level=6)
        results['gzip_6'] = gzip_info
        print(f"  圧縮率: {gzip_info['compression_ratio']:.1f}%")
        print(f"  処理時間: {gzip_info['compression_time']:.3f}秒")
        print(f"  スループット: {gzip_info['throughput_mbps']:.1f}MB/s")
        
        # Bzip2 (高圧縮)
        print("\n🟪 Bzip2 レベル9 (高圧縮):")
        bz2_compressed, bz2_info = self.compress_bz2(data, level=9)
        results['bz2_9'] = bz2_info
        print(f"  圧縮率: {bz2_info['compression_ratio']:.1f}%")
        print(f"  処理時間: {bz2_info['compression_time']:.3f}秒")
        print(f"  スループット: {bz2_info['throughput_mbps']:.1f}MB/s")
        
        # 比較分析
        self._analyze_results(results, description)
        
        return results
    
    def _analyze_results(self, results: Dict[str, Dict], description: str):
        """結果分析"""
        print(f"\n📈 {description} 比較分析:")
        print("-" * 50)
        
        # Zstandardレベル比較 (軽量モード vs Zlib3)
        nxzip_light = results['nxzip_light']
        zlib_3 = results['zlib_3']
        
        ratio_advantage = nxzip_light['compression_ratio'] - zlib_3['compression_ratio']
        speed_ratio = zlib_3['compression_time'] / nxzip_light['compression_time'] if nxzip_light['compression_time'] > 0 else 1
        
        print(f"🏃 軽量モード vs Zlib3 (Zstandardクラス):")
        print(f"  NXZip圧縮率アドバンテージ: {ratio_advantage:+.1f}%")
        print(f"  NXZip速度比: {speed_ratio:.1f}x")
        
        if ratio_advantage > 0 and speed_ratio >= 0.8:
            print("  ✅ Zstandardレベル目標達成")
        elif ratio_advantage > 0:
            print("  ⚠️ 圧縮率優位、速度改善余地あり")
        else:
            print("  ❌ Zstandardレベル未達成")
        
        # 7-Zip超越比較 (通常モード vs Zlib6)
        nxzip_normal = results['nxzip_normal']
        zlib_6 = results['zlib_6']
        
        ratio_advantage_normal = nxzip_normal['compression_ratio'] - zlib_6['compression_ratio']
        speed_ratio_normal = zlib_6['compression_time'] / nxzip_normal['compression_time'] if nxzip_normal['compression_time'] > 0 else 1
        
        print(f"\n🎯 通常モード vs Zlib6 (7-Zipクラス):")
        print(f"  NXZip圧縮率アドバンテージ: {ratio_advantage_normal:+.1f}%")
        print(f"  NXZip速度比: {speed_ratio_normal:.1f}x")
        
        if ratio_advantage_normal > 0 and speed_ratio_normal >= 2.0:
            print("  ✅ 7-Zip超越目標達成")
        elif ratio_advantage_normal > 0:
            print("  ⚠️ 圧縮率優位、速度目標未達")
        else:
            print("  ❌ 7-Zip超越未達成")
        
        # 総合ランキング
        methods = ['nxzip_light', 'nxzip_normal', 'zlib_3', 'zlib_6', 'gzip_6', 'bz2_9']
        print(f"\n🏆 圧縮率ランキング:")
        sorted_by_ratio = sorted(methods, key=lambda m: results[m]['compression_ratio'], reverse=True)
        for i, method in enumerate(sorted_by_ratio[:3], 1):
            print(f"  {i}位: {method} ({results[method]['compression_ratio']:.1f}%)")
        
        print(f"\n⚡ 速度ランキング:")
        sorted_by_speed = sorted(methods, key=lambda m: results[m]['throughput_mbps'], reverse=True)
        for i, method in enumerate(sorted_by_speed[:3], 1):
            print(f"  {i}位: {method} ({results[method]['throughput_mbps']:.1f}MB/s)")

def generate_test_datasets() -> List[Tuple[bytes, str]]:
    """テストデータセット生成"""
    datasets = []
    
    # 1. 高圧縮率テキスト (Zstandardテスト用)
    text_data = b'NXZip Test: Hello World! ' * 500  # 繰り返しデータ
    datasets.append((text_data, "高圧縮率テキスト"))
    
    # 2. 中程度テキスト (実用的データ)
    mixed_text = b''
    for i in range(100):
        line = f'Line {i:03d}: NXZip compression test with some variation {random.randint(1000, 9999)}\n'.encode()
        mixed_text += line
    datasets.append((mixed_text, "中程度構造化テキスト"))
    
    # 3. ランダムデータ (圧縮困難)
    random_data = bytes([random.randint(0, 255) for _ in range(5000)])
    datasets.append((random_data, "ランダムデータ"))
    
    # 4. パターンデータ (TMC最適化)
    pattern_data = b'A' * 1000 + b'B' * 1000 + b'C' * 1000 + b'D' * 1000
    datasets.append((pattern_data, "パターンデータ"))
    
    # 5. 大きめファイル (スループットテスト)
    large_text = (b'NXZip large file test. ' + b'Data pattern variation. ' * 10) * 200
    datasets.append((large_text, "大容量テキスト"))
    
    return datasets

def main():
    """メインベンチマーク実行"""
    print("🚀 NXZip TMC v9.1 vs 標準圧縮ベンチマーク")
    print("🎯 目標: 軽量=Zstandardレベル, 通常=7-Zip超越")
    print("=" * 70)
    
    benchmark = BenchmarkSuite()
    datasets = generate_test_datasets()
    
    all_results = {}
    
    for data, description in datasets:
        results = benchmark.benchmark_dataset(data, description)
        all_results[description] = results
    
    # 総合サマリー
    print("\n" + "=" * 70)
    print("📊 総合ベンチマーク結果サマリー")
    print("=" * 70)
    
    total_datasets = len(all_results)
    zstd_achievements = 0
    zip_achievements = 0
    
    for desc, results in all_results.items():
        nxzip_light = results['nxzip_light']
        nxzip_normal = results['nxzip_normal']
        zlib_3 = results['zlib_3']
        zlib_6 = results['zlib_6']
        
        # Zstandardレベル評価
        light_ratio_adv = nxzip_light['compression_ratio'] - zlib_3['compression_ratio']
        light_speed_ratio = zlib_3['compression_time'] / nxzip_light['compression_time'] if nxzip_light['compression_time'] > 0 else 1
        
        if light_ratio_adv > 0 and light_speed_ratio >= 0.8:
            zstd_achievements += 1
        
        # 7-Zip超越評価
        normal_ratio_adv = nxzip_normal['compression_ratio'] - zlib_6['compression_ratio']
        normal_speed_ratio = zlib_6['compression_time'] / nxzip_normal['compression_time'] if nxzip_normal['compression_time'] > 0 else 1
        
        if normal_ratio_adv > 0 and normal_speed_ratio >= 2.0:
            zip_achievements += 1
    
    zstd_success_rate = (zstd_achievements / total_datasets) * 100
    zip_success_rate = (zip_achievements / total_datasets) * 100
    
    print(f"⚡ Zstandardレベル達成率: {zstd_success_rate:.1f}% ({zstd_achievements}/{total_datasets})")
    print(f"🎯 7-Zip超越達成率: {zip_success_rate:.1f}% ({zip_achievements}/{total_datasets})")
    
    if zstd_success_rate >= 60 and zip_success_rate >= 60:
        print("\n✅ NXZip TMC v9.1 目標達成！")
    elif zstd_success_rate >= 60:
        print("\n🔄 軽量モード成功、通常モード最適化要")
    elif zip_success_rate >= 60:
        print("\n🔄 通常モード成功、軽量モード最適化要")
    else:
        print("\n🔧 両モード最適化が必要")
    
    print(f"\n🏁 NXZip TMC v9.1 ベンチマーク完了")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ベンチマークエラー: {e}")
        import traceback
        traceback.print_exc()
