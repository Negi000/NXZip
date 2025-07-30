#!/usr/bin/env python3
"""
TMC v6.0 vs 標準圧縮アルゴリズム ベンチマークテスト
7-Zip, Zstandard, LZMA2, LZ4等との詳細比較評価
"""

import os
import sys
import time
import subprocess
import tempfile
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# TMC Engine インポート
sys.path.append(os.path.join(os.path.dirname(__file__), 'nxzip', 'engine'))
from nxzip.engine.nexus_tmc_v4_unified import NEXUSTMCEngineV4

# 各種圧縮ライブラリのインポート
try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False

import zlib
import lzma
import bz2


class CompressionBenchmark:
    """圧縮アルゴリズム総合ベンチマーク"""
    
    def __init__(self):
        self.tmc_engine = NEXUSTMCEngineV4()
        self.results = []
        
        # 利用可能な圧縮器をチェック
        self.compressors = self._init_compressors()
        print(f"🔧 利用可能な圧縮器: {list(self.compressors.keys())}")
        
        # 7-Zipの利用可能性をチェック
        self.seven_zip_available = self._check_7zip_availability()
        if self.seven_zip_available:
            print("🔧 7-Zip利用可能")
        else:
            print("⚠️ 7-Zip未検出 - 内蔵アルゴリズムのみでテスト")
    
    def _init_compressors(self) -> Dict[str, callable]:
        """利用可能な圧縮器を初期化"""
        compressors = {
            'TMC_v6.0': self._compress_tmc,
            'LZMA2': self._compress_lzma2,
            'Zlib_Default': self._compress_zlib_default,
            'Zlib_Best': self._compress_zlib_best,
            'BZ2': self._compress_bz2,
        }
        
        if ZSTD_AVAILABLE:
            compressors.update({
                'Zstd_Fast': self._compress_zstd_fast,
                'Zstd_Default': self._compress_zstd_default,
                'Zstd_Best': self._compress_zstd_best,
                'Zstd_Ultra': self._compress_zstd_ultra,
            })
        
        if LZ4_AVAILABLE:
            compressors.update({
                'LZ4_Fast': self._compress_lz4_fast,
                'LZ4_HC': self._compress_lz4_hc,
            })
        
        return compressors
    
    def _check_7zip_availability(self) -> bool:
        """7-Zipの利用可能性をチェック"""
        try:
            result = subprocess.run(['7z'], capture_output=True, timeout=5)
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            try:
                # Windows環境での別パスをチェック
                result = subprocess.run(['7za'], capture_output=True, timeout=5)
                return True
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
                return False
    
    def generate_test_datasets(self) -> Dict[str, bytes]:
        """多様なテストデータセットを生成"""
        datasets = {}
        
        # 1. 系列整数データ（TMCが得意）
        sequential_ints = np.arange(0, 10000, dtype=np.int32)
        datasets['Sequential_Integers_40KB'] = sequential_ints.tobytes()
        
        # 2. 浮動小数点数値データ
        float_data = np.linspace(0, 1000, 10000, dtype=np.float32)
        datasets['Float_Numbers_40KB'] = float_data.tobytes()
        
        # 3. 構造化数値データ（TMCの多モデルが有効）
        structured_nums = []
        for i in range(2500):
            structured_nums.extend([i, i*2, i*i % 1000, (i*3) % 100])
        datasets['Structured_Numbers_40KB'] = np.array(structured_nums, dtype=np.int32).tobytes()
        
        # 4. テキストデータ（BWTが有効）
        text_content = """
        The quick brown fox jumps over the lazy dog. This is a sample text for compression testing.
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
        In publishing and graphic design, Lorem ipsum is a placeholder text commonly used to demonstrate the visual form of a document.
        """ * 200
        datasets['Text_Data_40KB'] = text_content.encode('utf-8')[:40960]
        
        # 5. 高反復データ
        repetitive_data = b"ABCDEFGH" * 5120
        datasets['Repetitive_Binary_40KB'] = repetitive_data
        
        # 6. ランダムデータ（圧縮困難）
        np.random.seed(42)
        random_data = np.random.randint(0, 256, 40960, dtype=np.uint8)
        datasets['Random_Data_40KB'] = random_data.tobytes()
        
        # 7. 実データ風（画像ライクデータ）
        image_like = []
        for y in range(200):
            for x in range(200):
                # 簡易的な画像パターン
                pixel = int(128 + 64 * np.sin(x/20) * np.cos(y/20))
                image_like.append(max(0, min(255, pixel)))
        datasets['Image_Like_40KB'] = bytes(image_like)
        
        # 8. 時系列データ
        time_series = []
        for i in range(10000):
            value = 100 + 50 * np.sin(i/100) + 10 * np.random.randn()
            time_series.append(int(max(0, min(65535, value))))
        datasets['Time_Series_40KB'] = np.array(time_series, dtype=np.uint16).tobytes()
        
        print(f"📊 生成されたテストデータセット: {len(datasets)}種類")
        for name, data in datasets.items():
            print(f"   {name}: {len(data)} bytes")
        
        return datasets
    
    def benchmark_single_dataset(self, name: str, data: bytes) -> Dict[str, Any]:
        """単一データセットの圧縮ベンチマーク"""
        print(f"\n🧪 ベンチマークテスト: {name} ({len(data)} bytes)")
        print("=" * 60)
        
        results = []
        
        # 各圧縮器でテスト
        for comp_name, compressor in self.compressors.items():
            try:
                start_time = time.perf_counter()
                compressed_data, decompressed_data = compressor(data)
                end_time = time.perf_counter()
                
                compression_time = end_time - start_time
                
                # 正確性検証
                if decompressed_data != data:
                    print(f"❌ {comp_name}: データ整合性エラー")
                    continue
                
                # 統計計算
                original_size = len(data)
                compressed_size = len(compressed_data)
                compression_ratio = (1 - compressed_size / original_size) * 100
                compression_speed_mbps = (original_size / (1024 * 1024)) / compression_time if compression_time > 0 else 0
                
                result = {
                    'compressor': comp_name,
                    'original_size': original_size,
                    'compressed_size': compressed_size,
                    'compression_ratio': compression_ratio,
                    'compression_time': compression_time,
                    'compression_speed_mbps': compression_speed_mbps,
                    'successful': True
                }
                
                results.append(result)
                
                print(f"✅ {comp_name:15s}: {original_size:6d} -> {compressed_size:6d} bytes "
                      f"({compression_ratio:6.2f}%) {compression_time*1000:6.1f}ms "
                      f"({compression_speed_mbps:5.1f} MB/s)")
                
            except Exception as e:
                print(f"❌ {comp_name}: エラー - {str(e)}")
                results.append({
                    'compressor': comp_name,
                    'successful': False,
                    'error': str(e)
                })
        
        # 7-Zipテスト（利用可能な場合）
        if self.seven_zip_available:
            try:
                result_7z = self._test_7zip(data, name)
                if result_7z:
                    results.append(result_7z)
                    print(f"✅ {'7-Zip':15s}: {result_7z['original_size']:6d} -> {result_7z['compressed_size']:6d} bytes "
                          f"({result_7z['compression_ratio']:6.2f}%) {result_7z['compression_time']*1000:6.1f}ms "
                          f"({result_7z['compression_speed_mbps']:5.1f} MB/s)")
            except Exception as e:
                print(f"❌ 7-Zip: エラー - {str(e)}")
        
        return {
            'dataset_name': name,
            'dataset_size': len(data),
            'results': results
        }
    
    def _compress_tmc(self, data: bytes) -> Tuple[bytes, bytes]:
        """TMC v6.0圧縮"""
        compressed, _ = self.tmc_engine.compress_tmc(data)
        decompressed, _ = self.tmc_engine.decompress_tmc(compressed)
        return compressed, decompressed
    
    def _compress_lzma2(self, data: bytes) -> Tuple[bytes, bytes]:
        """LZMA2圧縮"""
        compressed = lzma.compress(data, preset=6)
        decompressed = lzma.decompress(compressed)
        return compressed, decompressed
    
    def _compress_zlib_default(self, data: bytes) -> Tuple[bytes, bytes]:
        """Zlib デフォルト圧縮"""
        compressed = zlib.compress(data, level=6)
        decompressed = zlib.decompress(compressed)
        return compressed, decompressed
    
    def _compress_zlib_best(self, data: bytes) -> Tuple[bytes, bytes]:
        """Zlib 最高圧縮"""
        compressed = zlib.compress(data, level=9)
        decompressed = zlib.decompress(compressed)
        return compressed, decompressed
    
    def _compress_bz2(self, data: bytes) -> Tuple[bytes, bytes]:
        """BZ2圧縮"""
        compressed = bz2.compress(data, compresslevel=9)
        decompressed = bz2.decompress(compressed)
        return compressed, decompressed
    
    def _compress_zstd_fast(self, data: bytes) -> Tuple[bytes, bytes]:
        """Zstd 高速圧縮"""
        compressor = zstd.ZstdCompressor(level=1)
        decompressor = zstd.ZstdDecompressor()
        compressed = compressor.compress(data)
        decompressed = decompressor.decompress(compressed)
        return compressed, decompressed
    
    def _compress_zstd_default(self, data: bytes) -> Tuple[bytes, bytes]:
        """Zstd デフォルト圧縮"""
        compressor = zstd.ZstdCompressor(level=3)
        decompressor = zstd.ZstdDecompressor()
        compressed = compressor.compress(data)
        decompressed = decompressor.decompress(compressed)
        return compressed, decompressed
    
    def _compress_zstd_best(self, data: bytes) -> Tuple[bytes, bytes]:
        """Zstd 高圧縮"""
        compressor = zstd.ZstdCompressor(level=19)
        decompressor = zstd.ZstdDecompressor()
        compressed = compressor.compress(data)
        decompressed = decompressor.decompress(compressed)
        return compressed, decompressed
    
    def _compress_zstd_ultra(self, data: bytes) -> Tuple[bytes, bytes]:
        """Zstd 超高圧縮"""
        compressor = zstd.ZstdCompressor(level=22)
        decompressor = zstd.ZstdDecompressor()
        compressed = compressor.compress(data)
        decompressed = decompressor.decompress(compressed)
        return compressed, decompressed
    
    def _compress_lz4_fast(self, data: bytes) -> Tuple[bytes, bytes]:
        """LZ4 高速圧縮"""
        compressed = lz4.frame.compress(data, compression_level=1)
        decompressed = lz4.frame.decompress(compressed)
        return compressed, decompressed
    
    def _compress_lz4_hc(self, data: bytes) -> Tuple[bytes, bytes]:
        """LZ4 高圧縮"""
        compressed = lz4.frame.compress(data, compression_level=12)
        decompressed = lz4.frame.decompress(compressed)
        return compressed, decompressed
    
    def _test_7zip(self, data: bytes, dataset_name: str) -> Optional[Dict[str, Any]]:
        """7-Zipによる圧縮テスト"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                input_file = os.path.join(temp_dir, f"{dataset_name}.bin")
                output_file = os.path.join(temp_dir, f"{dataset_name}.7z")
                
                # データをファイルに書き込み
                with open(input_file, 'wb') as f:
                    f.write(data)
                
                # 7-Zip圧縮
                start_time = time.perf_counter()
                cmd = ['7z', 'a', '-t7z', '-mx=9', output_file, input_file]
                try:
                    result = subprocess.run(cmd, capture_output=True, timeout=60, check=True)
                except FileNotFoundError:
                    # 7zaを試行
                    cmd[0] = '7za'
                    result = subprocess.run(cmd, capture_output=True, timeout=60, check=True)
                
                compression_time = time.perf_counter() - start_time
                
                # 圧縮サイズを取得
                if os.path.exists(output_file):
                    compressed_size = os.path.getsize(output_file)
                    original_size = len(data)
                    compression_ratio = (1 - compressed_size / original_size) * 100
                    compression_speed_mbps = (original_size / (1024 * 1024)) / compression_time if compression_time > 0 else 0
                    
                    return {
                        'compressor': '7-Zip',
                        'original_size': original_size,
                        'compressed_size': compressed_size,
                        'compression_ratio': compression_ratio,
                        'compression_time': compression_time,
                        'compression_speed_mbps': compression_speed_mbps,
                        'successful': True
                    }
                
        except Exception as e:
            print(f"7-Zipテストエラー: {e}")
            return None
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """総合ベンチマークの実行"""
        print("🚀 TMC v6.0 vs 標準圧縮アルゴリズム 総合ベンチマーク開始")
        print("=" * 80)
        
        # テストデータセット生成
        datasets = self.generate_test_datasets()
        
        # 各データセットでベンチマーク実行
        benchmark_results = []
        
        for dataset_name, dataset_data in datasets.items():
            result = self.benchmark_single_dataset(dataset_name, dataset_data)
            benchmark_results.append(result)
        
        # 総合統計を計算
        summary = self._calculate_summary_statistics(benchmark_results)
        
        # 結果をレポート
        self._generate_benchmark_report(benchmark_results, summary)
        
        return {
            'benchmark_results': benchmark_results,
            'summary_statistics': summary,
            'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'compressors_tested': list(self.compressors.keys()) + (['7-Zip'] if self.seven_zip_available else [])
        }
    
    def _calculate_summary_statistics(self, benchmark_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """総合統計の計算"""
        compressor_stats = {}
        
        for dataset_result in benchmark_results:
            for result in dataset_result['results']:
                if not result.get('successful', False):
                    continue
                
                comp_name = result['compressor']
                if comp_name not in compressor_stats:
                    compressor_stats[comp_name] = {
                        'total_original_size': 0,
                        'total_compressed_size': 0,
                        'total_compression_time': 0,
                        'compression_ratios': [],
                        'compression_speeds': [],
                        'datasets_tested': 0
                    }
                
                stats = compressor_stats[comp_name]
                stats['total_original_size'] += result['original_size']
                stats['total_compressed_size'] += result['compressed_size']
                stats['total_compression_time'] += result['compression_time']
                stats['compression_ratios'].append(result['compression_ratio'])
                stats['compression_speeds'].append(result['compression_speed_mbps'])
                stats['datasets_tested'] += 1
        
        # 平均値等を計算
        summary = {}
        for comp_name, stats in compressor_stats.items():
            if stats['datasets_tested'] > 0:
                summary[comp_name] = {
                    'datasets_tested': stats['datasets_tested'],
                    'total_original_size': stats['total_original_size'],
                    'total_compressed_size': stats['total_compressed_size'],
                    'overall_compression_ratio': (1 - stats['total_compressed_size'] / stats['total_original_size']) * 100,
                    'average_compression_ratio': np.mean(stats['compression_ratios']),
                    'average_compression_speed': np.mean(stats['compression_speeds']),
                    'total_compression_time': stats['total_compression_time']
                }
        
        return summary
    
    def _generate_benchmark_report(self, benchmark_results: List[Dict[str, Any]], summary: Dict[str, Any]):
        """ベンチマーク結果レポートの生成"""
        print("\n")
        print("🏆 TMC v6.0 ベンチマーク総合結果レポート")
        print("=" * 80)
        
        # 圧縮率ランキング
        print("\n📊 総合圧縮率ランキング:")
        print("-" * 50)
        sorted_by_ratio = sorted(summary.items(), key=lambda x: x[1]['overall_compression_ratio'], reverse=True)
        
        for rank, (comp_name, stats) in enumerate(sorted_by_ratio, 1):
            print(f"{rank:2d}. {comp_name:15s}: {stats['overall_compression_ratio']:6.2f}% "
                  f"(平均: {stats['average_compression_ratio']:6.2f}%)")
        
        # 速度ランキング
        print("\n⚡ 総合圧縮速度ランキング:")
        print("-" * 50)
        sorted_by_speed = sorted(summary.items(), key=lambda x: x[1]['average_compression_speed'], reverse=True)
        
        for rank, (comp_name, stats) in enumerate(sorted_by_speed, 1):
            print(f"{rank:2d}. {comp_name:15s}: {stats['average_compression_speed']:7.1f} MB/s "
                  f"(総時間: {stats['total_compression_time']*1000:6.1f}ms)")
        
        # TMC v6.0の性能分析
        if 'TMC_v6.0' in summary:
            tmc_stats = summary['TMC_v6.0']
            print(f"\n🔥 TMC v6.0 詳細分析:")
            print("-" * 30)
            print(f"   総合圧縮率: {tmc_stats['overall_compression_ratio']:.2f}%")
            print(f"   平均圧縮率: {tmc_stats['average_compression_ratio']:.2f}%")
            print(f"   平均圧縮速度: {tmc_stats['average_compression_speed']:.1f} MB/s")
            print(f"   処理データ量: {tmc_stats['total_original_size']/1024:.1f} KB")
            
            # 他の圧縮器との比較
            print(f"\n📈 TMC v6.0 vs 他の圧縮器:")
            print("-" * 35)
            
            for comp_name, stats in summary.items():
                if comp_name == 'TMC_v6.0':
                    continue
                
                ratio_diff = tmc_stats['overall_compression_ratio'] - stats['overall_compression_ratio']
                speed_ratio = tmc_stats['average_compression_speed'] / stats['average_compression_speed']
                
                if ratio_diff > 0:
                    ratio_status = f"+{ratio_diff:.2f}% 🔥"
                else:
                    ratio_status = f"{ratio_diff:.2f}%"
                
                if speed_ratio > 1:
                    speed_status = f"×{speed_ratio:.1f} 🚀"
                else:
                    speed_status = f"÷{1/speed_ratio:.1f}"
                
                print(f"   vs {comp_name:12s}: 圧縮率 {ratio_status}, 速度 {speed_status}")
        
        # データタイプ別の最適圧縮器
        print(f"\n🎯 データタイプ別最適圧縮器:")
        print("-" * 40)
        
        for dataset_result in benchmark_results:
            dataset_name = dataset_result['dataset_name']
            best_ratio = max(
                (r for r in dataset_result['results'] if r.get('successful', False)),
                key=lambda x: x['compression_ratio'],
                default=None
            )
            
            if best_ratio:
                print(f"   {dataset_name:25s}: {best_ratio['compressor']:15s} ({best_ratio['compression_ratio']:6.2f}%)")


def main():
    """メイン実行関数"""
    print("🧪 TMC v6.0 総合ベンチマークテスト")
    
    # ベンチマーク実行
    benchmark = CompressionBenchmark()
    results = benchmark.run_comprehensive_benchmark()
    
    # 結果をJSONファイルに保存
    output_file = "tmc_v6_benchmark_results.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n💾 詳細結果を {output_file} に保存しました")
    except Exception as e:
        print(f"⚠️ 結果保存エラー: {e}")
    
    print("\n🎉 ベンチマークテスト完了!")


if __name__ == "__main__":
    main()
