#!/usr/bin/env python3
"""
NEXUS TMC v2.0 vs 競合最終ベンチマーク
最適化済みTMCエンジンの性能評価
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import statistics

# 最適化TMCエンジンインポート
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from nexus_tmc_engine import NEXUSTMCEngine
    TMC_OPTIMIZED_AVAILABLE = True
    print("🚀 NEXUS TMC Engine v2.0 最適化版 準備完了")
except ImportError:
    print("⚠️ NEXUS TMC Engine v2.0 が利用できません")
    TMC_OPTIMIZED_AVAILABLE = False

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
    print("⚡ Zstandard Engine 準備完了")
except ImportError:
    print("⚠️ Zstandard ライブラリが利用できません")
    ZSTD_AVAILABLE = False

import lzma
import zlib
import bz2


class TMCOptimizedCompetitor:
    """TMC v2.0 最適化版競合テスト"""
    
    def __init__(self):
        self.name = "TMC-v2.0-Optimized"
        if TMC_OPTIMIZED_AVAILABLE:
            self.engine = NEXUSTMCEngine(max_workers=4)
        else:
            self.engine = None
    
    def compress(self, data: bytes, level: int = 6) -> Tuple[bytes, Dict[str, Any]]:
        """TMC v2.0圧縮テスト"""
        if not self.engine:
            return data, {'error': 'tmc_engine_not_available'}
        
        try:
            compressed_data, compression_info = self.engine.compress_tmc(data)
            
            result_info = {
                'original_size': len(data),
                'compressed_size': len(compressed_data),
                'compression_ratio': compression_info['compression_ratio'],
                'compression_time': compression_info['total_compression_time'],
                'compression_throughput_mb_s': compression_info['compression_throughput_mb_s'],
                'data_type': compression_info['data_type'],
                'transform_method': compression_info['transform_info']['transform_method'],
                'reversible': compression_info['reversible'],
                'format': 'tmc_v2'
            }
            
            return compressed_data, result_info
            
        except Exception as e:
            return data, {
                'error': str(e),
                'original_size': len(data)
            }
    
    def decompress(self, data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """TMC v2.0展開テスト"""
        if not self.engine:
            return data, {'error': 'tmc_engine_not_available'}
        
        try:
            decompressed_data, decompression_info = self.engine.decompress_tmc(data)
            
            return decompressed_data, decompression_info
            
        except Exception as e:
            return data, {'error': str(e)}
    
    def test_reversibility(self, data: bytes) -> Dict[str, Any]:
        """可逆性テスト"""
        if not self.engine:
            return {'error': 'tmc_engine_not_available'}
        
        try:
            return self.engine.test_reversibility(data, "benchmark_test")
        except Exception as e:
            return {'error': str(e)}


class StandardCompetitors:
    """標準圧縮アルゴリズム競合"""
    
    @staticmethod
    def test_zstandard(data: bytes, level: int = 6) -> Dict[str, Any]:
        """Zstandard性能テスト"""
        if not ZSTD_AVAILABLE:
            return {'error': 'zstd_not_available'}
        
        try:
            # 圧縮
            compression_start = time.perf_counter()
            cctx = zstd.ZstdCompressor(level=level)
            compressed = cctx.compress(data)
            compression_time = time.perf_counter() - compression_start
            
            # 展開
            decompression_start = time.perf_counter()
            dctx = zstd.ZstdDecompressor()
            decompressed = dctx.decompress(compressed)
            decompression_time = time.perf_counter() - decompression_start
            
            # 可逆性チェック
            reversible = (data == decompressed)
            
            return {
                'algorithm': 'Zstandard',
                'level': level,
                'original_size': len(data),
                'compressed_size': len(compressed),
                'compression_ratio': (1 - len(compressed) / len(data)) * 100,
                'compression_time': compression_time,
                'decompression_time': decompression_time,
                'compression_throughput_mb_s': (len(data) / 1024 / 1024) / compression_time if compression_time > 0 else 0,
                'decompression_throughput_mb_s': (len(decompressed) / 1024 / 1024) / decompression_time if decompression_time > 0 else 0,
                'reversible': reversible
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def test_lzma(data: bytes, preset: int = 6) -> Dict[str, Any]:
        """LZMA性能テスト"""
        try:
            # 圧縮
            compression_start = time.perf_counter()
            compressed = lzma.compress(data, preset=preset)
            compression_time = time.perf_counter() - compression_start
            
            # 展開
            decompression_start = time.perf_counter()
            decompressed = lzma.decompress(compressed)
            decompression_time = time.perf_counter() - decompression_start
            
            # 可逆性チェック
            reversible = (data == decompressed)
            
            return {
                'algorithm': 'LZMA',
                'preset': preset,
                'original_size': len(data),
                'compressed_size': len(compressed),
                'compression_ratio': (1 - len(compressed) / len(data)) * 100,
                'compression_time': compression_time,
                'decompression_time': decompression_time,
                'compression_throughput_mb_s': (len(data) / 1024 / 1024) / compression_time if compression_time > 0 else 0,
                'decompression_throughput_mb_s': (len(decompressed) / 1024 / 1024) / decompression_time if decompression_time > 0 else 0,
                'reversible': reversible
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def test_zlib(data: bytes, level: int = 6) -> Dict[str, Any]:
        """Zlib性能テスト"""
        try:
            # 圧縮
            compression_start = time.perf_counter()
            compressed = zlib.compress(data, level=level)
            compression_time = time.perf_counter() - compression_start
            
            # 展開
            decompression_start = time.perf_counter()
            decompressed = zlib.decompress(compressed)
            decompression_time = time.perf_counter() - decompression_start
            
            # 可逆性チェック
            reversible = (data == decompressed)
            
            return {
                'algorithm': 'Zlib',
                'level': level,
                'original_size': len(data),
                'compressed_size': len(compressed),
                'compression_ratio': (1 - len(compressed) / len(data)) * 100,
                'compression_time': compression_time,
                'decompression_time': decompression_time,
                'compression_throughput_mb_s': (len(data) / 1024 / 1024) / compression_time if compression_time > 0 else 0,
                'decompression_throughput_mb_s': (len(decompressed) / 1024 / 1024) / decompression_time if decompression_time > 0 else 0,
                'reversible': reversible
            }
            
        except Exception as e:
            return {'error': str(e)}


class TMCOptimizedBenchmark:
    """TMC v2.0最適化ベンチマーク"""
    
    def __init__(self):
        self.tmc_competitor = TMCOptimizedCompetitor()
        
    def generate_benchmark_datasets(self) -> Dict[str, bytes]:
        """ベンチマークデータセット生成"""
        datasets = {}
        
        # 1. テキストデータ（高反復）
        text_content = (
            "The NEXUS TMC Engine v2.0 represents revolutionary compression technology. "
            "Transform-Model-Code algorithms provide superior data structure understanding. "
            "Optimized differential encoding and wavelet transforms achieve exceptional compression ratios. "
            "Parallel processing ensures high-speed compression and decompression. "
        ) * 500
        datasets['optimized_text'] = text_content.encode('utf-8')
        
        # 2. 構造化データ（数値パターン）
        structured_pattern = []
        for i in range(2000):
            structured_pattern.extend([
                i & 0xFF,           # 低バイト
                (i >> 8) & 0xFF,    # 高バイト
                i % 256,            # 剰余
                (i * 3) & 0xFF      # 3倍値
            ])
        datasets['structured_numeric'] = bytes(structured_pattern)
        
        # 3. 時系列風データ
        time_series = []
        base_value = 128
        for i in range(4000):
            noise = (i % 7) - 3  # -3から+3のノイズ
            base_value = max(0, min(255, base_value + noise))
            time_series.append(base_value)
        datasets['time_series'] = bytes(time_series)
        
        # 4. JSON風構造データ
        json_content = '{"records": ['
        for i in range(1000):
            json_content += f'{{"id": {i}, "timestamp": "2024-01-{i%30+1:02d}", "value": {i*1.234:.3f}, "status": "{["active", "inactive"][i%2]}", "metadata": {{"category": "test", "priority": {i%5}}}}}'
            if i < 999:
                json_content += ', '
        json_content += ']}'
        datasets['json_structured'] = json_content.encode('utf-8')
        
        # 5. 混合バイナリデータ
        mixed_binary = bytearray()
        # ヘッダー部分（構造的）
        mixed_binary.extend(b'NEXUS_TMC_v2.0_BINARY_HEADER')
        mixed_binary.extend(struct.pack('<IIII', 0x12345678, len(datasets['optimized_text']), 42, 0xABCDEF))
        # データ部分（パターン性あり）
        for i in range(1000):
            mixed_binary.extend([(i * 7) % 256, (i * 11) % 256, (i * 13) % 256, i % 256])
        datasets['mixed_binary'] = bytes(mixed_binary)
        
        return datasets
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """包括的ベンチマーク実行"""
        print("🏁 TMC v2.0 最適化版 vs 競合 包括ベンチマーク開始")
        print("=" * 80)
        
        datasets = self.generate_benchmark_datasets()
        results = {}
        
        for dataset_name, test_data in datasets.items():
            print(f"\n📊 データセット: {dataset_name}")
            print(f"   サイズ: {len(test_data):,} bytes ({len(test_data)/1024:.1f} KB)")
            print("-" * 60)
            
            dataset_results = {}
            
            # TMC v2.0テスト
            print("🚀 TMC v2.0 Optimized:")
            tmc_result = self.tmc_competitor.test_reversibility(test_data)
            
            if 'error' not in tmc_result:
                print(f"   圧縮率: {tmc_result['compression_ratio']:6.2f}% | "
                      f"圧縮速度: {tmc_result['compression_throughput_mb_s']:5.1f}MB/s | "
                      f"展開速度: {tmc_result['decompression_throughput_mb_s']:5.1f}MB/s | "
                      f"可逆性: {'✅' if tmc_result['reversible'] else '❌'}")
                dataset_results['TMC_v2.0'] = tmc_result
            else:
                print(f"   ❌ エラー: {tmc_result['error']}")
                dataset_results['TMC_v2.0'] = tmc_result
            
            # Zstandard テスト
            print("⚡ Zstandard:")
            zstd_result = StandardCompetitors.test_zstandard(test_data, level=6)
            
            if 'error' not in zstd_result:
                print(f"   圧縮率: {zstd_result['compression_ratio']:6.2f}% | "
                      f"圧縮速度: {zstd_result['compression_throughput_mb_s']:5.1f}MB/s | "
                      f"展開速度: {zstd_result['decompression_throughput_mb_s']:5.1f}MB/s | "
                      f"可逆性: {'✅' if zstd_result['reversible'] else '❌'}")
                dataset_results['Zstandard'] = zstd_result
            else:
                print(f"   ❌ エラー: {zstd_result['error']}")
                dataset_results['Zstandard'] = zstd_result
            
            # LZMA テスト
            print("🗜️  LZMA:")
            lzma_result = StandardCompetitors.test_lzma(test_data, preset=6)
            
            if 'error' not in lzma_result:
                print(f"   圧縮率: {lzma_result['compression_ratio']:6.2f}% | "
                      f"圧縮速度: {lzma_result['compression_throughput_mb_s']:5.1f}MB/s | "
                      f"展開速度: {lzma_result['decompression_throughput_mb_s']:5.1f}MB/s | "
                      f"可逆性: {'✅' if lzma_result['reversible'] else '❌'}")
                dataset_results['LZMA'] = lzma_result
            else:
                print(f"   ❌ エラー: {lzma_result['error']}")
                dataset_results['LZMA'] = lzma_result
            
            # Zlib テスト
            print("📦 Zlib:")
            zlib_result = StandardCompetitors.test_zlib(test_data, level=6)
            
            if 'error' not in zlib_result:
                print(f"   圧縮率: {zlib_result['compression_ratio']:6.2f}% | "
                      f"圧縮速度: {zlib_result['compression_throughput_mb_s']:5.1f}MB/s | "
                      f"展開速度: {zlib_result['decompression_throughput_mb_s']:5.1f}MB/s | "
                      f"可逆性: {'✅' if zlib_result['reversible'] else '❌'}")
                dataset_results['Zlib'] = zlib_result
            else:
                print(f"   ❌ エラー: {zlib_result['error']}")
                dataset_results['Zlib'] = zlib_result
            
            results[dataset_name] = dataset_results
        
        # 総合分析
        analysis = self._analyze_results(results)
        
        return {
            'detailed_results': results,
            'analysis': analysis,
            'benchmark_timestamp': time.time(),
            'tmc_version': 'v2.0_optimized'
        }
    
    def _analyze_results(self, results: Dict) -> Dict[str, Any]:
        """結果分析"""
        analysis = {
            'algorithm_averages': {},
            'category_winners': {},
            'tmc_performance': {}
        }
        
        try:
            # アルゴリズム別平均計算
            algorithms = ['TMC_v2.0', 'Zstandard', 'LZMA', 'Zlib']
            
            for algorithm in algorithms:
                compression_ratios = []
                compression_speeds = []
                decompression_speeds = []
                reversibility_count = 0
                total_tests = 0
                
                for dataset_results in results.values():
                    if algorithm in dataset_results and 'error' not in dataset_results[algorithm]:
                        result = dataset_results[algorithm]
                        
                        compression_ratios.append(result.get('compression_ratio', 0))
                        compression_speeds.append(result.get('compression_throughput_mb_s', 0))
                        decompression_speeds.append(result.get('decompression_throughput_mb_s', 0))
                        
                        if result.get('reversible', False):
                            reversibility_count += 1
                        total_tests += 1
                
                if compression_ratios:
                    analysis['algorithm_averages'][algorithm] = {
                        'avg_compression_ratio': statistics.mean(compression_ratios),
                        'avg_compression_speed': statistics.mean(compression_speeds),
                        'avg_decompression_speed': statistics.mean(decompression_speeds),
                        'reversibility_rate': (reversibility_count / total_tests * 100) if total_tests > 0 else 0,
                        'test_count': total_tests
                    }
            
            # カテゴリー別勝者
            if analysis['algorithm_averages']:
                # 最高圧縮率
                best_compression = max(analysis['algorithm_averages'].items(),
                                     key=lambda x: x[1]['avg_compression_ratio'])
                analysis['category_winners']['best_compression'] = {
                    'algorithm': best_compression[0],
                    'ratio': best_compression[1]['avg_compression_ratio']
                }
                
                # 最高圧縮速度
                best_compression_speed = max(analysis['algorithm_averages'].items(),
                                           key=lambda x: x[1]['avg_compression_speed'])
                analysis['category_winners']['best_compression_speed'] = {
                    'algorithm': best_compression_speed[0],
                    'speed': best_compression_speed[1]['avg_compression_speed']
                }
                
                # 最高展開速度
                best_decompression_speed = max(analysis['algorithm_averages'].items(),
                                             key=lambda x: x[1]['avg_decompression_speed'])
                analysis['category_winners']['best_decompression_speed'] = {
                    'algorithm': best_decompression_speed[0],
                    'speed': best_decompression_speed[1]['avg_decompression_speed']
                }
                
                # TMC特別分析
                if 'TMC_v2.0' in analysis['algorithm_averages']:
                    tmc_data = analysis['algorithm_averages']['TMC_v2.0']
                    analysis['tmc_performance'] = {
                        'compression_rank': self._get_rank(analysis['algorithm_averages'], 'avg_compression_ratio', 'TMC_v2.0'),
                        'compression_speed_rank': self._get_rank(analysis['algorithm_averages'], 'avg_compression_speed', 'TMC_v2.0'),
                        'decompression_speed_rank': self._get_rank(analysis['algorithm_averages'], 'avg_decompression_speed', 'TMC_v2.0'),
                        'reversibility_rate': tmc_data['reversibility_rate'],
                        'overall_performance': tmc_data
                    }
        
        except Exception as e:
            analysis['error'] = str(e)
        
        return analysis
    
    def _get_rank(self, data: Dict, metric: str, target_algorithm: str) -> int:
        """指定メトリックでのランキング取得"""
        try:
            sorted_algorithms = sorted(data.items(), 
                                     key=lambda x: x[1][metric], 
                                     reverse=True)
            
            for rank, (algorithm, _) in enumerate(sorted_algorithms, 1):
                if algorithm == target_algorithm:
                    return rank
            
            return len(data)  # 見つからない場合は最下位
            
        except Exception:
            return 0
    
    def print_final_analysis(self, results: Dict[str, Any]):
        """最終分析レポート"""
        print("\n" + "=" * 80)
        print("🏆 TMC v2.0 最適化版 vs 競合 最終分析結果")
        print("=" * 80)
        
        analysis = results.get('analysis', {})
        
        # アルゴリズム別平均パフォーマンス
        print("\n📈 アルゴリズム別平均パフォーマンス:")
        averages = analysis.get('algorithm_averages', {})
        
        for algorithm, metrics in averages.items():
            icon = "🚀" if algorithm == "TMC_v2.0" else "⚡" if algorithm == "Zstandard" else "🗜️" if algorithm == "LZMA" else "📦"
            print(f"   {icon} {algorithm:12}: "
                  f"圧縮率 {metrics['avg_compression_ratio']:5.1f}% | "
                  f"圧縮速度 {metrics['avg_compression_speed']:5.1f}MB/s | "
                  f"展開速度 {metrics['avg_decompression_speed']:5.1f}MB/s | "
                  f"可逆性 {metrics['reversibility_rate']:5.1f}%")
        
        # カテゴリー別勝者
        print("\n🏅 カテゴリー別勝者:")
        winners = analysis.get('category_winners', {})
        
        if 'best_compression' in winners:
            winner = winners['best_compression']
            print(f"   🗜️  最高圧縮率: {winner['algorithm']} ({winner['ratio']:.1f}%)")
        
        if 'best_compression_speed' in winners:
            winner = winners['best_compression_speed']
            print(f"   ⚡ 最高圧縮速度: {winner['algorithm']} ({winner['speed']:.1f}MB/s)")
        
        if 'best_decompression_speed' in winners:
            winner = winners['best_decompression_speed']
            print(f"   🔄 最高展開速度: {winner['algorithm']} ({winner['speed']:.1f}MB/s)")
        
        # TMC特別レポート
        tmc_perf = analysis.get('tmc_performance', {})
        if tmc_perf:
            print(f"\n🚀 TMC v2.0 特別レポート:")
            print(f"   圧縮率ランキング: {tmc_perf.get('compression_rank', 'N/A')}位")
            print(f"   圧縮速度ランキング: {tmc_perf.get('compression_speed_rank', 'N/A')}位")
            print(f"   展開速度ランキング: {tmc_perf.get('decompression_speed_rank', 'N/A')}位")
            print(f"   可逆性成功率: {tmc_perf.get('reversibility_rate', 0):.1f}%")
        
        print(f"\n🎯 TMC v2.0 革新技術:")
        print(f"   ✓ データ構造自動分析システム")
        print(f"   ✓ 高度な差分符号化（1次・2次対応）")
        print(f"   ✓ ウェーブレット風周波数変換")
        print(f"   ✓ 特性別最適化圧縮戦略")
        print(f"   ✓ 並列処理による高速化")
        print(f"   ✓ 完全可逆性保証")
        print(f"   ✓ 安全なテキスト辞書前処理")


# メイン実行
if __name__ == "__main__":
    print("🔥🔥🔥 TMC v2.0 最適化版 最終ベンチマーク 🔥🔥🔥")
    print("Transform-Model-Code Optimized vs Industry Standards")
    print("=" * 80)
    
    import struct  # 忘れていたimport追加
    
    benchmark = TMCOptimizedBenchmark()
    
    try:
        results = benchmark.run_comprehensive_benchmark()
        benchmark.print_final_analysis(results)
        
        # 結果をJSONで保存
        output_file = Path(current_dir) / "tmc_v2_optimized_benchmark_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 最終ベンチマーク結果を保存: {output_file}")
        
        print(f"\n🏁 TMC v2.0 最適化ベンチマーク完了!")
        
    except Exception as e:
        print(f"\n❌ ベンチマークエラー: {e}")
        import traceback
        traceback.print_exc()
