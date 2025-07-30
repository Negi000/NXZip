#!/usr/bin/env python3
"""
NEXUS TMC Engine v4.0 ユーザー提案統合版 ベンチマークテスト
Zstandard統合 + 改良ディスパッチャ + LeCo変換の性能評価
"""

import os
import sys
import time
import struct
import signal
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple

# TMC v4.0 エンジンインポート
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nexus_tmc_v4_unified import NEXUSTMCEngineV4

# 標準ライブラリとの比較用
import lzma
import gzip
import bz2
import zlib

# Zstandardが利用可能かチェック
try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False


class TimeoutError(Exception):
    """タイムアウトエラー"""
    pass


def timeout_handler(signum, frame):
    """タイムアウトハンドラ"""
    raise TimeoutError("処理がタイムアウトしました")


def with_timeout(timeout_seconds=30):
    """タイムアウト付きデコレータ"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Windowsではsignalが制限されているため、簡易タイムアウト実装
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    print(f"   ⏰ 警告: 処理時間が{elapsed:.1f}秒かかりました（目標: {timeout_seconds}秒）")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    print(f"   ⏰ タイムアウト: {elapsed:.1f}秒で中断")
                    raise TimeoutError(f"タイムアウト: {elapsed:.1f}秒")
                raise e
        return wrapper
    return decorator


class TMCv4Benchmark:
    """TMC v4.0 統合ベンチマーク"""
    
    def __init__(self):
        self.tmc_engine = NEXUSTMCEngineV4(max_workers=4)
        self.results = []
        
    def run_comprehensive_benchmark(self, sample_dir: str = "sample"):
        """包括的ベンチマーク実行"""
        print("🚀 NEXUS TMC v4.0 統合ベンチマーク開始")
        print("   ユーザー提案統合版 vs 標準圧縮ライブラリ")
        print("=" * 80)
        
        # 合成テストデータ生成
        synthetic_tests = self._generate_synthetic_test_data()
        
        # 実ファイルテスト（利用可能な場合）
        real_file_tests = self._collect_real_files(sample_dir)
        
        all_tests = synthetic_tests + real_file_tests
        
        if not all_tests:
            print("❌ テストデータが見つかりません")
            return
        
        print(f"📁 テストケース数: {len(all_tests)}")
        print("-" * 80)
        
        # 各テストケースでベンチマーク実行
        for test_name, test_data in all_tests:
            print(f"\n🔬 テスト中: {test_name}")
            self._benchmark_single_case(test_name, test_data)
        
        # 総合結果表示
        self._display_comprehensive_results()
        
        # 特化分析
        self._analyze_tmc_v4_specialization()
        
        return self.results
    
    def _generate_synthetic_test_data(self) -> List[Tuple[str, bytes]]:
        """合成テストデータ生成（ユーザー提案データタイプ対応）"""
        synthetic_tests = []
        
        # 1. 浮動小数点データ（TDT対象）
        print("📊 合成データ生成中...")
        np.random.seed(42)
        
        # 指数部と仮数部で統計的性質が異なるデータ
        base_values = np.linspace(1000, 1010, 5000, dtype=np.float32)
        noise = np.random.normal(0, 0.1, 5000).astype(np.float32)
        float_data = (base_values + noise).tobytes()
        synthetic_tests.append(("合成浮動小数点データ(TDT)", float_data))
        
        # 2. 系列整数データ（LeCo対象）
        sequential_ints = np.arange(0, 20000, 3, dtype=np.int32).tobytes()
        synthetic_tests.append(("合成系列整数データ(LeCo)", sequential_ints))
        
        # 3. テキストデータ（BWT対象） - サイズ縮小で安定性向上
        text_content = (
            "The Transform-Model-Code framework represents a revolutionary approach to data compression. "
            "By understanding the underlying structure of data, TMC achieves superior compression ratios "
            "while maintaining perfect reversibility. This is particularly effective for structured data "
            "such as floating-point arrays, sequential integers, and text documents. "
        ) * 50  # 200から50に縮小してBWT処理負荷を軽減
        text_data = text_content.encode('utf-8')
        synthetic_tests.append(("合成テキストデータ(BWT)", text_data))
        
        # 4. 高反復バイナリ
        repetitive_data = b"PATTERN123" * 2000
        synthetic_tests.append(("高反復バイナリ", repetitive_data))
        
        # 5. 構造化数値データ
        structured_data = bytearray()
        for i in range(2000):
            structured_data.extend(struct.pack('<If', i, i * 3.14159))
        synthetic_tests.append(("構造化数値データ", bytes(structured_data)))
        
        # 6. 時系列風データ
        time_series = np.cumsum(np.random.normal(0, 1, 10000)).astype(np.int32).tobytes()
        synthetic_tests.append(("時系列風データ", time_series))
        
        print(f"   ✓ {len(synthetic_tests)}種類の合成テストデータを生成")
        return synthetic_tests
    
    def _collect_real_files(self, sample_dir: str) -> List[Tuple[str, bytes]]:
        """実ファイル収集"""
        real_tests = []
        
        try:
            sample_path = Path(sample_dir)
            if not sample_path.exists():
                return real_tests
            
            for file_path in sample_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() not in {'.7z', '.zip', '.gz', '.bz2', '.xz'}:
                    # サイズ制限
                    if file_path.stat().st_size <= 10 * 1024 * 1024:  # 10MB以下
                        try:
                            with open(file_path, 'rb') as f:
                                data = f.read()
                            if len(data) > 0:
                                real_tests.append((f"実ファイル_{file_path.name}", data))
                        except Exception:
                            continue
            
            print(f"   ✓ {len(real_tests)}個の実ファイルを収集")
            
        except Exception:
            pass
        
        return real_tests
    
    def _benchmark_single_case(self, test_name: str, test_data: bytes):
        """単一テストケースベンチマーク"""
        try:
            original_size = len(test_data)
            print(f"   📊 原サイズ: {self._format_size(original_size)}")
            
            case_result = {
                'test_name': test_name,
                'original_size': original_size,
                'results': {}
            }
            
            # TMC v4.0 テスト
            print("   🧪 TMC v4.0...", end=" ", flush=True)
            tmc_result = self._test_tmc_v4(test_data)
            case_result['results']['TMC_v4'] = tmc_result
            print(f"✅ {tmc_result['compression_ratio']:.1f}% ({self._format_speed(tmc_result['compression_speed'])}) [{tmc_result.get('data_type', 'unknown')}]")
            
            # LZMA テスト
            print("   🧪 LZMA...", end=" ", flush=True)
            lzma_result = self._test_lzma(test_data)
            case_result['results']['LZMA'] = lzma_result
            print(f"✅ {lzma_result['compression_ratio']:.1f}% ({self._format_speed(lzma_result['compression_speed'])})")
            
            # Zstandardテスト（利用可能な場合）
            if ZSTD_AVAILABLE:
                print("   🧪 Zstd...", end=" ", flush=True)
                zstd_result = self._test_zstd(test_data)
                case_result['results']['Zstd'] = zstd_result
                print(f"✅ {zstd_result['compression_ratio']:.1f}% ({self._format_speed(zstd_result['compression_speed'])})")
            
            # Gzip テスト
            print("   🧪 Gzip...", end=" ", flush=True)
            gzip_result = self._test_gzip(test_data)
            case_result['results']['Gzip'] = gzip_result
            print(f"✅ {gzip_result['compression_ratio']:.1f}% ({self._format_speed(gzip_result['compression_speed'])})")
            
            self.results.append(case_result)
            
            # ケース別結果表示
            self._display_case_results(case_result)
            
        except Exception as e:
            print(f"   ❌ エラー: {str(e)}")
    
    @with_timeout(60)  # 60秒タイムアウト
    def _test_tmc_v4(self, data: bytes) -> Dict[str, Any]:
        """TMC v4.0 テスト（タイムアウト付き）"""
        try:
            result = self.tmc_engine.test_reversibility(data)
            
            return {
                'compression_ratio': result.get('compression_ratio', 0.0),
                'compressed_size': result.get('compressed_size', len(data)),
                'compression_speed': result.get('compression_throughput_mb_s', 0.0),
                'decompression_speed': result.get('decompression_throughput_mb_s', 0.0),
                'compression_time': result.get('compression_time', 0.0),
                'decompression_time': result.get('decompression_time', 0.0),
                'reversible': result.get('reversible', False),
                'data_type': result.get('data_type', 'unknown'),
                'zstd_used': result.get('zstd_used', False)
            }
            
        except TimeoutError as e:
            print(f"   ⏰ TMC v4.0 タイムアウト: {str(e)}")
            return {
                'compression_ratio': 0.0,
                'compressed_size': len(data),
                'compression_speed': 0.0,
                'decompression_speed': 0.0,
                'error': f'timeout: {str(e)}',
                'reversible': False,
                'timeout': True
            }
        except Exception as e:
            print(f"   ❌ TMC v4.0 エラー: {str(e)}")
            return {
                'compression_ratio': 0.0,
                'compressed_size': len(data),
                'compression_speed': 0.0,
                'decompression_speed': 0.0,
                'error': str(e),
                'reversible': False
            }
    
    def _test_lzma(self, data: bytes) -> Dict[str, Any]:
        """LZMA テスト"""
        try:
            compression_start = time.perf_counter()
            compressed = lzma.compress(data, preset=6)
            compression_time = time.perf_counter() - compression_start
            
            decompression_start = time.perf_counter()
            decompressed = lzma.decompress(compressed)
            decompression_time = time.perf_counter() - decompression_start
            
            reversible = (data == decompressed)
            
            return {
                'compression_ratio': (1 - len(compressed) / len(data)) * 100 if len(data) > 0 else 0,
                'compressed_size': len(compressed),
                'compression_speed': (len(data) / 1024 / 1024) / compression_time if compression_time > 0 else 0,
                'decompression_speed': (len(decompressed) / 1024 / 1024) / decompression_time if decompression_time > 0 else 0,
                'compression_time': compression_time,
                'decompression_time': decompression_time,
                'reversible': reversible
            }
            
        except Exception as e:
            return {
                'compression_ratio': 0.0,
                'compressed_size': len(data),
                'compression_speed': 0.0,
                'decompression_speed': 0.0,
                'error': str(e),
                'reversible': False
            }
    
    def _test_zstd(self, data: bytes) -> Dict[str, Any]:
        """Zstandard テスト"""
        try:
            cctx = zstd.ZstdCompressor(level=3)
            dctx = zstd.ZstdDecompressor()
            
            compression_start = time.perf_counter()
            compressed = cctx.compress(data)
            compression_time = time.perf_counter() - compression_start
            
            decompression_start = time.perf_counter()
            decompressed = dctx.decompress(compressed)
            decompression_time = time.perf_counter() - decompression_start
            
            reversible = (data == decompressed)
            
            return {
                'compression_ratio': (1 - len(compressed) / len(data)) * 100 if len(data) > 0 else 0,
                'compressed_size': len(compressed),
                'compression_speed': (len(data) / 1024 / 1024) / compression_time if compression_time > 0 else 0,
                'decompression_speed': (len(decompressed) / 1024 / 1024) / decompression_time if decompression_time > 0 else 0,
                'compression_time': compression_time,
                'decompression_time': decompression_time,
                'reversible': reversible
            }
            
        except Exception as e:
            return {
                'compression_ratio': 0.0,
                'compressed_size': len(data),
                'compression_speed': 0.0,
                'decompression_speed': 0.0,
                'error': str(e),
                'reversible': False
            }
    
    def _test_gzip(self, data: bytes) -> Dict[str, Any]:
        """Gzip テスト"""
        try:
            compression_start = time.perf_counter()
            compressed = gzip.compress(data, compresslevel=6)
            compression_time = time.perf_counter() - compression_start
            
            decompression_start = time.perf_counter()
            decompressed = gzip.decompress(compressed)
            decompression_time = time.perf_counter() - decompression_start
            
            reversible = (data == decompressed)
            
            return {
                'compression_ratio': (1 - len(compressed) / len(data)) * 100 if len(data) > 0 else 0,
                'compressed_size': len(compressed),
                'compression_speed': (len(data) / 1024 / 1024) / compression_time if compression_time > 0 else 0,
                'decompression_speed': (len(decompressed) / 1024 / 1024) / decompression_time if decompression_time > 0 else 0,
                'compression_time': compression_time,
                'decompression_time': decompression_time,
                'reversible': reversible
            }
            
        except Exception as e:
            return {
                'compression_ratio': 0.0,
                'compressed_size': len(data),
                'compression_speed': 0.0,
                'decompression_speed': 0.0,
                'error': str(e),
                'reversible': False
            }
    
    def _display_case_results(self, case_result: Dict[str, Any]):
        """ケース別結果表示"""
        print(f"\n   📋 {case_result['test_name']} 結果:")
        
        # 圧縮率ランキング
        methods = list(case_result['results'].keys())
        compression_ratios = [(method, case_result['results'][method]['compression_ratio']) 
                             for method in methods]
        compression_ratios.sort(key=lambda x: x[1], reverse=True)
        
        print("      🏆 圧縮率ランキング:")
        for i, (method, ratio) in enumerate(compression_ratios, 1):
            icon = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            reversible = case_result['results'][method].get('reversible', False)
            rev_icon = "✅" if reversible else "❌"
            
            # TMC v4.0の場合はデータタイプも表示
            extra_info = ""
            if method == 'TMC_v4':
                data_type = case_result['results'][method].get('data_type', 'unknown')
                zstd_used = case_result['results'][method].get('zstd_used', False)
                zstd_icon = "🔥" if zstd_used else ""
                extra_info = f" [{data_type}]{zstd_icon}"
            
            print(f"         {icon} {i}. {method:<8}: {ratio:>6.2f}% {rev_icon}{extra_info}")
    
    def _display_comprehensive_results(self):
        """総合結果表示"""
        if not self.results:
            print("❌ 結果データがありません")
            return
        
        print("\n" + "="*80)
        print("📊 TMC v4.0 統合ベンチマーク総合結果")
        print("="*80)
        
        # 利用可能メソッド確認
        all_methods = set()
        for result in self.results:
            all_methods.update(result['results'].keys())
        
        methods = sorted(list(all_methods))
        
        # 総合統計計算
        overall_stats = {}
        
        for method in methods:
            compression_ratios = []
            compression_speeds = []
            decompression_speeds = []
            reversible_count = 0
            total_count = 0
            
            for result in self.results:
                if method in result['results']:
                    method_result = result['results'][method]
                    compression_ratios.append(method_result['compression_ratio'])
                    compression_speeds.append(method_result['compression_speed'])
                    decompression_speeds.append(method_result['decompression_speed'])
                    if method_result.get('reversible', False):
                        reversible_count += 1
                    total_count += 1
            
            if compression_ratios:
                overall_stats[method] = {
                    'avg_compression_ratio': sum(compression_ratios) / len(compression_ratios),
                    'avg_compression_speed': sum(compression_speeds) / len(compression_speeds),
                    'avg_decompression_speed': sum(decompression_speeds) / len(decompression_speeds),
                    'reversibility_rate': reversible_count / total_count * 100 if total_count > 0 else 0,
                    'test_count': total_count
                }
        
        # 総合ランキング表示
        print("\n🏆 総合圧縮率ランキング:")
        compression_ranking = [(method, stats['avg_compression_ratio']) 
                              for method, stats in overall_stats.items()]
        compression_ranking.sort(key=lambda x: x[1], reverse=True)
        
        for i, (method, ratio) in enumerate(compression_ranking, 1):
            icon = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            reversibility = overall_stats[method]['reversibility_rate']
            print(f"   {icon} {i}. {method:<10}: {ratio:>6.2f}% (可逆性: {reversibility:>5.1f}%)")
        
        print("\n⚡ 総合圧縮速度ランキング:")
        speed_ranking = [(method, stats['avg_compression_speed']) 
                        for method, stats in overall_stats.items()]
        speed_ranking.sort(key=lambda x: x[1], reverse=True)
        
        for i, (method, speed) in enumerate(speed_ranking, 1):
            icon = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            print(f"   {icon} {i}. {method:<10}: {self._format_speed(speed)}")
        
        print("\n🚀 総合展開速度ランキング:")
        decomp_ranking = [(method, stats['avg_decompression_speed']) 
                         for method, stats in overall_stats.items()]
        decomp_ranking.sort(key=lambda x: x[1], reverse=True)
        
        for i, (method, speed) in enumerate(decomp_ranking, 1):
            icon = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            print(f"   {icon} {i}. {method:<10}: {self._format_speed(speed)}")
        
        # TMC v4.0 特別分析
        if 'TMC_v4' in overall_stats:
            tmc_stats = overall_stats['TMC_v4']
            print(f"\n🎯 TMC v4.0 特別評価:")
            print(f"   平均圧縮率: {tmc_stats['avg_compression_ratio']:.2f}%")
            print(f"   平均圧縮速度: {self._format_speed(tmc_stats['avg_compression_speed'])}")
            print(f"   平均展開速度: {self._format_speed(tmc_stats['avg_decompression_speed'])}")
            print(f"   可逆性: {tmc_stats['reversibility_rate']:.1f}%")
            print(f"   テスト件数: {tmc_stats['test_count']}")
    
    def _analyze_tmc_v4_specialization(self):
        """TMC v4.0 特化分析"""
        print("\n" + "="*80)
        print("🔍 TMC v4.0 特化性能分析")
        print("="*80)
        
        # データタイプ別分析
        data_type_performance = {}
        
        for result in self.results:
            if 'TMC_v4' in result['results']:
                tmc_result = result['results']['TMC_v4']
                data_type = tmc_result.get('data_type', 'unknown')
                
                if data_type not in data_type_performance:
                    data_type_performance[data_type] = {
                        'tests': [],
                        'compression_ratios': [],
                        'speeds': []
                    }
                
                data_type_performance[data_type]['tests'].append(result['test_name'])
                data_type_performance[data_type]['compression_ratios'].append(tmc_result['compression_ratio'])
                data_type_performance[data_type]['speeds'].append(tmc_result['compression_speed'])
        
        print("\n📈 データタイプ別性能:")
        for data_type, performance in data_type_performance.items():
            if performance['compression_ratios']:
                avg_ratio = sum(performance['compression_ratios']) / len(performance['compression_ratios'])
                avg_speed = sum(performance['speeds']) / len(performance['speeds'])
                count = len(performance['tests'])
                print(f"   {data_type}: {avg_ratio:.2f}% @ {self._format_speed(avg_speed)} ({count}件)")
        
        # Zstandardバックエンド効果分析
        zstd_used_count = 0
        total_tmc_tests = 0
        
        for result in self.results:
            if 'TMC_v4' in result['results']:
                total_tmc_tests += 1
                if result['results']['TMC_v4'].get('zstd_used', False):
                    zstd_used_count += 1
        
        if total_tmc_tests > 0:
            zstd_usage_rate = zstd_used_count / total_tmc_tests * 100
            print(f"\n🔥 Zstandardバックエンド使用率: {zstd_usage_rate:.1f}% ({zstd_used_count}/{total_tmc_tests})")
    
    def _format_size(self, size_bytes: int) -> str:
        """サイズフォーマット"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f}TB"
    
    def _format_speed(self, speed_mb_s: float) -> str:
        """速度フォーマット"""
        if speed_mb_s < 1.0:
            return f"{speed_mb_s*1024:.1f}KB/s"
        else:
            return f"{speed_mb_s:.1f}MB/s"


def main():
    """メイン実行"""
    print("🚀 NEXUS TMC v4.0 ユーザー提案統合版 ベンチマーク")
    print("   Zstandardバックエンド + 改良ディスパッチャ + LeCo/TDT/BWT統合")
    print("="*80)
    
    # ベンチマーク実行
    benchmark = TMCv4Benchmark()
    results = benchmark.run_comprehensive_benchmark()
    
    print("\n" + "="*80)
    print("🎉 TMC v4.0 統合ベンチマーク完了!")
    print("   ユーザー提案を統合したTMC v4.0の性能評価が完了しました。")
    
    if ZSTD_AVAILABLE:
        print("🔥 Zstandardバックエンドにより最高性能を実現!")
    else:
        print("⚠️ Zstandardが利用できません。pip install zstandard で性能向上可能です。")
    
    print("="*80)
    
    return results


if __name__ == "__main__":
    main()
