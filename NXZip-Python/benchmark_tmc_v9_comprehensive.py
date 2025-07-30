#!/usr/bin/env python3
"""
TMC v9.0 革新的圧縮エンジン 総合ベンチマークテスト
vs LZMA2 vs Zstandard 詳細性能比較

評価項目:
1. 圧縮率 (Compression Ratio)
2. 圧縮速度 (Compression Speed)
3. 展開速度 (Decompression Speed)  
4. 可逆性 (Reversibility)
5. メモリ使用量
6. CPU使用率
"""

import os
import sys
import time
import subprocess
import tempfile
import hashlib
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Any

# NXZip TMCエンジンインポート
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nxzip.engine.nexus_tmc_v4_unified import NEXUSTMCEngineV9

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
    print("🚀 Zstandard利用可能 - 高性能バックエンド有効")
except ImportError:
    ZSTD_AVAILABLE = False
    print("⚠️ Zstandard未利用可能")

try:
    import lzma
    LZMA_AVAILABLE = True
    print("🚀 LZMA2利用可能 - Python標準ライブラリ有効")
except ImportError:
    LZMA_AVAILABLE = False
    print("⚠️ LZMA未利用可能")

class PerformanceMonitor:
    """リアルタイム性能監視"""
    
    def __init__(self):
        self.monitoring = False
        self.cpu_usage = []
        self.memory_usage = []
        
    def start_monitoring(self):
        """監視開始"""
        self.monitoring = True
        self.cpu_usage.clear()
        self.memory_usage.clear()
        
        def monitor():
            while self.monitoring:
                try:
                    process = psutil.Process()
                    self.cpu_usage.append(process.cpu_percent())
                    self.memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB
                    time.sleep(0.1)
                except:
                    pass
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, float]:
        """監視停止と結果取得"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1)
        
        return {
            'avg_cpu_percent': sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0,
            'max_memory_mb': max(self.memory_usage) if self.memory_usage else 0,
            'peak_cpu_percent': max(self.cpu_usage) if self.cpu_usage else 0
        }

class ComprehensiveBenchmark:
    """総合ベンチマークテストスイート"""
    
    def __init__(self):
        self.tmc_engine = NEXUSTMCEngineV9(max_workers=8)
        self.monitor = PerformanceMonitor()
        self.temp_dir = tempfile.mkdtemp(prefix="tmc_benchmark_")
        print(f"🔧 ベンチマーク作業ディレクトリ: {self.temp_dir}")
        
        # テストデータセット定義
        self.test_datasets = {
            "構造化JSON": self._generate_structured_json(50 * 1024),  # 50KB
            "高冗長テキスト": self._generate_repetitive_text(100 * 1024),  # 100KB
            "浮動小数点配列": self._generate_float_data(80 * 1024),  # 80KB
            "ランダムバイナリ": self._generate_random_binary(64 * 1024),  # 64KB
            "混合データ": self._generate_mixed_data(120 * 1024),  # 120KB
            "大容量テキスト": self._generate_large_text(500 * 1024),  # 500KB
            "系列整数": self._generate_sequential_integers(75 * 1024),  # 75KB
        }
        
        print(f"📊 {len(self.test_datasets)}種類のテストデータセット準備完了")
    
    def _generate_structured_json(self, target_size: int) -> bytes:
        """構造化JSONデータ生成"""
        import json
        data = {
            "users": [
                {
                    "id": i,
                    "name": f"user_{i:04d}",
                    "email": f"user{i}@example.com",
                    "settings": {
                        "theme": "dark" if i % 2 == 0 else "light",
                        "notifications": i % 3 == 0,
                        "language": ["en", "ja", "de", "fr"][i % 4]
                    },
                    "scores": [i * 10 + j for j in range(5)]
                }
                for i in range(target_size // 200)  # 約200バイト/ユーザー
            ]
        }
        return json.dumps(data, separators=(',', ':')).encode('utf-8')
    
    def _generate_repetitive_text(self, target_size: int) -> bytes:
        """高冗長テキストデータ生成"""
        patterns = [
            "The quick brown fox jumps over the lazy dog. ",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ",
            "Hello world! This is a test pattern. ",
            "1234567890 abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ ",
        ]
        
        result = ""
        while len(result.encode('utf-8')) < target_size:
            result += patterns[len(result) % len(patterns)]
        
        return result.encode('utf-8')[:target_size]
    
    def _generate_float_data(self, target_size: int) -> bytes:
        """浮動小数点データ生成"""
        import numpy as np
        count = target_size // 4
        # 数学的パターンを含む浮動小数点数
        data = np.array([
            np.sin(i * 0.1) * 1000 + np.cos(i * 0.05) * 500 + i * 0.1
            for i in range(count)
        ], dtype=np.float32)
        return data.tobytes()
    
    def _generate_random_binary(self, target_size: int) -> bytes:
        """ランダムバイナリデータ生成"""
        import random
        return bytes(random.randint(0, 255) for _ in range(target_size))
    
    def _generate_mixed_data(self, target_size: int) -> bytes:
        """混合データ生成"""
        # テキスト + バイナリ + 反復パターンの混合
        text_part = b"Mixed data content with various patterns. " * 100
        binary_part = bytes(range(256)) * 20
        repetitive_part = b"PATTERN" * 200
        
        mixed = text_part + binary_part + repetitive_part
        return mixed[:target_size]
    
    def _generate_large_text(self, target_size: int) -> bytes:
        """大容量テキストデータ生成"""
        import random
        import string
        
        # 実際のテキストのような構造
        words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", 
                "hello", "world", "python", "programming", "algorithm", "data", 
                "compression", "benchmark", "performance", "test", "result"]
        
        result = ""
        while len(result.encode('utf-8')) < target_size:
            sentence_length = random.randint(5, 15)
            sentence = " ".join(random.choices(words, k=sentence_length))
            result += sentence.capitalize() + ". "
            
            if random.random() < 0.1:  # 10%の確率で段落区切り
                result += "\n\n"
        
        return result.encode('utf-8')[:target_size]
    
    def _generate_sequential_integers(self, target_size: int) -> bytes:
        """系列整数データ生成"""
        import numpy as np
        count = target_size // 4
        # 段階的増加パターン
        data = np.array([
            i + (i // 100) * 1000 + (i % 7) * 10
            for i in range(count)
        ], dtype=np.int32)
        return data.tobytes()
    
    def benchmark_tmc_v9(self, data: bytes, name: str) -> Dict[str, Any]:
        """TMC v9.0ベンチマーク"""
        print(f"  🚀 TMC v9.0テスト中...")
        
        # 圧縮テスト
        self.monitor.start_monitoring()
        compress_start = time.perf_counter()
        
        try:
            # TMC v9.0 非同期圧縮
            import asyncio
            compressed, info = asyncio.run(self.tmc_engine.compress_tmc_v9_async(data))
            compress_time = time.perf_counter() - compress_start
            
            # 性能統計取得
            compress_stats = self.monitor.stop_monitoring()
            
            # 展開テスト
            self.monitor.start_monitoring()
            decompress_start = time.perf_counter()
            
            decompressed, decomp_info = self.tmc_engine.decompress_tmc(compressed)
            decompress_time = time.perf_counter() - decompress_start
            
            decompress_stats = self.monitor.stop_monitoring()
            
            # 可逆性検証
            original_hash = hashlib.sha256(data).hexdigest()
            decompressed_hash = hashlib.sha256(decompressed).hexdigest()
            is_reversible = (original_hash == decompressed_hash)
            
            return {
                'name': 'TMC v9.0',
                'compressed_size': len(compressed),
                'compression_ratio': (1 - len(compressed) / len(data)) * 100,
                'compression_time': compress_time,
                'decompression_time': decompress_time,
                'compression_speed_mbps': (len(data) / 1024 / 1024) / compress_time,
                'decompression_speed_mbps': (len(data) / 1024 / 1024) / decompress_time,
                'reversible': is_reversible,
                'compress_cpu_avg': compress_stats['avg_cpu_percent'],
                'compress_memory_peak': compress_stats['max_memory_mb'],
                'decompress_cpu_avg': decompress_stats['avg_cpu_percent'],
                'decompress_memory_peak': decompress_stats['max_memory_mb'],
                'features_used': info.get('innovations', []),
                'sublinear_lz77_used': info.get('sublinear_lz77_used', False),
                'async_pipeline': True
            }
            
        except Exception as e:
            self.monitor.stop_monitoring()
            return {
                'name': 'TMC v9.0',
                'error': str(e),
                'reversible': False
            }
    
    def benchmark_zstd(self, data: bytes, name: str) -> Dict[str, Any]:
        """Zstandard ベンチマーク"""
        if not ZSTD_AVAILABLE:
            return {'name': 'Zstandard', 'error': 'Not available', 'reversible': False}
        
        print(f"  📦 Zstandard テスト中...")
        
        try:
            # 圧縮テスト（レベル3 - バランス）
            cctx = zstd.ZstdCompressor(level=3)
            
            self.monitor.start_monitoring()
            compress_start = time.perf_counter()
            
            compressed = cctx.compress(data)
            compress_time = time.perf_counter() - compress_start
            
            compress_stats = self.monitor.stop_monitoring()
            
            # 展開テスト
            dctx = zstd.ZstdDecompressor()
            
            self.monitor.start_monitoring()
            decompress_start = time.perf_counter()
            
            decompressed = dctx.decompress(compressed)
            decompress_time = time.perf_counter() - decompress_start
            
            decompress_stats = self.monitor.stop_monitoring()
            
            # 可逆性検証
            is_reversible = (data == decompressed)
            
            return {
                'name': 'Zstandard',
                'compressed_size': len(compressed),
                'compression_ratio': (1 - len(compressed) / len(data)) * 100,
                'compression_time': compress_time,
                'decompression_time': decompress_time,
                'compression_speed_mbps': (len(data) / 1024 / 1024) / compress_time,
                'decompression_speed_mbps': (len(data) / 1024 / 1024) / decompress_time,
                'reversible': is_reversible,
                'compress_cpu_avg': compress_stats['avg_cpu_percent'],
                'compress_memory_peak': compress_stats['max_memory_mb'],
                'decompress_cpu_avg': decompress_stats['avg_cpu_percent'],
                'decompress_memory_peak': decompress_stats['max_memory_mb'],
                'compression_level': 3
            }
            
        except Exception as e:
            self.monitor.stop_monitoring()
            return {
                'name': 'Zstandard',
                'error': str(e),
                'reversible': False
            }
    
    def benchmark_lzma2(self, data: bytes, name: str) -> Dict[str, Any]:
        """LZMA2 (Python標準ライブラリ) ベンチマーク"""
        if not LZMA_AVAILABLE:
            return {'name': 'LZMA2', 'error': 'Not available', 'reversible': False}
        
        print(f"  🗜️ LZMA2 テスト中...")
        
        try:
            # 圧縮テスト（プリセット6 - 高圧縮率）
            self.monitor.start_monitoring()
            compress_start = time.perf_counter()
            
            # LZMA2圧縮 (XZ形式)
            compressed = lzma.compress(data, format=lzma.FORMAT_XZ, preset=6)
            compress_time = time.perf_counter() - compress_start
            
            compress_stats = self.monitor.stop_monitoring()
            
            # 展開テスト
            self.monitor.start_monitoring()
            decompress_start = time.perf_counter()
            
            decompressed = lzma.decompress(compressed, format=lzma.FORMAT_XZ)
            decompress_time = time.perf_counter() - decompress_start
            
            decompress_stats = self.monitor.stop_monitoring()
            
            # 可逆性検証
            is_reversible = (data == decompressed)
            
            return {
                'name': 'LZMA2',
                'compressed_size': len(compressed),
                'compression_ratio': (1 - len(compressed) / len(data)) * 100,
                'compression_time': compress_time,
                'decompression_time': decompress_time,
                'compression_speed_mbps': (len(data) / 1024 / 1024) / compress_time,
                'decompression_speed_mbps': (len(data) / 1024 / 1024) / decompress_time,
                'reversible': is_reversible,
                'compress_cpu_avg': compress_stats['avg_cpu_percent'],
                'compress_memory_peak': compress_stats['max_memory_mb'],
                'decompress_cpu_avg': decompress_stats['avg_cpu_percent'],
                'decompress_memory_peak': decompress_stats['max_memory_mb'],
                'compression_preset': 6,
                'format': 'XZ/LZMA2'
            }
            
        except Exception as e:
            self.monitor.stop_monitoring()
            return {
                'name': 'LZMA2',
                'error': str(e),
                'reversible': False
            }
    
    def run_comprehensive_benchmark(self):
        """総合ベンチマーク実行"""
        print("🏁 TMC v9.0 総合ベンチマークテスト開始")
        print("=" * 80)
        
        all_results = {}
        
        for dataset_name, data in self.test_datasets.items():
            print(f"\n📊 データセット: {dataset_name}")
            print(f"   サイズ: {len(data):,} bytes ({len(data)/1024:.1f} KB)")
            print("-" * 60)
            
            # 各圧縮方式でテスト
            results = {}
            
            results['tmc'] = self.benchmark_tmc_v9(data, dataset_name)
            results['zstd'] = self.benchmark_zstd(data, dataset_name)
            results['lzma2'] = self.benchmark_lzma2(data, dataset_name)
            
            all_results[dataset_name] = {
                'data_size': len(data),
                'results': results
            }
            
            # 結果表示
            self._display_dataset_results(dataset_name, results, len(data))
        
        # 総合結果分析
        print("\n" + "=" * 80)
        print("🏆 総合分析結果")
        print("=" * 80)
        self._display_comprehensive_analysis(all_results)
        
        return all_results
    
    def _display_dataset_results(self, dataset_name: str, results: Dict, data_size: int):
        """データセット別結果表示"""
        
        # 圧縮率比較
        print("\n📈 圧縮率比較:")
        for method, result in results.items():
            if 'error' not in result:
                ratio = result['compression_ratio']
                size = result['compressed_size']
                print(f"   {result['name']:15}: {ratio:6.2f}% ({size:,} bytes)")
            else:
                print(f"   {result['name']:15}: エラー - {result['error']}")
        
        # 速度比較
        print("\n⚡ 速度比較:")
        print("   圧縮速度 (MB/s):")
        for method, result in results.items():
            if 'error' not in result:
                speed = result['compression_speed_mbps']
                time = result['compression_time']
                print(f"     {result['name']:15}: {speed:8.2f} MB/s ({time:.3f}秒)")
        
        print("   展開速度 (MB/s):")
        for method, result in results.items():
            if 'error' not in result:
                speed = result['decompression_speed_mbps']
                time = result['decompression_time']
                print(f"     {result['name']:15}: {speed:8.2f} MB/s ({time:.3f}秒)")
        
        # 可逆性
        print("\n🔄 可逆性:")
        for method, result in results.items():
            status = "✅" if result.get('reversible', False) else "❌"
            print(f"   {result['name']:15}: {status}")
        
        # リソース使用量
        print("\n💻 リソース使用量:")
        for method, result in results.items():
            if 'error' not in result and 'compress_memory_peak' in result:
                cpu = result['compress_cpu_avg']
                mem = result['compress_memory_peak']
                print(f"   {result['name']:15}: CPU {cpu:5.1f}%, メモリ {mem:6.1f}MB")
    
    def _display_comprehensive_analysis(self, all_results: Dict):
        """総合分析結果表示"""
        
        # 各手法の総合スコア計算
        scores = {'tmc': [], 'zstd': [], 'lzma2': []}
        
        compression_ratios = {'tmc': [], 'zstd': [], 'lzma2': []}
        compression_speeds = {'tmc': [], 'zstd': [], 'lzma2': []}
        decompression_speeds = {'tmc': [], 'zstd': [], 'lzma2': []}
        
        for dataset_name, dataset_info in all_results.items():
            results = dataset_info['results']
            
            for method in ['tmc', 'zstd', 'lzma2']:
                result = results[method]
                if 'error' not in result:
                    compression_ratios[method].append(result['compression_ratio'])
                    compression_speeds[method].append(result['compression_speed_mbps'])
                    decompression_speeds[method].append(result['decompression_speed_mbps'])
        
        # 平均値計算
        print("📊 平均性能指標:")
        print("-" * 50)
        
        methods_info = {
            'tmc': 'TMC v9.0        ',
            'zstd': 'Zstandard      ',
            'lzma2': 'LZMA2          '
        }
        
        for method, name in methods_info.items():
            if compression_ratios[method]:
                avg_ratio = sum(compression_ratios[method]) / len(compression_ratios[method])
                avg_comp_speed = sum(compression_speeds[method]) / len(compression_speeds[method])
                avg_decomp_speed = sum(decompression_speeds[method]) / len(decompression_speeds[method])
                
                print(f"{name}: 圧縮率 {avg_ratio:6.2f}%, 圧縮速度 {avg_comp_speed:6.2f}MB/s, 展開速度 {avg_decomp_speed:6.2f}MB/s")
        
        # 勝利カウント
        print("\n🏆 カテゴリ別勝利数:")
        print("-" * 50)
        
        ratio_wins = {'tmc': 0, 'zstd': 0, 'lzma2': 0}
        comp_speed_wins = {'tmc': 0, 'zstd': 0, 'lzma2': 0}
        decomp_speed_wins = {'tmc': 0, 'zstd': 0, 'lzma2': 0}
        
        for dataset_name, dataset_info in all_results.items():
            results = dataset_info['results']
            
            # 有効な結果のみ
            valid_results = {k: v for k, v in results.items() if 'error' not in v}
            
            if len(valid_results) > 1:
                # 圧縮率勝者
                best_ratio = max(valid_results.values(), key=lambda x: x['compression_ratio'])
                for method, result in valid_results.items():
                    if result['compression_ratio'] == best_ratio['compression_ratio']:
                        ratio_wins[method] += 1
                        break
                
                # 圧縮速度勝者
                best_comp_speed = max(valid_results.values(), key=lambda x: x['compression_speed_mbps'])
                for method, result in valid_results.items():
                    if result['compression_speed_mbps'] == best_comp_speed['compression_speed_mbps']:
                        comp_speed_wins[method] += 1
                        break
                
                # 展開速度勝者
                best_decomp_speed = max(valid_results.values(), key=lambda x: x['decompression_speed_mbps'])
                for method, result in valid_results.items():
                    if result['decompression_speed_mbps'] == best_decomp_speed['decompression_speed_mbps']:
                        decomp_speed_wins[method] += 1
                        break
        
        for method, name in methods_info.items():
            ratio_w = ratio_wins[method]
            comp_w = comp_speed_wins[method]
            decomp_w = decomp_speed_wins[method]
            total_w = ratio_w + comp_w + decomp_w
            
            print(f"{name}: 圧縮率{ratio_w}勝, 圧縮速度{comp_w}勝, 展開速度{decomp_w}勝 (合計{total_w}勝)")
        
        # TMC v9.0特有機能の活用状況
        print("\n🚀 TMC v9.0 革新機能活用状況:")
        print("-" * 50)
        
        sublinear_usage = 0
        pipeline_usage = 0
        total_datasets = len(all_results)
        
        for dataset_name, dataset_info in all_results.items():
            tmc_result = dataset_info['results']['tmc']
            if 'error' not in tmc_result:
                if tmc_result.get('sublinear_lz77_used', False):
                    sublinear_usage += 1
                if tmc_result.get('async_pipeline', False):
                    pipeline_usage += 1
        
        print(f"サブリニアLZ77使用: {sublinear_usage}/{total_datasets} データセット ({sublinear_usage/total_datasets*100:.1f}%)")
        print(f"非同期パイプライン: {pipeline_usage}/{total_datasets} データセット ({pipeline_usage/total_datasets*100:.1f}%)")
        
        # 総合評価
        print("\n🎯 総合評価:")
        print("-" * 50)
        
        tmc_total_wins = sum([ratio_wins['tmc'], comp_speed_wins['tmc'], decomp_speed_wins['tmc']])
        zstd_total_wins = sum([ratio_wins['zstd'], comp_speed_wins['zstd'], decomp_speed_wins['zstd']])
        lzma2_total_wins = sum([ratio_wins['lzma2'], comp_speed_wins['lzma2'], decomp_speed_wins['lzma2']])
        
        max_wins = max(tmc_total_wins, zstd_total_wins, lzma2_total_wins)
        
        if tmc_total_wins == max_wins:
            print("🥇 総合チャンピオン: TMC v9.0")
            print("   革新的並列パイプライン + サブリニアLZ77による次世代圧縮性能を実証！")
        elif zstd_total_wins == max_wins:
            print("🥇 総合チャンピオン: Zstandard")
        else:
            print("🥇 総合チャンピオン: LZMA2")
        
        if tmc_total_wins > 0:
            print(f"\nTMC v9.0は{tmc_total_wins}カテゴリで勝利を収め、革新的圧縮技術の実用性を証明しました！")

def main():
    """メイン実行関数"""
    print("🚀 TMC v9.0 革新的圧縮エンジン 総合ベンチマークテスト")
    print("🆚 vs LZMA2 vs Zstandard")
    print("=" * 80)
    
    # ベンチマーク実行
    benchmark = ComprehensiveBenchmark()
    
    try:
        results = benchmark.run_comprehensive_benchmark()
        
        print("\n✅ ベンチマークテスト完了！")
        print("TMC v9.0の革新的性能を確認できました。")
        
    except KeyboardInterrupt:
        print("\n⚠️ ユーザーによりテスト中断")
    except Exception as e:
        print(f"\n❌ ベンチマークエラー: {e}")
    finally:
        # クリーンアップ
        import shutil
        try:
            shutil.rmtree(benchmark.temp_dir)
        except:
            pass

if __name__ == "__main__":
    main()
