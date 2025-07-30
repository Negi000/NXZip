#!/usr/bin/env python3
"""
NEXUS TMC Engine v3.0 完全版 ベンチマークテスト
7Z/Zstandardとの競合比較テスト
"""

import os
import sys
import time
import shutil
import lzma
import gzip
import bz2
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple

# TMC v3.0 エンジンインポート
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nexus_tmc_v3_complete import NEXUSTMCEngine


class CompetitiveBenchmark:
    """競合圧縮ツールとの比較ベンチマーク"""
    
    def __init__(self):
        self.tmc_engine = NEXUSTMCEngine(max_workers=4)
        self.results = []
        
    def run_comprehensive_benchmark(self, sample_dir: str = "sample"):
        """包括的ベンチマーク実行"""
        print("🚀 NEXUS TMC v3.0 vs 7Z/Zstandard 競合ベンチマーク開始")
        print("=" * 80)
        
        sample_path = Path(sample_dir)
        if not sample_path.exists():
            print(f"❌ サンプルディレクトリが見つかりません: {sample_dir}")
            return
        
        # テストファイル収集
        test_files = self._collect_test_files(sample_path)
        
        if not test_files:
            print("❌ テストファイルが見つかりません")
            return
        
        print(f"📁 テストファイル数: {len(test_files)}")
        print("-" * 80)
        
        # 各ファイルでベンチマーク実行
        for file_path in test_files:
            print(f"\n🔬 テスト中: {file_path.name}")
            self._benchmark_single_file(file_path)
        
        # 総合結果表示
        self._display_comprehensive_results()
        
        # パフォーマンス分析
        self._analyze_performance()
        
        return self.results
    
    def _collect_test_files(self, sample_path: Path) -> List[Path]:
        """テストファイル収集"""
        test_files = []
        
        # 対象拡張子
        target_extensions = {'.txt', '.png', '.jpg', '.wav', '.mp3', '.mp4', '.pdf', '.dat'}
        
        for file_path in sample_path.iterdir():
            if file_path.is_file():
                # 圧縮済みファイルは除外
                if file_path.suffix.lower() not in {'.7z', '.zip', '.gz', '.bz2', '.xz'}:
                    # サイズ制限（実用的なテストのため）
                    if file_path.stat().st_size <= 50 * 1024 * 1024:  # 50MB以下
                        test_files.append(file_path)
        
        return sorted(test_files, key=lambda x: x.stat().st_size)
    
    def _benchmark_single_file(self, file_path: Path):
        """単一ファイルベンチマーク"""
        try:
            # ファイル読み込み
            with open(file_path, 'rb') as f:
                data = f.read()
            
            if len(data) == 0:
                print(f"   ⚠️ 空ファイルをスキップ: {file_path.name}")
                return
            
            original_size = len(data)
            print(f"   📊 原サイズ: {self._format_size(original_size)}")
            
            file_result = {
                'filename': file_path.name,
                'original_size': original_size,
                'results': {}
            }
            
            # TMC v3.0 テスト
            print("   🧪 TMC v3.0...", end=" ", flush=True)
            tmc_result = self._test_tmc_v3(data)
            file_result['results']['TMC_v3'] = tmc_result
            print(f"✅ {tmc_result['compression_ratio']:.1f}% ({self._format_speed(tmc_result['compression_speed'])})")
            
            # LZMA (7Z equivalent) テスト
            print("   🧪 LZMA...", end=" ", flush=True)
            lzma_result = self._test_lzma(data)
            file_result['results']['LZMA'] = lzma_result
            print(f"✅ {lzma_result['compression_ratio']:.1f}% ({self._format_speed(lzma_result['compression_speed'])})")
            
            # Gzip テスト
            print("   🧪 Gzip...", end=" ", flush=True)
            gzip_result = self._test_gzip(data)
            file_result['results']['Gzip'] = gzip_result
            print(f"✅ {gzip_result['compression_ratio']:.1f}% ({self._format_speed(gzip_result['compression_speed'])})")
            
            # BZ2 テスト
            print("   🧪 BZ2...", end=" ", flush=True)
            bz2_result = self._test_bz2(data)
            file_result['results']['BZ2'] = bz2_result
            print(f"✅ {bz2_result['compression_ratio']:.1f}% ({self._format_speed(bz2_result['compression_speed'])})")
            
            self.results.append(file_result)
            
            # ファイル別結果表示
            self._display_file_results(file_result)
            
        except Exception as e:
            print(f"   ❌ エラー: {str(e)}")
    
    def _test_tmc_v3(self, data: bytes) -> Dict[str, Any]:
        """TMC v3.0 テスト"""
        try:
            # 可逆性テスト実行
            result = self.tmc_engine.test_reversibility(data)
            
            return {
                'compression_ratio': result.get('compression_ratio', 0.0),
                'compressed_size': result.get('compressed_size', len(data)),
                'compression_speed': result.get('compression_throughput_mb_s', 0.0),
                'decompression_speed': result.get('decompression_throughput_mb_s', 0.0),
                'compression_time': result.get('compression_time', 0.0),
                'decompression_time': result.get('decompression_time', 0.0),
                'reversible': result.get('reversible', False),
                'method': result.get('compression_info', {}).get('compression_method', 'unknown')
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
    
    def _test_lzma(self, data: bytes) -> Dict[str, Any]:
        """LZMA テスト"""
        try:
            # 圧縮
            compression_start = time.perf_counter()
            compressed = lzma.compress(data, preset=6)
            compression_time = time.perf_counter() - compression_start
            
            # 展開
            decompression_start = time.perf_counter()
            decompressed = lzma.decompress(compressed)
            decompression_time = time.perf_counter() - decompression_start
            
            # 可逆性確認
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
            # 圧縮
            compression_start = time.perf_counter()
            compressed = gzip.compress(data, compresslevel=6)
            compression_time = time.perf_counter() - compression_start
            
            # 展開
            decompression_start = time.perf_counter()
            decompressed = gzip.decompress(compressed)
            decompression_time = time.perf_counter() - decompression_start
            
            # 可逆性確認
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
    
    def _test_bz2(self, data: bytes) -> Dict[str, Any]:
        """BZ2 テスト"""
        try:
            # 圧縮
            compression_start = time.perf_counter()
            compressed = bz2.compress(data, compresslevel=6)
            compression_time = time.perf_counter() - compression_start
            
            # 展開
            decompression_start = time.perf_counter()
            decompressed = bz2.decompress(compressed)
            decompression_time = time.perf_counter() - decompression_start
            
            # 可逆性確認
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
    
    def _display_file_results(self, file_result: Dict[str, Any]):
        """ファイル別結果表示"""
        print(f"\n   📋 {file_result['filename']} 結果:")
        
        # 圧縮率ランキング
        methods = list(file_result['results'].keys())
        compression_ratios = [(method, file_result['results'][method]['compression_ratio']) 
                             for method in methods]
        compression_ratios.sort(key=lambda x: x[1], reverse=True)
        
        print("      🏆 圧縮率ランキング:")
        for i, (method, ratio) in enumerate(compression_ratios, 1):
            icon = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            print(f"         {icon} {i}. {method}: {ratio:.2f}%")
        
        # 速度比較
        print("      ⚡ 圧縮速度:")
        for method in methods:
            speed = file_result['results'][method]['compression_speed']
            print(f"         {method}: {self._format_speed(speed)}")
    
    def _display_comprehensive_results(self):
        """総合結果表示"""
        if not self.results:
            print("❌ 結果データがありません")
            return
        
        print("\n" + "="*80)
        print("📊 総合結果 - TMC v3.0 vs 競合他社")
        print("="*80)
        
        # 全体統計計算
        methods = ['TMC_v3', 'LZMA', 'Gzip', 'BZ2']
        overall_stats = {}
        
        for method in methods:
            compression_ratios = []
            compression_speeds = []
            decompression_speeds = []
            reversible_count = 0
            total_count = 0
            
            for file_result in self.results:
                if method in file_result['results']:
                    result = file_result['results'][method]
                    compression_ratios.append(result['compression_ratio'])
                    compression_speeds.append(result['compression_speed'])
                    decompression_speeds.append(result['decompression_speed'])
                    if result.get('reversible', False):
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
    
    def _analyze_performance(self):
        """パフォーマンス分析"""
        print("\n" + "="*80)
        print("🔍 詳細パフォーマンス分析")
        print("="*80)
        
        if not self.results:
            print("❌ 分析データがありません")
            return
        
        # TMC v3.0 の特性分析
        tmc_results = []
        for file_result in self.results:
            if 'TMC_v3' in file_result['results']:
                tmc_result = file_result['results']['TMC_v3']
                tmc_results.append({
                    'filename': file_result['filename'],
                    'original_size': file_result['original_size'],
                    'compression_ratio': tmc_result['compression_ratio'],
                    'compression_speed': tmc_result['compression_speed'],
                    'method': tmc_result.get('method', 'unknown')
                })
        
        if tmc_results:
            print("\n📈 TMC v3.0 詳細分析:")
            
            # ファイルサイズ別パフォーマンス
            small_files = [r for r in tmc_results if r['original_size'] < 1024*1024]  # 1MB未満
            medium_files = [r for r in tmc_results if 1024*1024 <= r['original_size'] < 10*1024*1024]  # 1-10MB
            large_files = [r for r in tmc_results if r['original_size'] >= 10*1024*1024]  # 10MB以上
            
            for category, files in [("小ファイル(<1MB)", small_files), 
                                   ("中ファイル(1-10MB)", medium_files), 
                                   ("大ファイル(>=10MB)", large_files)]:
                if files:
                    avg_ratio = sum(f['compression_ratio'] for f in files) / len(files)
                    avg_speed = sum(f['compression_speed'] for f in files) / len(files)
                    print(f"   {category}: {avg_ratio:.2f}% @ {self._format_speed(avg_speed)} ({len(files)}件)")
            
            # 使用メソッド分析
            method_counts = {}
            for result in tmc_results:
                method = result.get('method', 'unknown')
                method_counts[method] = method_counts.get(method, 0) + 1
            
            print(f"\n   🔧 使用圧縮メソッド:")
            for method, count in sorted(method_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = count / len(tmc_results) * 100
                print(f"      {method}: {count}件 ({percentage:.1f}%)")
        
        # 競合比較分析
        print(f"\n🎯 競合優位性分析:")
        tmc_wins = {'compression': 0, 'comp_speed': 0, 'decomp_speed': 0}
        total_comparisons = 0
        
        for file_result in self.results:
            if 'TMC_v3' in file_result['results'] and 'LZMA' in file_result['results']:
                tmc = file_result['results']['TMC_v3']
                lzma = file_result['results']['LZMA']
                
                if tmc['compression_ratio'] >= lzma['compression_ratio']:
                    tmc_wins['compression'] += 1
                if tmc['compression_speed'] >= lzma['compression_speed']:
                    tmc_wins['comp_speed'] += 1
                if tmc['decompression_speed'] >= lzma['decompression_speed']:
                    tmc_wins['decomp_speed'] += 1
                
                total_comparisons += 1
        
        if total_comparisons > 0:
            print(f"   vs LZMA (7Z相当):")
            print(f"      圧縮率で勝利: {tmc_wins['compression']}/{total_comparisons} ({tmc_wins['compression']/total_comparisons*100:.1f}%)")
            print(f"      圧縮速度で勝利: {tmc_wins['comp_speed']}/{total_comparisons} ({tmc_wins['comp_speed']/total_comparisons*100:.1f}%)")
            print(f"      展開速度で勝利: {tmc_wins['decomp_speed']}/{total_comparisons} ({tmc_wins['decomp_speed']/total_comparisons*100:.1f}%)")
        
        # 可逆性レポート
        tmc_reversible = sum(1 for r in self.results 
                           if 'TMC_v3' in r['results'] and r['results']['TMC_v3'].get('reversible', False))
        tmc_total = sum(1 for r in self.results if 'TMC_v3' in r['results'])
        
        print(f"\n✅ TMC v3.0 可逆性: {tmc_reversible}/{tmc_total} ({tmc_reversible/tmc_total*100 if tmc_total > 0 else 0:.1f}%)")
        
        # 推奨使用ケース
        print(f"\n💡 TMC v3.0 推奨使用ケース:")
        if tmc_wins['compression'] / total_comparisons > 0.5:
            print("   ✓ 高圧縮率が必要なアーカイブ用途")
        if tmc_wins['comp_speed'] / total_comparisons > 0.5:
            print("   ✓ 高速圧縮が必要なリアルタイム処理")
        if tmc_reversible / tmc_total >= 0.95:
            print("   ✓ データ完全性が重要なミッションクリティカル用途")
        
        print("   ✓ 多様なデータタイプの統合処理")
        print("   ✓ 適応的圧縮が必要な環境")
    
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
    print("🚀 NEXUS TMC v3.0 完全版 競合ベンチマーク")
    print("   vs 7Z(LZMA) / Gzip / BZ2")
    print("="*80)
    
    # ベンチマーク実行
    benchmark = CompetitiveBenchmark()
    results = benchmark.run_comprehensive_benchmark()
    
    print("\n" + "="*80)
    print("🎉 ベンチマーク完了!")
    print("   TMC v3.0 完全実装版のパフォーマンス評価が完了しました。")
    print("="*80)
    
    return results


if __name__ == "__main__":
    main()
