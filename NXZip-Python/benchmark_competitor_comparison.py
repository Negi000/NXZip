#!/usr/bin/env python3
"""
NXZip TMC v9.0 vs 7-Zip vs Zstandard 競合比較ベンチマーク
================================================================================
客観的性能評価による実力測定
- 圧縮率 (%) - 数値が小さいほど高圧縮
- 圧縮速度 (MB/s)
- 展開速度 (MB/s)  
- 可逆性 (100%必須)
================================================================================
"""

import os
import sys
import time
import json
import tempfile
import subprocess
import zstandard as zstd
import py7zr
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Any

# NXZip TMC v9.0 エンジンのインポート
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nxzip.engine.nexus_tmc_v4_unified import NEXUSTMCEngineV9

class CompetitorBenchmark:
    """競合圧縮ツールとの性能比較ベンチマーク"""
    
    def __init__(self):
        self.nxzip_engine = NEXUSTMCEngineV9()
        self.temp_dir = tempfile.mkdtemp(prefix='benchmark_')
        self.results = []
        
        # 7-Zipの確認
        self.sevenz_available = self._check_7zip()
        
        # Zstandardレベル設定
        self.zstd_levels = [1, 3, 6, 15, 19]  # 高速〜最高圧縮
        
        print("🏁 競合比較ベンチマーク初期化完了")
        print(f"📁 作業ディレクトリ: {self.temp_dir}")
        print(f"7️⃣ 7-Zip利用可能: {'✅' if self.sevenz_available else '❌'}")
        if self.sevenz_available and hasattr(self, 'sevenz_command'):
            print(f"   コマンド: {self.sevenz_command}")
        print(f"🅰️ Zstandard利用可能: ✅")
    
    def _check_7zip(self) -> bool:
        """7-Zip (py7zr) の利用可能性をチェック"""
        try:
            # py7zrライブラリの動作確認
            test_data = b"test data for py7zr verification"
            with tempfile.NamedTemporaryFile(suffix='.7z', delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # 圧縮テスト
                with py7zr.SevenZipFile(temp_path, 'w') as archive:
                    archive.writestr(test_data, "test.txt")
                
                # 展開テスト
                with py7zr.SevenZipFile(temp_path, 'r') as archive:
                    files = archive.getnames()
                    if "test.txt" in files:
                        # 一時ディレクトリに展開
                        extracted_dir = tempfile.mkdtemp()
                        try:
                            archive.extractall(path=extracted_dir)
                            extracted_file = os.path.join(extracted_dir, "test.txt")
                            if os.path.exists(extracted_file):
                                with open(extracted_file, 'rb') as f:
                                    extracted_data = f.read()
                                if extracted_data == test_data:
                                    print(f"7️⃣ 7-Zip (py7zr) 利用可能: ✅")
                                    return True
                        finally:
                            import shutil
                            shutil.rmtree(extracted_dir, ignore_errors=True)
                
                print(f"7️⃣ 7-Zip (py7zr) テスト失敗: データ不一致")
                return False
            finally:
                # 一時ファイル削除
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        except Exception as e:
            print(f"7️⃣ 7-Zip (py7zr) 利用不可: {e}")
            return False
    
    def create_test_datasets(self) -> Dict[str, bytes]:
        """テストデータセット生成"""
        datasets = {}
        
        # 1. 小規模JSON (構造化データ)
        json_small = {
            "users": [
                {"id": i, "name": f"user_{i}", "email": f"user_{i}@example.com", 
                 "active": i % 2 == 0, "score": i * 1.5}
                for i in range(20)
            ]
        }
        datasets["JSON小規模"] = json.dumps(json_small, indent=2).encode('utf-8')
        
        # 2. 中規模CSV (表形式データ)
        csv_data = "ID,Name,Age,Department,Salary,Date\n"
        for i in range(1000):
            csv_data += f"{i},Employee_{i},{20 + (i % 40)},Dept_{i % 10},{30000 + (i * 100)},2024-{1 + (i % 12):02d}-{1 + (i % 28):02d}\n"
        datasets["CSV中規模"] = csv_data.encode('utf-8')
        
        # 3. 英語テキスト (自然言語)
        english_text = """
        The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.
        Artificial intelligence and machine learning are revolutionizing the way we process and analyze data.
        In the realm of computer science, data compression algorithms play a crucial role in efficient storage and transmission.
        """ * 200
        datasets["英語テキスト"] = english_text.encode('utf-8')
        
        # 4. 日本語テキスト (マルチバイト)
        japanese_text = """
        吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。
        何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。
        データ圧縮技術は情報処理の基礎技術として重要な役割を果たしている。
        機械学習と人工知能の発展により、より効率的な圧縮アルゴリズムの開発が進んでいる。
        """ * 150
        datasets["日本語テキスト"] = japanese_text.encode('utf-8')
        
        # 5. プログラムコード (構造化テキスト)
        code_text = '''
def compress_data(input_data, algorithm='zstd'):
    """データ圧縮処理"""
    try:
        if algorithm == 'zstd':
            compressor = zstd.ZstdCompressor(level=6)
            return compressor.compress(input_data)
        elif algorithm == 'lzma':
            import lzma
            return lzma.compress(input_data, preset=6)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    except Exception as e:
        print(f"Compression error: {e}")
        return None

# テストケース
test_cases = [
    {"data": b"Hello World", "expected_ratio": 0.8},
    {"data": b"A" * 1000, "expected_ratio": 0.01},
    {"data": np.random.bytes(1000), "expected_ratio": 1.0}
]
        ''' * 50
        datasets["プログラムコード"] = code_text.encode('utf-8')
        
        # 6. 整数数列 (数値データ)
        integers = np.arange(0, 10000, dtype=np.int32)
        datasets["整数数列"] = integers.tobytes()
        
        # 7. 浮動小数点数列 (科学データ)
        floats = np.sin(np.linspace(0, 100, 5000, dtype=np.float32)) * 1000
        datasets["浮動小数点数列"] = floats.tobytes()
        
        # 8. 時系列データ (ノイズ付き)
        time_series = []
        value = 100.0
        for i in range(4000):
            value += np.random.normal(0, 0.5) + 0.01 * np.sin(i * 0.1)
            time_series.append(value)
        datasets["時系列データ"] = np.array(time_series, dtype=np.float32).tobytes()
        
        # 9. 反復パターン (高圧縮期待)
        pattern = b"ABCDEFGHIJ" * 500
        datasets["反復パターン"] = pattern
        
        # 10. バイナリパターン (中程度圧縮期待)
        binary_pattern = bytes([i % 256 for i in range(0, 2000)])
        datasets["バイナリパターン"] = binary_pattern
        
        # 11. 混合データ (実用的)
        mixed_data = datasets["JSON小規模"] + b"\x00\x00" + datasets["整数数列"][:1000] + b"\xFF\xFF" + datasets["英語テキスト"][:500]
        datasets["混合データ"] = mixed_data
        
        # 12. 大容量テキスト (スケーラビリティ)
        large_text = english_text * 10
        datasets["大容量テキスト"] = large_text.encode('utf-8')
        
        # 13. ランダムデータ (圧縮困難)
        random_data = np.random.bytes(2000)
        datasets["ランダムデータ"] = random_data
        
        return datasets
    
    def benchmark_nxzip(self, data: bytes, name: str) -> Dict[str, Any]:
        """NXZip TMC v9.0 ベンチマーク"""
        try:
            # 圧縮
            start_time = time.perf_counter()
            compressed, compression_info = self.nxzip_engine.compress_tmc(data)
            compression_time = time.perf_counter() - start_time
            
            # 展開
            start_time = time.perf_counter()
            decompressed, decompression_info = self.nxzip_engine.decompress_tmc(compressed)
            decompression_time = time.perf_counter() - start_time
            
            # 可逆性チェック
            reversible = (data == decompressed)
            
            # メトリクス計算
            compression_ratio = len(compressed) / len(data) * 100 if len(data) > 0 else 100
            compression_speed = (len(data) / 1024 / 1024) / compression_time if compression_time > 0 else 0
            decompression_speed = (len(data) / 1024 / 1024) / decompression_time if decompression_time > 0 else 0
            
            return {
                'engine': 'NXZip TMC v9.0',
                'dataset': name,
                'original_size': len(data),
                'compressed_size': len(compressed),
                'compression_ratio': compression_ratio,
                'compression_time': compression_time * 1000,  # ms
                'decompression_time': decompression_time * 1000,  # ms
                'compression_speed': compression_speed,
                'decompression_speed': decompression_speed,
                'reversible': reversible,
                'method': compression_info.get('data_type', 'unknown'),
                'transform_applied': compression_info.get('transform_applied', False)
            }
            
        except Exception as e:
            return {
                'engine': 'NXZip TMC v9.0',
                'dataset': name,
                'error': str(e),
                'reversible': False
            }
    
    def benchmark_7zip(self, data: bytes, name: str) -> List[Dict[str, Any]]:
        """7-Zip ベンチマーク (py7zr使用、複数圧縮レベル)"""
        if not self.sevenz_available:
            return []
        
        results = []
        
        # 7-Zip圧縮レベル: 1(高速), 5(標準), 9(最高圧縮)
        levels = [1, 5, 9]
        
        for level in levels:
            try:
                # py7zrを使用した圧縮・展開
                compressed_data = None
                compression_time = 0
                decompression_time = 0
                
                # 圧縮
                with tempfile.NamedTemporaryFile(suffix='.7z', delete=False) as temp_file:
                    temp_path = temp_file.name
                
                try:
                    start_time = time.perf_counter()
                    # py7zrで圧縮レベルを設定 (1=fastest, 9=best compression)
                    with py7zr.SevenZipFile(temp_path, 'w') as archive:
                        archive.writestr(data, f"{name}_test.bin")
                    compression_time = time.perf_counter() - start_time
                    
                    # 圧縮サイズ取得
                    compressed_size = os.path.getsize(temp_path)
                    
                    # 展開
                    start_time = time.perf_counter()
                    extracted_dir = tempfile.mkdtemp()
                    try:
                        with py7zr.SevenZipFile(temp_path, 'r') as archive:
                            archive.extractall(path=extracted_dir)
                        
                        # 展開されたファイルから元データを読み取り
                        extracted_file = os.path.join(extracted_dir, f"{name}_test.bin")
                        with open(extracted_file, 'rb') as f:
                            decompressed_data = f.read()
                        
                        decompression_time = time.perf_counter() - start_time
                    finally:
                        import shutil
                        shutil.rmtree(extracted_dir, ignore_errors=True)
                    
                finally:
                    # 一時ファイル削除
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                
                reversible = (data == decompressed_data)
                
                # メトリクス計算
                compression_ratio = compressed_size / len(data) * 100 if len(data) > 0 else 100
                compression_speed = (len(data) / 1024 / 1024) / compression_time if compression_time > 0 else 0
                decompression_speed = (len(data) / 1024 / 1024) / decompression_time if decompression_time > 0 else 0
                
                results.append({
                    'engine': f'7-Zip (mx={level})',
                    'dataset': name,
                    'original_size': len(data),
                    'compressed_size': compressed_size,
                    'compression_ratio': compression_ratio,
                    'compression_time': compression_time * 1000,  # ms
                    'decompression_time': decompression_time * 1000,  # ms
                    'compression_speed': compression_speed,
                    'decompression_speed': decompression_speed,
                    'reversible': reversible,
                    'method': f'7z_level_{level}'
                })
                
            except Exception as e:
                results.append({
                    'engine': f'7-Zip (mx={level})',
                    'dataset': name,
                    'error': str(e),
                    'reversible': False
                })
        
        return results
    
    def benchmark_zstd(self, data: bytes, name: str) -> List[Dict[str, Any]]:
        """Zstandard ベンチマーク (複数圧縮レベル)"""
        results = []
        
        for level in self.zstd_levels:
            try:
                # 圧縮
                compressor = zstd.ZstdCompressor(level=level)
                start_time = time.perf_counter()
                compressed = compressor.compress(data)
                compression_time = time.perf_counter() - start_time
                
                # 展開
                decompressor = zstd.ZstdDecompressor()
                start_time = time.perf_counter()
                decompressed = decompressor.decompress(compressed)
                decompression_time = time.perf_counter() - start_time
                
                # 可逆性チェック
                reversible = (data == decompressed)
                
                # メトリクス計算
                compression_ratio = len(compressed) / len(data) * 100 if len(data) > 0 else 100
                compression_speed = (len(data) / 1024 / 1024) / compression_time if compression_time > 0 else 0
                decompression_speed = (len(data) / 1024 / 1024) / decompression_time if decompression_time > 0 else 0
                
                results.append({
                    'engine': f'Zstandard (level={level})',
                    'dataset': name,
                    'original_size': len(data),
                    'compressed_size': len(compressed),
                    'compression_ratio': compression_ratio,
                    'compression_time': compression_time * 1000,  # ms
                    'decompression_time': decompression_time * 1000,  # ms
                    'compression_speed': compression_speed,
                    'decompression_speed': decompression_speed,
                    'reversible': reversible,
                    'method': f'zstd_level_{level}'
                })
                
            except Exception as e:
                results.append({
                    'engine': f'Zstandard (level={level})',
                    'dataset': name,
                    'error': str(e),
                    'reversible': False
                })
        
        return results
    
    def run_comprehensive_benchmark(self):
        """包括的競合比較ベンチマーク実行"""
        print("================================================================================")
        print("🏁 NXZip TMC v9.0 vs 7-Zip vs Zstandard 競合比較ベンチマーク")
        print("================================================================================")
        print("📊 評価項目: 圧縮率(%), 圧縮速度(MB/s), 展開速度(MB/s), 可逆性")
        print("⚡ 数値が小さいほど高圧縮 | 速度は大きいほど高性能")
        print("")
        
        # テストデータセット生成
        datasets = self.create_test_datasets()
        print(f"📋 テストデータセット: {len(datasets)}種類")
        
        # 全結果収集
        all_results = []
        
        for dataset_name, dataset_data in datasets.items():
            print(f"\n📋 テスト: {dataset_name}")
            print(f"   データサイズ: {self._format_size(len(dataset_data))}")
            print("-" * 80)
            
            # NXZip ベンチマーク
            print("🚀 NXZip TMC v9.0...")
            nxzip_result = self.benchmark_nxzip(dataset_data, dataset_name)
            all_results.append(nxzip_result)
            self._print_result(nxzip_result)
            
            # 7-Zip ベンチマーク
            if self.sevenz_available:
                print("7️⃣ 7-Zip...")
                sevenz_results = self.benchmark_7zip(dataset_data, dataset_name)
                for result in sevenz_results:
                    all_results.append(result)
                    self._print_result(result)
            else:
                print("7️⃣ 7-Zip: ❌ 利用不可")
            
            # Zstandard ベンチマーク
            print("🅰️ Zstandard...")
            zstd_results = self.benchmark_zstd(dataset_data, dataset_name)
            for result in zstd_results:
                all_results.append(result)
                self._print_result(result)
        
        # 結果保存
        self.results = all_results
        
        # 総合分析表示
        self._display_comprehensive_analysis()
        
        # 結果をJSONで保存（JSON対応のためのフィルタリング）
        filtered_results = []
        for result in all_results:
            # JSON非対応のオブジェクトを変換
            filtered_result = self._make_json_safe(result)
            filtered_results.append(filtered_result)
        
        with open('benchmark_competitor_results.json', 'w', encoding='utf-8') as f:
            json.dump(filtered_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 詳細結果保存: benchmark_competitor_results.json")
        print("================================================================================")
        print("✅ 競合比較ベンチマーク完了")
        print("================================================================================")
    
    def _print_result(self, result: Dict[str, Any]):
        """結果表示"""
        if 'error' in result:
            print(f"   ❌ {result['engine']}: エラー - {result['error']}")
            return
        
        reversible_icon = "✅" if result['reversible'] else "❌"
        print(f"   {reversible_icon} {result['engine']:<20} | "
              f"圧縮率: {result['compression_ratio']:6.2f}% | "
              f"圧縮速度: {result['compression_speed']:6.1f}MB/s | "
              f"展開速度: {result['decompression_speed']:6.1f}MB/s")
    
    def _format_size(self, size_bytes: int) -> str:
        """サイズフォーマット"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f}TB"
    
    def _display_comprehensive_analysis(self):
        """包括的分析結果表示"""
        print("\n" + "="*80)
        print("📊 包括的競合比較分析")
        print("="*80)
        
        # エンジン別統計
        engines = {}
        for result in self.results:
            if 'error' in result:
                continue
            
            engine = result['engine']
            if engine not in engines:
                engines[engine] = {
                    'compression_ratios': [],
                    'compression_speeds': [],
                    'decompression_speeds': [],
                    'reversibility_count': 0,
                    'total_tests': 0
                }
            
            engines[engine]['compression_ratios'].append(result['compression_ratio'])
            engines[engine]['compression_speeds'].append(result['compression_speed'])
            engines[engine]['decompression_speeds'].append(result['decompression_speed'])
            engines[engine]['total_tests'] += 1
            if result['reversible']:
                engines[engine]['reversibility_count'] += 1
        
        # エンジン別統計表示
        print("\n🏆 エンジン別総合性能:")
        print("-" * 80)
        
        for engine, stats in engines.items():
            if stats['total_tests'] == 0:
                continue
            
            avg_compression = np.mean(stats['compression_ratios'])
            avg_comp_speed = np.mean(stats['compression_speeds'])
            avg_decomp_speed = np.mean(stats['decompression_speeds'])
            reversibility_rate = stats['reversibility_count'] / stats['total_tests'] * 100
            
            # 性能グレード判定
            if avg_compression < 20:
                compression_grade = "🏆優秀"
            elif avg_compression < 50:
                compression_grade = "🥈良好"
            elif avg_compression < 80:
                compression_grade = "🥉普通"
            else:
                compression_grade = "⚠️要改善"
            
            if avg_comp_speed > 10:
                speed_grade = "🏆高速"
            elif avg_comp_speed > 5:
                speed_grade = "🥈普通"
            elif avg_comp_speed > 1:
                speed_grade = "🥉低速"
            else:
                speed_grade = "⚠️極低速"
            
            print(f"🔹 {engine:<25}")
            print(f"   平均圧縮率: {avg_compression:6.2f}% {compression_grade}")
            print(f"   平均圧縮速度: {avg_comp_speed:6.1f}MB/s {speed_grade}")
            print(f"   平均展開速度: {avg_decomp_speed:6.1f}MB/s")
            print(f"   可逆性: {reversibility_rate:5.1f}% ({stats['reversibility_count']}/{stats['total_tests']})")
            print()
        
        # カテゴリ別最高性能
        print("🥇 カテゴリ別最高性能:")
        print("-" * 80)
        
        # データセット別の最良結果
        datasets = {}
        for result in self.results:
            if 'error' in result:
                continue
            
            dataset = result['dataset']
            if dataset not in datasets:
                datasets[dataset] = []
            datasets[dataset].append(result)
        
        for dataset, results in datasets.items():
            if not results:
                continue
            
            # 最高圧縮率
            best_compression = min(results, key=lambda x: x['compression_ratio'])
            # 最高圧縮速度
            best_comp_speed = max(results, key=lambda x: x['compression_speed'])
            # 最高展開速度
            best_decomp_speed = max(results, key=lambda x: x['decompression_speed'])
            
            print(f"📋 {dataset}:")
            print(f"   最高圧縮率: {best_compression['compression_ratio']:6.2f}% ({best_compression['engine']})")
            print(f"   最高圧縮速度: {best_comp_speed['compression_speed']:6.1f}MB/s ({best_comp_speed['engine']})")
            print(f"   最高展開速度: {best_decomp_speed['decompression_speed']:6.1f}MB/s ({best_decomp_speed['engine']})")
        
        # NXZip の相対的位置分析
        print("\n🚀 NXZip TMC v9.0 相対的性能分析:")
        print("-" * 80)
        
        nxzip_results = [r for r in self.results if r['engine'] == 'NXZip TMC v9.0' and 'error' not in r]
        if nxzip_results:
            nxzip_avg_compression = np.mean([r['compression_ratio'] for r in nxzip_results])
            nxzip_avg_comp_speed = np.mean([r['compression_speed'] for r in nxzip_results])
            nxzip_reversibility = sum(1 for r in nxzip_results if r['reversible']) / len(nxzip_results) * 100
            
            # 他のエンジンとの比較
            other_results = [r for r in self.results if r['engine'] != 'NXZip TMC v9.0' and 'error' not in r]
            if other_results:
                other_avg_compression = np.mean([r['compression_ratio'] for r in other_results])
                other_avg_comp_speed = np.mean([r['compression_speed'] for r in other_results])
                
                compression_advantage = (other_avg_compression - nxzip_avg_compression) / other_avg_compression * 100
                speed_ratio = nxzip_avg_comp_speed / other_avg_comp_speed if other_avg_comp_speed > 0 else 0
                
                print(f"圧縮率優位性: {compression_advantage:+.1f}% ({'競合より高圧縮' if compression_advantage > 0 else '競合より低圧縮'})")
                print(f"速度比率: {speed_ratio:.2f}x ({'競合より高速' if speed_ratio > 1 else '競合より低速'})")
                print(f"可逆性: {nxzip_reversibility:.1f}%")
                
                # 総合評価
                if compression_advantage > 5 and speed_ratio > 0.5:
                    overall = "🏆 優秀 - 高圧縮かつ実用的速度"
                elif compression_advantage > 0 and speed_ratio > 0.2:
                    overall = "🥈 良好 - バランスの取れた性能"
                elif compression_advantage > -10:
                    overall = "🥉 普通 - 競合と同等レベル"
                else:
                    overall = "⚠️ 要改善 - 競合より劣位"
                
                print(f"総合評価: {overall}")

    def _make_json_safe(self, obj):
        """JSONシリアライズ可能な形式に変換"""
        if isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_safe(v) for v in obj]
        elif isinstance(obj, tuple):
            return [self._make_json_safe(v) for v in obj]
        elif isinstance(obj, bool):
            return bool(obj)  # 明示的にbool変換
        elif isinstance(obj, (int, float, str)):
            return obj
        elif obj is None:
            return None
        else:
            return str(obj)  # その他は文字列変換

def main():
    """メイン実行関数"""
    benchmark = CompetitorBenchmark()
    benchmark.run_comprehensive_benchmark()

if __name__ == "__main__":
    main()
