#!/usr/bin/env python3
"""
TMC v6.0 vs 7-Zip特化ベンチマーク
実際の7-Zipバイナリとの詳細比較テスト
"""

import os
import sys
import time
import subprocess
import tempfile
import numpy as np
from pathlib import Path

# TMC Engine インポート
sys.path.append(os.path.join(os.path.dirname(__file__), 'nxzip', 'engine'))
from nxzip.engine.nexus_tmc_v4_unified import NEXUSTMCEngineV4


class SevenZipComparisonBenchmark:
    """7-Zip特化比較ベンチマーク"""
    
    def __init__(self):
        self.tmc_engine = NEXUSTMCEngineV4()
        self.check_7zip()
    
    def check_7zip(self):
        """7-Zipの利用可能性詳細チェック"""
        self.seven_zip_cmd = None
        
        # 複数の7-Zip実行ファイルを試行
        candidates = ['7z', '7za', '7zr']
        
        for cmd in candidates:
            try:
                result = subprocess.run([cmd], capture_output=True, timeout=5)
                self.seven_zip_cmd = cmd
                print(f"✅ 7-Zip found: {cmd}")
                break
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
                continue
        
        if not self.seven_zip_cmd:
            print("❌ 7-Zipが見つかりません。内蔵アルゴリズムのみでテストします。")
            # Windows環境での7-Zipの標準パスをチェック
            standard_paths = [
                r"C:\Program Files\7-Zip\7z.exe",
                r"C:\Program Files (x86)\7-Zip\7z.exe"
            ]
            
            for path in standard_paths:
                if os.path.exists(path):
                    self.seven_zip_cmd = path
                    print(f"✅ 7-Zip found at: {path}")
                    break
    
    def create_specialized_datasets(self) -> dict:
        """TMC v6.0が得意なデータセットを重点的に生成"""
        datasets = {}
        
        # 1. 高度系列整数データ（TMCの真価発揮）
        print("📊 高度系列整数データセット生成中...")
        sequential_large = np.arange(0, 50000, dtype=np.int32)
        datasets['Large_Sequential_200KB'] = sequential_large.tobytes()
        
        # 2. 数学的系列データ
        math_sequence = []
        for i in range(25000):
            math_sequence.append(i)           # linear component
            math_sequence.append(i * i % 1000) # quadratic component
        datasets['Mathematical_Sequence_200KB'] = np.array(math_sequence, dtype=np.int32).tobytes()
        
        # 3. 科学計算風浮動小数点データ
        print("📊 科学計算風データセット生成中...")
        scientific_floats = []
        for i in range(25000):
            value = 1000.0 + np.sin(i/100) * 100 + np.cos(i/50) * 50 + np.random.normal(0, 0.1)
            scientific_floats.append(value)
        datasets['Scientific_Floats_100KB'] = np.array(scientific_floats, dtype=np.float32).tobytes()
        
        # 4. IoTセンサーデータ風
        print("📊 IoTセンサーデータセット生成中...")
        sensor_data = []
        base_temp = 20.0
        for i in range(25000):
            # 温度センサーの値をシミュレート
            temp = base_temp + 5 * np.sin(i/1440) + np.random.normal(0, 0.5)  # 日周期
            sensor_data.append(temp)
        datasets['IoT_Sensor_Data_100KB'] = np.array(sensor_data, dtype=np.float32).tobytes()
        
        # 5. 時系列整数データ
        print("📊 時系列整数データセット生成中...")
        timeseries_ints = []
        base_value = 1000
        for i in range(50000):
            # ランダムウォーク + トレンド
            change = np.random.randint(-5, 6)  # -5 to +5
            base_value += change
            base_value = max(0, min(base_value, 10000))  # bounds
            timeseries_ints.append(base_value)
        datasets['Timeseries_Integers_200KB'] = np.array(timeseries_ints, dtype=np.int32).tobytes()
        
        print(f"✅ 特化データセット {len(datasets)}種類生成完了")
        for name, data in datasets.items():
            print(f"   {name}: {len(data):,} bytes")
        
        return datasets
    
    def test_7zip_detailed(self, data: bytes, dataset_name: str, compression_level: int = 9) -> dict:
        """7-Zipの詳細テスト（複数圧縮方式）"""
        if not self.seven_zip_cmd:
            return None
        
        results = {}
        
        # 複数の7-Zip圧縮方式をテスト
        test_methods = [
            ('7z_LZMA2', ['-t7z', f'-mx={compression_level}', '-m0=LZMA2']),
            ('7z_LZMA', ['-t7z', f'-mx={compression_level}', '-m0=LZMA']),
            ('7z_PPMd', ['-t7z', f'-mx={compression_level}', '-m0=PPMd']),
            ('ZIP_Deflate', ['-tzip', f'-mx={compression_level}']),
        ]
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                input_file = os.path.join(temp_dir, f"{dataset_name}.bin")
                
                # データをファイルに書き込み
                with open(input_file, 'wb') as f:
                    f.write(data)
                
                for method_name, method_args in test_methods:
                    try:
                        output_file = os.path.join(temp_dir, f"{dataset_name}_{method_name}.7z")
                        
                        # 7-Zip圧縮実行
                        start_time = time.perf_counter()
                        cmd = [self.seven_zip_cmd, 'a'] + method_args + [output_file, input_file]
                        
                        result = subprocess.run(cmd, capture_output=True, timeout=120, check=True)
                        compression_time = time.perf_counter() - start_time
                        
                        # 圧縮サイズを取得
                        if os.path.exists(output_file):
                            compressed_size = os.path.getsize(output_file)
                            original_size = len(data)
                            compression_ratio = (1 - compressed_size / original_size) * 100
                            compression_speed = (original_size / (1024 * 1024)) / compression_time if compression_time > 0 else 0
                            
                            results[method_name] = {
                                'original_size': original_size,
                                'compressed_size': compressed_size,
                                'compression_ratio': compression_ratio,
                                'compression_time': compression_time,
                                'compression_speed_mbps': compression_speed,
                                'successful': True
                            }
                            
                            print(f"  ✅ {method_name:15s}: {original_size:8,} -> {compressed_size:8,} bytes "
                                  f"({compression_ratio:6.2f}%) {compression_time*1000:7.1f}ms")
                        
                    except Exception as e:
                        print(f"  ❌ {method_name}: {str(e)}")
                        results[method_name] = {'successful': False, 'error': str(e)}
        
        except Exception as e:
            print(f"7-Zip詳細テストエラー: {e}")
            return None
        
        return results
    
    def run_specialized_comparison(self):
        """特化データセットでの詳細比較"""
        print("🚀 TMC v6.0 vs 7-Zip 特化比較ベンチマーク開始")
        print("=" * 70)
        
        datasets = self.create_specialized_datasets()
        
        comparison_results = []
        
        for dataset_name, dataset_data in datasets.items():
            print(f"\n🧪 詳細比較テスト: {dataset_name} ({len(dataset_data):,} bytes)")
            print("-" * 60)
            
            # TMC v6.0テスト
            print("TMC v6.0 圧縮テスト:")
            start_time = time.perf_counter()
            tmc_compressed, tmc_info = self.tmc_engine.compress_tmc(dataset_data)
            tmc_compression_time = time.perf_counter() - start_time
            
            # 可逆性確認
            tmc_decompressed, _ = self.tmc_engine.decompress_tmc(tmc_compressed)
            tmc_reversible = (dataset_data == tmc_decompressed)
            
            tmc_result = {
                'method': 'TMC_v6.0',
                'original_size': len(dataset_data),
                'compressed_size': len(tmc_compressed),
                'compression_ratio': (1 - len(tmc_compressed) / len(dataset_data)) * 100,
                'compression_time': tmc_compression_time,
                'compression_speed_mbps': (len(dataset_data) / (1024 * 1024)) / tmc_compression_time if tmc_compression_time > 0 else 0,
                'reversible': tmc_reversible,
                'data_type': tmc_info.get('data_type', 'unknown'),
                'transform_method': tmc_info.get('transform_info', {}).get('method', 'none')
            }
            
            print(f"  ✅ TMC v6.0        : {len(dataset_data):8,} -> {len(tmc_compressed):8,} bytes "
                  f"({tmc_result['compression_ratio']:6.2f}%) {tmc_compression_time*1000:7.1f}ms")
            print(f"      データタイプ: {tmc_result['data_type']}, 変換: {tmc_result['transform_method']}")
            
            # 7-Zip詳細テスト
            if self.seven_zip_cmd:
                print("7-Zip 詳細圧縮テスト:")
                seven_zip_results = self.test_7zip_detailed(dataset_data, dataset_name)
            else:
                seven_zip_results = {}
            
            # 結果比較
            print(f"\n📊 {dataset_name} 比較結果:")
            print("-" * 40)
            
            all_results = {'TMC_v6.0': tmc_result}
            if seven_zip_results:
                all_results.update(seven_zip_results)
            
            # 圧縮率でソート
            sorted_results = sorted(
                [(name, result) for name, result in all_results.items() if result.get('successful', True)],
                key=lambda x: x[1].get('compression_ratio', 0),
                reverse=True
            )
            
            for rank, (method_name, result) in enumerate(sorted_results, 1):
                ratio = result.get('compression_ratio', 0)
                speed = result.get('compression_speed_mbps', 0)
                crown = "🏆" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else ""
                print(f"  {rank}. {crown} {method_name:15s}: {ratio:6.2f}% ({speed:5.1f} MB/s)")
            
            comparison_results.append({
                'dataset_name': dataset_name,
                'dataset_size': len(dataset_data),
                'tmc_result': tmc_result,
                'seven_zip_results': seven_zip_results,
                'best_method': sorted_results[0][0] if sorted_results else None,
                'best_ratio': sorted_results[0][1].get('compression_ratio', 0) if sorted_results else 0
            })
        
        # 総合分析
        self.generate_detailed_analysis(comparison_results)
        
        return comparison_results
    
    def generate_detailed_analysis(self, results):
        """詳細分析レポートの生成"""
        print(f"\n🏆 TMC v6.0 vs 7-Zip 詳細比較分析")
        print("=" * 50)
        
        tmc_wins = 0
        total_tests = len(results)
        
        print(f"\n📈 データセット別勝敗:")
        print("-" * 30)
        
        for result in results:
            dataset_name = result['dataset_name']
            best_method = result['best_method']
            best_ratio = result['best_ratio']
            tmc_ratio = result['tmc_result']['compression_ratio']
            
            if best_method == 'TMC_v6.0':
                tmc_wins += 1
                status = "🏆 TMC勝利"
            else:
                diff = best_ratio - tmc_ratio
                status = f"敗北 (-{diff:.2f}%)"
            
            print(f"  {dataset_name:25s}: {tmc_ratio:6.2f}% {status}")
        
        win_rate = tmc_wins / total_tests * 100
        print(f"\n🎯 TMC v6.0 勝率: {tmc_wins}/{total_tests} ({win_rate:.1f}%)")
        
        # TMCが勝利したケースの分析
        if tmc_wins > 0:
            print(f"\n🔥 TMC v6.0優位領域:")
            print("-" * 25)
            for result in results:
                if result['best_method'] == 'TMC_v6.0':
                    tmc_res = result['tmc_result']
                    print(f"  {result['dataset_name']:25s}: {tmc_res['data_type']} + {tmc_res['transform_method']}")


def main():
    """メイン実行"""
    benchmark = SevenZipComparisonBenchmark()
    results = benchmark.run_specialized_comparison()
    print("\n🎉 TMC v6.0 vs 7-Zip 特化比較完了!")


if __name__ == "__main__":
    main()
