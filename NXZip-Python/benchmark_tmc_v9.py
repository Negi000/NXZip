#!/usr/bin/env python3
"""
TMC v9.0 包括的ベンチマーク評価システム
vs 7-Zip (LZMA2) & Zstandard
評価項目: 圧縮率、圧縮速度、展開速度、可逆性
"""

import os
import sys
import time
import zlib
import lzma
import subprocess
import hashlib
import json
import tempfile
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Any

try:
    import psutil
except ImportError:
    print("⚠️ psutil未利用 - pip install psutil でインストール可能")
    psutil = None

# NXZipエンジンのインポート
try:
    from nxzip.engine.nexus_tmc_v4_unified import NEXUSTMCEngineV9
except ImportError:
    # パスを手動追加
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    from nxzip.engine.nexus_tmc_v4_unified import NEXUSTMCEngineV9

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False
    print("⚠️ Zstandard未利用 - pip install zstandard でインストール可能")


class ComprehensiveBenchmark:
    """TMC v9.0 包括的ベンチマーク評価システム"""
    
    def __init__(self):
        self.results = {}
        self.test_data_paths = []
        self.temp_dir = Path(tempfile.mkdtemp(prefix="tmc_benchmark_"))
        
        # TMC v9.0エンジン初期化
        self.tmc_engine = NEXUSTMCEngineV9(max_workers=4, chunk_size=2*1024*1024)
        
        # 7-Zipパスの検索
        self.sevenzip_path = self._find_7zip_executable()
        
        print("🏁 TMC v9.0 ベンチマーク評価システム初期化完了")
        print(f"📁 一時ディレクトリ: {self.temp_dir}")
        print(f"🔧 7-Zip実行ファイル: {self.sevenzip_path}")
        print(f"📊 Zstandard利用可能: {ZSTD_AVAILABLE}")
    
    def _find_7zip_executable(self) -> str:
        """7-Zip実行ファイルを検索"""
        possible_paths = [
            r"C:\Program Files\7-Zip\7z.exe",
            r"C:\Program Files (x86)\7-Zip\7z.exe",
            "7z",  # PATH環境変数
            "7za"
        ]
        
        for path in possible_paths:
            try:
                result = subprocess.run([path, "--help"], 
                                      capture_output=True, timeout=5)
                if result.returncode == 0:
                    return path
            except:
                continue
        
        print("⚠️ 7-Zip実行ファイルが見つかりません")
        return None
    
    def prepare_test_data(self) -> List[Tuple[str, bytes, str]]:
        """多様なテストデータの準備"""
        test_datasets = []
        
        # 1. テキストデータ（高圧縮率期待）
        text_data = self._generate_text_data()
        test_datasets.append(("Text_Repetitive", text_data, "高反復テキスト"))
        
        # 2. JSONデータ（構造化データ）
        json_data = self._generate_json_data()
        test_datasets.append(("JSON_Structured", json_data, "構造化JSON"))
        
        # 3. バイナリデータ（低圧縮率）
        binary_data = self._generate_binary_data()
        test_datasets.append(("Binary_Random", binary_data, "ランダムバイナリ"))
        
        # 4. 数値データ（LeCo効果期待）
        numeric_data = self._generate_numeric_data()
        test_datasets.append(("Numeric_Sequence", numeric_data, "数値シーケンス"))
        
        # 5. 画像風データ（ピクセル類似性）
        image_like_data = self._generate_image_like_data()
        test_datasets.append(("Image_Like", image_like_data, "画像風データ"))
        
        # 6. 既存ファイルがあれば追加
        sample_dir = Path("NXZip-Python/sample")
        if sample_dir.exists():
            for file_path in sample_dir.glob("*"):
                if file_path.is_file() and file_path.stat().st_size < 50 * 1024 * 1024:
                    try:
                        with open(file_path, 'rb') as f:
                            file_data = f.read()
                        test_datasets.append((f"File_{file_path.name}", file_data, f"実ファイル: {file_path.name}"))
                    except:
                        continue
        
        print(f"📋 テストデータセット準備完了: {len(test_datasets)}種類")
        for name, data, desc in test_datasets:
            print(f"  - {name}: {len(data):,} bytes ({desc})")
        
        return test_datasets
    
    def _generate_text_data(self) -> bytes:
        """反復性の高いテキストデータ生成"""
        base_text = "The quick brown fox jumps over the lazy dog. " * 100
        patterns = [
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 50,
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 200,
            "1234567890" * 500,
            "Hello World! " * 1000
        ]
        
        full_text = base_text + "\n".join(patterns) * 10
        return full_text.encode('utf-8')
    
    def _generate_json_data(self) -> bytes:
        """構造化JSONデータ生成"""
        json_structure = {
            "users": [
                {
                    "id": i,
                    "name": f"User_{i:04d}",
                    "email": f"user{i}@example.com",
                    "score": i * 1.5,
                    "metadata": {
                        "created_at": "2024-01-01T00:00:00Z",
                        "updated_at": "2024-12-31T23:59:59Z",
                        "tags": ["tag1", "tag2", "tag3"] * (i % 3 + 1)
                    }
                } for i in range(1000)
            ],
            "config": {
                "version": "1.0.0",
                "settings": {
                    "compression": True,
                    "encryption": False,
                    "backup": True
                } 
            }
        }
        
        return json.dumps(json_structure, ensure_ascii=False, indent=2).encode('utf-8')
    
    def _generate_binary_data(self) -> bytes:
        """ランダムバイナリデータ生成（低圧縮率）"""
        import random
        random.seed(42)  # 再現性のため
        return bytes([random.randint(0, 255) for _ in range(1024 * 1024)])  # 1MB
    
    def _generate_numeric_data(self) -> bytes:
        """数値シーケンスデータ生成（LeCo効果期待）"""
        import struct
        numbers = []
        
        # 等差数列
        for i in range(0, 100000, 3):
            numbers.append(i)
        
        # フィボナッチ数列
        a, b = 1, 1
        for _ in range(10000):
            numbers.append(a)
            a, b = b, a + b
            if a > 2**30:  # オーバーフロー防止
                a, b = 1, 1
        
        # 整数を4バイトバイナリに変換
        binary_data = b''.join(struct.pack('<I', num & 0xFFFFFFFF) for num in numbers)
        return binary_data
    
    def _generate_image_like_data(self) -> bytes:
        """画像風データ生成（ピクセル隣接類似性）"""
        width, height = 256, 256
        data = bytearray()
        
        for y in range(height):
            for x in range(width):
                # グラデーション + ノイズ
                base_value = int(255 * (x + y) / (width + height))
                noise = (x * y) % 30 - 15  # 少しのノイズ
                pixel_value = max(0, min(255, base_value + noise))
                
                # RGB (3バイト/ピクセル)
                data.extend([pixel_value, pixel_value//2, pixel_value//3])
        
        return bytes(data)
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """包括的ベンチマーク実行"""
        print("\n🚀 TMC v9.0 包括的ベンチマーク評価開始")
        print("=" * 80)
        
        test_datasets = self.prepare_test_data()
        
        benchmark_results = {
            'test_info': {
                'engine_version': 'TMC v9.0 Unified',
                'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'system_info': self._get_system_info(),
                'datasets_count': len(test_datasets)
            },
            'results': {}
        }
        
        # 各データセットでテスト実行
        for dataset_name, test_data, description in test_datasets:
            print(f"\n📊 データセット: {dataset_name} ({description})")
            print(f"📏 データサイズ: {len(test_data):,} bytes ({len(test_data)/1024/1024:.2f} MB)")
            
            # データ整合性チェック用ハッシュ
            original_hash = hashlib.sha256(test_data).hexdigest()
            
            dataset_results = {
                'original_size': len(test_data),
                'original_hash': original_hash,
                'description': description,
                'compressors': {}
            }
            
            # 1. TMC v9.0テスト
            tmc_result = self._test_tmc_v9(test_data, dataset_name)
            dataset_results['compressors']['TMC_v9'] = tmc_result
            
            # 2. 7-Zipテスト (LZMA2)
            if self.sevenzip_path:
                sevenzip_result = self._test_7zip(test_data, dataset_name)
                dataset_results['compressors']['7-Zip_LZMA2'] = sevenzip_result
            
            # 3. Zstandardテスト
            if ZSTD_AVAILABLE:
                zstd_result = self._test_zstandard(test_data, dataset_name)
                dataset_results['compressors']['Zstandard'] = zstd_result
            
            # 4. 標準zlibテスト（参考）
            zlib_result = self._test_zlib(test_data, dataset_name)
            dataset_results['compressors']['zlib'] = zlib_result
            
            benchmark_results['results'][dataset_name] = dataset_results
            
            # 中間結果表示
            self._print_dataset_summary(dataset_name, dataset_results)
        
        # 総合結果分析
        self._analyze_overall_results(benchmark_results)
        
        # 結果をJSONファイルに保存
        result_file = self.temp_dir / "tmc_v9_benchmark_results.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            # JSON シリアライゼーション対応
            json_safe_results = self._make_json_serializable(benchmark_results)
            json.dump(json_safe_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 詳細結果を保存: {result_file}")
        
        return benchmark_results
    
    def _test_tmc_v9(self, data: bytes, dataset_name: str) -> Dict[str, Any]:
        """TMC v9.0エンジンテスト"""
        print("  🧠 TMC v9.0 テスト実行中...")
        
        try:
            # 圧縮テスト
            compress_start = time.perf_counter()
            compressed_data, compression_info = self.tmc_engine.compress_tmc(data)
            compress_time = time.perf_counter() - compress_start
            
            # 展開テスト
            decompress_start = time.perf_counter()
            decompressed_data, decompress_info = self.tmc_engine.decompress_tmc(compressed_data)
            decompress_time = time.perf_counter() - decompress_start
            
            # 整合性チェック
            is_lossless = (data == decompressed_data)
            compression_ratio = len(compressed_data) / len(data)
            
            # 速度計算（ゼロ除算対策）
            compression_speed = (len(data) / (1024 * 1024)) / compress_time if compress_time > 0 else 0.0
            decompression_speed = (len(data) / (1024 * 1024)) / decompress_time if decompress_time > 0 else 0.0
            
            result = {
                'compressed_size': len(compressed_data),
                'compression_ratio': compression_ratio,
                'compression_time': compress_time,
                'decompression_time': decompress_time,
                'compression_speed_mbps': compression_speed,
                'decompression_speed_mbps': decompression_speed,
                'is_lossless': is_lossless,
                'space_saving_percent': (1 - compression_ratio) * 100,
                'compression_info': compression_info,
                'status': 'success'
            }
            
            print(f"    ✅ 圧縮率: {compression_ratio:.3f} ({result['space_saving_percent']:.1f}% 削減)")
            print(f"    ⚡ 圧縮速度: {result['compression_speed_mbps']:.1f} MB/s")
            print(f"    ⚡ 展開速度: {result['decompression_speed_mbps']:.1f} MB/s")
            print(f"    🔍 可逆性: {'✅ 完全' if is_lossless else '❌ 不完全'}")
            
        except Exception as e:
            result = {
                'status': 'error',
                'error': str(e),
                'compression_ratio': float('inf'),
                'is_lossless': False
            }
            print(f"    ❌ TMC v9.0 エラー: {e}")
        
        return result
    
    def _test_7zip(self, data: bytes, dataset_name: str) -> Dict[str, Any]:
        """7-Zip (LZMA2) テスト"""
        print("  📦 7-Zip (LZMA2) テスト実行中...")
        
        try:
            # 一時ファイル作成
            input_file = self.temp_dir / f"{dataset_name}_input.bin"
            compressed_file = self.temp_dir / f"{dataset_name}_7z.7z"
            output_file = self.temp_dir / f"{dataset_name}_7z_output.bin"
            
            # 入力ファイル書き込み
            with open(input_file, 'wb') as f:
                f.write(data)
            
            # 圧縮実行
            compress_start = time.perf_counter()
            compress_result = subprocess.run([
                self.sevenzip_path, 'a', '-t7z', '-m0=lzma2', '-mx=9',
                str(compressed_file), str(input_file)
            ], capture_output=True, timeout=300)
            compress_time = time.perf_counter() - compress_start
            
            if compress_result.returncode != 0:
                raise Exception(f"7-Zip圧縮エラー: {compress_result.stderr.decode()}")
            
            # 圧縮ファイルサイズ取得
            compressed_size = compressed_file.stat().st_size
            
            # 展開実行
            decompress_start = time.perf_counter()
            decompress_result = subprocess.run([
                self.sevenzip_path, 'e', str(compressed_file), 
                f'-o{self.temp_dir}', '-y'
            ], capture_output=True, timeout=300)
            decompress_time = time.perf_counter() - decompress_start
            
            if decompress_result.returncode != 0:
                raise Exception(f"7-Zip展開エラー: {decompress_result.stderr.decode()}")
            
            # 展開結果読み込み
            extracted_file = self.temp_dir / f"{dataset_name}_input.bin"  # 7-zipは元のファイル名で展開
            with open(extracted_file, 'rb') as f:
                decompressed_data = f.read()
            
            # 整合性チェック
            is_lossless = (data == decompressed_data)
            compression_ratio = compressed_size / len(data)
            
            # 速度計算（ゼロ除算対策）
            compression_speed = (len(data) / (1024 * 1024)) / compress_time if compress_time > 0 else 0.0
            decompression_speed = (len(data) / (1024 * 1024)) / decompress_time if decompress_time > 0 else 0.0
            
            result = {
                'compressed_size': compressed_size,
                'compression_ratio': compression_ratio,
                'compression_time': compress_time,
                'decompression_time': decompress_time,
                'compression_speed_mbps': compression_speed,
                'decompression_speed_mbps': decompression_speed,
                'is_lossless': is_lossless,
                'space_saving_percent': (1 - compression_ratio) * 100,
                'status': 'success'
            }
            
            print(f"    ✅ 圧縮率: {compression_ratio:.3f} ({result['space_saving_percent']:.1f}% 削減)")
            print(f"    ⚡ 圧縮速度: {result['compression_speed_mbps']:.1f} MB/s")
            print(f"    ⚡ 展開速度: {result['decompression_speed_mbps']:.1f} MB/s")
            print(f"    🔍 可逆性: {'✅ 完全' if is_lossless else '❌ 不完全'}")
            
            # 一時ファイルクリーンアップ
            for temp_file in [input_file, compressed_file, extracted_file]:
                if temp_file.exists():
                    temp_file.unlink()
            
        except Exception as e:
            result = {
                'status': 'error',
                'error': str(e),
                'compression_ratio': float('inf'),
                'is_lossless': False
            }
            print(f"    ❌ 7-Zip エラー: {e}")
        
        return result
    
    def _test_zstandard(self, data: bytes, dataset_name: str) -> Dict[str, Any]:
        """Zstandard テスト"""
        print("  🔥 Zstandard テスト実行中...")
        
        try:
            # 圧縮テスト (最高圧縮レベル)
            compressor = zstd.ZstdCompressor(level=22)  # 最高圧縮率
            
            compress_start = time.perf_counter()
            compressed_data = compressor.compress(data)
            compress_time = time.perf_counter() - compress_start
            
            # 展開テスト
            decompressor = zstd.ZstdDecompressor()
            
            decompress_start = time.perf_counter()
            decompressed_data = decompressor.decompress(compressed_data)
            decompress_time = time.perf_counter() - decompress_start
            
            # 整合性チェック
            is_lossless = (data == decompressed_data)
            compression_ratio = len(compressed_data) / len(data)
            
            result = {
                'compressed_size': len(compressed_data),
                'compression_ratio': compression_ratio,
                'compression_time': compress_time,
                'decompression_time': decompress_time,
                'compression_speed_mbps': (len(data) / compress_time) / (1024 * 1024),
                'decompression_speed_mbps': (len(data) / decompress_time) / (1024 * 1024),
                'is_lossless': is_lossless,
                'space_saving_percent': (1 - compression_ratio) * 100,
                'status': 'success'
            }
            
            print(f"    ✅ 圧縮率: {compression_ratio:.3f} ({result['space_saving_percent']:.1f}% 削減)")
            print(f"    ⚡ 圧縮速度: {result['compression_speed_mbps']:.1f} MB/s")
            print(f"    ⚡ 展開速度: {result['decompression_speed_mbps']:.1f} MB/s")
            print(f"    🔍 可逆性: {'✅ 完全' if is_lossless else '❌ 不完全'}")
            
        except Exception as e:
            result = {
                'status': 'error', 
                'error': str(e),
                'compression_ratio': float('inf'),
                'is_lossless': False
            }
            print(f"    ❌ Zstandard エラー: {e}")
        
        return result
    
    def _test_zlib(self, data: bytes, dataset_name: str) -> Dict[str, Any]:
        """標準zlib テスト（参考）"""
        print("  📋 zlib (参考) テスト実行中...")
        
        try:
            # 圧縮テスト
            compress_start = time.perf_counter()
            compressed_data = zlib.compress(data, level=9)
            compress_time = time.perf_counter() - compress_start
            
            # 展開テスト
            decompress_start = time.perf_counter()
            decompressed_data = zlib.decompress(compressed_data)
            decompress_time = time.perf_counter() - decompress_start
            
            # 整合性チェック
            is_lossless = (data == decompressed_data)
            compression_ratio = len(compressed_data) / len(data)
            
            # 速度計算（ゼロ除算対策）
            compression_speed = (len(data) / (1024 * 1024)) / compress_time if compress_time > 0 else 0.0
            decompression_speed = (len(data) / (1024 * 1024)) / decompress_time if decompress_time > 0 else 0.0
            
            result = {
                'compressed_size': len(compressed_data),
                'compression_ratio': compression_ratio,
                'compression_time': compress_time,
                'decompression_time': decompress_time,
                'compression_speed_mbps': compression_speed,
                'decompression_speed_mbps': decompression_speed,
                'is_lossless': is_lossless,
                'space_saving_percent': (1 - compression_ratio) * 100,
                'status': 'success'
            }
            
            print(f"    ✅ 圧縮率: {compression_ratio:.3f} ({result['space_saving_percent']:.1f}% 削減)")
            
        except Exception as e:
            result = {
                'status': 'error',
                'error': str(e),
                'compression_ratio': float('inf'),
                'is_lossless': False
            }
            print(f"    ❌ zlib エラー: {e}")
        
        return result
    
    def _print_dataset_summary(self, dataset_name: str, results: Dict[str, Any]):
        """データセット結果サマリー表示"""
        print(f"\n📈 {dataset_name} 結果サマリー:")
        print("-" * 60)
        
        compressors = results['compressors']
        
        # 圧縮率ランキング
        valid_compressors = {name: comp for name, comp in compressors.items() 
                           if comp.get('status') == 'success'}
        
        if valid_compressors:
            # 圧縮率順（小さい方が良い）
            compression_ranking = sorted(valid_compressors.items(), 
                                       key=lambda x: x[1]['compression_ratio'])
            
            print("🏆 圧縮率ランキング (良い順):")
            for i, (name, comp) in enumerate(compression_ranking, 1):
                ratio = comp['compression_ratio']
                saving = comp['space_saving_percent']
                print(f"  {i}. {name}: {ratio:.3f} ({saving:.1f}% 削減)")
            
            # 圧縮速度ランキング
            speed_ranking = sorted(valid_compressors.items(),
                                 key=lambda x: x[1]['compression_speed_mbps'], reverse=True)
            
            print("\n⚡ 圧縮速度ランキング (速い順):")
            for i, (name, comp) in enumerate(speed_ranking, 1):
                speed = comp['compression_speed_mbps']
                print(f"  {i}. {name}: {speed:.1f} MB/s")
    
    def _analyze_overall_results(self, benchmark_results: Dict[str, Any]):
        """総合結果分析"""
        print("\n" + "=" * 80)
        print("🎯 TMC v9.0 総合ベンチマーク結果分析")
        print("=" * 80)
        
        all_results = benchmark_results['results']
        compressor_names = set()
        
        # 利用可能な圧縮器を収集
        for dataset_results in all_results.values():
            compressor_names.update(dataset_results['compressors'].keys())
        
        # 各圧縮器の総合スコア計算
        compressor_scores = {}
        
        for compressor in compressor_names:
            compression_ratios = []
            compression_speeds = []
            decompression_speeds = []
            lossless_count = 0
            total_count = 0
            
            for dataset_name, dataset_results in all_results.items():
                if compressor in dataset_results['compressors']:
                    comp_result = dataset_results['compressors'][compressor]
                    
                    if comp_result.get('status') == 'success':
                        compression_ratios.append(comp_result['compression_ratio'])
                        compression_speeds.append(comp_result['compression_speed_mbps'])
                        decompression_speeds.append(comp_result['decompression_speed_mbps'])
                        
                        if comp_result['is_lossless']:
                            lossless_count += 1
                        total_count += 1
            
            if compression_ratios:
                avg_compression_ratio = sum(compression_ratios) / len(compression_ratios)
                avg_compression_speed = sum(compression_speeds) / len(compression_speeds)
                avg_decompression_speed = sum(decompression_speeds) / len(decompression_speeds)
                lossless_rate = lossless_count / total_count if total_count > 0 else 0
                
                # 総合スコア計算 (低い方が良い)
                # 圧縮率50%, 圧縮速度25%, 展開速度20%, 可逆性5%
                normalized_compression = avg_compression_ratio  # 小さい方が良い
                normalized_comp_speed = 1.0 / (avg_compression_speed + 0.1)  # 大きい方が良い→小さくする
                normalized_decomp_speed = 1.0 / (avg_decompression_speed + 0.1)  # 大きい方が良い→小さくする
                normalized_lossless = 1.0 - lossless_rate  # 高い方が良い→小さくする
                
                total_score = (normalized_compression * 0.5 +
                             normalized_comp_speed * 0.25 +
                             normalized_decomp_speed * 0.2 +
                             normalized_lossless * 0.05)
                
                compressor_scores[compressor] = {
                    'avg_compression_ratio': avg_compression_ratio,
                    'avg_compression_speed': avg_compression_speed,
                    'avg_decompression_speed': avg_decompression_speed,
                    'lossless_rate': lossless_rate,
                    'total_score': total_score,
                    'test_count': total_count
                }
        
        # 結果表示
        print("\n📊 総合パフォーマンス比較:")
        print("-" * 80)
        
        # スコア順でソート (低い方が良い)
        ranked_compressors = sorted(compressor_scores.items(), key=lambda x: x[1]['total_score'])
        
        for i, (name, scores) in enumerate(ranked_compressors, 1):
            print(f"\n🏅 {i}位: {name}")
            print(f"   📦 平均圧縮率: {scores['avg_compression_ratio']:.3f}")
            print(f"   ⚡ 平均圧縮速度: {scores['avg_compression_speed']:.1f} MB/s")
            print(f"   ⚡ 平均展開速度: {scores['avg_decompression_speed']:.1f} MB/s")
            print(f"   🔍 可逆性成功率: {scores['lossless_rate']:.1%}")
            print(f"   🎯 総合スコア: {scores['total_score']:.3f}")
            print(f"   📋 テスト数: {scores['test_count']}")
        
        # TMC v9.0の詳細分析
        if 'TMC_v9' in compressor_scores:
            tmc_scores = compressor_scores['TMC_v9']
            print(f"\n🧠 TMC v9.0 詳細分析:")
            print("-" * 40)
            
            # 他の圧縮器との比較
            for comp_name, comp_scores in compressor_scores.items():
                if comp_name != 'TMC_v9' and comp_scores['test_count'] > 0:
                    compression_improvement = (comp_scores['avg_compression_ratio'] - tmc_scores['avg_compression_ratio']) / comp_scores['avg_compression_ratio'] * 100
                    speed_comparison = tmc_scores['avg_compression_speed'] / comp_scores['avg_compression_speed']
                    
                    print(f"  vs {comp_name}:")
                    print(f"    圧縮率改善: {compression_improvement:+.1f}%")
                    print(f"    速度比: {speed_comparison:.2f}x")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """システム情報取得"""
        try:
            if psutil:
                return {
                    'cpu_count': psutil.cpu_count(),
                    'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                    'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                    'python_version': sys.version,
                    'platform': sys.platform
                }
            else:
                return {
                    'cpu_count': os.cpu_count(),
                    'python_version': sys.version,
                    'platform': sys.platform
                }
        except:
            return {'error': 'system_info_unavailable'}
    
    def _make_json_serializable(self, obj):
        """オブジェクトをJSONシリアライゼーション可能な形式に変換"""
        import numpy as np
        
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, bool):
            return obj
        elif isinstance(obj, (int, float, str, type(None))):
            return obj
        else:
            # 未知の型は文字列表現に変換
            return str(obj)
    
    def cleanup(self):
        """ベンチマーク後のクリーンアップ"""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
            print(f"🧹 一時ディレクトリクリーンアップ完了: {self.temp_dir}")
        except Exception as e:
            print(f"⚠️ クリーンアップエラー: {e}")


def main():
    """メイン実行関数"""
    print("🚀 TMC v9.0 包括的ベンチマーク評価システム")
    print("📊 vs 7-Zip (LZMA2) & Zstandard")
    print("🎯 評価項目: 圧縮率、圧縮速度、展開速度、可逆性")
    print("=" * 80)
    
    benchmark = ComprehensiveBenchmark()
    
    try:
        # ベンチマーク実行
        results = benchmark.run_comprehensive_benchmark()
        
        print("\n🎉 ベンチマーク評価完了！")
        print("📈 詳細結果は上記および保存されたJSONファイルを参照してください。")
        
    except Exception as e:
        print(f"❌ ベンチマーク実行エラー: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # クリーンアップ
        benchmark.cleanup()


if __name__ == "__main__":
    main()
