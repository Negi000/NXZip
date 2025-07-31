#!/usr/bin/env python3
"""
包括的な圧縮エンジン比較テスト
通常モード、軽量モード、Zstandard、7Zip (LZMA2) の完全比較
"""
import time
import os
import sys
import gc
import tempfile
import hashlib
from pathlib import Path
from typing import Tuple, Dict, List, Any

# 外部ライブラリ
import zstandard as zstd

# 7Zipライブラリの安全なインポート
py7zr_available = False
try:
    import py7zr
    py7zr_available = True
    print("✅ py7zr ライブラリが利用可能")
except ImportError:
    print("⚠️ py7zr ライブラリが見つかりません (pip install py7zr)")

# NXZipエンジンの安全なインポート
nexus_available = False
try:
    sys.path.insert(0, '.')
    from normal_mode import NEXUSTMCNormal, NEXUSTMCLightweight
    nexus_available = True
    print("✅ NEXUS TMC エンジンが利用可能")
except ImportError as e:
    print(f"⚠️ NEXUS TMC エンジンのインポートに失敗: {e}")

class ComprehensiveCompressionBenchmark:
    """包括的圧縮ベンチマーク"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.results = {}
        
        # エンジンの初期化
        self.engines = {}
        
        if nexus_available:
            self.engines['NEXUS_Normal'] = NEXUSTMCNormal()
            self.engines['NEXUS_Lightweight'] = NEXUSTMCLightweight()
            
        # Zstandard エンジン（複数レベル）
        self.engines['Zstd_Level3'] = ZstdEngine(level=3)
        self.engines['Zstd_Level6'] = ZstdEngine(level=6)
        self.engines['Zstd_Level10'] = ZstdEngine(level=10)
        
        # 7Zip エンジン
        if py7zr_available:
            self.engines['7Zip_LZMA2'] = SevenZipEngine()
        
        if self.verbose:
            print(f"📊 利用可能エンジン: {len(self.engines)} 個")
            for name in self.engines.keys():
                print(f"  - {name}")
    
    def run_comprehensive_test(self, test_files: List[str] = None) -> Dict:
        """包括的テストの実行"""
        if test_files is None:
            test_files = self._get_test_files()
        
        print(f"\n🚀 包括的圧縮テスト開始 ({len(test_files)} ファイル)")
        print("=" * 60)
        
        all_results = {}
        
        for file_path in test_files:
            if not os.path.exists(file_path):
                print(f"⚠️ ファイルが見つかりません: {file_path}")
                continue
                
            print(f"\n📁 テストファイル: {os.path.basename(file_path)}")
            print(f"   サイズ: {os.path.getsize(file_path):,} bytes")
            
            # ファイルを読み込み
            try:
                with open(file_path, 'rb') as f:
                    original_data = f.read()
            except Exception as e:
                print(f"❌ ファイル読み込みエラー: {e}")
                continue
            
            file_results = {}
            
            # 各エンジンでテスト
            for engine_name, engine in self.engines.items():
                print(f"\n  🔧 {engine_name} でテスト中...")
                result = self._test_engine(engine, engine_name, original_data)
                file_results[engine_name] = result
                
                if result['success']:
                    print(f"    ✅ 圧縮率: {result['compression_ratio']:.1%}")
                    print(f"    ⚡ 圧縮速度: {result['compression_speed']:.1f} MB/s")
                    print(f"    🚀 展開速度: {result['decompression_speed']:.1f} MB/s")
                    print(f"    🔍 可逆性: {'✅' if result['lossless'] else '❌'}")
                else:
                    print(f"    ❌ エラー: {result.get('error', 'Unknown error')}")
            
            all_results[os.path.basename(file_path)] = file_results
            
        # 結果の保存と表示
        self._save_results(all_results)
        self._display_summary(all_results)
        
        return all_results
    
    def _test_engine(self, engine, engine_name: str, original_data: bytes) -> Dict:
        """個別エンジンのテスト"""
        result = {
            'success': False,
            'original_size': len(original_data),
            'compressed_size': 0,
            'compression_ratio': 0.0,
            'compression_time': 0.0,
            'decompression_time': 0.0,
            'compression_speed': 0.0,
            'decompression_speed': 0.0,
            'lossless': False,
            'error': None
        }
        
        try:
            # 圧縮テスト
            gc.collect()  # メモリをクリーンアップ
            start_time = time.time()
            
            if hasattr(engine, 'compress_normal'):
                compressed_data, meta = engine.compress_normal(original_data)
            elif hasattr(engine, 'compress_fast'):
                compressed_data, meta = engine.compress_fast(original_data)
            else:
                compressed_data, meta = engine.compress(original_data)
            
            compression_time = time.time() - start_time
            
            # 展開テスト
            gc.collect()
            start_time = time.time()
            
            if hasattr(engine, 'decompress_normal'):
                decompressed_data = engine.decompress_normal(compressed_data, meta)
            elif hasattr(engine, 'decompress_fast'):
                decompressed_data = engine.decompress_fast(compressed_data, meta)
            else:
                decompressed_data = engine.decompress(compressed_data, meta)
            
            decompression_time = time.time() - start_time
            
            # 可逆性チェック
            lossless = (original_data == decompressed_data)
            
            # 結果計算
            original_size_mb = len(original_data) / (1024 * 1024)
            
            result.update({
                'success': True,
                'compressed_size': len(compressed_data),
                'compression_ratio': 1.0 - (len(compressed_data) / len(original_data)),
                'compression_time': compression_time,
                'decompression_time': decompression_time,
                'compression_speed': original_size_mb / compression_time if compression_time > 0 else 0,
                'decompression_speed': original_size_mb / decompression_time if decompression_time > 0 else 0,
                'lossless': lossless
            })
            
        except Exception as e:
            result['error'] = str(e)
            
        return result
    
    def _get_test_files(self) -> List[str]:
        """テストファイルの自動検出"""
        test_files = []
        sample_dir = Path("./sample")
        
        if sample_dir.exists():
            # サンプルディレクトリからファイルを取得
            for ext in ['.txt', '.jpg', '.png', '.mp4', '.mp3', '.wav']:
                test_files.extend(sample_dir.glob(f"*{ext}"))
        
        # デフォルトファイルを追加
        default_files = [
            "./README.md",
            "./PROJECT_STATUS.md",
            "./TECHNICAL.md"
        ]
        
        for file_path in default_files:
            if os.path.exists(file_path):
                test_files.append(file_path)
        
        return [str(f) for f in test_files]
    
    def _save_results(self, results: Dict):
        """結果をファイルに保存"""
        timestamp = int(time.time())
        output_file = f"comprehensive_benchmark_results_{timestamp}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("NEXUS TMC 包括的圧縮ベンチマーク結果\n")
            f.write("=" * 50 + "\n\n")
            
            for file_name, file_results in results.items():
                f.write(f"📁 ファイル: {file_name}\n")
                f.write("-" * 30 + "\n")
                
                for engine_name, result in file_results.items():
                    if result['success']:
                        f.write(f"🔧 {engine_name}:\n")
                        f.write(f"  圧縮率: {result['compression_ratio']:.1%}\n")
                        f.write(f"  圧縮速度: {result['compression_speed']:.1f} MB/s\n")
                        f.write(f"  展開速度: {result['decompression_speed']:.1f} MB/s\n")
                        f.write(f"  可逆性: {'✅' if result['lossless'] else '❌'}\n")
                    else:
                        f.write(f"❌ {engine_name}: {result.get('error', 'エラー')}\n")
                    f.write("\n")
                f.write("\n")
        
        print(f"\n💾 結果を保存しました: {output_file}")
    
    def _display_summary(self, results: Dict):
        """結果サマリーの表示"""
        print("\n" + "=" * 60)
        print("📊 総合結果サマリー")
        print("=" * 60)
        
        # エンジン別平均性能
        engine_stats = {}
        
        for file_results in results.values():
            for engine_name, result in file_results.items():
                if result['success']:
                    if engine_name not in engine_stats:
                        engine_stats[engine_name] = {
                            'compression_ratios': [],
                            'compression_speeds': [],
                            'decompression_speeds': [],
                            'lossless_count': 0,
                            'total_count': 0
                        }
                    
                    stats = engine_stats[engine_name]
                    stats['compression_ratios'].append(result['compression_ratio'])
                    stats['compression_speeds'].append(result['compression_speed'])
                    stats['decompression_speeds'].append(result['decompression_speed'])
                    if result['lossless']:
                        stats['lossless_count'] += 1
                    stats['total_count'] += 1
        
        # 平均値を計算して表示
        for engine_name, stats in engine_stats.items():
            if stats['total_count'] > 0:
                avg_compression = sum(stats['compression_ratios']) / len(stats['compression_ratios'])
                avg_comp_speed = sum(stats['compression_speeds']) / len(stats['compression_speeds'])
                avg_decomp_speed = sum(stats['decompression_speeds']) / len(stats['decompression_speeds'])
                lossless_rate = stats['lossless_count'] / stats['total_count']
                
                print(f"\n🔧 {engine_name}:")
                print(f"  平均圧縮率: {avg_compression:.1%}")
                print(f"  平均圧縮速度: {avg_comp_speed:.1f} MB/s")
                print(f"  平均展開速度: {avg_decomp_speed:.1f} MB/s")
                print(f"  可逆性成功率: {lossless_rate:.1%}")

class ZstdEngine:
    """Zstandard圧縮エンジン"""
    
    def __init__(self, level=6):
        self.level = level
        self.compressor = zstd.ZstdCompressor(level=level)
        self.decompressor = zstd.ZstdDecompressor()
    
    def compress(self, data: bytes) -> Tuple[bytes, Dict]:
        compressed = self.compressor.compress(data)
        meta = {'method': 'zstd', 'level': self.level}
        return compressed, meta
    
    def decompress(self, compressed: bytes, meta: Dict) -> bytes:
        return self.decompressor.decompress(compressed)

class SevenZipEngine:
    """7Zip LZMA2圧縮エンジン"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def compress(self, data: bytes) -> Tuple[bytes, Dict]:
        """7Zipで圧縮"""
        temp_input = os.path.join(self.temp_dir, "input.dat")
        temp_output = os.path.join(self.temp_dir, "output.7z")
        
        try:
            # 入力データを一時ファイルに保存
            with open(temp_input, 'wb') as f:
                f.write(data)
            
            # 7Zipで圧縮
            with py7zr.SevenZipFile(temp_output, 'w') as archive:
                archive.write(temp_input, "data")
            
            # 圧縮されたデータを読み込み
            with open(temp_output, 'rb') as f:
                compressed = f.read()
            
            meta = {'method': '7zip', 'temp_files': [temp_input, temp_output]}
            return compressed, meta
            
        except Exception as e:
            raise Exception(f"7Zip圧縮エラー: {e}")
    
    def decompress(self, compressed: bytes, meta: Dict) -> bytes:
        """7Zipで展開"""
        temp_archive = os.path.join(self.temp_dir, "decomp.7z")
        temp_extract_dir = os.path.join(self.temp_dir, "extract")
        
        try:
            # 圧縮データを一時ファイルに保存
            with open(temp_archive, 'wb') as f:
                f.write(compressed)
            
            # 7Zipで展開
            os.makedirs(temp_extract_dir, exist_ok=True)
            
            with py7zr.SevenZipFile(temp_archive, 'r') as archive:
                archive.extractall(temp_extract_dir)
            
            # 展開されたデータを読み込み
            extracted_file = os.path.join(temp_extract_dir, "data")
            with open(extracted_file, 'rb') as f:
                decompressed = f.read()
            
            return decompressed
            
        except Exception as e:
            raise Exception(f"7Zip展開エラー: {e}")

def main():
    """メイン実行関数"""
    print("🚀 NEXUS TMC 包括的圧縮ベンチマーク")
    print("=" * 50)
    
    benchmark = ComprehensiveCompressionBenchmark(verbose=True)
    
    # テストファイルの指定（必要に応じて変更）
    test_files = [
        "./README.md",
        "./PROJECT_STATUS.md",
        "./sample/出庫実績明細_202412.txt"
    ]
    
    # 実際に存在するファイルのみをテスト
    existing_files = [f for f in test_files if os.path.exists(f)]
    
    if not existing_files:
        print("⚠️ テストファイルが見つかりません。サンプルファイルを自動検出します。")
        existing_files = None
    
    # ベンチマーク実行
    results = benchmark.run_comprehensive_test(existing_files)
    
    print("\n✅ 包括的テスト完了!")

if __name__ == "__main__":
    main()
