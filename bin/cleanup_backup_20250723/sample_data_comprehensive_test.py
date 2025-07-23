#!/usr/bin/env python3
"""
NXZip Sample Data 包括的テストスイート
4つの高性能エンジンを使用してsampleディレクトリの完全可逆性と性能をテスト
"""

import os
import sys
import subprocess
import hashlib
import time
import json
from datetime import datetime
import tempfile
import shutil
from pathlib import Path

class SampleDataTester:
    def __init__(self):
        self.bin_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.bin_dir)
        self.sample_dir = os.path.join(self.project_root, "NXZip-Python", "sample")
        
        self.results = {
            "test_date": datetime.now().isoformat(),
            "sample_directory": self.sample_dir,
            "engines": {},
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "reversibility_rate": 0.0,
                "total_compression_ratio": 0.0,
                "average_compression_time": 0.0,
                "average_decompression_time": 0.0
            }
        }
        
        # 4つの高性能エンジンとその対応フォーマット
        self.engines = {
            "nexus_quantum.py": {
                "name": "量子圧縮エンジン",
                "formats": ["png", "jpg", "jpeg"],
                "description": "画像用量子圧縮・93.8%理論値達成率"
            },
            "nexus_phase8_turbo.py": {
                "name": "AI強化動画エンジン", 
                "formats": ["mp4", "avi", "mkv", "mov"],
                "description": "動画用AI強化・40.2%圧縮率"
            },
            "nexus_optimal_balance.py": {
                "name": "構造破壊型テキストエンジン",
                "formats": ["txt", "json", "xml", "csv"],
                "description": "テキスト用構造破壊型・99.9%圧縮率"
            },
            "nexus_lightning_fast.py": {
                "name": "超高速音声エンジン",
                "formats": ["mp3", "wav", "flac", "aac"],
                "description": "音声用超高速・79.1%/100%圧縮率"
            }
        }
        
    def calculate_file_hash(self, file_path):
        """ファイルのSHA256ハッシュを計算"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            print(f"❌ ハッシュ計算エラー: {e}")
            return None
    
    def get_sample_files(self):
        """sampleディレクトリから適切なテストファイルを取得"""
        if not os.path.exists(self.sample_dir):
            print(f"❌ サンプルディレクトリが見つかりません: {self.sample_dir}")
            return {}
        
        sample_files = {}
        
        # 既存の.nxzファイルを除外し、オリジナルファイルのみを取得
        for file_path in Path(self.sample_dir).rglob("*"):
            if file_path.is_file():
                filename = file_path.name
                file_ext = file_path.suffix[1:].lower()
                
                # .nxz, .7z, .restored.*, .verified.*ファイルを除外
                if (not filename.endswith('.nxz') and 
                    not filename.endswith('.7z') and 
                    not '.restored.' in filename and 
                    not '.verified.' in filename and
                    file_ext in ['png', 'jpg', 'jpeg', 'mp4', 'avi', 'mkv', 'mov', 'mp3', 'wav', 'flac', 'aac', 'txt', 'json', 'xml', 'csv']):
                    
                    # ファイルサイズチェック（空ファイルを除外）
                    if file_path.stat().st_size > 0:
                        sample_files[filename] = str(file_path)
        
        return sample_files
    
    def test_engine_with_sample(self, engine_file, test_file_path, file_format):
        """単一エンジンでサンプルファイルをテスト"""
        engine_path = os.path.join(self.bin_dir, engine_file)
        engine_info = self.engines[engine_file]
        filename = os.path.basename(test_file_path)
        
        print(f"\n🔍 テスト開始:")
        print(f"   エンジン: {engine_info['name']} ({engine_file})")
        print(f"   ファイル: {filename}")
        print(f"   フォーマット: {file_format.upper()}")
        
        # オリジナルファイル情報
        original_hash = self.calculate_file_hash(test_file_path)
        if not original_hash:
            return False, "オリジナルファイルハッシュ計算失敗"
        
        original_size = os.path.getsize(test_file_path)
        print(f"   オリジナルサイズ: {original_size:,} bytes")
        print(f"   オリジナルハッシュ: {original_hash[:16]}...")
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # テストファイルを一時ディレクトリにコピー
                temp_test_file = os.path.join(temp_dir, filename)
                shutil.copy2(test_file_path, temp_test_file)
                
                # 圧縮テスト
                print(f"   📦 圧縮処理中...")
                start_time = time.time()
                
                # エンジンを実行
                result = subprocess.run([
                    sys.executable, engine_path, temp_test_file
                ], capture_output=True, text=True, cwd=temp_dir, timeout=300)
                
                compress_time = time.time() - start_time
                
                if result.returncode != 0:
                    error_msg = f"圧縮失敗: {result.stderr.strip()}"
                    print(f"   ❌ {error_msg}")
                    return False, error_msg
                
                # 圧縮ファイルを探す
                compressed_files = []
                
                # temp_dirで検索
                for file in os.listdir(temp_dir):
                    if file.endswith('.nxz'):
                        compressed_files.append(os.path.join(temp_dir, file))
                
                # bin_dirでも検索（エンジンがそこに出力する場合）
                for file in os.listdir(self.bin_dir):
                    if file.endswith('.nxz') and filename.replace('.', '_') in file:
                        source_path = os.path.join(self.bin_dir, file)
                        dest_path = os.path.join(temp_dir, file)
                        shutil.move(source_path, dest_path)
                        compressed_files.append(dest_path)
                
                if not compressed_files:
                    error_msg = "圧縮ファイルが見つからない"
                    print(f"   ❌ {error_msg}")
                    print(f"   temp_dir contents: {os.listdir(temp_dir)}")
                    print(f"   stdout: {result.stdout}")
                    return False, error_msg
                
                compressed_file = compressed_files[0]
                compressed_size = os.path.getsize(compressed_file)
                compression_ratio = (1 - compressed_size / original_size) * 100
                
                print(f"   ✅ 圧縮完了: {compressed_size:,} bytes")
                print(f"   📊 圧縮率: {compression_ratio:.1f}%")
                print(f"   ⏱️ 圧縮時間: {compress_time:.3f}秒")
                
                # 復元テスト
                print(f"   📂 復元処理中...")
                start_time = time.time()
                
                # 統合復元エンジンを使用
                unified_decompressor = os.path.join(self.bin_dir, "nxzip_unified_wrapper.py")
                if os.path.exists(unified_decompressor):
                    # 統合エンジンで復元
                    result = subprocess.run([
                        sys.executable, unified_decompressor, compressed_file, "--decompress"
                    ], capture_output=True, text=True, cwd=temp_dir, timeout=300)
                else:
                    # 個別エンジンで復元試行
                    result = subprocess.run([
                        sys.executable, engine_path, compressed_file, "--decompress"
                    ], capture_output=True, text=True, cwd=temp_dir, timeout=300)
                
                decompress_time = time.time() - start_time
                
                if result.returncode != 0:
                    error_msg = f"復元失敗: {result.stderr.strip()}"
                    print(f"   ❌ {error_msg}")
                    return False, error_msg
                
                # 復元ファイルを探す
                restored_files = []
                
                # temp_dirで復元ファイル検索
                for file in os.listdir(temp_dir):
                    if (not file.endswith('.nxz') and 
                        file != filename and 
                        ('restored' in file.lower() or 'decompressed' in file.lower() or 
                         file.startswith('output_') or file.endswith('_restored'))):
                        full_path = os.path.join(temp_dir, file)
                        if os.path.getsize(full_path) > 0:
                            restored_files.append(full_path)
                
                # bin_dirでも検索
                for file in os.listdir(self.bin_dir):
                    if (('restored' in file.lower() or 'decompressed' in file.lower() or 
                         'output' in file.lower()) and filename.replace('.', '_') in file):
                        source_path = os.path.join(self.bin_dir, file)
                        dest_path = os.path.join(temp_dir, f"restored_{file}")
                        if os.path.exists(source_path):
                            shutil.move(source_path, dest_path)
                            restored_files.append(dest_path)
                
                if not restored_files:
                    error_msg = "復元ファイルが見つからない"
                    print(f"   ❌ {error_msg}")
                    print(f"   temp_dir contents: {os.listdir(temp_dir)}")
                    print(f"   stdout: {result.stdout}")
                    return False, error_msg
                
                # 最適な復元ファイルを選択（サイズが一致するものを優先）
                restored_file = restored_files[0]
                for rf in restored_files:
                    if os.path.getsize(rf) == original_size:
                        restored_file = rf
                        break
                
                print(f"   ⏱️ 復元時間: {decompress_time:.3f}秒")
                
                # ハッシュ比較による可逆性検証
                restored_hash = self.calculate_file_hash(restored_file)
                restored_size = os.path.getsize(restored_file)
                
                print(f"   復元サイズ: {restored_size:,} bytes")
                print(f"   復元ハッシュ: {restored_hash[:16] if restored_hash else 'None'}...")
                
                # 可逆性判定
                is_reversible = (restored_hash == original_hash and restored_size == original_size)
                
                if is_reversible:
                    print(f"   ✅ 完全可逆性確認: データ完全一致")
                    return True, {
                        "compression_ratio": compression_ratio,
                        "compress_time": compress_time,
                        "decompress_time": decompress_time,
                        "original_size": original_size,
                        "compressed_size": compressed_size,
                        "restored_size": restored_size,
                        "total_time": compress_time + decompress_time,
                        "speed_mbps": (original_size / (1024*1024)) / (compress_time + decompress_time) if (compress_time + decompress_time) > 0 else 0
                    }
                else:
                    error_details = []
                    if restored_size != original_size:
                        error_details.append(f"サイズ不一致: {original_size} → {restored_size}")
                    if restored_hash != original_hash:
                        error_details.append(f"ハッシュ不一致")
                    
                    error_msg = "可逆性検証失敗: " + ", ".join(error_details)
                    print(f"   ❌ {error_msg}")
                    return False, error_msg
                    
        except subprocess.TimeoutExpired:
            error_msg = "処理タイムアウト（5分）"
            print(f"   ❌ {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"テスト実行エラー: {str(e)}"
            print(f"   ❌ {error_msg}")
            return False, error_msg
    
    def run_comprehensive_sample_test(self):
        """サンプルデータを使用した包括的テスト実行"""
        print("🧪 NXZip Sample Data 包括的テストスイート開始")
        print("=" * 70)
        print(f"📁 サンプルディレクトリ: {self.sample_dir}")
        
        # サンプルファイル取得
        print(f"\n📝 サンプルファイル収集中...")
        sample_files = self.get_sample_files()
        
        if not sample_files:
            print("❌ テスト可能なサンプルファイルが見つかりません")
            return
        
        print(f"   ✅ {len(sample_files)}個のテストファイルを発見")
        for filename, filepath in sample_files.items():
            size_mb = os.path.getsize(filepath) / (1024*1024)
            print(f"     - {filename} ({size_mb:.2f} MB)")
        
        # エンジン別テスト実行
        total_tests = 0
        passed_tests = 0
        total_compression_ratio = 0
        total_compress_time = 0
        total_decompress_time = 0
        
        for engine_file, engine_info in self.engines.items():
            engine_path = os.path.join(self.bin_dir, engine_file)
            
            if not os.path.exists(engine_path):
                print(f"\n⚠️  エンジンファイルが見つかりません: {engine_file}")
                continue
            
            print(f"\n🚀 {engine_info['name']} 性能テスト開始")
            print(f"   ファイル: {engine_file}")
            print(f"   説明: {engine_info['description']}")
            print("-" * 50)
            
            engine_results = {
                "name": engine_info["name"],
                "description": engine_info["description"],
                "supported_formats": engine_info["formats"],
                "tests": {},
                "engine_summary": {
                    "tests_run": 0,
                    "tests_passed": 0,
                    "average_compression_ratio": 0,
                    "average_speed_mbps": 0,
                    "total_processing_time": 0
                }
            }
            
            engine_tests = 0
            engine_passed = 0
            engine_compression_total = 0
            engine_speed_total = 0
            engine_time_total = 0
            
            # サンプルファイルでテスト実行
            for filename, filepath in sample_files.items():
                file_ext = os.path.splitext(filename)[1][1:].lower()
                
                # エンジンが対応するフォーマットかチェック
                if file_ext not in engine_info["formats"]:
                    continue
                
                total_tests += 1
                engine_tests += 1
                
                success, result = self.test_engine_with_sample(engine_file, filepath, file_ext)
                
                if success:
                    passed_tests += 1
                    engine_passed += 1
                    
                    metrics = result
                    engine_compression_total += metrics["compression_ratio"]
                    engine_speed_total += metrics["speed_mbps"]
                    engine_time_total += metrics["total_time"]
                    total_compression_ratio += metrics["compression_ratio"]
                    total_compress_time += metrics["compress_time"]
                    total_decompress_time += metrics["decompress_time"]
                    
                    engine_results["tests"][filename] = {
                        "status": "PASS",
                        "metrics": metrics
                    }
                else:
                    engine_results["tests"][filename] = {
                        "status": "FAIL",
                        "error": result
                    }
            
            # エンジン統計計算
            if engine_tests > 0:
                engine_results["engine_summary"]["tests_run"] = engine_tests
                engine_results["engine_summary"]["tests_passed"] = engine_passed
                engine_results["engine_summary"]["success_rate"] = (engine_passed / engine_tests) * 100
                
                if engine_passed > 0:
                    engine_results["engine_summary"]["average_compression_ratio"] = engine_compression_total / engine_passed
                    engine_results["engine_summary"]["average_speed_mbps"] = engine_speed_total / engine_passed
                    engine_results["engine_summary"]["total_processing_time"] = engine_time_total
            
            self.results["engines"][engine_file] = engine_results
        
        # 全体統計計算
        self.results["summary"]["total_tests"] = total_tests
        self.results["summary"]["passed"] = passed_tests
        self.results["summary"]["failed"] = total_tests - passed_tests
        self.results["summary"]["reversibility_rate"] = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        if passed_tests > 0:
            self.results["summary"]["total_compression_ratio"] = total_compression_ratio / passed_tests
            self.results["summary"]["average_compression_time"] = total_compress_time / passed_tests
            self.results["summary"]["average_decompression_time"] = total_decompress_time / passed_tests
        
        # 結果表示
        self.display_comprehensive_results()
        
        # 結果保存
        self.save_test_results()
    
    def display_comprehensive_results(self):
        """包括的テスト結果表示"""
        print("\n" + "=" * 70)
        print("🏆 NXZip Sample Data 包括的テスト結果")
        print("=" * 70)
        
        summary = self.results["summary"]
        print(f"\n📊 総合結果:")
        print(f"   総テスト数: {summary['total_tests']}")
        print(f"   成功: {summary['passed']}")
        print(f"   失敗: {summary['failed']}")
        print(f"   可逆性達成率: {summary['reversibility_rate']:.1f}%")
        print(f"   平均圧縮率: {summary['total_compression_ratio']:.1f}%")
        print(f"   平均圧縮時間: {summary['average_compression_time']:.3f}秒")
        print(f"   平均復元時間: {summary['average_decompression_time']:.3f}秒")
        
        # エンジン別詳細結果
        for engine_file, engine_data in self.results["engines"].items():
            print(f"\n🔧 {engine_data['name']} 詳細結果")
            print(f"   エンジンファイル: {engine_file}")
            
            summary = engine_data["engine_summary"]
            if summary["tests_run"] > 0:
                print(f"   実行テスト数: {summary['tests_run']}")
                print(f"   成功テスト数: {summary['tests_passed']}")
                print(f"   成功率: {summary.get('success_rate', 0):.1f}%")
                print(f"   平均圧縮率: {summary['average_compression_ratio']:.1f}%")
                print(f"   平均処理速度: {summary['average_speed_mbps']:.2f} MB/s")
                print(f"   総処理時間: {summary['total_processing_time']:.3f}秒")
                
                print(f"\n   個別ファイル結果:")
                for filename, test_result in engine_data["tests"].items():
                    status_icon = "✅" if test_result["status"] == "PASS" else "❌"
                    print(f"     {status_icon} {filename}: {test_result['status']}")
                    
                    if test_result["status"] == "PASS" and "metrics" in test_result:
                        metrics = test_result["metrics"]
                        print(f"        圧縮率: {metrics['compression_ratio']:.1f}% | "
                              f"速度: {metrics['speed_mbps']:.2f} MB/s | "
                              f"時間: {metrics['total_time']:.3f}s")
            else:
                print("   対応フォーマットのテストファイルなし")
        
        # 最終判定
        print(f"\n🎯 最終評価:")
        if summary['reversibility_rate'] == 100.0:
            print(f"🎉 完全可逆性達成: すべてのテストが成功！")
            print(f"💎 品質保証: NXZipは完全な可逆性を実現")
        elif summary['reversibility_rate'] >= 95.0:
            print(f"🌟 優秀な可逆性: {summary['reversibility_rate']:.1f}%の高い成功率")
        elif summary['reversibility_rate'] >= 80.0:
            print(f"✨ 良好な可逆性: {summary['reversibility_rate']:.1f}%の成功率")
        else:
            print(f"⚠️  改善の余地: {summary['reversibility_rate']:.1f}%の成功率")
        
        if summary['total_compression_ratio'] > 0:
            print(f"📈 圧縮性能: 平均{summary['total_compression_ratio']:.1f}%の圧縮率を達成")
    
    def save_test_results(self):
        """テスト結果をJSONファイルに保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.bin_dir, f"sample_data_test_report_{timestamp}.json")
        
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            print(f"\n💾 テスト結果保存: {output_file}")
        except Exception as e:
            print(f"\n❌ 結果保存エラー: {e}")

def main():
    """メイン実行関数"""
    tester = SampleDataTester()
    tester.run_comprehensive_sample_test()

if __name__ == "__main__":
    main()
