#!/usr/bin/env python3
"""
NXZip 完全可逆性テストスイート
4つの高性能エンジンの可逆性を包括的にテスト
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

class ReversibilityTester:
    def __init__(self):
        self.bin_dir = os.path.dirname(os.path.abspath(__file__))
        self.test_data_dir = os.path.join(os.path.dirname(self.bin_dir), "test-data")
        self.results = {
            "test_date": datetime.now().isoformat(),
            "engines": {},
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "reversibility_rate": 0.0
            }
        }
        
        # 4つの高性能エンジン
        self.engines = {
            "nexus_quantum.py": {
                "name": "量子圧縮エンジン",
                "formats": ["png", "jpg", "jpeg"],
                "description": "画像用量子圧縮・93.8%理論値達成率"
            },
            "nexus_phase8_turbo.py": {
                "name": "AI強化動画エンジン", 
                "formats": ["mp4", "avi", "mkv"],
                "description": "動画用AI強化・40.2%圧縮率"
            },
            "nexus_optimal_balance.py": {
                "name": "構造破壊型テキストエンジン",
                "formats": ["txt", "json", "xml", "csv"],
                "description": "テキスト用構造破壊型・99.9%圧縮率"
            },
            "nexus_lightning_fast.py": {
                "name": "超高速音声エンジン",
                "formats": ["mp3", "wav", "flac"],
                "description": "音声用超高速・79.1%/100%圧縮率"
            }
        }
        
    def calculate_md5(self, file_path):
        """ファイルのMD5ハッシュを計算"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            print(f"❌ MD5計算エラー: {e}")
            return None
    
    def create_test_files(self):
        """テスト用ファイルを作成"""
        test_files = {}
        
        # テキストファイル
        text_content = """
        これは可逆性テスト用のサンプルテキストです。
        日本語、English、123456789、記号!@#$%^&*()
        改行やタブ、スペースの処理も確認します。
        
        特殊文字: äöü ñ ç œ 中文 한글 العربية
        長いテキスト: """ + "A" * 1000 + """
        リピートパターン: """ + "PATTERN" * 100
        
        test_files["sample_text.txt"] = os.path.join(self.test_data_dir, "sample_text.txt")
        with open(test_files["sample_text.txt"], "w", encoding="utf-8") as f:
            f.write(text_content)
        
        # JSON形式テスト
        json_content = {
            "test": "reversibility",
            "data": [1, 2, 3, {"nested": "value"}],
            "unicode": "日本語テスト",
            "numbers": [3.14159, 2.71828],
            "boolean": True,
            "null": None
        }
        test_files["sample_data.json"] = os.path.join(self.test_data_dir, "sample_data.json")
        with open(test_files["sample_data.json"], "w", encoding="utf-8") as f:
            json.dump(json_content, f, ensure_ascii=False, indent=2)
            
        # CSV形式テスト
        csv_content = """名前,年齢,職業,備考
田中太郎,25,エンジニア,"Python, Java"
山田花子,30,デザイナー,"Photoshop, Illustrator"
佐藤次郎,35,マネージャー,"プロジェクト管理"
鈴木美咲,28,アナリスト,"データ分析, SQL"
"""
        test_files["sample_data.csv"] = os.path.join(self.test_data_dir, "sample_data.csv")
        with open(test_files["sample_data.csv"], "w", encoding="utf-8") as f:
            f.write(csv_content)
        
        return test_files
    
    def test_engine_reversibility(self, engine_file, test_file, file_format):
        """単一エンジンの可逆性をテスト"""
        engine_path = os.path.join(self.bin_dir, engine_file)
        engine_info = self.engines[engine_file]
        
        print(f"\n🔍 テスト開始: {engine_info['name']}")
        print(f"   エンジン: {engine_file}")
        print(f"   ファイル: {os.path.basename(test_file)}")
        print(f"   フォーマット: {file_format.upper()}")
        
        # オリジナルファイルのハッシュ
        original_hash = self.calculate_md5(test_file)
        if not original_hash:
            return False, "オリジナルファイルハッシュ計算失敗"
        
        original_size = os.path.getsize(test_file)
        print(f"   オリジナルサイズ: {original_size:,} bytes")
        print(f"   オリジナルハッシュ: {original_hash}")
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # 圧縮テスト
                print(f"   📦 圧縮中...")
                start_time = time.time()
                
                result = subprocess.run([
                    sys.executable, engine_path, test_file
                ], capture_output=True, text=True, cwd=temp_dir)
                
                compress_time = time.time() - start_time
                
                if result.returncode != 0:
                    error_msg = f"圧縮失敗: {result.stderr}"
                    print(f"   ❌ {error_msg}")
                    return False, error_msg
                
                # 圧縮ファイルを探す
                compressed_files = [f for f in os.listdir(temp_dir) if f.endswith('.nxz')]
                if not compressed_files:
                    # binディレクトリも確認
                    bin_compressed = [f for f in os.listdir(self.bin_dir) if f.endswith('.nxz')]
                    if bin_compressed:
                        # binディレクトリから移動
                        for cf in bin_compressed:
                            shutil.move(os.path.join(self.bin_dir, cf), temp_dir)
                        compressed_files = bin_compressed
                
                if not compressed_files:
                    error_msg = "圧縮ファイルが見つからない"
                    print(f"   ❌ {error_msg}")
                    return False, error_msg
                
                compressed_file = os.path.join(temp_dir, compressed_files[0])
                compressed_size = os.path.getsize(compressed_file)
                compression_ratio = (1 - compressed_size / original_size) * 100
                
                print(f"   ✅ 圧縮完了: {compressed_size:,} bytes ({compression_ratio:.1f}%圧縮)")
                print(f"   ⏱️ 圧縮時間: {compress_time:.3f}秒")
                
                # 復元テスト
                print(f"   📂 復元中...")
                start_time = time.time()
                
                # 復元用の統合エンジン使用を試行
                decompressor_path = os.path.join(self.bin_dir, "nxzip_final_decompressor.py")
                if os.path.exists(decompressor_path):
                    result = subprocess.run([
                        sys.executable, decompressor_path, compressed_file
                    ], capture_output=True, text=True, cwd=temp_dir)
                else:
                    # エンジン固有の復元
                    result = subprocess.run([
                        sys.executable, engine_path, compressed_file, "--decompress"
                    ], capture_output=True, text=True, cwd=temp_dir)
                
                decompress_time = time.time() - start_time
                
                if result.returncode != 0:
                    error_msg = f"復元失敗: {result.stderr}"
                    print(f"   ❌ {error_msg}")
                    return False, error_msg
                
                # 復元ファイルを探す
                restored_files = []
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if not file.endswith('.nxz') and file != os.path.basename(test_file):
                            full_path = os.path.join(root, file)
                            if os.path.getsize(full_path) > 0:
                                restored_files.append(full_path)
                
                # binディレクトリも確認
                for file in os.listdir(self.bin_dir):
                    if file.startswith("restored_") or file.startswith("decompressed_"):
                        full_path = os.path.join(self.bin_dir, file)
                        if os.path.getsize(full_path) > 0:
                            restored_files.append(full_path)
                            # 一時ディレクトリに移動
                            shutil.move(full_path, temp_dir)
                
                if not restored_files:
                    error_msg = "復元ファイルが見つからない"
                    print(f"   ❌ {error_msg}")
                    return False, error_msg
                
                # 最も適切な復元ファイルを選択
                restored_file = restored_files[0]
                for rf in restored_files:
                    if os.path.getsize(rf) == original_size:
                        restored_file = rf
                        break
                
                print(f"   ⏱️ 復元時間: {decompress_time:.3f}秒")
                
                # ハッシュ比較
                restored_hash = self.calculate_md5(restored_file)
                restored_size = os.path.getsize(restored_file)
                
                print(f"   復元サイズ: {restored_size:,} bytes")
                print(f"   復元ハッシュ: {restored_hash}")
                
                if restored_hash == original_hash:
                    print(f"   ✅ 完全可逆性確認: ハッシュ一致")
                    return True, {
                        "compression_ratio": compression_ratio,
                        "compress_time": compress_time,
                        "decompress_time": decompress_time,
                        "original_size": original_size,
                        "compressed_size": compressed_size,
                        "restored_size": restored_size
                    }
                else:
                    error_msg = f"ハッシュ不一致: オリジナル={original_hash}, 復元={restored_hash}"
                    print(f"   ❌ {error_msg}")
                    return False, error_msg
                    
        except Exception as e:
            error_msg = f"テスト実行エラー: {str(e)}"
            print(f"   ❌ {error_msg}")
            return False, error_msg
    
    def run_comprehensive_test(self):
        """包括的可逆性テスト実行"""
        print("🧪 NXZip 完全可逆性テストスイート開始")
        print("=" * 60)
        
        # テストディレクトリ確保
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # テストファイル作成
        print("\n📝 テストファイル作成中...")
        test_files = self.create_test_files()
        
        # 既存テストファイルも追加
        existing_files = {
            "huge_test.txt": os.path.join(self.test_data_dir, "huge_test.txt"),
            "large_test.txt": os.path.join(self.test_data_dir, "large_test.txt"),
            "sample_text.txt": os.path.join(self.test_data_dir, "sample_text.txt"),
            "test.txt": os.path.join(self.test_data_dir, "test.txt")
        }
        
        for name, path in existing_files.items():
            if os.path.exists(path):
                test_files[name] = path
        
        print(f"   ✅ {len(test_files)}個のテストファイル準備完了")
        
        # エンジン別テスト実行
        total_tests = 0
        passed_tests = 0
        
        for engine_file, engine_info in self.engines.items():
            engine_path = os.path.join(self.bin_dir, engine_file)
            
            if not os.path.exists(engine_path):
                print(f"\n⚠️  エンジンファイルが見つかりません: {engine_file}")
                continue
            
            print(f"\n🚀 {engine_info['name']} テスト開始")
            print(f"   説明: {engine_info['description']}")
            
            engine_results = {
                "name": engine_info["name"],
                "description": engine_info["description"],
                "tests": {}
            }
            
            # テストファイルでテスト実行
            for test_name, test_path in test_files.items():
                if not os.path.exists(test_path):
                    continue
                
                file_ext = os.path.splitext(test_path)[1][1:].lower()
                if file_ext not in engine_info["formats"] and not any(fmt in test_name.lower() for fmt in engine_info["formats"]):
                    continue
                
                total_tests += 1
                success, result = self.test_engine_reversibility(engine_file, test_path, file_ext)
                
                if success:
                    passed_tests += 1
                    engine_results["tests"][test_name] = {
                        "status": "PASS",
                        "metrics": result
                    }
                else:
                    engine_results["tests"][test_name] = {
                        "status": "FAIL",
                        "error": result
                    }
            
            self.results["engines"][engine_file] = engine_results
        
        # 統計計算
        self.results["summary"]["total_tests"] = total_tests
        self.results["summary"]["passed"] = passed_tests
        self.results["summary"]["failed"] = total_tests - passed_tests
        self.results["summary"]["reversibility_rate"] = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # 結果表示
        self.display_results()
        
        # 結果保存
        self.save_results()
    
    def display_results(self):
        """テスト結果表示"""
        print("\n" + "=" * 60)
        print("🏆 NXZip 完全可逆性テスト結果")
        print("=" * 60)
        
        summary = self.results["summary"]
        print(f"\n📊 総合結果:")
        print(f"   総テスト数: {summary['total_tests']}")
        print(f"   成功: {summary['passed']}")
        print(f"   失敗: {summary['failed']}")
        print(f"   可逆性達成率: {summary['reversibility_rate']:.1f}%")
        
        # エンジン別結果
        for engine_file, engine_data in self.results["engines"].items():
            print(f"\n🔧 {engine_data['name']} ({engine_file})")
            
            tests = engine_data["tests"]
            passed = sum(1 for t in tests.values() if t["status"] == "PASS")
            total = len(tests)
            
            if total > 0:
                success_rate = passed / total * 100
                print(f"   テスト成功率: {passed}/{total} ({success_rate:.1f}%)")
                
                for test_name, test_result in tests.items():
                    status_icon = "✅" if test_result["status"] == "PASS" else "❌"
                    print(f"   {status_icon} {test_name}: {test_result['status']}")
                    
                    if test_result["status"] == "PASS" and "metrics" in test_result:
                        metrics = test_result["metrics"]
                        print(f"      圧縮率: {metrics['compression_ratio']:.1f}%")
                        print(f"      処理時間: {metrics['compress_time']:.3f}s + {metrics['decompress_time']:.3f}s")
            else:
                print("   対応テストファイルなし")
        
        # 最終判定
        if summary['reversibility_rate'] == 100.0:
            print(f"\n🎉 完全可逆性達成: すべてのテストが成功しました！")
        elif summary['reversibility_rate'] >= 90.0:
            print(f"\n🎯 高い可逆性: {summary['reversibility_rate']:.1f}%の成功率を達成")
        elif summary['reversibility_rate'] >= 50.0:
            print(f"\n⚠️  部分的可逆性: {summary['reversibility_rate']:.1f}%の成功率")
        else:
            print(f"\n❌ 可逆性に課題: {summary['reversibility_rate']:.1f}%の成功率")
    
    def save_results(self):
        """テスト結果をJSONファイルに保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.bin_dir, f"reversibility_test_report_{timestamp}.json")
        
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            print(f"\n💾 テスト結果保存: {output_file}")
        except Exception as e:
            print(f"\n❌ 結果保存エラー: {e}")

def main():
    """メイン実行関数"""
    tester = ReversibilityTester()
    tester.run_comprehensive_test()

if __name__ == "__main__":
    main()
