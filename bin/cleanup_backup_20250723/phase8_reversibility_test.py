#!/usr/bin/env python3
"""
Phase8 Turbo エンジン専用可逆性テスト
確実に動作する Phase8 Turbo エンジンの完全可逆性を検証
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

class Phase8ReversibilityTest:
    def __init__(self):
        self.bin_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.bin_dir)
        self.sample_dir = os.path.join(self.project_root, "NXZip-Python", "sample")
        self.engine_file = "nexus_phase8_turbo.py"
        
        self.results = {
            "test_date": datetime.now().isoformat(),
            "engine": "Phase8 Turbo AI強化エンジン",
            "tests": {},
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "reversibility_rate": 0.0
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
            print(f"ERROR: ハッシュ計算エラー: {e}")
            return None
    
    def test_file_reversibility(self, test_file_path):
        """単一ファイルの完全可逆性をテスト"""
        filename = os.path.basename(test_file_path)
        
        print(f"\n=== {filename} 可逆性テスト ===")
        
        # オリジナルファイル情報
        original_hash = self.calculate_file_hash(test_file_path)
        if not original_hash:
            return False, "オリジナルファイルハッシュ計算失敗"
        
        original_size = os.path.getsize(test_file_path)
        print(f"オリジナルサイズ: {original_size:,} bytes")
        print(f"オリジナルハッシュ: {original_hash[:16]}...")
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # テストファイルを一時ディレクトリにコピー
                temp_test_file = os.path.join(temp_dir, filename)
                shutil.copy2(test_file_path, temp_test_file)
                
                # Phase8 Turboエンジンで圧縮
                engine_path = os.path.join(self.bin_dir, self.engine_file)
                
                print("圧縮中...")
                start_time = time.time()
                
                result = subprocess.run([
                    sys.executable, engine_path, temp_test_file
                ], capture_output=True, text=True, cwd=temp_dir, timeout=300)
                
                compress_time = time.time() - start_time
                
                if result.returncode != 0:
                    error_msg = f"圧縮失敗: {result.stderr.strip()}"
                    print(f"ERROR: {error_msg}")
                    return False, error_msg
                
                # 圧縮ファイルを探す
                nxz_files = list(Path(temp_dir).glob("*.nxz"))
                if not nxz_files:
                    error_msg = "圧縮ファイル(.nxz)が見つからない"
                    print(f"ERROR: {error_msg}")
                    return False, error_msg
                
                compressed_file = nxz_files[0]
                compressed_size = compressed_file.stat().st_size
                compression_ratio = (1 - compressed_size / original_size) * 100
                
                print(f"圧縮完了: {compressed_size:,} bytes ({compression_ratio:.1f}%圧縮)")
                print(f"圧縮時間: {compress_time:.3f}秒")
                
                # 復元テスト - NXZファイルから手動復元
                print("復元中...")
                start_time = time.time()
                
                # NXZファイルからデータ復元
                with open(compressed_file, 'rb') as f:
                    nxz_data = f.read()
                
                # マジックヘッダーをチェック (NXZ8T)
                magic_header = b'NXZ8T'
                if not nxz_data.startswith(magic_header):
                    error_msg = "無効なNXZファイル形式"
                    print(f"ERROR: {error_msg}")
                    return False, error_msg
                
                # 圧縮データを取得
                compressed_data = nxz_data[len(magic_header):]
                
                # LZMA復元を試行
                try:
                    import lzma
                    restored_data = lzma.decompress(compressed_data)
                except:
                    # LZMAが失敗した場合はzlibを試行
                    try:
                        import zlib
                        restored_data = zlib.decompress(compressed_data)
                    except Exception as e:
                        error_msg = f"復元失敗: {str(e)}"
                        print(f"ERROR: {error_msg}")
                        return False, error_msg
                
                decompress_time = time.time() - start_time
                
                # 復元ファイルを保存
                restored_file = os.path.join(temp_dir, f"restored_{filename}")
                with open(restored_file, 'wb') as f:
                    f.write(restored_data)
                
                print(f"復元時間: {decompress_time:.3f}秒")
                
                # ハッシュ比較
                restored_hash = self.calculate_file_hash(restored_file)
                restored_size = len(restored_data)
                
                print(f"復元サイズ: {restored_size:,} bytes")
                print(f"復元ハッシュ: {restored_hash[:16] if restored_hash else 'None'}...")
                
                # 可逆性判定
                size_match = (restored_size == original_size)
                hash_match = (restored_hash == original_hash)
                
                if size_match and hash_match:
                    print(f"SUCCESS: 完全可逆性確認 - データ完全一致")
                    return True, {
                        "compression_ratio": compression_ratio,
                        "compress_time": compress_time,
                        "decompress_time": decompress_time,
                        "original_size": original_size,
                        "compressed_size": compressed_size,
                        "restored_size": restored_size,
                        "total_time": compress_time + decompress_time
                    }
                else:
                    error_details = []
                    if not size_match:
                        error_details.append(f"サイズ不一致: {original_size} → {restored_size}")
                    if not hash_match:
                        error_details.append(f"ハッシュ不一致")
                    
                    error_msg = "可逆性検証失敗: " + ", ".join(error_details)
                    print(f"ERROR: {error_msg}")
                    return False, error_msg
                    
        except subprocess.TimeoutExpired:
            error_msg = "処理タイムアウト（5分）"
            print(f"ERROR: {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"テスト実行エラー: {str(e)}"
            print(f"ERROR: {error_msg}")
            return False, error_msg
    
    def run_comprehensive_test(self):
        """Phase8 Turbo エンジンの包括的可逆性テスト"""
        print("Phase8 Turbo AI強化エンジン 完全可逆性テストスイート")
        print("=" * 60)
        
        # サンプルファイル取得
        sample_files = []
        if os.path.exists(self.sample_dir):
            for file_path in Path(self.sample_dir).rglob("*"):
                if file_path.is_file():
                    filename = file_path.name
                    # テスト対象ファイルを選択
                    if (not filename.endswith('.nxz') and 
                        not filename.endswith('.7z') and 
                        not '.restored.' in filename and 
                        not '.verified.' in filename and
                        file_path.stat().st_size > 0):
                        sample_files.append(str(file_path))
        
        if not sample_files:
            print("ERROR: テスト可能なサンプルファイルが見つかりません")
            return
        
        print(f"テストファイル数: {len(sample_files)}")
        
        # 各ファイルでテスト実行
        total_tests = 0
        passed_tests = 0
        
        for sample_file in sample_files:
            filename = os.path.basename(sample_file)
            print(f"\n[{total_tests + 1}/{len(sample_files)}] {filename}")
            
            total_tests += 1
            success, result = self.test_file_reversibility(sample_file)
            
            if success:
                passed_tests += 1
                self.results["tests"][filename] = {
                    "status": "PASS",
                    "metrics": result
                }
            else:
                self.results["tests"][filename] = {
                    "status": "FAIL", 
                    "error": result
                }
        
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
        print("Phase8 Turbo AI強化エンジン 可逆性テスト結果")
        print("=" * 60)
        
        summary = self.results["summary"]
        print(f"\n総合結果:")
        print(f"  総テスト数: {summary['total_tests']}")
        print(f"  成功: {summary['passed']}")
        print(f"  失敗: {summary['failed']}")
        print(f"  可逆性達成率: {summary['reversibility_rate']:.1f}%")
        
        # 個別ファイル結果
        print(f"\n個別ファイル結果:")
        for filename, test_result in self.results["tests"].items():
            status_icon = "✅" if test_result["status"] == "PASS" else "❌"
            print(f"  {status_icon} {filename}: {test_result['status']}")
            
            if test_result["status"] == "PASS" and "metrics" in test_result:
                metrics = test_result["metrics"]
                print(f"     圧縮率: {metrics['compression_ratio']:.1f}%")
                print(f"     処理時間: {metrics['total_time']:.3f}秒")
        
        # 最終判定
        if summary['reversibility_rate'] == 100.0:
            print(f"\n🎉 完全可逆性達成: すべてのテストが成功しました！")
            print(f"💎 Phase8 Turbo AI強化エンジンは完全な可逆性を実現")
        elif summary['reversibility_rate'] >= 95.0:
            print(f"\n🌟 優秀な可逆性: {summary['reversibility_rate']:.1f}%の高い成功率")
        elif summary['reversibility_rate'] >= 80.0:
            print(f"\n✨ 良好な可逆性: {summary['reversibility_rate']:.1f}%の成功率")
        else:
            print(f"\n⚠️  改善の余地: {summary['reversibility_rate']:.1f}%の成功率")
    
    def save_results(self):
        """テスト結果をJSONファイルに保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.bin_dir, f"phase8_reversibility_test_{timestamp}.json")
        
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            print(f"\nテスト結果保存: {output_file}")
        except Exception as e:
            print(f"\nERROR: 結果保存エラー: {e}")

def main():
    tester = Phase8ReversibilityTest()
    tester.run_comprehensive_test()

if __name__ == "__main__":
    main()
