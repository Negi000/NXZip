#!/usr/bin/env python3
"""
NXZip Simple Engine Test
4つの高性能エンジンの簡単な動作確認とテスト
Unicode問題を回避した安全なテスト
"""

import os
import sys
import subprocess
import hashlib
import time
import tempfile
import shutil
from pathlib import Path

class SimpleEngineTest:
    def __init__(self):
        self.bin_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.bin_dir)
        self.sample_dir = os.path.join(self.project_root, "NXZip-Python", "sample")
        
        # エンジンリスト
        self.engines = [
            "nexus_quantum.py",
            "nexus_phase8_turbo.py", 
            "nexus_optimal_balance.py",
            "nexus_lightning_fast.py"
        ]
        
    def test_engine_help(self, engine_file):
        """エンジンのヘルプを表示して動作確認"""
        engine_path = os.path.join(self.bin_dir, engine_file)
        
        if not os.path.exists(engine_path):
            print(f"SKIP: {engine_file} - ファイルが見つかりません")
            return False
            
        print(f"\n=== {engine_file} テスト ===")
        
        try:
            # 引数なしで実行
            result = subprocess.run([
                sys.executable, engine_path
            ], capture_output=True, text=True, timeout=10)
            
            print(f"Return Code: {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            
            return True
            
        except subprocess.TimeoutExpired:
            print("TIMEOUT: 10秒でタイムアウト")
            return False
        except Exception as e:
            print(f"ERROR: {e}")
            return False
    
    def test_engine_with_file(self, engine_file, test_file):
        """エンジンでファイルを圧縮テスト"""
        engine_path = os.path.join(self.bin_dir, engine_file)
        
        if not os.path.exists(engine_path):
            print(f"SKIP: {engine_file} - エンジンファイルが見つかりません")
            return False
            
        if not os.path.exists(test_file):
            print(f"SKIP: テストファイルが見つかりません - {test_file}")
            return False
        
        filename = os.path.basename(test_file)
        file_size = os.path.getsize(test_file)
        
        print(f"\n=== {engine_file} + {filename} テスト ===")
        print(f"ファイルサイズ: {file_size:,} bytes")
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # テストファイルをコピー
                temp_test_file = os.path.join(temp_dir, filename)
                shutil.copy2(test_file, temp_test_file)
                
                # エンジン実行 - 複数の方法を試行
                test_methods = [
                    [sys.executable, engine_path, temp_test_file],
                    [sys.executable, engine_path, "compress", temp_test_file],
                    [sys.executable, engine_path, temp_test_file, "compress"]
                ]
                
                for i, cmd in enumerate(test_methods):
                    print(f"\n試行 {i+1}: {' '.join(cmd)}")
                    
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, 
                        cwd=temp_dir, timeout=60
                    )
                    
                    print(f"Return Code: {result.returncode}")
                    if result.stdout:
                        print(f"STDOUT: {result.stdout[:500]}")
                    if result.stderr:
                        print(f"STDERR: {result.stderr[:500]}")
                    
                    # 成功した場合、圧縮ファイルをチェック
                    if result.returncode == 0:
                        # .nxzファイルを探す
                        nxz_files = list(Path(temp_dir).glob("*.nxz"))
                        if nxz_files:
                            nxz_file = nxz_files[0]
                            compressed_size = nxz_file.stat().st_size
                            compression_ratio = (1 - compressed_size / file_size) * 100
                            print(f"SUCCESS: 圧縮成功")
                            print(f"圧縮ファイル: {nxz_file.name}")
                            print(f"圧縮サイズ: {compressed_size:,} bytes")
                            print(f"圧縮率: {compression_ratio:.1f}%")
                            return True
                        else:
                            print("WARNING: 圧縮ファイル(.nxz)が見つかりません")
                            # temp_dirの内容を確認
                            print(f"temp_dir contents: {list(Path(temp_dir).iterdir())}")
                
                print("FAIL: すべての試行が失敗")
                return False
                
        except subprocess.TimeoutExpired:
            print("TIMEOUT: 60秒でタイムアウト")
            return False
        except Exception as e:
            print(f"ERROR: {e}")
            return False
    
    def find_sample_files(self):
        """サンプルディレクトリから適切なテストファイルを見つける"""
        if not os.path.exists(self.sample_dir):
            print(f"サンプルディレクトリが見つかりません: {self.sample_dir}")
            return []
        
        sample_files = []
        
        # 各フォーマットから1つずつピックアップ
        target_files = {
            "jpg": None,
            "png": None, 
            "mp4": None,
            "txt": None,
            "wav": None,
            "mp3": None
        }
        
        for file_path in Path(self.sample_dir).rglob("*"):
            if file_path.is_file():
                filename = file_path.name
                file_ext = file_path.suffix[1:].lower()
                
                # .nxz, .7z, .restored.*などを除外
                if (not filename.endswith('.nxz') and 
                    not filename.endswith('.7z') and 
                    not '.restored.' in filename and 
                    not '.verified.' in filename):
                    
                    if file_ext in target_files and target_files[file_ext] is None:
                        if file_path.stat().st_size > 0:
                            target_files[file_ext] = str(file_path)
        
        # 見つかったファイルをリストに追加
        for ext, path in target_files.items():
            if path:
                sample_files.append(path)
        
        return sample_files
    
    def run_comprehensive_test(self):
        """包括的エンジンテスト実行"""
        print("NXZip Simple Engine Test Suite")
        print("=" * 50)
        
        # エンジンの基本動作確認
        print("\n1. エンジン基本動作確認")
        print("-" * 30)
        
        working_engines = []
        for engine in self.engines:
            if self.test_engine_help(engine):
                working_engines.append(engine)
        
        print(f"\n動作確認済みエンジン: {len(working_engines)}/{len(self.engines)}")
        
        # サンプルファイルを使用したテスト
        print("\n2. サンプルファイルテスト")
        print("-" * 30)
        
        sample_files = self.find_sample_files()
        print(f"見つかったサンプルファイル: {len(sample_files)}")
        
        for sample_file in sample_files:
            filename = os.path.basename(sample_file)
            print(f"\nテストファイル: {filename}")
            
            # 各エンジンでテスト
            for engine in working_engines:
                success = self.test_engine_with_file(engine, sample_file)
                if success:
                    print(f"  {engine}: SUCCESS")
                else:
                    print(f"  {engine}: FAIL")
        
        print("\n" + "=" * 50)
        print("テスト完了")

def main():
    tester = SimpleEngineTest()
    tester.run_comprehensive_test()

if __name__ == "__main__":
    main()
