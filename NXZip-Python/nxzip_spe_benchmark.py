#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NXZip SPE統合版 包括的ベンチマーク
通常モード vs 軽量モード vs 7-Zip vs Zstandard
"""

import os
import sys
import time
import hashlib
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Any

# NXZipコンポーネントをインポート
sys.path.append(os.path.dirname(__file__))
from nxzip.formats.enhanced_nxz import SuperNXZipFile

def benchmark_test():
    """包括的ベンチマークテスト"""
    print("🚀 NXZip SPE統合版 包括的ベンチマーク")
    print("=" * 80)
    print("📊 通常モード vs 軽量モード vs 7-Zip vs Zstandard")
    print("=" * 80)
    
    # テストデータセット
    test_datasets = [
        ("小テキスト", "Hello, World! こんにちは世界！\n" * 100),
        ("繰り返し", "ABCDEFGHIJ" * 5000),
        ("日本語", "日本語のテキストデータです。圧縮テストを行います。\n" * 1000),
        ("ゼロ埋め", "\x00" * 50000),
    ]
    
    results = {}
    
    for dataset_name, text_data in test_datasets:
        data = text_data.encode('utf-8') if isinstance(text_data, str) else text_data
        original_size = len(data)
        original_hash = hashlib.sha256(data).hexdigest()
        
        print(f"\n📋 データセット: {dataset_name}")
        print(f"📊 元サイズ: {original_size:,} bytes")
        print("-" * 60)
        
        dataset_results = {}
        
        # 1. NXZip通常モード
        print("🔄 NXZip通常モード 評価中...")
        try:
            nxz_normal = SuperNXZipFile(lightweight_mode=False)
            
            start_time = time.time()
            compressed = nxz_normal.create_archive(data, show_progress=False)
            compress_time = time.time() - start_time
            
            start_time = time.time()
            decompressed = nxz_normal.extract_archive(compressed, show_progress=False)
            decompress_time = time.time() - start_time
            
            # 検証
            restored_hash = hashlib.sha256(decompressed).hexdigest()
            reversible = (original_hash == restored_hash)
            compression_ratio = (1 - len(compressed) / original_size) * 100
            
            dataset_results['NXZip通常'] = {
                'compressed_size': len(compressed),
                'compression_ratio': compression_ratio,
                'compress_speed': original_size / compress_time / 1024 / 1024,
                'decompress_speed': original_size / decompress_time / 1024 / 1024,
                'reversible': reversible
            }
            
            print(f"   圧縮率: {compression_ratio:6.1f}%")
            print(f"   圧縮速度: {original_size / compress_time / 1024 / 1024:6.1f} MB/s")
            print(f"   展開速度: {original_size / decompress_time / 1024 / 1024:6.1f} MB/s")
            print(f"   可逆性: {'✅' if reversible else '❌'}")
            
        except Exception as e:
            print(f"   ❌ エラー: {e}")
            dataset_results['NXZip通常'] = {'error': str(e)}
        
        # 2. NXZip軽量モード
        print("🔄 NXZip軽量モード 評価中...")
        try:
            nxz_light = SuperNXZipFile(lightweight_mode=True)
            
            start_time = time.time()
            compressed = nxz_light.create_archive(data, show_progress=False)
            compress_time = time.time() - start_time
            
            start_time = time.time()
            decompressed = nxz_light.extract_archive(compressed, show_progress=False)
            decompress_time = time.time() - start_time
            
            # 検証
            restored_hash = hashlib.sha256(decompressed).hexdigest()
            reversible = (original_hash == restored_hash)
            compression_ratio = (1 - len(compressed) / original_size) * 100
            
            dataset_results['NXZip軽量'] = {
                'compressed_size': len(compressed),
                'compression_ratio': compression_ratio,
                'compress_speed': original_size / compress_time / 1024 / 1024,
                'decompress_speed': original_size / decompress_time / 1024 / 1024,
                'reversible': reversible
            }
            
            print(f"   圧縮率: {compression_ratio:6.1f}%")
            print(f"   圧縮速度: {original_size / compress_time / 1024 / 1024:6.1f} MB/s")
            print(f"   展開速度: {original_size / decompress_time / 1024 / 1024:6.1f} MB/s")
            print(f"   可逆性: {'✅' if reversible else '❌'}")
            
        except Exception as e:
            print(f"   ❌ エラー: {e}")
            dataset_results['NXZip軽量'] = {'error': str(e)}
        
        # 3. 7-Zip
        print("🔄 7-Zip 評価中...")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                input_file = Path(temp_dir) / "input.dat"
                archive_file = Path(temp_dir) / "archive.7z"
                
                input_file.write_bytes(data)
                
                # 圧縮
                start_time = time.time()
                result = subprocess.run([
                    "7z", "a", "-t7z", "-mx=6", str(archive_file), str(input_file)
                ], capture_output=True, text=True, timeout=30)
                compress_time = time.time() - start_time
                
                if result.returncode != 0:
                    raise Exception("7z compression failed")
                
                compressed_size = archive_file.stat().st_size
                input_file.unlink()  # 元ファイルを削除
                
                # 展開
                start_time = time.time()
                result = subprocess.run([
                    "7z", "e", str(archive_file), f"-o{temp_dir}", "-y"
                ], capture_output=True, text=True, timeout=30)
                decompress_time = time.time() - start_time
                
                if result.returncode != 0:
                    raise Exception("7z extraction failed")
                
                # 検証
                decompressed = input_file.read_bytes()
                restored_hash = hashlib.sha256(decompressed).hexdigest()
                reversible = (original_hash == restored_hash)
                compression_ratio = (1 - compressed_size / original_size) * 100
                
                dataset_results['7-Zip'] = {
                    'compressed_size': compressed_size,
                    'compression_ratio': compression_ratio,
                    'compress_speed': original_size / compress_time / 1024 / 1024,
                    'decompress_speed': original_size / decompress_time / 1024 / 1024,
                    'reversible': reversible
                }
                
                print(f"   圧縮率: {compression_ratio:6.1f}%")
                print(f"   圧縮速度: {original_size / compress_time / 1024 / 1024:6.1f} MB/s")
                print(f"   展開速度: {original_size / decompress_time / 1024 / 1024:6.1f} MB/s")
                print(f"   可逆性: {'✅' if reversible else '❌'}")
                
        except Exception as e:
            print(f"   ❌ エラー: {e}")
            dataset_results['7-Zip'] = {'error': str(e)}
        
        # 4. Zstandard (zlib代替)
        print("🔄 Zstandard 評価中...")
        try:
            try:
                import zstandard as zstd
                
                # 圧縮
                cctx = zstd.ZstdCompressor(level=6)
                start_time = time.time()
                compressed = cctx.compress(data)
                compress_time = time.time() - start_time
                
                # 展開
                dctx = zstd.ZstdDecompressor()
                start_time = time.time()
                decompressed = dctx.decompress(compressed)
                decompress_time = time.time() - start_time
                
            except ImportError:
                # zstandardがない場合はzlibを使用
                import zlib
                
                # 圧縮
                start_time = time.time()
                compressed = zlib.compress(data, level=6)
                compress_time = time.time() - start_time
                
                # 展開
                start_time = time.time()
                decompressed = zlib.decompress(compressed)
                decompress_time = time.time() - start_time
            
            # 検証
            restored_hash = hashlib.sha256(decompressed).hexdigest()
            reversible = (original_hash == restored_hash)
            compression_ratio = (1 - len(compressed) / original_size) * 100
            
            dataset_results['Zstandard'] = {
                'compressed_size': len(compressed),
                'compression_ratio': compression_ratio,
                'compress_speed': original_size / compress_time / 1024 / 1024,
                'decompress_speed': original_size / decompress_time / 1024 / 1024,
                'reversible': reversible
            }
            
            print(f"   圧縮率: {compression_ratio:6.1f}%")
            print(f"   圧縮速度: {original_size / compress_time / 1024 / 1024:6.1f} MB/s")
            print(f"   展開速度: {original_size / decompress_time / 1024 / 1024:6.1f} MB/s")
            print(f"   可逆性: {'✅' if reversible else '❌'}")
            
        except Exception as e:
            print(f"   ❌ エラー: {e}")
            dataset_results['Zstandard'] = {'error': str(e)}
        
        results[dataset_name] = dataset_results
    
    # 総合サマリー
    print(f"\n🏆 総合評価サマリー")
    print("=" * 80)
    
    algorithms = ['NXZip通常', 'NXZip軽量', '7-Zip', 'Zstandard']
    
    print(f"{'アルゴリズム':<12} {'平均圧縮率':<10} {'平均圧縮速度':<12} {'平均展開速度':<12} {'可逆性':<8}")
    print("-" * 80)
    
    for algo in algorithms:
        ratios = []
        compress_speeds = []
        decompress_speeds = []
        reversible_count = 0
        total_count = 0
        
        for dataset_name, dataset_results in results.items():
            if algo in dataset_results and 'error' not in dataset_results[algo]:
                result = dataset_results[algo]
                ratios.append(result['compression_ratio'])
                compress_speeds.append(result['compress_speed'])
                decompress_speeds.append(result['decompress_speed'])
                if result['reversible']:
                    reversible_count += 1
                total_count += 1
        
        if ratios:
            avg_ratio = sum(ratios) / len(ratios)
            avg_compress_speed = sum(compress_speeds) / len(compress_speeds)
            avg_decompress_speed = sum(decompress_speeds) / len(decompress_speeds)
            reversible_rate = (reversible_count / total_count) * 100
            
            print(f"{algo:<12} "
                  f"{avg_ratio:8.1f}% "
                  f"{avg_compress_speed:10.1f} MB/s "
                  f"{avg_decompress_speed:10.1f} MB/s "
                  f"{reversible_rate:6.1f}%")
        else:
            print(f"{algo:<12} {'N/A':>8} {'N/A':>12} {'N/A':>12} {'N/A':>8}")
    
    print(f"\n✨ ベンチマーク完了！")
    return results

if __name__ == "__main__":
    benchmark_test()
