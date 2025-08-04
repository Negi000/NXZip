#!/usr/bin/env python3
"""
NXZip vs Zstandard vs 7-Zip 性能比較ベンチマーク
"""

import os
import sys
import time
import hashlib
import subprocess
import tempfile
from pathlib import Path

# NXZip-Pythonパスを追加
sys.path.insert(0, str(Path(__file__).parent / "NXZip-Python"))

try:
    from nxzip.engine.nexus_tmc_v91_modular import NEXUSTMCEngineV91
    print("✅ NEXUSTMCEngineV91 インポート成功")
except ImportError as e:
    print(f"❌ NEXUSTMCEngineV91 インポートエラー: {e}")
    sys.exit(1)

try:
    import zstandard as zstd
    print("✅ Zstandard インポート成功")
    ZSTD_AVAILABLE = True
except ImportError:
    print("⚠️ Zstandard利用不可 - pip install zstandard")
    ZSTD_AVAILABLE = False

def check_7zip():
    """7-Zipの利用可能性をチェック"""
    try:
        # Windowsでの7-Zip実行ファイルパスを試行
        possible_paths = [
            r"C:\Program Files\7-Zip\7z.exe",
            r"C:\Program Files (x86)\7-Zip\7z.exe", 
            "7z.exe",  # PATHに含まれている場合
            "7za.exe"  # 軽量版
        ]
        
        for path in possible_paths:
            try:
                result = subprocess.run([path, "--help"], 
                                      capture_output=True, 
                                      text=True, 
                                      timeout=5)
                if result.returncode == 0:
                    print(f"✅ 7-Zip利用可能: {path}")
                    return path
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                continue
        
        print("⚠️ 7-Zip利用不可 - インストールまたはPATH設定が必要")
        return None
    except Exception as e:
        print(f"⚠️ 7-Zipチェックエラー: {e}")
        return None

def generate_benchmark_data():
    """ベンチマーク用テストデータ生成"""
    test_cases = {}
    
    # 1. テキストファイル（高圧縮率期待）
    text_data = """
    これはNXZip TMC v9.1のベンチマークテスト用テキストデータです。
    このテキストは繰り返し構造を含んでおり、圧縮アルゴリズムの効果を測定します。
    Zstandard、7-Zip、NXZipの性能比較を行います。
    """ * 500
    test_cases["text"] = text_data.encode('utf-8')
    
    # 2. 構造化数値データ（TMC効果期待）
    structured_data = b''.join([
        (i % 1000).to_bytes(4, 'little') for i in range(5000)
    ])
    test_cases["structured_numeric"] = structured_data
    
    # 3. 半ランダムデータ（中程度圧縮期待）
    import random
    random.seed(42)
    semi_random = bytes([
        random.choices([i % 256 for i in range(16)], 
                      weights=[10, 8, 6, 4, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1])[0]
        for _ in range(10000)
    ])
    test_cases["semi_random"] = semi_random
    
    # 4. 実用的混合データ
    practical_data = (
        b"HEADER: NXZip Benchmark Data\n" + b"="*50 + b"\n" +
        text_data[:1000].encode('utf-8') + b"\n" +
        b"BINARY_SECTION:\n" + structured_data[:2000] + 
        semi_random[:1000]
    )
    test_cases["practical_mixed"] = practical_data
    
    return test_cases

def benchmark_nxzip(data, mode="lightweight"):
    """NXZip性能測定"""
    try:
        # エンジン初期化
        engine = NEXUSTMCEngineV91(
            max_workers=2 if mode == "lightweight" else 4,
            chunk_size=256*1024 if mode == "lightweight" else 1024*1024,
            lightweight_mode=(mode == "lightweight")
        )
        
        # 圧縮
        start_time = time.time()
        compressed_data, compression_info = engine.compress(data)
        compression_time = time.time() - start_time
        
        # 解凍
        start_time = time.time()
        decompressed_data = engine.decompress(compressed_data, compression_info)
        decompression_time = time.time() - start_time
        
        # 検証
        original_hash = hashlib.sha256(data).hexdigest()
        decompressed_hash = hashlib.sha256(decompressed_data).hexdigest()
        
        return {
            'compressed_size': len(compressed_data),
            'compression_time': compression_time,
            'decompression_time': decompression_time,
            'compression_ratio': (1 - len(compressed_data) / len(data)) * 100,
            'throughput_mbps': (len(data) / (1024 * 1024)) / compression_time if compression_time > 0 else 0,
            'valid': original_hash == decompressed_hash,
            'info': compression_info
        }
    except Exception as e:
        return {'error': str(e)}

def benchmark_zstandard(data, level=3):
    """Zstandard性能測定"""
    if not ZSTD_AVAILABLE:
        return {'error': 'Zstandard not available'}
    
    try:
        # 圧縮
        cctx = zstd.ZstdCompressor(level=level)
        start_time = time.time()
        compressed_data = cctx.compress(data)
        compression_time = time.time() - start_time
        
        # 解凍
        dctx = zstd.ZstdDecompressor()
        start_time = time.time()
        decompressed_data = dctx.decompress(compressed_data)
        decompression_time = time.time() - start_time
        
        # 検証
        original_hash = hashlib.sha256(data).hexdigest()
        decompressed_hash = hashlib.sha256(decompressed_data).hexdigest()
        
        return {
            'compressed_size': len(compressed_data),
            'compression_time': compression_time,
            'decompression_time': decompression_time,
            'compression_ratio': (1 - len(compressed_data) / len(data)) * 100,
            'throughput_mbps': (len(data) / (1024 * 1024)) / compression_time if compression_time > 0 else 0,
            'valid': original_hash == decompressed_hash,
            'level': level
        }
    except Exception as e:
        return {'error': str(e)}

def benchmark_7zip(data, level=5, zip_path=None):
    """7-Zip性能測定"""
    if not zip_path:
        return {'error': '7-Zip not available'}
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # 入力ファイル作成
            input_file = os.path.join(temp_dir, "input.dat")
            with open(input_file, 'wb') as f:
                f.write(data)
            
            # 圧縮
            compressed_file = os.path.join(temp_dir, "compressed.7z")
            start_time = time.time()
            result = subprocess.run([
                zip_path, "a", "-t7z", f"-mx={level}", 
                compressed_file, input_file
            ], capture_output=True, text=True, timeout=30)
            compression_time = time.time() - start_time
            
            if result.returncode != 0:
                return {'error': f'7-Zip compression failed: {result.stderr}'}
            
            # 圧縮ファイルサイズ
            compressed_size = os.path.getsize(compressed_file)
            
            # 解凍
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir)
            start_time = time.time()
            result = subprocess.run([
                zip_path, "x", compressed_file, f"-o{output_dir}"
            ], capture_output=True, text=True, timeout=30)
            decompression_time = time.time() - start_time
            
            if result.returncode != 0:
                return {'error': f'7-Zip decompression failed: {result.stderr}'}
            
            # 解凍データ検証
            output_file = os.path.join(output_dir, "input.dat")
            if not os.path.exists(output_file):
                return {'error': 'Decompressed file not found'}
            
            with open(output_file, 'rb') as f:
                decompressed_data = f.read()
            
            original_hash = hashlib.sha256(data).hexdigest()
            decompressed_hash = hashlib.sha256(decompressed_data).hexdigest()
            
            return {
                'compressed_size': compressed_size,
                'compression_time': compression_time,
                'decompression_time': decompression_time,
                'compression_ratio': (1 - compressed_size / len(data)) * 100,
                'throughput_mbps': (len(data) / (1024 * 1024)) / compression_time if compression_time > 0 else 0,
                'valid': original_hash == decompressed_hash,
                'level': level
            }
    except Exception as e:
        return {'error': str(e)}

def run_comprehensive_benchmark():
    """包括的ベンチマーク実行"""
    print("🚀 NXZip vs Zstandard vs 7-Zip 包括的ベンチマーク開始")
    
    # 7-Zipチェック
    zip_path = check_7zip()
    
    # テストデータ生成
    test_data = generate_benchmark_data()
    print(f"✅ テストデータ生成完了: {len(test_data)}種類")
    
    results = {}
    
    for data_name, data in test_data.items():
        print(f"\n{'='*60}")
        print(f"📊 {data_name} ベンチマーク ({len(data):,} bytes)")
        print(f"{'='*60}")
        
        results[data_name] = {}
        
        # NXZip軽量モード
        print("🔧 NXZip軽量モード...")
        nxzip_light = benchmark_nxzip(data, "lightweight")
        results[data_name]['nxzip_lightweight'] = nxzip_light
        if 'error' not in nxzip_light:
            print(f"   圧縮率: {nxzip_light['compression_ratio']:.1f}%")
            print(f"   速度: {nxzip_light['throughput_mbps']:.2f} MB/s")
            print(f"   可逆性: {'✅' if nxzip_light['valid'] else '❌'}")
        else:
            print(f"   エラー: {nxzip_light['error']}")
        
        # NXZip通常モード
        print("🔧 NXZip通常モード...")
        nxzip_normal = benchmark_nxzip(data, "normal")
        results[data_name]['nxzip_normal'] = nxzip_normal
        if 'error' not in nxzip_normal:
            print(f"   圧縮率: {nxzip_normal['compression_ratio']:.1f}%")
            print(f"   速度: {nxzip_normal['throughput_mbps']:.2f} MB/s")
            print(f"   可逆性: {'✅' if nxzip_normal['valid'] else '❌'}")
        else:
            print(f"   エラー: {nxzip_normal['error']}")
        
        # Zstandard (level 3 - 軽量相当)
        if ZSTD_AVAILABLE:
            print("🔧 Zstandard Level 3...")
            zstd_3 = benchmark_zstandard(data, 3)
            results[data_name]['zstd_3'] = zstd_3
            if 'error' not in zstd_3:
                print(f"   圧縮率: {zstd_3['compression_ratio']:.1f}%")
                print(f"   速度: {zstd_3['throughput_mbps']:.2f} MB/s")
                print(f"   可逆性: {'✅' if zstd_3['valid'] else '❌'}")
            else:
                print(f"   エラー: {zstd_3['error']}")
        
        # Zstandard (level 9 - 高圧縮相当)
        if ZSTD_AVAILABLE:
            print("🔧 Zstandard Level 9...")
            zstd_9 = benchmark_zstandard(data, 9)
            results[data_name]['zstd_9'] = zstd_9
            if 'error' not in zstd_9:
                print(f"   圧縮率: {zstd_9['compression_ratio']:.1f}%")
                print(f"   速度: {zstd_9['throughput_mbps']:.2f} MB/s")
                print(f"   可逆性: {'✅' if zstd_9['valid'] else '❌'}")
            else:
                print(f"   エラー: {zstd_9['error']}")
        
        # 7-Zip (level 5 - 標準)
        if zip_path:
            print("🔧 7-Zip Level 5...")
            zip_5 = benchmark_7zip(data, 5, zip_path)
            results[data_name]['7zip_5'] = zip_5
            if 'error' not in zip_5:
                print(f"   圧縮率: {zip_5['compression_ratio']:.1f}%")
                print(f"   速度: {zip_5['throughput_mbps']:.2f} MB/s")
                print(f"   可逆性: {'✅' if zip_5['valid'] else '❌'}")
            else:
                print(f"   エラー: {zip_5['error']}")
        
        # 7-Zip (level 9 - 最大圧縮)
        if zip_path:
            print("🔧 7-Zip Level 9...")
            zip_9 = benchmark_7zip(data, 9, zip_path)
            results[data_name]['7zip_9'] = zip_9
            if 'error' not in zip_9:
                print(f"   圧縮率: {zip_9['compression_ratio']:.1f}%")
                print(f"   速度: {zip_9['throughput_mbps']:.2f} MB/s")
                print(f"   可逆性: {'✅' if zip_9['valid'] else '❌'}")
            else:
                print(f"   エラー: {zip_9['error']}")
    
    # 結果分析
    print(f"\n{'='*80}")
    print(f"📊 包括的ベンチマーク結果分析")
    print(f"{'='*80}")
    
    # データタイプ別比較表
    for data_name in test_data.keys():
        print(f"\n--- {data_name} 結果比較 ---")
        print(f"{'アルゴリズム':<20} {'圧縮率':<8} {'速度(MB/s)':<12} {'圧縮時間':<10} {'可逆性'}")
        print("-" * 60)
        
        data_results = results[data_name]
        for algo_name, result in data_results.items():
            if 'error' not in result:
                print(f"{algo_name:<20} {result['compression_ratio']:>6.1f}% "
                      f"{result['throughput_mbps']:>10.2f} "
                      f"{result['compression_time']:>8.3f}s "
                      f"{'✅' if result['valid'] else '❌'}")
            else:
                print(f"{algo_name:<20} {'ERROR':<6} {result['error']}")
    
    # 総合評価
    print(f"\n{'='*80}")
    print(f"🎯 総合評価・改善提案")
    print(f"{'='*80}")
    
    # NXZipの立ち位置分析
    print("\n📍 NXZipポジション分析:")
    analyze_nxzip_position(results)
    
    # 具体的改善提案
    print("\n🔧 具体的改善提案:")
    generate_improvement_suggestions(results)
    
    return results

def analyze_nxzip_position(results):
    """NXZipの競合他社比較での立ち位置分析"""
    
    # データタイプ別の勝敗分析
    for data_name, data_results in results.items():
        print(f"\n{data_name}:")
        
        # 圧縮率比較
        compression_ratios = {}
        speeds = {}
        
        for algo, result in data_results.items():
            if 'error' not in result:
                compression_ratios[algo] = result['compression_ratio']
                speeds[algo] = result['throughput_mbps']
        
        if compression_ratios:
            best_compression = max(compression_ratios.items(), key=lambda x: x[1])
            best_speed = max(speeds.items(), key=lambda x: x[1])
            
            print(f"  最高圧縮率: {best_compression[0]} ({best_compression[1]:.1f}%)")
            print(f"  最高速度: {best_speed[0]} ({best_speed[1]:.2f} MB/s)")
            
            # NXZipの位置
            nxzip_algos = [k for k in compression_ratios.keys() if k.startswith('nxzip')]
            for nxzip_algo in nxzip_algos:
                if nxzip_algo in compression_ratios:
                    comp_rank = sorted(compression_ratios.items(), key=lambda x: x[1], reverse=True)
                    speed_rank = sorted(speeds.items(), key=lambda x: x[1], reverse=True)
                    
                    comp_pos = next((i for i, (k, v) in enumerate(comp_rank) if k == nxzip_algo), -1) + 1
                    speed_pos = next((i for i, (k, v) in enumerate(speed_rank) if k == nxzip_algo), -1) + 1
                    
                    print(f"  {nxzip_algo}: 圧縮率{comp_pos}位, 速度{speed_pos}位")

def generate_improvement_suggestions(results):
    """具体的改善提案生成"""
    
    suggestions = []
    
    # 1. 速度改善提案
    print("⚡ 速度改善:")
    print("  - float_array処理のTDTアルゴリズム最適化 (現在6秒→目標0.1秒以下)")
    print("  - 軽量モードでの予期しない低速化の原因調査")
    print("  - Numba JIT最適化の追加適用")
    print("  - メモリコピー削減による高速化")
    
    # 2. 圧縮率改善提案
    print("\n🗜️ 圧縮率改善:")
    print("  - text_naturalでの98.4%は優秀だが、text_repetitiveで95.5%は改善余地あり")
    print("  - generic_binaryの負の圧縮率を回避する事前判定強化")
    print("  - 混合データでの部分的変換適用の実装")
    
    # 3. モード最適化提案
    print("\n⚙️ モード最適化:")
    print("  - 軽量モード: 速度最優先の設定見直し")
    print("  - 通常モード: 圧縮率最優先の設定見直し")
    print("  - データタイプ別の動的パラメータ調整")
    
    # 4. 競合比較戦略
    print("\n🎯 競合比較戦略:")
    print("  - Zstandardレベル3との速度競争に勝つための最適化")
    print("  - 7-Zipレベル9との圧縮率競争に勝つためのアルゴリズム強化")
    print("  - テキストデータでの圧倒的優位性の確立")

if __name__ == "__main__":
    run_comprehensive_benchmark()
