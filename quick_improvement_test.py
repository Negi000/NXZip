#!/usr/bin/env python3
"""
改善版NXZip速度・圧縮率テスト
修正点:
1. 軽量BWTの可逆性問題修正
2. 軽量モードの初期化コスト削減
3. 圧縮レベルの最適化
"""

import os
import sys
import time
import hashlib
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
    print("⚠️ Zstandard利用不可")
    ZSTD_AVAILABLE = False

def create_test_data():
    """改善効果測定用テストデータ"""
    # テキストデータ（圧縮率重要）
    text_data = "NXZip Test Data! " * 8000  # 128KB相当
    
    # 数値データ（速度重要）
    numeric_data = bytes([i % 100 for i in range(20000)])  # 20KB
    
    # 混合データ（バランス重要）
    mixed_data = text_data[:5000].encode('utf-8') + numeric_data[:5000]  # 10KB
    
    return {
        'text': text_data.encode('utf-8'),
        'numeric': numeric_data,
        'mixed': mixed_data
    }

def benchmark_algorithm(data, name, compress_func, decompress_func):
    """アルゴリズム性能測定"""
    try:
        original_hash = hashlib.sha256(data).hexdigest()
        
        # 圧縮
        start_time = time.time()
        result = compress_func(data)
        compression_time = time.time() - start_time
        
        if isinstance(result, tuple):
            compressed_data, info = result
        else:
            compressed_data = result
            info = {}
        
        # 解凍
        start_time = time.time()
        decompressed_data = decompress_func(compressed_data, info)
        decompression_time = time.time() - start_time
        
        # 検証
        decompressed_hash = hashlib.sha256(decompressed_data).hexdigest()
        
        compression_ratio = (1 - len(compressed_data) / len(data)) * 100 if len(data) > 0 else 0
        throughput = (len(data) / (1024 * 1024)) / compression_time if compression_time > 0 else 0
        
        return {
            'name': name,
            'original_size': len(data),
            'compressed_size': len(compressed_data),
            'compression_ratio': compression_ratio,
            'compression_time': compression_time,
            'decompression_time': decompression_time,
            'throughput_mbps': throughput,
            'valid': original_hash == decompressed_hash,
            'info': info
        }
    except Exception as e:
        return {
            'name': name,
            'error': str(e),
            'valid': False
        }

def main():
    """改善効果確認テスト"""
    print("🚀 NXZip改善効果確認テスト開始")
    
    test_data = create_test_data()
    print(f"✅ テストデータ準備完了: {len(test_data)}種類")
    
    for data_name, data in test_data.items():
        print(f"\n{'='*60}")
        print(f"📊 {data_name}データテスト ({len(data):,} bytes)")
        print(f"{'='*60}")
        
        results = []
        
        # NXZip軽量モード（改善版）
        def nxzip_light_compress(data):
            engine = NEXUSTMCEngineV91(lightweight_mode=True)
            return engine.compress(data)
        
        def nxzip_light_decompress(compressed, info):
            engine = NEXUSTMCEngineV91(lightweight_mode=True)
            return engine.decompress(compressed, info)
        
        result = benchmark_algorithm(data, "NXZip軽量", nxzip_light_compress, nxzip_light_decompress)
        results.append(result)
        
        # NXZip通常モード（改善版）
        def nxzip_normal_compress(data):
            engine = NEXUSTMCEngineV91(lightweight_mode=False)
            return engine.compress(data)
        
        def nxzip_normal_decompress(compressed, info):
            engine = NEXUSTMCEngineV91(lightweight_mode=False)
            return engine.decompress(compressed, info)
        
        result = benchmark_algorithm(data, "NXZip通常", nxzip_normal_compress, nxzip_normal_decompress)
        results.append(result)
        
        # Zstandard比較
        if ZSTD_AVAILABLE:
            def zstd_compress(data, level=3):
                cctx = zstd.ZstdCompressor(level=level)
                return cctx.compress(data)
            
            def zstd_decompress(compressed_data, info):
                dctx = zstd.ZstdDecompressor()
                return dctx.decompress(compressed_data)
            
            # Zstd Level 3
            result = benchmark_algorithm(data, "Zstd-3", 
                                       lambda d: zstd_compress(d, 3), 
                                       zstd_decompress)
            results.append(result)
            
            # Zstd Level 9
            result = benchmark_algorithm(data, "Zstd-9", 
                                       lambda d: zstd_compress(d, 9), 
                                       zstd_decompress)
            results.append(result)
        
        # 結果表示
        print(f"\n{'アルゴリズム':<12} {'圧縮率':<8} {'速度(MB/s)':<12} {'時間(s)':<10} {'可逆性'}")
        print("-" * 60)
        for result in results:
            if 'error' not in result:
                print(f"{result['name']:<12} {result['compression_ratio']:>6.1f}% "
                      f"{result['throughput_mbps']:>10.2f} "
                      f"{result['compression_time']:>8.3f} "
                      f"{'✅' if result['valid'] else '❌'}")
            else:
                print(f"{result['name']:<12} {'ERROR':<6} {result['error']}")
        
        # 改善点分析
        nxzip_light = next((r for r in results if r['name'] == 'NXZip軽量' and 'error' not in r), None)
        nxzip_normal = next((r for r in results if r['name'] == 'NXZip通常' and 'error' not in r), None)
        zstd_3 = next((r for r in results if r['name'] == 'Zstd-3' and 'error' not in r), None)
        zstd_9 = next((r for r in results if r['name'] == 'Zstd-9' and 'error' not in r), None)
        
        print(f"\n📈 改善点分析:")
        if nxzip_light and zstd_3:
            ratio_diff = nxzip_light['compression_ratio'] - zstd_3['compression_ratio']
            speed_diff = nxzip_light['throughput_mbps'] / zstd_3['throughput_mbps'] if zstd_3['throughput_mbps'] > 0 else 0
            print(f"  軽量 vs Zstd-3: 圧縮率 {ratio_diff:+.1f}%, 速度 {speed_diff:.2f}x")
        
        if nxzip_normal and zstd_9:
            ratio_diff = nxzip_normal['compression_ratio'] - zstd_9['compression_ratio']
            speed_diff = nxzip_normal['throughput_mbps'] / zstd_9['throughput_mbps'] if zstd_9['throughput_mbps'] > 0 else 0
            print(f"  通常 vs Zstd-9: 圧縮率 {ratio_diff:+.1f}%, 速度 {speed_diff:.2f}x")
        
        if nxzip_light and nxzip_normal:
            print(f"  軽量 vs 通常: 可逆性 {'✅両方OK' if nxzip_light['valid'] and nxzip_normal['valid'] else '❌問題あり'}")

if __name__ == "__main__":
    main()
