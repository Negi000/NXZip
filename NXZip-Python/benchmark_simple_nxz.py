#!/usr/bin/env python3
"""
NXZ統合圧縮 シンプルベンチマーク
SPE + 基本圧縮 vs Zstandard
動作確実版
"""

import os
import time
import zlib
import lzma
import py7zr
import zstandard as zstd
from pathlib import Path
from typing import Dict, Any, List, Tuple

# NXZip コンポーネント
from nxzip.engine.spe_core_jit import SPECoreJIT


class SimpleNXZBenchmark:
    """シンプルなNXZ統合ベンチマーク"""
    
    def __init__(self):
        print("🚀 NXZ統合ベンチマーク (シンプル版) 初期化...")
        self.spe_core = SPECoreJIT()
    
    def compress_spe_zlib(self, data: bytes, level: int = 6) -> Tuple[bytes, Dict[str, Any]]:
        """SPE + zlib 圧縮"""
        start_time = time.time()
        
        # Phase 1: zlib圧縮
        compressed_data = zlib.compress(data, level=level)
        
        # Phase 2: SPE構造保持暗号化
        spe_data = self.spe_core.apply_transform(compressed_data)
        
        total_time = time.time() - start_time
        
        return spe_data, {
            'method': 'SPE + zlib',
            'original_size': len(data),
            'compressed_size': len(spe_data),
            'compression_time': total_time,
            'compression_ratio': (1 - len(spe_data) / len(data)) * 100
        }
    
    def decompress_spe_zlib(self, spe_data: bytes, info: Dict[str, Any]) -> bytes:
        """SPE + zlib 展開"""
        start_time = time.time()
        
        # Phase 1: SPE逆変換
        compressed_data = self.spe_core.reverse_transform(spe_data)
        
        # Phase 2: zlib展開
        original_data = zlib.decompress(compressed_data)
        
        info['decompression_time'] = time.time() - start_time
        return original_data
    
    def compress_spe_lzma(self, data: bytes, level: int = 6) -> Tuple[bytes, Dict[str, Any]]:
        """SPE + LZMA 圧縮"""
        start_time = time.time()
        
        # Phase 1: LZMA圧縮
        compressed_data = lzma.compress(data, preset=level)
        
        # Phase 2: SPE構造保持暗号化
        spe_data = self.spe_core.apply_transform(compressed_data)
        
        total_time = time.time() - start_time
        
        return spe_data, {
            'method': 'SPE + LZMA',
            'original_size': len(data),
            'compressed_size': len(spe_data),
            'compression_time': total_time,
            'compression_ratio': (1 - len(spe_data) / len(data)) * 100
        }
    
    def decompress_spe_lzma(self, spe_data: bytes, info: Dict[str, Any]) -> bytes:
        """SPE + LZMA 展開"""
        start_time = time.time()
        
        # Phase 1: SPE逆変換
        compressed_data = self.spe_core.reverse_transform(spe_data)
        
        # Phase 2: LZMA展開
        original_data = lzma.decompress(compressed_data)
        
        info['decompression_time'] = time.time() - start_time
        return original_data


def benchmark_zlib(data: bytes, level: int = 6) -> Dict[str, Any]:
    """zlib単体ベンチマーク"""
    try:
        # 圧縮
        start_time = time.time()
        compressed_data = zlib.compress(data, level=level)
        compression_time = time.time() - start_time
        
        # 展開
        start_decomp = time.time()
        decompressed_data = zlib.decompress(compressed_data)
        decompression_time = time.time() - start_decomp
        
        # 検証
        if decompressed_data != data:
            return {'error': 'zlib 可逆性エラー'}
        
        return {
            'method': f'zlib (level {level})',
            'original_size': len(data),
            'compressed_size': len(compressed_data),
            'compression_time': compression_time,
            'decompression_time': decompression_time,
            'compression_ratio': (1 - len(compressed_data) / len(data)) * 100
        }
    
    except Exception as e:
        return {'error': f'zlib エラー: {e}'}


def benchmark_lzma(data: bytes, level: int = 6) -> Dict[str, Any]:
    """LZMA単体ベンチマーク"""
    try:
        # 圧縮
        start_time = time.time()
        compressed_data = lzma.compress(data, preset=level)
        compression_time = time.time() - start_time
        
        # 展開
        start_decomp = time.time()
        decompressed_data = lzma.decompress(compressed_data)
        decompression_time = time.time() - start_decomp
        
        # 検証
        if decompressed_data != data:
            return {'error': 'LZMA 可逆性エラー'}
        
        return {
            'method': f'LZMA (preset {level})',
            'original_size': len(data),
            'compressed_size': len(compressed_data),
            'compression_time': compression_time,
            'decompression_time': decompression_time,
            'compression_ratio': (1 - len(compressed_data) / len(data)) * 100
        }
    
    except Exception as e:
        return {'error': f'LZMA エラー: {e}'}


def benchmark_7zip(data: bytes) -> Dict[str, Any]:
    """7-Zip (py7zr) ベンチマーク"""
    try:
        import io
        
        # 圧縮
        start_time = time.time()
        buffer = io.BytesIO()
        with py7zr.SevenZipFile(buffer, 'w') as archive:
            archive.writestr(data, "test_file")
        compressed_data = buffer.getvalue()
        compression_time = time.time() - start_time
        
        # 展開
        start_decomp = time.time()
        buffer.seek(0)
        with py7zr.SevenZipFile(buffer, 'r') as archive:
            extracted = archive.readall()
            decompressed_data = extracted["test_file"].read()
        decompression_time = time.time() - start_decomp
        
        # 検証
        if decompressed_data != data:
            return {'error': '7-Zip 可逆性エラー'}
        
        return {
            'method': '7-Zip (py7zr)',
            'original_size': len(data),
            'compressed_size': len(compressed_data),
            'compression_time': compression_time,
            'decompression_time': decompression_time,
            'compression_ratio': (1 - len(compressed_data) / len(data)) * 100
        }
    
    except Exception as e:
        return {'error': f'7-Zip エラー: {e}'}


def benchmark_zstandard(data: bytes, level: int = 19) -> Dict[str, Any]:
    """Zstandard ベンチマーク"""
    try:
        # 圧縮
        start_time = time.time()
        cctx = zstd.ZstdCompressor(level=level)
        compressed_data = cctx.compress(data)
        compression_time = time.time() - start_time
        
        # 展開
        start_decomp = time.time()
        dctx = zstd.ZstdDecompressor()
        decompressed_data = dctx.decompress(compressed_data)
        decompression_time = time.time() - start_decomp
        
        # 検証
        if decompressed_data != data:
            return {'error': 'Zstandard 可逆性エラー'}
        
        return {
            'method': f'Zstandard (level {level})',
            'original_size': len(data),
            'compressed_size': len(compressed_data),
            'compression_time': compression_time,
            'decompression_time': decompression_time,
            'compression_ratio': (1 - len(compressed_data) / len(data)) * 100
        }
    
    except Exception as e:
        return {'error': f'Zstandard エラー: {e}'}


def run_simple_benchmark():
    """シンプルベンチマーク実行"""
    print("🚀 NXZ統合圧縮 シンプルベンチマーク")
    print("SPE + 基本圧縮 vs 標準圧縮アルゴリズム")
    print("=" * 60)
    
    # テストファイル収集
    sample_dir = Path("sample")
    test_files = []
    for ext in ['.txt', '.jpg', '.png', '.mp4', '.wav', '.mp3']:
        test_files.extend(sample_dir.glob(f"*{ext}"))
    
    if not test_files:
        print("❌ テストファイルが見つかりません")
        return
    
    print(f"📂 テスト対象: {len(test_files)} ファイル\n")
    
    benchmark = SimpleNXZBenchmark()
    results = []
    
    for file_path in test_files:
        print(f"📁 ベンチマーク実行: {file_path.name}")
        print(f"   ファイルサイズ: {file_path.stat().st_size:,} bytes")
        
        # ファイル読み込み
        with open(file_path, 'rb') as f:
            data = f.read()
        
        # 大きすぎるファイルはスキップ
        if len(data) > 50 * 1024 * 1024:  # 50MB超はスキップ
            print("   ⚠️  ファイルサイズが大きすぎるためスキップ")
            continue
        
        file_results = {'file': file_path.name, 'size': len(data), 'methods': {}}
        
        # 1. SPE + zlib
        try:
            print("   🔧 SPE + zlib...")
            compressed, info = benchmark.compress_spe_zlib(data, level=6)
            decompressed = benchmark.decompress_spe_zlib(compressed, info)
            
            if decompressed == data:
                file_results['methods']['SPE_zlib'] = info
                print(f"      ✅ 圧縮率: {info['compression_ratio']:.1f}%, 圧縮: {info['compression_time']:.2f}s, 展開: {info['decompression_time']:.2f}s")
            else:
                print("      ❌ 可逆性エラー")
        except Exception as e:
            print(f"      ❌ エラー: {e}")
        
        # 2. SPE + LZMA
        try:
            print("   🔧 SPE + LZMA...")
            compressed, info = benchmark.compress_spe_lzma(data, level=6)
            decompressed = benchmark.decompress_spe_lzma(compressed, info)
            
            if decompressed == data:
                file_results['methods']['SPE_LZMA'] = info
                print(f"      ✅ 圧縮率: {info['compression_ratio']:.1f}%, 圧縮: {info['compression_time']:.2f}s, 展開: {info['decompression_time']:.2f}s")
            else:
                print("      ❌ 可逆性エラー")
        except Exception as e:
            print(f"      ❌ エラー: {e}")
        
        # 3. zlib単体
        print("   📦 zlib...")
        zlib_result = benchmark_zlib(data, level=6)
        if 'error' not in zlib_result:
            file_results['methods']['zlib'] = zlib_result
            print(f"      ✅ 圧縮率: {zlib_result['compression_ratio']:.1f}%, 圧縮: {zlib_result['compression_time']:.2f}s, 展開: {zlib_result['decompression_time']:.2f}s")
        else:
            print(f"      ❌ {zlib_result['error']}")
        
        # 4. LZMA単体
        print("   📦 LZMA...")
        lzma_result = benchmark_lzma(data, level=6)
        if 'error' not in lzma_result:
            file_results['methods']['LZMA'] = lzma_result
            print(f"      ✅ 圧縮率: {lzma_result['compression_ratio']:.1f}%, 圧縮: {lzma_result['compression_time']:.2f}s, 展開: {lzma_result['decompression_time']:.2f}s")
        else:
            print(f"      ❌ {lzma_result['error']}")
        
        # 5. 7-Zip
        print("   📦 7-Zip...")
        zip7_result = benchmark_7zip(data)
        if 'error' not in zip7_result:
            file_results['methods']['7-Zip'] = zip7_result
            print(f"      ✅ 圧縮率: {zip7_result['compression_ratio']:.1f}%, 圧縮: {zip7_result['compression_time']:.2f}s, 展開: {zip7_result['decompression_time']:.2f}s")
        else:
            print(f"      ❌ {zip7_result['error']}")
        
        # 6. Zstandard
        print("   🗜️  Zstandard...")
        zstd_result = benchmark_zstandard(data, level=19)
        if 'error' not in zstd_result:
            file_results['methods']['Zstandard'] = zstd_result
            print(f"      ✅ 圧縮率: {zstd_result['compression_ratio']:.1f}%, 圧縮: {zstd_result['compression_time']:.2f}s, 展開: {zstd_result['decompression_time']:.2f}s")
        else:
            print(f"      ❌ {zstd_result['error']}")
        
        results.append(file_results)
        print()
    
    # 結果総合分析
    print("\n" + "=" * 80)
    print("📊 総合結果分析")
    print("=" * 80)
    
    methods = ['SPE_zlib', 'SPE_LZMA', 'zlib', 'LZMA', '7-Zip', 'Zstandard']
    
    print(f"{'ファイル':<25} {'サイズ':<12} {'手法':<15} {'圧縮率':<8} {'圧縮時間':<8} {'展開時間':<8}")
    print("-" * 80)
    
    for result in results:
        for method in methods:
            if method in result['methods']:
                info = result['methods'][method]
                print(f"{result['file']:<25} {result['size']:<12,} {method:<15} "
                      f"{info['compression_ratio']:.1f}%{'':<4} {info['compression_time']:.2f}s{'':<4} "
                      f"{info.get('decompression_time', 0):.2f}s")
    
    # 平均性能計算
    print("\n📈 平均性能:")
    for method in methods:
        ratios = []
        comp_times = []
        decomp_times = []
        
        for result in results:
            if method in result['methods']:
                info = result['methods'][method]
                ratios.append(info['compression_ratio'])
                comp_times.append(info['compression_time'])
                decomp_times.append(info.get('decompression_time', 0))
        
        if ratios:
            print(f"{method:<15}: 平均圧縮率 {sum(ratios)/len(ratios):.1f}%, "
                  f"平均圧縮時間 {sum(comp_times)/len(comp_times):.2f}s, "
                  f"平均展開時間 {sum(decomp_times)/len(decomp_times):.2f}s")
    
    # SPE効果分析
    print("\n🔍 SPE構造保持暗号化の効果:")
    for result in results:
        if 'SPE_zlib' in result['methods'] and 'zlib' in result['methods']:
            spe_ratio = result['methods']['SPE_zlib']['compression_ratio']
            base_ratio = result['methods']['zlib']['compression_ratio']
            effect = spe_ratio - base_ratio
            print(f"{result['file']:<25}: SPE効果 {effect:+.1f}% (zlib基準)")
        
        if 'SPE_LZMA' in result['methods'] and 'LZMA' in result['methods']:
            spe_ratio = result['methods']['SPE_LZMA']['compression_ratio']
            base_ratio = result['methods']['LZMA']['compression_ratio']
            effect = spe_ratio - base_ratio
            print(f"{result['file']:<25}: SPE効果 {effect:+.1f}% (LZMA基準)")


if __name__ == "__main__":
    run_simple_benchmark()
