#!/usr/bin/env python3
"""
NXZ統合圧縮 完全ベンチマーク
SPE + TMC v9.1 + NXZ vs 標準圧縮アルゴリズム
完全統合版
"""

import os
import time
import zlib
import lzma
import zstandard as zstd
from pathlib import Path
from typing import Dict, Any, List, Tuple

# NXZip 完全統合コンポーネント
from nxzip.engine.spe_core_jit import SPECoreJIT
from nxzip.engine.nexus_tmc_v91_modular import NEXUSTMCEngineV91
from nxzip.formats.enhanced_nxz import SuperNXZipFile

try:
    import py7zr
    PY7ZR_AVAILABLE = True
except ImportError:
    PY7ZR_AVAILABLE = False


class NXZIntegratedBenchmark:
    """NXZ統合ベンチマーク - 完全版"""
    
    def __init__(self):
        print("🚀 NXZ統合ベンチマーク (完全版) 初期化...")
        self.spe_core = SPECoreJIT()
        # TMC v9.1はNXZファイル内で自動初期化されるため、直接は使わない
        self.nxz_file = SuperNXZipFile()
    
    def compress_nxz_full(self, data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """完全なNXZ統合圧縮 (SPE + TMC v9.1 + Enhanced NXZ v2.0)"""
        start_time = time.time()
        
        try:
            # Enhanced NXZ v2.0による完全統合圧縮
            nxz_data = self.nxz_file.create_archive(data, show_progress=False)
            
            total_time = time.time() - start_time
            
            return nxz_data, {
                'method': 'NXZ v2.0 統合',
                'original_size': len(data),
                'compressed_size': len(nxz_data),
                'compression_time': total_time,
                'compression_ratio': (1 - len(nxz_data) / len(data)) * 100
            }
        except Exception as e:
            # フォールバック: 基本的なSPE + zlib
            print(f"⚠️  NXZ統合エラー ({e}), SPE+zlib フォールバック...")
            compressed_data = zlib.compress(data)
            spe_data = self.spe_core.apply_transform(compressed_data)
            
            total_time = time.time() - start_time
            
            return spe_data, {
                'method': 'SPE + zlib (フォールバック)',
                'original_size': len(data),
                'compressed_size': len(spe_data),
                'compression_time': total_time,
                'compression_ratio': (1 - len(spe_data) / len(data)) * 100
            }
    
    def decompress_nxz_full(self, nxz_data: bytes, info: Dict[str, Any]) -> bytes:
        """完全なNXZ統合展開"""
        start_time = time.time()
        
        try:
            if info['method'] == 'NXZ v2.0 統合':
                # Enhanced NXZ v2.0による完全統合展開
                original_data = self.nxz_file.extract_archive(nxz_data, show_progress=False)
            else:
                # フォールバック: SPE + zlib展開
                compressed_data = self.spe_core.reverse_transform(nxz_data)
                original_data = zlib.decompress(compressed_data)
            
            info['decompression_time'] = time.time() - start_time
            return original_data
        
        except Exception as e:
            print(f"⚠️  NXZ展開エラー: {e}")
            info['decompression_time'] = time.time() - start_time
            return b''  # エラー時は空データ
    
    def compress_spe_tmc_direct(self, data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """直接的なSPE + TMC v9.1組み合わせ（NXZフォーマットなし）"""
        start_time = time.time()
        
        try:
            # TMC v9.1エンジンを直接使用
            tmc_engine = NEXUSTMCEngineV91()
            compressed_data, tmc_info = tmc_engine.compress(data)
            
            # SPE構造保持暗号化
            spe_data = self.spe_core.apply_transform(compressed_data)
            
            total_time = time.time() - start_time
            
            return spe_data, {
                'method': 'SPE + TMC v9.1 (直接)',
                'original_size': len(data),
                'compressed_size': len(spe_data),
                'compression_time': total_time,
                'compression_ratio': (1 - len(spe_data) / len(data)) * 100,
                'tmc_info': tmc_info
            }
        
        except Exception as e:
            print(f"⚠️  SPE+TMC直接エラー ({e}), SPE+zlib フォールバック...")
            # フォールバック
            compressed_data = zlib.compress(data)
            spe_data = self.spe_core.apply_transform(compressed_data)
            
            total_time = time.time() - start_time
            
            return spe_data, {
                'method': 'SPE + zlib (TMCフォールバック)',
                'original_size': len(data),
                'compressed_size': len(spe_data),
                'compression_time': total_time,
                'compression_ratio': (1 - len(spe_data) / len(data)) * 100
            }
    
    def decompress_spe_tmc_direct(self, spe_data: bytes, info: Dict[str, Any]) -> bytes:
        """直接的なSPE + TMC v9.1展開"""
        start_time = time.time()
        
        try:
            # SPE逆変換
            compressed_data = self.spe_core.reverse_transform(spe_data)
            
            if 'tmc_info' in info and info['method'] == 'SPE + TMC v9.1 (直接)':
                # TMC v9.1展開
                tmc_engine = NEXUSTMCEngineV91()
                original_data = tmc_engine.decompress(compressed_data, info['tmc_info'])
            else:
                # フォールバック: zlib展開
                original_data = zlib.decompress(compressed_data)
            
            info['decompression_time'] = time.time() - start_time
            return original_data
        
        except Exception as e:
            print(f"⚠️  SPE+TMC展開エラー: {e}")
            info['decompression_time'] = time.time() - start_time
            return b''


def benchmark_zlib(data: bytes, level: int = 6) -> Dict[str, Any]:
    """zlib単体ベンチマーク"""
    try:
        start_time = time.time()
        compressed_data = zlib.compress(data, level=level)
        compression_time = time.time() - start_time
        
        start_decomp = time.time()
        decompressed_data = zlib.decompress(compressed_data)
        decompression_time = time.time() - start_decomp
        
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
        start_time = time.time()
        compressed_data = lzma.compress(data, preset=level)
        compression_time = time.time() - start_time
        
        start_decomp = time.time()
        decompressed_data = lzma.decompress(compressed_data)
        decompression_time = time.time() - start_decomp
        
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


def benchmark_zstandard(data: bytes, level: int = 19) -> Dict[str, Any]:
    """Zstandard ベンチマーク"""
    try:
        start_time = time.time()
        cctx = zstd.ZstdCompressor(level=level)
        compressed_data = cctx.compress(data)
        compression_time = time.time() - start_time
        
        start_decomp = time.time()
        dctx = zstd.ZstdDecompressor()
        decompressed_data = dctx.decompress(compressed_data)
        decompression_time = time.time() - start_decomp
        
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


def benchmark_7zip(data: bytes) -> Dict[str, Any]:
    """7-Zip (py7zr) ベンチマーク"""
    if not PY7ZR_AVAILABLE:
        return {'error': 'py7zr ライブラリがインストールされていません'}
    
    try:
        import io
        
        start_time = time.time()
        buffer = io.BytesIO()
        with py7zr.SevenZipFile(buffer, 'w') as archive:
            archive.writestr(data, "test_file")
        compressed_data = buffer.getvalue()
        compression_time = time.time() - start_time
        
        start_decomp = time.time()
        buffer.seek(0)
        with py7zr.SevenZipFile(buffer, 'r') as archive:
            extracted = archive.readall()
            decompressed_data = extracted["test_file"].read()
        decompression_time = time.time() - start_decomp
        
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


def run_integrated_benchmark():
    """NXZ統合ベンチマーク実行"""
    print("🚀 NXZ統合圧縮 完全ベンチマーク")
    print("SPE + TMC v9.1 + Enhanced NXZ vs 標準圧縮アルゴリズム")
    print("=" * 70)
    
    # テストファイル収集
    sample_dir = Path("sample")
    test_files = []
    for ext in ['.txt', '.jpg', '.png', '.mp4', '.wav', '.mp3']:
        test_files.extend(sample_dir.glob(f"*{ext}"))
    
    if not test_files:
        print("❌ テストファイルが見つかりません")
        return
    
    print(f"📂 テスト対象: {len(test_files)} ファイル\n")
    
    benchmark = NXZIntegratedBenchmark()
    results = []
    
    for file_path in test_files:
        print(f"📁 ベンチマーク実行: {file_path.name}")
        print(f"   ファイルサイズ: {file_path.stat().st_size:,} bytes")
        
        # ファイル読み込み
        with open(file_path, 'rb') as f:
            data = f.read()
        
        # 大きすぎるファイルはスキップ
        if len(data) > 30 * 1024 * 1024:  # 30MB超はスキップ
            print("   ⚠️  ファイルサイズが大きすぎるためスキップ")
            continue
        
        file_results = {'file': file_path.name, 'size': len(data), 'methods': {}}
        
        # 1. NXZ v2.0 完全統合
        try:
            print("   🔥 NXZ v2.0 完全統合 (SPE + TMC + Enhanced NXZ)...")
            compressed, info = benchmark.compress_nxz_full(data)
            decompressed = benchmark.decompress_nxz_full(compressed, info)
            
            if decompressed == data:
                file_results['methods']['NXZ_Full'] = info
                print(f"      ✅ 圧縮率: {info['compression_ratio']:.1f}%, 圧縮: {info['compression_time']:.2f}s, 展開: {info['decompression_time']:.2f}s")
            else:
                print("      ❌ 可逆性エラー")
        except Exception as e:
            print(f"      ❌ エラー: {e}")
        
        # 2. SPE + TMC v9.1 直接組み合わせ
        try:
            print("   🔧 SPE + TMC v9.1 (直接組み合わせ)...")
            compressed, info = benchmark.compress_spe_tmc_direct(data)
            decompressed = benchmark.decompress_spe_tmc_direct(compressed, info)
            
            if decompressed == data:
                file_results['methods']['SPE_TMC_Direct'] = info
                print(f"      ✅ 圧縮率: {info['compression_ratio']:.1f}%, 圧縮: {info['compression_time']:.2f}s, 展開: {info['decompression_time']:.2f}s")
            else:
                print("      ❌ 可逆性エラー")
        except Exception as e:
            print(f"      ❌ エラー: {e}")
        
        # 3-7. 標準アルゴリズム
        print("   📦 zlib...")
        zlib_result = benchmark_zlib(data, level=6)
        if 'error' not in zlib_result:
            file_results['methods']['zlib'] = zlib_result
            print(f"      ✅ 圧縮率: {zlib_result['compression_ratio']:.1f}%, 圧縮: {zlib_result['compression_time']:.2f}s, 展開: {zlib_result['decompression_time']:.2f}s")
        else:
            print(f"      ❌ {zlib_result['error']}")
        
        print("   📦 LZMA...")
        lzma_result = benchmark_lzma(data, level=6)
        if 'error' not in lzma_result:
            file_results['methods']['LZMA'] = lzma_result
            print(f"      ✅ 圧縮率: {lzma_result['compression_ratio']:.1f}%, 圧縮: {lzma_result['compression_time']:.2f}s, 展開: {lzma_result['decompression_time']:.2f}s")
        else:
            print(f"      ❌ {lzma_result['error']}")
        
        print("   🗜️  Zstandard...")
        zstd_result = benchmark_zstandard(data, level=19)
        if 'error' not in zstd_result:
            file_results['methods']['Zstandard'] = zstd_result
            print(f"      ✅ 圧縮率: {zstd_result['compression_ratio']:.1f}%, 圧縮: {zstd_result['compression_time']:.2f}s, 展開: {zstd_result['decompression_time']:.2f}s")
        else:
            print(f"      ❌ {zstd_result['error']}")
        
        if PY7ZR_AVAILABLE:
            print("   📦 7-Zip...")
            zip_result = benchmark_7zip(data)
            if 'error' not in zip_result:
                file_results['methods']['7-Zip'] = zip_result
                print(f"      ✅ 圧縮率: {zip_result['compression_ratio']:.1f}%, 圧縮: {zip_result['compression_time']:.2f}s, 展開: {zip_result['decompression_time']:.2f}s")
            else:
                print(f"      ❌ {zip_result['error']}")
        
        results.append(file_results)
        print()
    
    # 結果総合分析
    print("\n" + "=" * 90)
    print("📊 総合結果分析")
    print("=" * 90)
    
    methods = ['NXZ_Full', 'SPE_TMC_Direct', 'zlib', 'LZMA', 'Zstandard']
    if PY7ZR_AVAILABLE:
        methods.append('7-Zip')
    
    print(f"{'ファイル':<25} {'サイズ':<12} {'手法':<20} {'圧縮率':<8} {'圧縮時間':<8} {'展開時間':<8}")
    print("-" * 90)
    
    for result in results:
        for method in methods:
            if method in result['methods']:
                info = result['methods'][method]
                print(f"{result['file']:<25} {result['size']:<12,} {method:<20} "
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
            print(f"{method:<20}: 平均圧縮率 {sum(ratios)/len(ratios):.1f}%, "
                  f"平均圧縮時間 {sum(comp_times)/len(comp_times):.2f}s, "
                  f"平均展開時間 {sum(decomp_times)/len(decomp_times):.2f}s")
    
    # NXZ統合効果分析
    print("\n🔥 NXZ統合技術の効果:")
    nxz_wins = 0
    total_comparisons = 0
    
    for result in results:
        if 'NXZ_Full' in result['methods']:
            nxz_ratio = result['methods']['NXZ_Full']['compression_ratio']
            print(f"\n{result['file']}:")
            
            for method in ['zlib', 'LZMA', 'Zstandard', '7-Zip']:
                if method in result['methods']:
                    other_ratio = result['methods'][method]['compression_ratio']
                    diff = nxz_ratio - other_ratio
                    if diff > 0:
                        nxz_wins += 1
                        print(f"  📈 vs {method}: +{diff:.1f}% (NXZ勝利)")
                    else:
                        print(f"  📉 vs {method}: {diff:.1f}%")
                    total_comparisons += 1
    
    if total_comparisons > 0:
        win_rate = (nxz_wins / total_comparisons) * 100
        print(f"\n🏆 NXZ統合技術の勝率: {win_rate:.1f}% ({nxz_wins}/{total_comparisons})")


if __name__ == "__main__":
    run_integrated_benchmark()
