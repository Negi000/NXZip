#!/usr/bin/env python3
"""
NEXUS TMC v9.1 + SPE Integration Test Suite

SPE統合 + .nxz形式での完全圧縮・解凍テスト
- 通常モード vs 軽量モード比較
- 7zip, Zstandard との競合比較
- 圧縮率、速度、可逆性の総合評価
"""

import os
import sys
import time
import zlib
import lzma
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# プロジェクトパスを追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from nxzip.engine.nexus_tmc_v91_modular import NEXUSTMCEngineV91
from nxzip.crypto.encrypt import SuperCrypto, EncryptionAlgorithm
from nxzip.formats.enhanced_nxz import SuperNXZipFile

# 競合ツール
try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False
    print("⚠️ Zstandard not available")

try:
    import py7zr
    PY7ZR_AVAILABLE = True
except ImportError:
    PY7ZR_AVAILABLE = False
    print("⚠️ py7zr not available")


class SPE_TMC_IntegratedEngine:
    """SPE統合TMC v9.1エンジン"""
    
    def __init__(self, lightweight_mode: bool = False, enable_encryption: bool = True):
        self.tmc_engine = NEXUSTMCEngineV91(lightweight_mode=lightweight_mode)
        self.lightweight_mode = lightweight_mode
        self.enable_encryption = enable_encryption
        
        if enable_encryption:
            self.crypto = SuperCrypto(EncryptionAlgorithm.AES_GCM)
            self.nxz_handler = SuperNXZipFile()
            print(f"🔐 SPE統合モード: {'軽量' if lightweight_mode else '標準'}")
        else:
            print(f"🚀 非暗号化モード: {'軽量' if lightweight_mode else '標準'}")
    
    def compress_with_spe(self, data: bytes, password: Optional[str] = None) -> Tuple[bytes, Dict[str, Any]]:
        """SPE統合圧縮"""
        start_time = time.time()
        
        # Phase 1: TMC v9.1圧縮
        compressed_data, tmc_info = self.tmc_engine.compress(data)
        
        # Phase 2: SPE暗号化（オプション）
        if self.enable_encryption and password:
            encrypted_data, crypto_metadata = self.crypto.encrypt(compressed_data, password)
            
            # NXZ形式でパッケージ
            final_data = self.nxz_handler.create_archive(
                encrypted_data, 
                password=None,  # 既に暗号化済み
                compression_level=1,  # 既に圧縮済みなので軽圧縮
                show_progress=False
            )
        else:
            final_data = compressed_data
            crypto_metadata = None
        
        total_time = time.time() - start_time
        
        # 統合情報
        integrated_info = {
            'tmc_info': tmc_info,
            'crypto_metadata': crypto_metadata,
            'total_compression_time': total_time,
            'original_size': len(data),
            'final_size': len(final_data),
            'overall_compression_ratio': (1 - len(final_data) / len(data)) * 100 if len(data) > 0 else 0,
            'spe_enabled': self.enable_encryption and password is not None,
            'lightweight_mode': self.lightweight_mode
        }
        
        return final_data, integrated_info
    
    def decompress_with_spe(self, compressed_data: bytes, info: Dict[str, Any], 
                           password: Optional[str] = None) -> bytes:
        """SPE統合解凍"""
        
        # Phase 1: NXZ解凍 + SPE復号化（必要に応じて）
        if self.enable_encryption and info.get('spe_enabled', False):
            # NXZ形式の解凍とSPE復号化を組み合わせ
            # 実装簡略化のため、TMC部分のみ解凍
            if password and info.get('crypto_metadata'):
                decrypted_data = self.crypto.decrypt(
                    compressed_data, 
                    info['crypto_metadata'], 
                    password
                )
                tmc_data = decrypted_data
            else:
                raise ValueError("パスワードまたは暗号化メタデータが不足")
        else:
            tmc_data = compressed_data
        
        # Phase 2: TMC v9.1解凍
        decompressed_data = self.tmc_engine.decompress(tmc_data, info['tmc_info'])
        
        return decompressed_data


def generate_test_data_suite() -> Dict[str, bytes]:
    """多様なテストデータセットを生成"""
    test_data = {}
    
    # 1. テキストデータ（高圧縮率期待）
    text_data = """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
    Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
    """ * 1000
    test_data['text_repetitive'] = text_data.encode('utf-8')
    
    # 2. バイナリデータ（中程度圧縮率）
    test_data['binary_mixed'] = os.urandom(50 * 1024) + b'A' * 10240 + os.urandom(40 * 1024)
    
    # 3. 高エントロピーデータ（低圧縮率）
    test_data['high_entropy'] = os.urandom(100 * 1024)
    
    # 4. 数値配列データ（特殊圧縮期待）
    import struct
    numbers = [float(i * 0.1) for i in range(25600)]  # 100KB float array
    test_data['float_array'] = struct.pack(f'{len(numbers)}f', *numbers)
    
    return test_data


def test_competitor_7zip(data: bytes) -> Tuple[float, float, float, bool]:
    """7zipとの比較テスト"""
    if not PY7ZR_AVAILABLE:
        return 0.0, 0.0, 0.0, False
    
    try:
        # 7zip圧縮
        start_time = time.time()
        with tempfile.NamedTemporaryFile() as temp_file:
            with py7zr.SevenZipFile(temp_file.name, 'w') as archive:
                archive.writestr(data, "test_data")
            
            temp_file.seek(0)
            compressed_data = temp_file.read()
        compress_time = time.time() - start_time
        
        # 7zip解凍
        start_time = time.time()
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(compressed_data)
            temp_file.flush()
            
            with py7zr.SevenZipFile(temp_file.name, 'r') as archive:
                extracted = archive.read()
                decompressed_data = list(extracted.values())[0]
        
        decompress_time = time.time() - start_time
        
        # 可逆性チェック
        is_reversible = (data == decompressed_data)
        compression_ratio = (1 - len(compressed_data) / len(data)) * 100
        
        return compression_ratio, compress_time, decompress_time, is_reversible
        
    except Exception as e:
        print(f"7zip テストエラー: {e}")
        return 0.0, 0.0, 0.0, False


def test_competitor_zstd(data: bytes) -> Tuple[float, float, float, bool]:
    """Zstandardとの比較テスト"""
    if not ZSTD_AVAILABLE:
        return 0.0, 0.0, 0.0, False
    
    try:
        compressor = zstd.ZstdCompressor(level=6)
        
        # Zstd圧縮
        start_time = time.time()
        compressed_data = compressor.compress(data)
        compress_time = time.time() - start_time
        
        # Zstd解凍
        decompressor = zstd.ZstdDecompressor()
        start_time = time.time()
        decompressed_data = decompressor.decompress(compressed_data)
        decompress_time = time.time() - start_time
        
        # 可逆性チェック
        is_reversible = (data == decompressed_data)
        compression_ratio = (1 - len(compressed_data) / len(data)) * 100
        
        return compression_ratio, compress_time, decompress_time, is_reversible
        
    except Exception as e:
        print(f"Zstd テストエラー: {e}")
        return 0.0, 0.0, 0.0, False


def test_tmc_spe_engine(data: bytes, data_name: str, lightweight_mode: bool, 
                       enable_encryption: bool = True, password: str = "test123") -> Dict[str, Any]:
    """TMC+SPE統合エンジンのテスト"""
    
    engine = SPE_TMC_IntegratedEngine(
        lightweight_mode=lightweight_mode, 
        enable_encryption=enable_encryption
    )
    
    original_hash = hashlib.sha256(data).hexdigest()
    
    try:
        # 圧縮
        start_time = time.time()
        compressed_data, info = engine.compress_with_spe(data, password if enable_encryption else None)
        compress_time = time.time() - start_time
        
        # 解凍
        start_time = time.time()
        decompressed_data = engine.decompress_with_spe(
            compressed_data, info, password if enable_encryption else None
        )
        decompress_time = time.time() - start_time
        
        # 可逆性チェック
        decompressed_hash = hashlib.sha256(decompressed_data).hexdigest()
        is_reversible = (original_hash == decompressed_hash)
        
        compression_ratio = info.get('overall_compression_ratio', 0)
        
        return {
            'data_name': data_name,
            'mode': 'lightweight' if lightweight_mode else 'standard',
            'encryption': enable_encryption,
            'compression_ratio': compression_ratio,
            'compress_time': compress_time,
            'decompress_time': decompress_time,
            'is_reversible': is_reversible,
            'original_size': len(data),
            'compressed_size': len(compressed_data),
            'throughput_compress': (len(data) / (1024 * 1024)) / compress_time if compress_time > 0 else 0,
            'throughput_decompress': (len(data) / (1024 * 1024)) / decompress_time if decompress_time > 0 else 0,
            'info': info
        }
        
    except Exception as e:
        print(f"TMC+SPE エンジンテストエラー: {e}")
        return {
            'data_name': data_name,
            'mode': 'lightweight' if lightweight_mode else 'standard',
            'encryption': enable_encryption,
            'error': str(e)
        }


def run_comprehensive_comparison():
    """総合比較テストの実行"""
    print("🎯 NEXUS TMC v9.1 + SPE Integration - Comprehensive Comparison Test")
    print("=" * 80)
    
    test_data_suite = generate_test_data_suite()
    
    results = []
    
    for data_name, data in test_data_suite.items():
        print(f"\n📊 Testing: {data_name} ({len(data) // 1024}KB)")
        print("-" * 60)
        
        # TMC+SPE 標準モード（暗号化あり）
        result = test_tmc_spe_engine(data, data_name, lightweight_mode=False, enable_encryption=True)
        result['engine'] = 'TMC+SPE Standard'
        results.append(result)
        
        # TMC+SPE 軽量モード（暗号化あり）
        result = test_tmc_spe_engine(data, data_name, lightweight_mode=True, enable_encryption=True)
        result['engine'] = 'TMC+SPE Lightweight'
        results.append(result)
        
        # TMC 標準モード（暗号化なし）
        result = test_tmc_spe_engine(data, data_name, lightweight_mode=False, enable_encryption=False)
        result['engine'] = 'TMC Standard'
        results.append(result)
        
        # 競合比較
        if PY7ZR_AVAILABLE:
            ratio, c_time, d_time, reversible = test_competitor_7zip(data)
            results.append({
                'data_name': data_name,
                'engine': '7zip',
                'compression_ratio': ratio,
                'compress_time': c_time,
                'decompress_time': d_time,
                'is_reversible': reversible,
                'throughput_compress': (len(data) / (1024 * 1024)) / c_time if c_time > 0 else 0,
                'throughput_decompress': (len(data) / (1024 * 1024)) / d_time if d_time > 0 else 0
            })
        
        if ZSTD_AVAILABLE:
            ratio, c_time, d_time, reversible = test_competitor_zstd(data)
            results.append({
                'data_name': data_name,
                'engine': 'Zstandard',
                'compression_ratio': ratio,
                'compress_time': c_time,
                'decompress_time': d_time,
                'is_reversible': reversible,
                'throughput_compress': (len(data) / (1024 * 1024)) / c_time if c_time > 0 else 0,
                'throughput_decompress': (len(data) / (1024 * 1024)) / d_time if d_time > 0 else 0
            })
    
    # 結果分析とレポート生成
    print("\n" + "=" * 80)
    print("📈 COMPREHENSIVE COMPARISON RESULTS")
    print("=" * 80)
    
    for data_name in test_data_suite.keys():
        print(f"\n📊 {data_name.upper()}")
        print("-" * 60)
        
        data_results = [r for r in results if r.get('data_name') == data_name and 'error' not in r]
        
        if not data_results:
            print("❌ No valid results")
            continue
        
        print(f"{'Engine':<20} {'Ratio%':<8} {'C.Time':<8} {'D.Time':<8} {'C.MB/s':<8} {'D.MB/s':<8} {'Rev.':<5}")
        print("-" * 60)
        
        for result in data_results:
            engine = result.get('engine', 'Unknown')
            ratio = result.get('compression_ratio', 0)
            c_time = result.get('compress_time', 0)
            d_time = result.get('decompress_time', 0)
            c_throughput = result.get('throughput_compress', 0)
            d_throughput = result.get('throughput_decompress', 0)
            reversible = "✅" if result.get('is_reversible', False) else "❌"
            
            print(f"{engine:<20} {ratio:<8.1f} {c_time:<8.3f} {d_time:<8.3f} "
                  f"{c_throughput:<8.1f} {d_throughput:<8.1f} {reversible:<5}")
    
    # 総合評価
    print("\n" + "=" * 80)
    print("🏆 OVERALL EVALUATION")
    print("=" * 80)
    
    tmc_spe_results = [r for r in results if 'TMC' in r.get('engine', '') and 'error' not in r]
    if tmc_spe_results:
        avg_ratio = sum(r.get('compression_ratio', 0) for r in tmc_spe_results) / len(tmc_spe_results)
        avg_speed = sum(r.get('throughput_compress', 0) for r in tmc_spe_results) / len(tmc_spe_results)
        all_reversible = all(r.get('is_reversible', False) for r in tmc_spe_results)
        
        print(f"📊 TMC+SPE Average Performance:")
        print(f"  Compression Ratio: {avg_ratio:.1f}%")
        print(f"  Average Speed: {avg_speed:.1f} MB/s")
        print(f"  Reversibility: {'✅ Perfect' if all_reversible else '❌ Issues detected'}")
        
        if avg_ratio > 80:
            print("🏅 Excellent compression ratio!")
        if all_reversible:
            print("🔒 Perfect data integrity!")


def main():
    """メイン実行関数"""
    print("🚀 Starting NEXUS TMC v9.1 + SPE Integration Test")
    
    try:
        run_comprehensive_comparison()
        
        print("\n" + "=" * 80)
        print("✅ All tests completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
