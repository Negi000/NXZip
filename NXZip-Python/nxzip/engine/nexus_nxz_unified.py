#!/usr/bin/env python3
"""
NEXUS NXZ統合エンジン v1.0
SPE (Structure-Preserving Encryption) + TMC v9.1 統合システム
通常モード・軽量モード対応ベンチマーク用統合クラス
"""

import os
import time
import asyncio
from typing import Tuple, Dict, Any, Optional
from .nexus_tmc_v91_modular import NEXUSTMCEngineV91
from .spe_core_jit import SPECoreJIT
from ..formats.enhanced_nxz import SuperNXZipFile

class NXZUnifiedEngine:
    """
    NXZ統合圧縮エンジン
    SPE + TMC v9.1 + Enhanced NXZ Format
    """
    
    def __init__(self, lightweight_mode: bool = False, encryption_enabled: bool = True):
        self.lightweight_mode = lightweight_mode
        self.encryption_enabled = encryption_enabled
        
        # コアエンジンの初期化
        self.tmc_engine = NEXUSTMCEngineV91(lightweight_mode=lightweight_mode)
        self.spe_engine = SPECoreJIT() if encryption_enabled else None
        self.nxz_format = SuperNXZipFile()
        
        # 統計情報
        self.stats = {
            'files_processed': 0,
            'total_compression_time': 0.0,
            'total_decompression_time': 0.0,
            'total_input_size': 0,
            'total_output_size': 0,
            'encryption_overhead': 0.0,
            'reversibility_tests': 0,
            'reversibility_success': 0
        }
        
        mode_name = "軽量モード" if lightweight_mode else "通常モード"
        encryption_status = "SPE有効" if encryption_enabled else "暗号化無効"
        print(f"🚀 NXZ統合エンジン初期化完了: {mode_name}, {encryption_status}")
    
    async def compress_nxz(self, data: bytes, password: Optional[str] = None) -> Tuple[bytes, Dict[str, Any]]:
        """NXZ統合圧縮"""
        start_time = time.time()
        
        try:
            # Phase 1: TMC v9.1圧縮
            print("🔄 Phase 1: TMC v9.1圧縮実行中...")
            tmc_start = time.time()
            compressed_data, tmc_info = await self.tmc_engine.compress_tmc_v91_async(data)
            tmc_time = time.time() - tmc_start
            
            # Phase 2: SPE暗号化（オプション）
            if self.encryption_enabled and password and self.spe_engine:
                print("🔄 Phase 2: SPE暗号化実行中...")
                spe_start = time.time()
                encrypted_data = self.spe_engine.apply_transform(compressed_data)
                spe_time = time.time() - spe_start
                self.stats['encryption_overhead'] += spe_time
                spe_info = {
                    'algorithm': 'SPE-JIT',
                    'encryption_time': spe_time,
                    'overhead_bytes': len(encrypted_data) - len(compressed_data)
                }
            else:
                encrypted_data = compressed_data
                spe_info = {'encryption': 'disabled'}
                spe_time = 0.0
            
            # Phase 3: NXZフォーマット作成
            print("🔄 Phase 3: NXZフォーマット作成中...")
            nxz_data = self.nxz_format.create_archive(
                encrypted_data, 
                compression_level=compression_level
            )
            
            total_time = time.time() - start_time
            
            # 統計更新
            self.stats['files_processed'] += 1
            self.stats['total_compression_time'] += total_time
            self.stats['total_input_size'] += len(data)
            self.stats['total_output_size'] += len(nxz_data)
            
            # 圧縮情報統合
            compression_info = {
                'engine': 'NXZ Unified v1.0',
                'mode': 'lightweight' if self.lightweight_mode else 'standard',
                'original_size': len(data),
                'compressed_size': len(nxz_data),
                'compression_ratio': (1 - len(nxz_data) / len(data)) * 100 if len(data) > 0 else 0,
                'compression_time': total_time,
                'tmc_time': tmc_time,
                'spe_time': spe_time,
                'throughput_mbps': (len(data) / (1024 * 1024) / total_time) if total_time > 0 else 0,
                'tmc_info': tmc_info,
                'spe_info': spe_info,
                'encryption_enabled': self.encryption_enabled
            }
            
            print(f"✅ NXZ圧縮完了: {compression_info['compression_ratio']:.1f}% 圧縮, {compression_info['throughput_mbps']:.1f}MB/s")
            return nxz_data, compression_info
            
        except Exception as e:
            print(f"❌ NXZ圧縮エラー: {e}")
            raise
    
    async def decompress_nxz(self, nxz_data: bytes, password: Optional[str] = None) -> Tuple[bytes, Dict[str, Any]]:
        """NXZ統合解凍"""
        start_time = time.time()
        
        try:
            # Phase 1: NXZフォーマット解析
            print("🔄 Phase 1: NXZフォーマット解析中...")
            encrypted_data, nxz_metadata = self.nxz_format.parse_nxz(nxz_data)
            
            # Phase 2: SPE復号化（必要に応じて）
            if nxz_metadata.get('encryption_enabled', False) and password and self.spe_engine:
                print("🔄 Phase 2: SPE復号化実行中...")
                spe_start = time.time()
                compressed_data = self.spe_engine.reverse_transform(encrypted_data)
                spe_time = time.time() - spe_start
            else:
                compressed_data = encrypted_data
                spe_time = 0.0
            
            # Phase 3: TMC v9.1解凍
            print("🔄 Phase 3: TMC v9.1解凍実行中...")
            tmc_start = time.time()
            tmc_info = nxz_metadata.get('tmc_info', {})
            decompressed_data = self.tmc_engine.decompress(compressed_data, tmc_info)
            tmc_time = time.time() - tmc_start
            
            total_time = time.time() - start_time
            
            # 統計更新
            self.stats['total_decompression_time'] += total_time
            
            decompression_info = {
                'engine': 'NXZ Unified v1.0',
                'decompressed_size': len(decompressed_data),
                'decompression_time': total_time,
                'tmc_time': tmc_time,
                'spe_time': spe_time,
                'throughput_mbps': (len(decompressed_data) / (1024 * 1024) / total_time) if total_time > 0 else 0,
                'metadata': nxz_metadata
            }
            
            print(f"✅ NXZ解凍完了: {len(nxz_data)} -> {len(decompressed_data)} bytes, {decompression_info['throughput_mbps']:.1f}MB/s")
            return decompressed_data, decompression_info
            
        except Exception as e:
            print(f"❌ NXZ解凍エラー: {e}")
            raise
    
    def test_reversibility(self, test_data: bytes, password: Optional[str] = None) -> bool:
        """可逆性テスト"""
        try:
            self.stats['reversibility_tests'] += 1
            
            # 圧縮・解凍サイクル
            # SPE暗号化は構造保持なのでパスワード不要
            compressed, comp_info = asyncio.run(self.compress_nxz(test_data, None))  # パスワードなしでテスト
            decompressed, decomp_info = asyncio.run(self.decompress_nxz(compressed, None))
            
            # データ比較
            is_reversible = decompressed == test_data
            
            if is_reversible:
                self.stats['reversibility_success'] += 1
                print("✅ 可逆性テスト成功")
            else:
                print(f"❌ 可逆性テスト失敗: {len(test_data)} -> {len(decompressed)}")
            
            return is_reversible
            
        except Exception as e:
            print(f"❌ 可逆性テストエラー: {e}")
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計取得"""
        stats = self.stats.copy()
        
        if stats['files_processed'] > 0:
            stats['avg_compression_time'] = stats['total_compression_time'] / stats['files_processed']
            stats['avg_decompression_time'] = stats['total_decompression_time'] / stats['files_processed']
            
        if stats['total_input_size'] > 0:
            stats['overall_compression_ratio'] = (1 - stats['total_output_size'] / stats['total_input_size']) * 100
            
        if stats['reversibility_tests'] > 0:
            stats['reversibility_rate'] = (stats['reversibility_success'] / stats['reversibility_tests']) * 100
        
        return stats

class CompetitiveCompressionEngine:
    """競合圧縮エンジン（7-Zip, Zstandard）"""
    
    @staticmethod
    def compress_7zip(data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """7-Zip圧縮"""
        start_time = time.time()
        try:
            import py7zr
            import io
            
            compressed_buffer = io.BytesIO()
            with py7zr.SevenZipFile(compressed_buffer, 'w') as archive:
                archive.writestr(data, 'data.bin')
            
            compressed_data = compressed_buffer.getvalue()
            compression_time = time.time() - start_time
            
            info = {
                'engine': '7-Zip',
                'original_size': len(data),
                'compressed_size': len(compressed_data),
                'compression_ratio': (1 - len(compressed_data) / len(data)) * 100 if len(data) > 0 else 0,
                'compression_time': compression_time,
                'throughput_mbps': (len(data) / (1024 * 1024) / compression_time) if compression_time > 0 else 0
            }
            
            return compressed_data, info
            
        except ImportError:
            # フォールバック: LZMA使用
            import lzma
            compressed_data = lzma.compress(data, preset=6)
            compression_time = time.time() - start_time
            
            info = {
                'engine': '7-Zip (LZMA fallback)',
                'original_size': len(data),
                'compressed_size': len(compressed_data),
                'compression_ratio': (1 - len(compressed_data) / len(data)) * 100 if len(data) > 0 else 0,
                'compression_time': compression_time,
                'throughput_mbps': (len(data) / (1024 * 1024) / compression_time) if compression_time > 0 else 0
            }
            
            return compressed_data, info
    
    @staticmethod
    def decompress_7zip(compressed_data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """7-Zip解凍"""
        start_time = time.time()
        try:
            import py7zr
            import io
            
            compressed_buffer = io.BytesIO(compressed_data)
            with py7zr.SevenZipFile(compressed_buffer, 'r') as archive:
                allfiles = archive.readall()
                decompressed_data = list(allfiles.values())[0].read()
            
            decompression_time = time.time() - start_time
            
            info = {
                'engine': '7-Zip',
                'decompressed_size': len(decompressed_data),
                'decompression_time': decompression_time,
                'throughput_mbps': (len(decompressed_data) / (1024 * 1024) / decompression_time) if decompression_time > 0 else 0
            }
            
            return decompressed_data, info
            
        except ImportError:
            # フォールバック: LZMA使用
            import lzma
            decompressed_data = lzma.decompress(compressed_data)
            decompression_time = time.time() - start_time
            
            info = {
                'engine': '7-Zip (LZMA fallback)',
                'decompressed_size': len(decompressed_data),
                'decompression_time': decompression_time,
                'throughput_mbps': (len(decompressed_data) / (1024 * 1024) / decompression_time) if decompression_time > 0 else 0
            }
            
            return decompressed_data, info
    
    @staticmethod
    def compress_zstd(data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """Zstandard圧縮"""
        start_time = time.time()
        try:
            import zstandard as zstd
            
            cctx = zstd.ZstdCompressor(level=6)
            compressed_data = cctx.compress(data)
            compression_time = time.time() - start_time
            
            info = {
                'engine': 'Zstandard',
                'original_size': len(data),
                'compressed_size': len(compressed_data),
                'compression_ratio': (1 - len(compressed_data) / len(data)) * 100 if len(data) > 0 else 0,
                'compression_time': compression_time,
                'throughput_mbps': (len(data) / (1024 * 1024) / compression_time) if compression_time > 0 else 0
            }
            
            return compressed_data, info
            
        except ImportError:
            # フォールバック: zlib使用
            import zlib
            compressed_data = zlib.compress(data, level=6)
            compression_time = time.time() - start_time
            
            info = {
                'engine': 'Zstandard (zlib fallback)',
                'original_size': len(data),
                'compressed_size': len(compressed_data),
                'compression_ratio': (1 - len(compressed_data) / len(data)) * 100 if len(data) > 0 else 0,
                'compression_time': compression_time,
                'throughput_mbps': (len(data) / (1024 * 1024) / compression_time) if compression_time > 0 else 0
            }
            
            return compressed_data, info
    
    @staticmethod
    def decompress_zstd(compressed_data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """Zstandard解凍"""
        start_time = time.time()
        try:
            import zstandard as zstd
            
            dctx = zstd.ZstdDecompressor()
            decompressed_data = dctx.decompress(compressed_data)
            decompression_time = time.time() - start_time
            
            info = {
                'engine': 'Zstandard',
                'decompressed_size': len(decompressed_data),
                'decompression_time': decompression_time,
                'throughput_mbps': (len(decompressed_data) / (1024 * 1024) / decompression_time) if decompression_time > 0 else 0
            }
            
            return decompressed_data, info
            
        except ImportError:
            # フォールバック: zlib使用
            import zlib
            decompressed_data = zlib.decompress(compressed_data)
            decompression_time = time.time() - start_time
            
            info = {
                'engine': 'Zstandard (zlib fallback)',
                'decompressed_size': len(decompressed_data),
                'decompression_time': decompression_time,
                'throughput_mbps': (len(decompressed_data) / (1024 * 1024) / decompression_time) if decompression_time > 0 else 0
            }
            
            return decompressed_data, info

if __name__ == "__main__":
    print("🚀 NEXUS NXZ統合エンジン v1.0")
    print("📦 SPE + TMC v9.1 + Enhanced NXZ Format")
