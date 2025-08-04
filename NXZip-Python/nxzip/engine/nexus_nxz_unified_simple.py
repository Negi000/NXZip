#!/usr/bin/env python3
"""
NEXUS NXZ統合エンジン v2.0 - Simple & Effective
SPE + TMC v9.1統合システム - ベンチマーク特化版
"""

import time
from typing import Tuple, Dict, Any, Optional

# 必要なコンポーネント
from .nexus_tmc_v91_modular import NEXUSTMCEngineV91
from .spe_core_jit import SPECoreJIT


class NXZUnifiedEngine:
    """
    NXZ統合圧縮エンジン v2.0
    ベンチマーク用シンプル統合版
    """
    
    def __init__(self):
        """エンジン初期化"""
        print("🚀 NXZ統合エンジン v2.0 初期化...")
        
        # コアエンジン初期化
        self.tmc_engine = NEXUSTMCEngineV91()
        self.spe_core = SPECoreJIT()
        
        # 統計情報
        self.stats = {
            'total_files': 0,
            'compression_time': 0.0,
            'decompression_time': 0.0
        }
    
    def compress_nxz(self, data: bytes, compression_level: int = 6, 
                     lightweight_mode: bool = False) -> bytes:
        """
        NXZ統合圧縮
        SPE + TMC v9.1 + Enhanced NXZ
        """
        start_time = time.time()
        print(f"🚀 NXZ統合圧縮開始 (サイズ: {len(data):,} bytes)")
        
        try:
            # Phase 1: TMC v9.1 圧縮
            print("🔄 Phase 1: TMC v9.1 圧縮...")
            if lightweight_mode:
                # 軽量モード: 高速圧縮（基本圧縮のみ）
                compressed_data, tmc_info = self.tmc_engine.core_compressor.compress_core(data, method='zlib')
            else:
                # 通常モード: 高圧縮（フルTMC v9.1）
                compressed_data, tmc_info = self.tmc_engine.compress(data)
            
            compression_ratio = (1 - len(compressed_data) / len(data)) * 100
            print(f"📊 TMC圧縮完了: {len(compressed_data):,} bytes ({compression_ratio:.1f}% 削減)")
            
            # Phase 2: SPE暗号化
            print("🔄 Phase 2: SPE構造保持暗号化...")
            spe_data = self.spe_core.apply_transform(compressed_data)
            print(f"🔒 SPE変換完了: {len(spe_data):,} bytes")
            
            # Phase 3: 簡単なヘッダー付加（ベンチマーク用）
            header = self._create_simple_header(len(data), len(compressed_data), len(spe_data), tmc_info)
            final_data = header + spe_data
            
            total_time = time.time() - start_time
            total_ratio = (1 - len(final_data) / len(data)) * 100
            
            print(f"✅ NXZ統合圧縮完了!")
            print(f"📈 最終圧縮率: {total_ratio:.1f}% ({len(data):,} → {len(final_data):,} bytes)")
            print(f"⚡ 処理時間: {total_time:.2f}秒")
            
            # 統計更新
            self.stats['total_files'] += 1
            self.stats['compression_time'] += total_time
            
            return final_data
            
        except Exception as e:
            print(f"❌ 圧縮エラー: {e}")
            raise
    
    def decompress_nxz(self, nxz_data: bytes) -> bytes:
        """
        NXZ統合展開
        Enhanced NXZ + SPE + TMC v9.1
        """
        start_time = time.time()
        print(f"🔓 NXZ統合展開開始 (サイズ: {len(nxz_data):,} bytes)")
        
        try:
            # Phase 1: ヘッダー解析
            header_info = self._parse_simple_header(nxz_data[:128])  # より大きなヘッダー
            header_size = header_info['header_size']
            spe_data = nxz_data[header_size:]
            
            print(f"📊 ヘッダー解析: 原サイズ {header_info['original_size']:,} bytes")
            
            # Phase 2: SPE逆変換
            print("🔄 Phase 2: SPE逆変換...")
            compressed_data = self.spe_core.reverse_transform(spe_data)
            print(f"🔓 SPE逆変換完了: {len(compressed_data):,} bytes")
            
            # Phase 3: TMC v9.1 展開（メタデータ使用）
            print("🔄 Phase 3: TMC v9.1 展開...")
            tmc_info = header_info['tmc_info']
            if tmc_info.get('method') in ['zlib', 'lzma', 'bz2']:
                # 軽量モードの場合（基本圧縮）
                original_data = self._decompress_core(compressed_data, tmc_info)
            else:
                # 通常モード（フルTMC v9.1）
                original_data = self.tmc_engine.decompress(compressed_data, tmc_info)
            
            total_time = time.time() - start_time
            
            print(f"✅ NXZ統合展開完了!")
            print(f"📈 展開サイズ: {len(original_data):,} bytes")
            print(f"⚡ 処理時間: {total_time:.2f}秒")
            
            # 統計更新
            self.stats['decompression_time'] += total_time
            
            # サイズ検証
            if len(original_data) != header_info['original_size']:
                print(f"⚠️  サイズ不一致: 期待 {header_info['original_size']:,}, 実際 {len(original_data):,}")
            
            return original_data
            
        except Exception as e:
            print(f"❌ 展開エラー: {e}")
            raise
    
    def _create_simple_header(self, original_size: int, compressed_size: int, 
                             encrypted_size: int) -> bytes:
        """簡単なヘッダー作成（ベンチマーク用）"""
        header = bytearray(32)
        
        # マジックナンバー "NXZU" (4 bytes)
        header[0:4] = b"NXZU"
        
        # サイズ情報 (24 bytes: 各8バイト)
        import struct
        struct.pack_into('<QQQ', header, 4, original_size, compressed_size, encrypted_size)
        
        # 予約領域 (4 bytes)
        header[28:32] = b'\x00' * 4
        
        return bytes(header)
    
    def _parse_simple_header(self, header: bytes) -> Dict[str, int]:
        """簡単なヘッダー解析"""
        if len(header) != 32:
            raise ValueError("不正なヘッダーサイズ")
        
        if header[0:4] != b"NXZU":
            raise ValueError("不正なマジックナンバー")
        
        import struct
        original_size, compressed_size, encrypted_size = struct.unpack('<QQQ', header[4:28])
        
        return {
            'original_size': original_size,
            'compressed_size': compressed_size,
            'encrypted_size': encrypted_size
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        return self.stats.copy()
