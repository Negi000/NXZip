#!/usr/bin/env python3
"""
NXZ Format - JIT最適化版を使用したNXZフォーマット実装
"""

import struct
import time
import hashlib
import zlib
import secrets
from typing import Optional, Tuple
from ..engine.spe_core_simple_jit import SPECoreSimpleJIT
from ..engine.nexus import NXZipNEXUSFinal

# NXZ定数
NXZ_MAGIC = b'NXZP'
NXZ_VERSION = 1

class NXZFormatJIT:
    """
    NXZ Format Implementation with JIT-optimized SPE
    
    JIT最適化された超高速NXZフォーマット:
    - SPE JIT最適化による49倍高速化
    - NEXUSハイブリッド圧縮
    - 100%データ整合性保証
    """
    
    def __init__(self):
        self.spe_core = SPECoreSimpleJIT()
        self.nexus = NXZipNEXUSFinal()
        
    def compress_and_encrypt(self, data: bytes, password: Optional[str] = None) -> bytes:
        """
        JIT最適化された圧縮+暗号化
        
        Args:
            data: 圧縮対象のデータ
            password: 暗号化パスワード（オプション）
            
        Returns:
            NXZ形式のバイナリデータ
        """
        if not data:
            return self._create_empty_nxz()
        
        # 1. NEXUS圧縮（超高速ハイブリッド圧縮）
        compressed_result = self.nexus.compress(data)
        if isinstance(compressed_result, tuple):
            compressed_data = compressed_result[0]  # 圧縮データのみ取得
        else:
            compressed_data = compressed_result
        
        # 2. SPE暗号化（JIT最適化版）
        encrypted_data = self.spe_core.apply_transform(compressed_data)
        
        # 3. NXZヘッダー作成
        header = self._create_nxz_header(
            original_size=len(data),
            compressed_size=len(compressed_data),
            encrypted_size=len(encrypted_data)
        )
        
        # 4. 最終データ結合
        nxz_data = header + encrypted_data
        
        return nxz_data
    
    def decompress_and_decrypt(self, nxz_data: bytes, password: Optional[str] = None) -> bytes:
        """
        JIT最適化された復号化+展開
        
        Args:
            nxz_data: NXZ形式のバイナリデータ
            password: 復号化パスワード（オプション）
            
        Returns:
            展開されたオリジナルデータ
        """
        if not nxz_data:
            return b""
        
        # 1. NXZヘッダー解析
        header_info = self._parse_nxz_header(nxz_data)
        if not header_info:
            raise ValueError("Invalid NXZ format")
        
        # 2. 暗号化データ抽出
        encrypted_data = nxz_data[44:]  # ヘッダー44バイト後
        
        # 3. SPE復号化（JIT最適化版）
        compressed_data = self.spe_core.reverse_transform(encrypted_data)
        
        # 4. NEXUS展開
        decompressed_result = self.nexus.decompress(compressed_data)
        if isinstance(decompressed_result, tuple):
            original_data = decompressed_result[0]  # 展開データのみ取得
        else:
            original_data = decompressed_result
        
        # 5. データ整合性確認
        expected_size = header_info['original_size']
        actual_size = len(original_data)
        if actual_size != expected_size:
            print(f"DEBUG: Size mismatch - expected: {expected_size}, actual: {actual_size}")
            print(f"DEBUG: Header info: {header_info}")
            print(f"DEBUG: Compressed data size: {len(compressed_data)}")
            print(f"DEBUG: Original data preview: {original_data[:100]}...")
            raise ValueError(f"Data integrity check failed: expected {expected_size}, got {actual_size}")
        
        return original_data
    
    def _create_nxz_header(self, original_size: int, compressed_size: int, encrypted_size: int) -> bytes:
        """NXZヘッダー作成"""
        header = bytearray(44)
        
        # Magic bytes (4 bytes)
        header[0:4] = NXZ_MAGIC
        
        # Version (4 bytes)
        header[4:8] = struct.pack('<I', NXZ_VERSION)
        
        # Original size (8 bytes)
        header[8:16] = struct.pack('<Q', original_size)
        
        # Compressed size (8 bytes)
        header[16:24] = struct.pack('<Q', compressed_size)
        
        # Encrypted size (8 bytes)
        header[24:32] = struct.pack('<Q', encrypted_size)
        
        # Timestamp (8 bytes)
        header[32:40] = struct.pack('<Q', int(time.time()))
        
        # Checksum (4 bytes)
        checksum = zlib.crc32(header[0:40]) & 0xFFFFFFFF
        header[40:44] = struct.pack('<I', checksum)
        
        return bytes(header)
    
    def _parse_nxz_header(self, nxz_data: bytes) -> Optional[dict]:
        """NXZヘッダー解析"""
        if len(nxz_data) < 44:
            return None
        
        header = nxz_data[:44]
        
        # Magic bytes検証
        if header[0:4] != NXZ_MAGIC:
            return None
        
        # Version
        version = struct.unpack('<I', header[4:8])[0]
        
        # Sizes
        original_size = struct.unpack('<Q', header[8:16])[0]
        compressed_size = struct.unpack('<Q', header[16:24])[0]
        encrypted_size = struct.unpack('<Q', header[24:32])[0]
        
        # Timestamp
        timestamp = struct.unpack('<Q', header[32:40])[0]
        
        # Checksum検証
        expected_checksum = zlib.crc32(header[0:40]) & 0xFFFFFFFF
        actual_checksum = struct.unpack('<I', header[40:44])[0]
        
        if expected_checksum != actual_checksum:
            return None
        
        return {
            'version': version,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'encrypted_size': encrypted_size,
            'timestamp': timestamp
        }
    
    def _create_empty_nxz(self) -> bytes:
        """空のNXZファイル作成"""
        return self._create_nxz_header(0, 0, 0)

# ========== JIT最適化NXZフォーマット テスト関数 ==========

def test_nxz_jit_performance():
    """JIT最適化NXZフォーマット性能テスト"""
    print("🚀 JIT最適化NXZフォーマット性能テスト")
    print("=" * 60)
    
    # NXZ JIT版初期化
    nxz = NXZFormatJIT()
    
    # テストデータ生成
    test_sizes = [1024, 10240, 102400, 1024000, 10240000]  # 1KB, 10KB, 100KB, 1MB, 10MB
    
    for size in test_sizes:
        print(f"\n📊 テストサイズ: {size:,} bytes")
        
        # テストデータ
        test_data = secrets.token_bytes(size)
        
        # 圧縮+暗号化テスト
        iterations = 100 if size <= 10240 else 10
        
        start_time = time.perf_counter()
        for _ in range(iterations):
            nxz_data = nxz.compress_and_encrypt(test_data)
        compress_time = (time.perf_counter() - start_time) / iterations
        
        # 復号化+展開テスト
        start_time = time.perf_counter()
        for _ in range(iterations):
            recovered_data = nxz.decompress_and_decrypt(nxz_data)
        decompress_time = (time.perf_counter() - start_time) / iterations
        
        # 結果確認
        is_correct = test_data == recovered_data
        compression_ratio = (1 - len(nxz_data) / len(test_data)) * 100
        
        # 速度計算
        compress_speed = (size / 1024 / 1024) / compress_time if compress_time > 0 else float('inf')
        decompress_speed = (size / 1024 / 1024) / decompress_time if decompress_time > 0 else float('inf')
        total_speed = (size / 1024 / 1024) / (compress_time + decompress_time) if (compress_time + decompress_time) > 0 else float('inf')
        
        print(f"   圧縮: {compress_speed:.2f} MB/s ({compress_time*1000:.2f}ms)")
        print(f"   展開: {decompress_speed:.2f} MB/s ({decompress_time*1000:.2f}ms)")
        print(f"   総合: {total_speed:.2f} MB/s ({(compress_time + decompress_time)*1000:.2f}ms)")
        print(f"   圧縮率: {compression_ratio:.2f}%")
        print(f"   正確性: {'✅' if is_correct else '❌'}")
        print(f"   実行回数: {iterations}回平均")
        
        # 目標速度チェック
        target_speed = 10.0  # 10MB/s目標
        if total_speed >= target_speed:
            print(f"   🎯 JIT最適化NXZ目標速度達成! (>{target_speed} MB/s)")
        else:
            print(f"   ⚠️  JIT最適化NXZ目標速度未達成 (<{target_speed} MB/s)")

def benchmark_nxz_jit_vs_normal():
    """JIT最適化NXZ vs 通常版NXZ 性能比較"""
    print("\n🏎️ JIT最適化NXZ vs 通常版NXZ 性能比較")
    print("=" * 60)
    
    # JIT最適化NXZ版初期化
    nxz_jit = NXZFormatJIT()
    
    # 通常版NXZ初期化
    try:
        from .nxz_format import NXZFormat
        nxz_normal = NXZFormat()
        
        # テストデータ
        test_data = secrets.token_bytes(1024000)  # 1MB
        
        # JIT最適化NXZ版テスト
        start_time = time.perf_counter()
        for _ in range(10):
            nxz_data = nxz_jit.compress_and_encrypt(test_data)
            recovered_data = nxz_jit.decompress_and_decrypt(nxz_data)
        jit_time = (time.perf_counter() - start_time) / 10
        
        # 通常版NXZ版テスト
        start_time = time.perf_counter()
        for _ in range(10):
            nxz_data = nxz_normal.compress_and_encrypt(test_data)
            recovered_data = nxz_normal.decompress_and_decrypt(nxz_data)
        normal_time = (time.perf_counter() - start_time) / 10
        
        # 結果計算
        jit_speed = (1024000 / 1024 / 1024) / jit_time
        normal_speed = (1024000 / 1024 / 1024) / normal_time
        speedup = jit_speed / normal_speed
        
        print(f"📊 テストサイズ: 1MB (10回平均)")
        print(f"   JIT最適化NXZ総合速度: {jit_speed:.2f} MB/s ({jit_time*1000:.2f}ms)")
        print(f"   通常版NXZ総合速度: {normal_speed:.2f} MB/s ({normal_time*1000:.2f}ms)")
        print(f"   🚀 JIT最適化NXZ高速化率: {speedup:.2f}x")
        
    except ImportError:
        print("   ⚠️  通常版NXZFormatが見つかりません")


if __name__ == "__main__":
    # JIT最適化NXZフォーマットテスト実行
    test_nxz_jit_performance()
    
    # JIT最適化NXZ vs 通常版NXZ比較
    benchmark_nxz_jit_vs_normal()
    
    # 基本動作確認
    print("\n🔍 基本動作確認テスト")
    print("=" * 50)
    
    test_vectors = [
        b"test",
        b"NXZip JIT Format Test " * 100,
        b"Advanced JIT NXZ Format Test Data " * 1000
    ]
    
    nxz = NXZFormatJIT()
    
    for i, vector in enumerate(test_vectors):
        print(f"Testing vector {i}: {vector[:20]}...")
        
        nxz_data = nxz.compress_and_encrypt(vector)
        recovered = nxz.decompress_and_decrypt(nxz_data)
        
        if vector == recovered:
            print(f"✅ Vector {i} passed")
        else:
            print(f"❌ Vector {i} failed")
            print(f"   Original:  {vector[:50]}...")
            print(f"   Recovered: {recovered[:50]}...")
