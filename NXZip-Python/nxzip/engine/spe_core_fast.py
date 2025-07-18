#!/usr/bin/env python3
"""
SPE Core - Structure-Preserving Encryption (Ultra High Performance Version)
NXZip の核となる暗号化技術 - 超高速化版

超高速化のポイント:
1. 単純化されたアルゴリズム
2. インライン演算による最適化
3. 最小限のメモリアクセス
4. 段階数削減
5. 直接配列操作
"""

import hashlib
import secrets
import struct
import time
from typing import List, Optional

class SPECore:
    """
    Structure-Preserving Encryption Core (Ultra High Performance)
    
    超高速化された3段階SPE変換:
    - 最小限の演算で最大のセキュリティ
    - インライン最適化
    - 直接配列操作
    """
    
    def __init__(self):
        # 超高速化: 最小限の初期化
        self._initialize_ultra_fast()
        
    def _initialize_ultra_fast(self) -> None:
        """超高速初期化"""
        # 超高速化: 単純なマスターキー
        self._master_key = hashlib.blake2b(
            b"NXZip_SPE_Ultra_Fast_v3.0" + struct.pack('<Q', int(time.time()) // 3600),
            digest_size=32
        ).digest()
        
        # 超高速化: 事前計算済みテーブル
        self._init_fast_tables()
    
    def _init_fast_tables(self) -> None:
        """超高速テーブル初期化"""
        # 1. 超高速XORキー
        self._xor_key = self._master_key
        
        # 2. 超高速シフト値（固定）
        self._shift_values = [(self._master_key[i] & 0x07) + 1 for i in range(32)]
        
        # 3. 超高速置換テーブル
        seed = int.from_bytes(self._master_key[:8], 'little')
        self._fast_sbox = []
        for i in range(256):
            seed = (seed * 1103515245 + 12345) & 0xFFFFFFFF
            self._fast_sbox.append(seed & 0xFF)
        
        # 4. 逆置換テーブル
        self._fast_inverse_sbox = [0] * 256
        for i in range(256):
            self._fast_inverse_sbox[self._fast_sbox[i]] = i
    
    def apply_transform(self, data: bytes) -> bytes:
        """超高速化された3段階SPE変換"""
        if not data:
            return data
            
        result = bytearray(data)
        original_len = len(result)
        
        # 超高速化: 3段階のみ
        result = self._ultra_fast_stage1(result, original_len)
        result = self._ultra_fast_stage2(result)
        result = self._ultra_fast_stage3(result)
        
        return bytes(result)
    
    def reverse_transform(self, data: bytes) -> bytes:
        """超高速化された3段階SPE逆変換"""
        if not data:
            return data
            
        result = bytearray(data)
        
        # 逆順で超高速変換
        result = self._reverse_ultra_fast_stage3(result)
        result = self._reverse_ultra_fast_stage2(result)
        result = self._reverse_ultra_fast_stage1(result)
        
        return bytes(result)
    
    # ========== 超高速化された変換段階 ==========
    
    def _ultra_fast_stage1(self, data: bytearray, original_len: int) -> bytearray:
        """Stage 1: 超高速前処理"""
        # 超高速化: 最小限のパディング
        padding_size = (8 - (original_len % 8)) % 8
        data.extend([padding_size] * padding_size)
        data.extend(struct.pack('<Q', original_len))
        return data
    
    def _reverse_ultra_fast_stage1(self, data: bytearray) -> bytearray:
        """Stage 1 超高速逆変換"""
        if len(data) < 8:
            return data
        original_len = struct.unpack('<Q', data[-8:])[0]
        return data[:original_len]
    
    def _ultra_fast_stage2(self, data: bytearray) -> bytearray:
        """Stage 2: 超高速置換+XOR（最適化版）"""
        # 超高速化: 事前計算済みテーブル
        sbox = self._fast_sbox
        xor_key = self._xor_key
        
        # 超高速化: 32バイトずつ処理（最大並列化）
        i = 0
        data_len = len(data)
        
        # 32バイトブロック処理
        while i + 32 <= data_len:
            # 32バイト並列処理（極限最適化）
            for j in range(32):
                idx = i + j
                data[idx] = sbox[data[idx]] ^ xor_key[j]
            i += 32
        
        # 残りバイト処理
        while i < data_len:
            data[i] = sbox[data[i]] ^ xor_key[i & 0x1F]
            i += 1
        
        return data
    
    def _reverse_ultra_fast_stage2(self, data: bytearray) -> bytearray:
        """Stage 2 超高速逆変換（最適化版）"""
        inverse_sbox = self._fast_inverse_sbox
        xor_key = self._xor_key
        
        # 超高速化: 32バイトずつ処理（最大並列化）
        i = 0
        data_len = len(data)
        
        # 32バイトブロック処理
        while i + 32 <= data_len:
            # 32バイト並列処理（極限最適化）
            for j in range(32):
                idx = i + j
                data[idx] = inverse_sbox[data[idx] ^ xor_key[j]]
            i += 32
        
        # 残りバイト処理
        while i < data_len:
            data[i] = inverse_sbox[data[i] ^ xor_key[i & 0x1F]]
            i += 1
        
        return data
    
    def _ultra_fast_stage3(self, data: bytearray) -> bytearray:
        """Stage 3: 超高速シフト+交換（最適化版）"""
        # 超高速化: 事前計算済みシフト値
        shift_values = self._shift_values
        
        # 超高速化: 32バイトずつ処理（最大並列化）
        i = 0
        data_len = len(data)
        
        # 32バイトブロック処理
        while i + 32 <= data_len:
            # 32バイト並列処理（極限最適化）
            for j in range(32):
                idx = i + j
                shift_val = shift_values[j]
                data[idx] = ((data[idx] << shift_val) | (data[idx] >> (8 - shift_val))) & 0xFF
            i += 32
        
        # 残りバイト処理
        while i < data_len:
            shift_val = shift_values[i & 0x1F]
            data[i] = ((data[i] << shift_val) | (data[i] >> (8 - shift_val))) & 0xFF
            i += 1
        
        return data
    
    def _reverse_ultra_fast_stage3(self, data: bytearray) -> bytearray:
        """Stage 3 超高速逆変換（最適化版）"""
        shift_values = self._shift_values
        
        # 超高速化: 32バイトずつ処理（最大並列化）
        i = 0
        data_len = len(data)
        
        # 32バイトブロック処理
        while i + 32 <= data_len:
            # 32バイト並列処理（極限最適化）
            for j in range(32):
                idx = i + j
                shift_val = shift_values[j]
                data[idx] = ((data[idx] >> shift_val) | (data[idx] << (8 - shift_val))) & 0xFF
            i += 32
        
        # 残りバイト処理
        while i < data_len:
            shift_val = shift_values[i & 0x1F]
            data[i] = ((data[i] >> shift_val) | (data[i] << (8 - shift_val))) & 0xFF
            i += 1
        
        return data


# ========== 超高速化テスト関数 ==========

def test_spe_performance():
    """SPE超高速化テスト"""
    print("🚀 SPE超高速化性能テスト")
    print("=" * 50)
    
    # テストデータ生成
    test_sizes = [1024, 10240, 102400, 1024000, 10240000]  # 1KB, 10KB, 100KB, 1MB, 10MB
    
    for size in test_sizes:
        print(f"\n📊 テストサイズ: {size:,} bytes")
        
        # テストデータ
        test_data = secrets.token_bytes(size)
        
        # SPE Core初期化
        spe = SPECore()
        
        # 暗号化テスト（複数回実行で精度向上）
        iterations = 1000 if size <= 1024 else (100 if size <= 10240 else 10)
        
        start_time = time.perf_counter()
        for _ in range(iterations):
            encrypted_data = spe.apply_transform(test_data)
        encryption_time = (time.perf_counter() - start_time) / iterations
        
        # 復号化テスト（複数回実行で精度向上）
        start_time = time.perf_counter()
        for _ in range(iterations):
            decrypted_data = spe.reverse_transform(encrypted_data)
        decryption_time = (time.perf_counter() - start_time) / iterations
        
        # 結果確認
        is_correct = test_data == decrypted_data
        
        # 速度計算（安全な除算）
        if encryption_time > 0:
            encryption_speed = (size / 1024 / 1024) / encryption_time
        else:
            encryption_speed = float('inf')
        
        if decryption_time > 0:
            decryption_speed = (size / 1024 / 1024) / decryption_time
        else:
            decryption_speed = float('inf')
        
        print(f"   暗号化: {encryption_speed:.2f} MB/s ({encryption_time*1000:.4f}ms)")
        print(f"   復号化: {decryption_speed:.2f} MB/s ({decryption_time*1000:.4f}ms)")
        print(f"   正確性: {'✅' if is_correct else '❌'}")
        print(f"   実行回数: {iterations}回平均")
        
        # 目標速度チェック
        target_speed = 10.0  # 10MB/s目標
        if encryption_speed >= target_speed and decryption_speed >= target_speed:
            print(f"   🎯 目標速度達成! (>{target_speed} MB/s)")
        else:
            print(f"   ⚠️  目標速度未達成 (<{target_speed} MB/s)")


if __name__ == "__main__":
    # 超高速化テスト実行
    test_spe_performance()
    
    # 基本動作確認
    print("\n🔍 基本動作確認テスト")
    print("=" * 50)
    
    test_vectors = [
        b"test",
        b"NXZip SPE Core Test " * 100,
        b"Advanced Security Test Data " * 1000
    ]
    
    spe = SPECore()
    
    for i, vector in enumerate(test_vectors):
        print(f"Testing vector {i}: {vector[:20]}...")
        
        encrypted = spe.apply_transform(vector)
        decrypted = spe.reverse_transform(encrypted)
        
        if vector == decrypted:
            print(f"✅ Vector {i} passed")
        else:
            print(f"❌ Vector {i} failed")
            print(f"   Original:  {vector[:50]}...")
            print(f"   Decrypted: {decrypted[:50]}...")
