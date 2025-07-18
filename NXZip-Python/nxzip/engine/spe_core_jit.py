#!/usr/bin/env python3
"""
SPE Core - Structure-Preserving Encryption (JIT Optimized Version)
NXZip の核となる暗号化技術 - NumbaによるJIT最適化版

JIT最適化のポイント:
1. numbaによる機械語コンパイル
2. nopython=True（純粋C速度）
3. 並列処理（parallel=True）
4. 高度な最適化指令
5. メモリ効率の最大化
"""

import hashlib
import secrets
import struct
import time
import numpy as np
from typing import List, Optional
from numba import jit, prange, types
from numba.core import config
import numba

# NumbaのJIT最適化設定
try:
    config.THREADING_LAYER = 'tbb'  # Intel TBBによる並列化
    numba.set_num_threads(8)  # 8スレッド並列実行
except Exception:
    # TBBが利用できない場合は通常の並列化を使用
    try:
        numba.set_num_threads(4)  # 4スレッド並列実行
    except Exception:
        pass  # 並列化なしで続行

# ========== JIT最適化関数 ==========

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _jit_ultra_fast_stage1(data: np.ndarray, original_len: int) -> np.ndarray:
    """JIT最適化された超高速Stage1"""
    # パディングサイズ計算
    padding_size = (8 - (original_len % 8)) % 8
    
    # 新しい配列作成
    new_len = original_len + padding_size + 8
    result = np.empty(new_len, dtype=np.uint8)
    
    # オリジナルデータコピー
    result[:original_len] = data[:original_len]
    
    # パディング追加
    for i in prange(padding_size):
        result[original_len + i] = padding_size
    
    # 長さ情報追加（little endian手動実装）
    length_value = original_len
    for i in range(8):
        result[new_len - 8 + i] = (length_value >> (i * 8)) & 0xFF
    
    return result

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _jit_reverse_stage1(data: np.ndarray) -> np.ndarray:
    """JIT最適化された超高速Stage1逆変換"""
    if len(data) < 8:
        return data
    
    # 長さ情報読み取り
    original_len = 0
    for i in range(8):
        original_len |= (int(data[len(data) - 8 + i]) << (i * 8))
    
    return data[:original_len]

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _jit_ultra_fast_stage2(data: np.ndarray, sbox: np.ndarray, xor_key: np.ndarray) -> np.ndarray:
    """JIT最適化された超高速Stage2（置換+XOR）"""
    data_len = len(data)
    
    # 64バイトずつ並列処理（最大並列化）
    for i in prange(0, data_len, 64):
        end_idx = min(i + 64, data_len)
        
        # 64バイトブロック並列処理
        for j in range(i, end_idx):
            key_idx = j & 0x1F  # % 32
            data[j] = sbox[data[j]] ^ xor_key[key_idx]
    
    return data

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _jit_reverse_stage2(data: np.ndarray, inverse_sbox: np.ndarray, xor_key: np.ndarray) -> np.ndarray:
    """JIT最適化された超高速Stage2逆変換"""
    data_len = len(data)
    
    # 64バイトずつ並列処理（最大並列化）
    for i in prange(0, data_len, 64):
        end_idx = min(i + 64, data_len)
        
        # 64バイトブロック並列処理
        for j in range(i, end_idx):
            key_idx = j & 0x1F  # % 32
            data[j] = inverse_sbox[data[j] ^ xor_key[key_idx]]
    
    return data

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _jit_ultra_fast_stage3(data: np.ndarray, shift_values: np.ndarray) -> np.ndarray:
    """JIT最適化された超高速Stage3（シフト）"""
    data_len = len(data)
    
    # 64バイトずつ並列処理（最大並列化）
    for i in prange(0, data_len, 64):
        end_idx = min(i + 64, data_len)
        
        # 64バイトブロック並列処理
        for j in range(i, end_idx):
            shift_idx = j & 0x1F  # % 32
            shift_val = shift_values[shift_idx]
            data[j] = ((data[j] << shift_val) | (data[j] >> (8 - shift_val))) & 0xFF
    
    return data

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _jit_reverse_stage3(data: np.ndarray, shift_values: np.ndarray) -> np.ndarray:
    """JIT最適化された超高速Stage3逆変換"""
    data_len = len(data)
    
    # 64バイトずつ並列処理（最大並列化）
    for i in prange(0, data_len, 64):
        end_idx = min(i + 64, data_len)
        
        # 64バイトブロック並列処理
        for j in range(i, end_idx):
            shift_idx = j & 0x1F  # % 32
            shift_val = shift_values[shift_idx]
            data[j] = ((data[j] >> shift_val) | (data[j] << (8 - shift_val))) & 0xFF
    
    return data

# ========== SPE Core JIT最適化クラス ==========

class SPECoreJIT:
    """
    Structure-Preserving Encryption Core (JIT Optimized)
    
    NumbaによるJIT最適化された3段階SPE変換:
    - 機械語レベルの最適化
    - 並列処理による最大化
    - nopython=True（純粋C速度）
    """
    
    def __init__(self):
        # JIT最適化: 最小限の初期化
        self._initialize_jit_optimized()
        
    def _initialize_jit_optimized(self) -> None:
        """JIT最適化初期化"""
        # 超高速化: 単純なマスターキー
        self._master_key = hashlib.blake2b(
            b"NXZip_SPE_JIT_v1.0" + struct.pack('<Q', int(time.time()) // 3600),
            digest_size=32
        ).digest()
        
        # JIT最適化: 事前計算済みNumpy配列
        self._init_jit_tables()
    
    def _init_jit_tables(self) -> None:
        """JIT最適化テーブル初期化"""
        # 1. JIT最適化XORキー（numpy配列）
        self._xor_key = np.frombuffer(self._master_key, dtype=np.uint8)
        
        # 2. JIT最適化シフト値（numpy配列）
        shift_list = [(self._master_key[i] & 0x07) + 1 for i in range(32)]
        self._shift_values = np.array(shift_list, dtype=np.uint8)
        
        # 3. JIT最適化置換テーブル（numpy配列）
        seed = int.from_bytes(self._master_key[:8], 'little')
        sbox_list = []
        for i in range(256):
            seed = (seed * 1103515245 + 12345) & 0xFFFFFFFF
            sbox_list.append(seed & 0xFF)
        self._fast_sbox = np.array(sbox_list, dtype=np.uint8)
        
        # 4. JIT最適化逆置換テーブル（numpy配列）
        inverse_sbox_list = [0] * 256
        for i in range(256):
            inverse_sbox_list[self._fast_sbox[i]] = i
        self._fast_inverse_sbox = np.array(inverse_sbox_list, dtype=np.uint8)
    
    def apply_transform(self, data: bytes) -> bytes:
        """JIT最適化された3段階SPE変換"""
        if not data:
            return data
        
        # numpy配列に変換
        data_array = np.frombuffer(data, dtype=np.uint8).copy()
        original_len = len(data_array)
        
        # JIT最適化された3段階変換
        result = _jit_ultra_fast_stage1(data_array, original_len)
        result = _jit_ultra_fast_stage2(result, self._fast_sbox, self._xor_key)
        result = _jit_ultra_fast_stage3(result, self._shift_values)
        
        return result.tobytes()
    
    def reverse_transform(self, data: bytes) -> bytes:
        """JIT最適化された3段階SPE逆変換"""
        if not data:
            return data
        
        # numpy配列に変換
        data_array = np.frombuffer(data, dtype=np.uint8).copy()
        
        # JIT最適化された逆変換（逆順）
        result = _jit_reverse_stage3(data_array, self._shift_values)
        result = _jit_reverse_stage2(result, self._fast_inverse_sbox, self._xor_key)
        result = _jit_reverse_stage1(result)
        
        return result.tobytes()

# ========== JIT最適化テスト関数 ==========

def test_jit_spe_performance():
    """JIT最適化SPE性能テスト"""
    print("🚀 JIT最適化SPE性能テスト（NumbaによるJIT最適化）")
    print("=" * 60)
    
    # JIT初期化（最初の実行で最適化）
    print("📊 JIT初期化中...")
    spe = SPECoreJIT()
    test_data = b"JIT initialization test"
    _ = spe.apply_transform(test_data)
    _ = spe.reverse_transform(_)
    print("✅ JIT初期化完了")
    
    # テストデータ生成
    test_sizes = [1024, 10240, 102400, 1024000, 10240000]  # 1KB, 10KB, 100KB, 1MB, 10MB
    
    for size in test_sizes:
        print(f"\n📊 テストサイズ: {size:,} bytes")
        
        # テストデータ
        test_data = secrets.token_bytes(size)
        
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
        target_speed = 20.0  # 20MB/s目標（JIT最適化）
        if encryption_speed >= target_speed and decryption_speed >= target_speed:
            print(f"   🎯 JIT目標速度達成! (>{target_speed} MB/s)")
        else:
            print(f"   ⚠️  JIT目標速度未達成 (<{target_speed} MB/s)")

def benchmark_jit_vs_normal():
    """JIT最適化 vs 通常版 性能比較"""
    print("\n🏎️ JIT最適化 vs 通常版 性能比較")
    print("=" * 60)
    
    # JIT版初期化
    spe_jit = SPECoreJIT()
    
    # 通常版初期化
    try:
        from .spe_core_fast import SPECore
        spe_normal = SPECore()
        
        # テストデータ
        test_data = secrets.token_bytes(1024000)  # 1MB
        
        # JIT版テスト
        start_time = time.perf_counter()
        for _ in range(10):
            encrypted_data = spe_jit.apply_transform(test_data)
            decrypted_data = spe_jit.reverse_transform(encrypted_data)
        jit_time = (time.perf_counter() - start_time) / 10
        
        # 通常版テスト
        start_time = time.perf_counter()
        for _ in range(10):
            encrypted_data = spe_normal.apply_transform(test_data)
            decrypted_data = spe_normal.reverse_transform(encrypted_data)
        normal_time = (time.perf_counter() - start_time) / 10
        
        # 結果計算
        jit_speed = (1024000 / 1024 / 1024) / jit_time
        normal_speed = (1024000 / 1024 / 1024) / normal_time
        speedup = jit_speed / normal_speed
        
        print(f"📊 テストサイズ: 1MB (10回平均)")
        print(f"   JIT版総合速度: {jit_speed:.2f} MB/s ({jit_time*1000:.2f}ms)")
        print(f"   通常版総合速度: {normal_speed:.2f} MB/s ({normal_time*1000:.2f}ms)")
        print(f"   🚀 JIT高速化率: {speedup:.2f}x")
        
    except ImportError:
        print("   ⚠️  通常版SPECoreが見つかりません")


if __name__ == "__main__":
    # JIT最適化テスト実行
    test_jit_spe_performance()
    
    # JIT vs 通常版比較
    benchmark_jit_vs_normal()
    
    # 基本動作確認
    print("\n🔍 基本動作確認テスト")
    print("=" * 50)
    
    test_vectors = [
        b"test",
        b"NXZip SPE Core JIT Test " * 100,
        b"Advanced JIT Security Test Data " * 1000
    ]
    
    spe = SPECoreJIT()
    
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
