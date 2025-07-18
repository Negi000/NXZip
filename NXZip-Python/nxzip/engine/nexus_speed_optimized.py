#!/usr/bin/env python3
"""
NEXUS Speed-Optimized Engine - 速度維持で圧縮率向上
高速性能を犠牲にせず、圧縮率を向上させる最適化エンジン
"""

import lzma
import zlib
import time
import struct
import secrets
import hashlib
from typing import Tuple, Optional

class NexusSpeedOptimized:
    """
    NEXUS Speed-Optimized Engine
    
    速度維持原則:
    - 圧縮時間: 現在の速度を維持 (>100 MB/s)
    - 圧縮率改善: 軽量な前処理で効果最大化
    - メモリ効率: ストリーミング処理
    """
    
    def __init__(self):
        # 高速初期化
        self._init_fast_tables()
    
    def _init_fast_tables(self):
        """高速テーブル初期化"""
        # 軽量な前処理テーブル
        self._byte_freq_table = [0] * 256
        self._pattern_cache = {}
        
        # 高速圧縮設定
        self._lzma_preset = 6  # 速度と圧縮率のバランス
        self._zlib_level = 6   # 高速圧縮レベル
    
    def compress(self, data: bytes) -> Tuple[bytes, str]:
        """
        速度維持の改良圧縮
        """
        if not data:
            return data, 'none'
        
        # 高速データ分析（10KB以上でのみ実行）
        if len(data) > 10240:
            preprocessed = self._fast_preprocess(data)
        else:
            preprocessed = data
        
        # 高速圧縮方式選択
        if len(preprocessed) < 1024:
            # 小さなデータ: zlib高速
            compressed = self._fast_zlib_compress(preprocessed)
            method = 'zlib_fast'
        elif len(preprocessed) < 102400:  # 100KB未満
            # 中サイズ: LZMA中速
            compressed = self._fast_lzma_compress(preprocessed)
            method = 'lzma_fast'
        else:
            # 大サイズ: ハイブリッド高速
            compressed = self._hybrid_fast_compress(preprocessed)
            method = 'hybrid_fast'
        
        # 軽量ヘッダー追加
        method_byte = method.encode('utf-8')
        header = bytes([len(method_byte)]) + method_byte
        
        return header + compressed, method
    
    def decompress(self, data: bytes) -> bytes:
        """
        高速展開
        """
        if not data:
            return data
        
        # ヘッダー読み取り
        method_len = data[0]
        method = data[1:1+method_len].decode('utf-8')
        compressed_data = data[1+method_len:]
        
        # 高速展開
        if method == 'zlib_fast':
            return self._fast_zlib_decompress(compressed_data)
        elif method == 'lzma_fast':
            return self._fast_lzma_decompress(compressed_data)
        elif method == 'hybrid_fast':
            return self._hybrid_fast_decompress(compressed_data)
        else:
            return compressed_data
    
    def _fast_preprocess(self, data: bytes) -> bytes:
        """
        軽量前処理（速度重視）
        """
        # 高速バイト頻度分析
        freq = [0] * 256
        for byte in data[:min(4096, len(data))]:  # 先頭4KBのみ分析
            freq[byte] += 1
        
        # 最頻出バイト検出
        max_freq = max(freq)
        if max_freq > len(data) * 0.05:  # 5%以上の頻度
            most_frequent = freq.index(max_freq)
            # 簡単な置換（高速）
            if most_frequent != 0:
                return data.replace(bytes([most_frequent]), b'\x00')
        
        return data
    
    def _fast_zlib_compress(self, data: bytes) -> bytes:
        """高速zlib圧縮"""
        return zlib.compress(data, level=self._zlib_level)
    
    def _fast_zlib_decompress(self, data: bytes) -> bytes:
        """高速zlib展開"""
        return zlib.decompress(data)
    
    def _fast_lzma_compress(self, data: bytes) -> bytes:
        """高速LZMA圧縮"""
        return lzma.compress(data, format=lzma.FORMAT_XZ, preset=self._lzma_preset)
    
    def _fast_lzma_decompress(self, data: bytes) -> bytes:
        """高速LZMA展開"""
        return lzma.decompress(data, format=lzma.FORMAT_XZ)
    
    def _hybrid_fast_compress(self, data: bytes) -> bytes:
        """
        ハイブリッド高速圧縮
        """
        # 段階1: 軽量前処理
        preprocessed = self._fast_preprocess(data)
        
        # 段階2: 高速圧縮
        # データサイズに応じて最適な圧縮を選択
        if len(preprocessed) > 1024000:  # 1MB以上
            # 大容量: zlib高速
            return self._fast_zlib_compress(preprocessed)
        else:
            # 中容量: LZMA高速
            return self._fast_lzma_compress(preprocessed)
    
    def _hybrid_fast_decompress(self, data: bytes) -> bytes:
        """
        ハイブリッド高速展開
        """
        # 自動判別して展開
        try:
            # LZMA試行
            return lzma.decompress(data, format=lzma.FORMAT_XZ)
        except:
            # zlib試行
            return zlib.decompress(data)

# ========== 速度重視テスト ==========

def test_speed_optimized_compression():
    """速度重視圧縮テスト"""
    print("⚡ NEXUS Speed-Optimized - 速度維持圧縮率向上テスト")
    print("=" * 70)
    
    # 対象ファイル
    test_file = r"C:\Users\241822\Desktop\新しいフォルダー (2)\需要引当予測リスト クエリ.txt"
    
    try:
        # ファイル読み込み
        with open(test_file, 'rb') as f:
            original_data = f.read()
    except FileNotFoundError:
        print("⚠️ テストファイルが見つかりません。サンプルデータでテストします。")
        # サンプルデータ生成
        original_data = b"NXZip Speed Test Data " * 50000
    
    original_size = len(original_data)
    print(f"📊 テストデータサイズ: {original_size:,} bytes ({original_size/1024/1024:.2f} MB)")
    
    # 速度最適化エンジン初期化
    speed_engine = NexusSpeedOptimized()
    
    # 速度重視圧縮テスト
    print("\n🚀 高速圧縮実行中...")
    start_time = time.perf_counter()
    compressed_data, method = speed_engine.compress(original_data)
    compress_time = time.perf_counter() - start_time
    
    # 圧縮結果分析
    compressed_size = len(compressed_data)
    compression_ratio = (1 - compressed_size / original_size) * 100
    compress_speed = (original_size / 1024 / 1024) / compress_time
    
    print(f"✅ 高速圧縮完了!")
    print(f"   🔸 圧縮方式: {method}")
    print(f"   🔸 圧縮率: {compression_ratio:.2f}%")
    print(f"   🔸 圧縮速度: {compress_speed:.2f} MB/s")
    print(f"   🔸 圧縮時間: {compress_time:.3f}秒")
    
    # 速度基準チェック
    target_speed = 100.0  # 100MB/s維持目標
    if compress_speed >= target_speed:
        print(f"   🎯 速度目標達成! (>{target_speed} MB/s)")
    else:
        print(f"   ⚠️ 速度目標未達成 (<{target_speed} MB/s)")
    
    # 高速展開テスト
    print("\n⚡ 高速展開実行中...")
    start_time = time.perf_counter()
    decompressed_data = speed_engine.decompress(compressed_data)
    decompress_time = time.perf_counter() - start_time
    
    # 展開結果分析
    decompress_speed = (original_size / 1024 / 1024) / decompress_time
    is_correct = original_data == decompressed_data
    
    print(f"✅ 高速展開完了!")
    print(f"   🔸 展開速度: {decompress_speed:.2f} MB/s")
    print(f"   🔸 展開時間: {decompress_time:.3f}秒")
    print(f"   🔸 データ整合性: {'✅ 完全一致' if is_correct else '❌ エラー'}")
    
    # 従来比較（推定値）
    print(f"\n📊 従来NXZip比較 (推定)")
    print("=" * 50)
    
    # 既知の従来結果
    old_ratio = 95.88  # 従来の圧縮率
    old_speed = 123.55  # 従来の速度
    
    ratio_improvement = compression_ratio - old_ratio
    speed_maintenance = (compress_speed / old_speed) * 100
    
    print(f"🔸 従来圧縮率: {old_ratio:.2f}%")
    print(f"🔸 改良圧縮率: {compression_ratio:.2f}%")
    print(f"🔸 圧縮率改善: {ratio_improvement:+.2f}%")
    print(f"🔸 速度維持率: {speed_maintenance:.1f}%")
    
    # 7z比較
    zz_size = 3084928  # 7zサイズ
    vs_7z_ratio = compressed_size / zz_size
    
    print(f"\n🏆 7z比較結果:")
    print(f"   🔸 7zサイズ: {zz_size/1024/1024:.2f} MB")
    print(f"   🔸 改良NXZ: {compressed_size/1024/1024:.2f} MB")
    print(f"   🔸 7z対比: {vs_7z_ratio:.2f}倍")
    
    if vs_7z_ratio < 1.0:
        print(f"   🎉 7z超越達成! ({(1-vs_7z_ratio)*100:.1f}%改善)")
    else:
        print(f"   📈 7z差: {(vs_7z_ratio-1)*100:.1f}%")
    
    # 総合評価
    print(f"\n🎯 NEXUS Speed-Optimized 総合評価")
    print("=" * 50)
    print(f"⚡ 速度維持: {'✅' if compress_speed >= target_speed else '❌'}")
    print(f"📈 圧縮率改善: {'✅' if compression_ratio > old_ratio else '❌'}")
    print(f"🔍 データ整合性: {'✅' if is_correct else '❌'}")
    print(f"🏆 実用性: {'✅ 高速+高圧縮' if compress_speed >= target_speed and compression_ratio > old_ratio else '⚠️ 要改善'}")
    
    return compress_speed >= target_speed, compression_ratio, vs_7z_ratio

# ========== 複数サイズテスト ==========

def test_multiple_sizes():
    """複数サイズでの速度テスト"""
    print("\n📊 複数サイズ速度テスト")
    print("=" * 50)
    
    engine = NexusSpeedOptimized()
    
    test_sizes = [1024, 10240, 102400, 1024000]  # 1KB, 10KB, 100KB, 1MB
    
    for size in test_sizes:
        print(f"\n🔍 サイズ: {size:,} bytes")
        
        # テストデータ生成
        test_data = secrets.token_bytes(size)
        
        # 圧縮テスト
        start_time = time.perf_counter()
        compressed, method = engine.compress(test_data)
        compress_time = time.perf_counter() - start_time
        
        # 展開テスト
        start_time = time.perf_counter()
        decompressed = engine.decompress(compressed)
        decompress_time = time.perf_counter() - start_time
        
        # 結果計算
        ratio = (1 - len(compressed) / len(test_data)) * 100
        compress_speed = (size / 1024 / 1024) / compress_time if compress_time > 0 else float('inf')
        decompress_speed = (size / 1024 / 1024) / decompress_time if decompress_time > 0 else float('inf')
        
        print(f"   方式: {method}")
        print(f"   圧縮率: {ratio:.2f}%")
        print(f"   圧縮速度: {compress_speed:.2f} MB/s")
        print(f"   展開速度: {decompress_speed:.2f} MB/s")
        print(f"   整合性: {'✅' if test_data == decompressed else '❌'}")

if __name__ == "__main__":
    # 速度重視テスト
    test_speed_optimized_compression()
    
    # 複数サイズテスト
    test_multiple_sizes()
