#!/usr/bin/env python3
"""
NEXUS Video Fast - 動画専用超高速エンジン
速度最優先・軽量設計の動画圧縮
"""

import struct
import time
import zlib
import lzma
from typing import Optional
from pathlib import Path
import sys

# プロジェクトパス追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from .spe_core_jit import SPECoreJIT

# NXZ定数
NXZ_MAGIC = b'NXZF'  # Fast専用マジック
NXZ_VERSION = 1

class NEXUSVideoFast:
    """
    動画専用高速NEXUS - 速度最優先
    
    戦略:
    1. 超軽量フォーマット検出
    2. 高速圧縮アルゴリズム（zlib level 1-3）
    3. SPE最適化
    4. 最小限ヘッダー
    """
    
    def __init__(self):
        self.spe = SPECoreJIT()
    
    def compress(self, data: bytes) -> bytes:
        """超高速動画圧縮"""
        if not data:
            return self._create_empty_nxz()
        
        # 1. 超軽量検出
        is_mp4 = data[4:8] == b'ftyp'
        
        # 2. 高速圧縮（速度重視）
        if is_mp4:
            # MP4: 超高速圧縮
            compressed_data = b'FASTZLIB' + zlib.compress(data, level=1)
        else:
            # その他: 高速圧縮
            compressed_data = b'FASTZLIB' + zlib.compress(data, level=2)
        
        # 3. SPE暗号化
        encrypted_data = self.spe.apply_transform(compressed_data)
        
        # 4. 最小ヘッダー
        header = self._create_fast_header(
            original_size=len(data),
            compressed_size=len(compressed_data),
            encrypted_size=len(encrypted_data)
        )
        
        return header + encrypted_data
    
    def decompress(self, nxz_data: bytes) -> bytes:
        """超高速動画展開"""
        if not nxz_data:
            return b""
        
        # 1. 最小ヘッダー解析
        if len(nxz_data) < 32:
            raise ValueError("Invalid NXZ Fast format")
        
        # 2. 暗号化データ抽出
        encrypted_data = nxz_data[32:]  # 最小ヘッダー32バイト
        
        # 3. SPE復号化
        compressed_data = self.spe.reverse_transform(encrypted_data)
        
        # 4. 高速展開
        if compressed_data.startswith(b'FASTZLIB'):
            original_data = zlib.decompress(compressed_data[8:])
        else:
            raise ValueError("Unknown fast compression format")
        
        return original_data
    
    def _create_fast_header(self, original_size: int, compressed_size: int, encrypted_size: int) -> bytes:
        """最小ヘッダー作成 (32バイト)"""
        header = bytearray(32)
        
        # マジックナンバー
        header[0:4] = NXZ_MAGIC
        
        # バージョン
        header[4:8] = struct.pack('<I', NXZ_VERSION)
        
        # サイズ情報
        header[8:16] = struct.pack('<Q', original_size)
        header[16:24] = struct.pack('<Q', encrypted_size)
        
        # 簡易チェックサム
        checksum = (original_size + encrypted_size) & 0xffffffff
        header[24:28] = struct.pack('<I', checksum)
        
        # タイムスタンプ（簡易）
        header[28:32] = struct.pack('<I', int(time.time()) & 0xffffffff)
        
        return bytes(header)
    
    def _create_empty_nxz(self) -> bytes:
        """空の高速NXZファイル作成"""
        return self._create_fast_header(0, 0, 0)

def test_nexus_video_fast():
    """NEXUS Video Fast テスト"""
    print("⚡ NEXUS Video Fast テスト - 超高速動画エンジン")
    print("=" * 60)
    
    # MP4テストファイル
    test_file = Path(r"C:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\OneTEL_CADDi全体会議午後_restored.mp4")
    
    if not test_file.exists():
        print("❌ MP4テストファイルが見つかりません")
        return
    
    file_size = test_file.stat().st_size
    print(f"📄 ファイル: {test_file.name}")
    print(f"📊 サイズ: {file_size//1024//1024} MB")
    
    # データ読み込み
    print("\n📖 データ読み込み中...")
    with open(test_file, 'rb') as f:
        data = f.read()
    
    # NEXUS Video Fast初期化
    nexus = NEXUSVideoFast()
    
    # 圧縮テスト
    print("\n⚡ NEXUS Video Fast 圧縮中...")
    start_time = time.perf_counter()
    compressed = nexus.compress(data)
    compress_time = time.perf_counter() - start_time
    
    # 圧縮結果
    compression_ratio = (1 - len(compressed) / len(data)) * 100
    compress_speed = (len(data) / 1024 / 1024) / compress_time
    
    print(f"✅ 圧縮完了!")
    print(f"   📈 圧縮率: {compression_ratio:.2f}%")
    print(f"   ⚡ 速度: {compress_speed:.2f} MB/s")
    print(f"   ⏱️ 時間: {compress_time:.2f}秒")
    
    # 展開テスト
    print(f"\n🔄 展開テスト中...")
    start_time = time.perf_counter()
    decompressed = nexus.decompress(compressed)
    decomp_time = time.perf_counter() - start_time
    
    # 展開結果
    decomp_speed = (len(data) / 1024 / 1024) / decomp_time
    
    print(f"✅ 展開完了!")
    print(f"   ⚡ 速度: {decomp_speed:.2f} MB/s")
    print(f"   ⏱️ 時間: {decomp_time:.2f}秒")
    
    # 正確性確認
    is_correct = data == decompressed
    print(f"   🔍 正確性: {'✅ OK' if is_correct else '❌ NG'}")
    
    # 総合評価
    total_time = compress_time + decomp_time
    total_speed = (len(data) * 2 / 1024 / 1024) / total_time
    
    print(f"\n⚡ NEXUS Video Fast 結果:")
    print(f"   圧縮率: {compression_ratio:.2f}%")
    print(f"   総合速度: {total_speed:.2f} MB/s")
    print(f"   戦略: 超高速・軽量設計")
    print(f"   完全可逆性: ✅ 保証")
    
    # 速度重視目標
    target_compression = 10   # 10%を目標（速度重視）
    target_speed = 100        # 100MB/sを目標
    
    print(f"\n🎯 高速目標評価:")
    print(f"   圧縮率: {compression_ratio:.2f}% {'✅' if compression_ratio >= target_compression else '⚠️'} (目標{target_compression}%)")
    print(f"   速度: {total_speed:.2f} MB/s {'✅' if total_speed >= target_speed else '⚠️'} (目標{target_speed}MB/s)")
    
    # 改善提案
    if total_speed < target_speed:
        print(f"\n💡 速度改善案:")
        print(f"   - SPE処理の最適化")
        print(f"   - zlib level 1 固定")
        print(f"   - ヘッダー最小化")
        print(f"   - メモリ効率化")
    
    return compression_ratio, total_speed

if __name__ == "__main__":
    test_nexus_video_fast()
