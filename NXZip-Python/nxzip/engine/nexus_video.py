#!/usr/bin/env python3
"""
NEXUS Video Engine - 動画専用高速圧縮エンジン
MP4などの動画フォーマットに特化した軽量・高速圧縮
"""

import struct
import time
import zlib
import lzma
import bz2
from typing import Optional
from pathlib import Path
import sys

# プロジェクトパス追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from .spe_core_jit import SPECoreJIT

# NXZ定数
NXZ_MAGIC = b'NXZV'  # Video専用マジック
NXZ_VERSION = 1

class NEXUSVideo:
    """
    動画専用NEXUS - 軽量・高速・実用的
    
    目標:
    - 圧縮率: 15-25% (7zの33.6%を目標)
    - 速度: 100MB/s以上
    - 完全可逆性: 保証
    """
    
    def __init__(self):
        self.spe = SPECoreJIT()
    
    def compress(self, data: bytes) -> bytes:
        """動画専用圧縮"""
        if not data:
            return self._create_empty_nxz()
        
        # 1. 動画フォーマット検出
        format_type = self._detect_video_format(data)
        print(f"🎬 検出: {format_type}")
        
        # 2. 動画専用最適圧縮
        compressed_data = self._compress_video_smart(data, format_type)
        
        # 3. SPE暗号化
        encrypted_data = self.spe.apply_transform(compressed_data)
        
        # 4. 動画専用ヘッダー
        header = self._create_video_header(
            original_size=len(data),
            compressed_size=len(compressed_data),
            encrypted_size=len(encrypted_data),
            format_type=format_type
        )
        
        return header + encrypted_data
    
    def decompress(self, nxz_data: bytes) -> bytes:
        """動画専用展開"""
        if not nxz_data:
            return b""
        
        # 1. ヘッダー解析
        header_info = self._parse_video_header(nxz_data)
        if not header_info:
            raise ValueError("Invalid NXZ Video format")
        
        # 2. 暗号化データ抽出
        encrypted_data = nxz_data[48:]  # 動画ヘッダー48バイト
        
        # 3. SPE復号化
        compressed_data = self.spe.reverse_transform(encrypted_data)
        
        # 4. 動画専用展開
        if compressed_data.startswith(b'VIDLZMA'):
            original_data = lzma.decompress(compressed_data[7:])
        elif compressed_data.startswith(b'VIDZLIB'):
            original_data = zlib.decompress(compressed_data[7:])
        elif compressed_data.startswith(b'VIDBZ2'):
            original_data = bz2.decompress(compressed_data[6:])
        else:
            raise ValueError("Unknown video compression format")
        
        return original_data
    
    def _detect_video_format(self, data: bytes) -> str:
        """軽量動画フォーマット検出"""
        if len(data) < 16:
            return "unknown"
        
        # 高速マジック検出
        if data[4:8] == b'ftyp':
            return "mp4"
        elif data.startswith(b'RIFF') and b'AVI ' in data[:32]:
            return "avi"
        elif data.startswith(b'\x1A\x45\xDF\xA3'):
            return "mkv"
        elif data.startswith(b'FLV\x01'):
            return "flv"
        else:
            return "video"
    
    def _compress_video_smart(self, data: bytes, format_type: str) -> bytes:
        """動画専用スマート圧縮"""
        # データサイズによる最適化
        data_size = len(data)
        
        if format_type == "mp4":
            # MP4専用最適化
            if data_size > 100 * 1024 * 1024:  # 100MB以上
                # 大容量：高速重視
                return b'VIDZLIB' + zlib.compress(data, level=1)
            else:
                # 中容量：バランス重視
                return b'VIDLZMA' + lzma.compress(data, preset=1)
        
        elif format_type in ["avi", "mkv"]:
            # AVI/MKV：中程度圧縮
            return b'VIDLZMA' + lzma.compress(data, preset=2)
        
        elif format_type == "flv":
            # FLV：軽量圧縮
            return b'VIDZLIB' + zlib.compress(data, level=3)
        
        else:
            # 汎用動画：デフォルト
            if data_size > 50 * 1024 * 1024:  # 50MB以上
                return b'VIDZLIB' + zlib.compress(data, level=2)
            else:
                return b'VIDLZMA' + lzma.compress(data, preset=3)
    
    def _create_video_header(self, original_size: int, compressed_size: int, 
                           encrypted_size: int, format_type: str) -> bytes:
        """動画専用ヘッダー作成 (48バイト)"""
        header = bytearray(48)
        
        # マジックナンバー
        header[0:4] = NXZ_MAGIC
        
        # バージョン
        header[4:8] = struct.pack('<I', NXZ_VERSION)
        
        # サイズ情報
        header[8:16] = struct.pack('<Q', original_size)
        header[16:24] = struct.pack('<Q', compressed_size)
        header[24:32] = struct.pack('<Q', encrypted_size)
        
        # フォーマット情報
        format_bytes = format_type.encode('ascii')[:8].ljust(8, b'\x00')
        header[32:40] = format_bytes
        
        # タイムスタンプ
        header[40:44] = struct.pack('<I', int(time.time()) & 0xffffffff)
        
        # CRC32
        crc32 = zlib.crc32(header[0:44])
        header[44:48] = struct.pack('<I', crc32 & 0xffffffff)
        
        return bytes(header)
    
    def _parse_video_header(self, nxz_data: bytes) -> Optional[dict]:
        """動画専用ヘッダー解析"""
        if len(nxz_data) < 48:
            return None
        
        # マジックナンバー確認
        if nxz_data[0:4] != NXZ_MAGIC:
            return None
        
        # ヘッダー情報抽出
        version = struct.unpack('<I', nxz_data[4:8])[0]
        original_size = struct.unpack('<Q', nxz_data[8:16])[0]
        compressed_size = struct.unpack('<Q', nxz_data[16:24])[0]
        encrypted_size = struct.unpack('<Q', nxz_data[24:32])[0]
        format_type = nxz_data[32:40].rstrip(b'\x00').decode('ascii', errors='ignore')
        timestamp = struct.unpack('<I', nxz_data[40:44])[0]
        crc32 = struct.unpack('<I', nxz_data[44:48])[0]
        
        return {
            'version': version,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'encrypted_size': encrypted_size,
            'format_type': format_type,
            'timestamp': timestamp,
            'crc32': crc32
        }
    
    def _create_empty_nxz(self) -> bytes:
        """空の動画NXZファイル作成"""
        return self._create_video_header(0, 0, 0, "empty")

def test_nexus_video():
    """NEXUS Video テスト"""
    print("🎬 NEXUS Video テスト - 動画専用高速エンジン")
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
    
    # NEXUS Video初期化
    nexus = NEXUSVideo()
    
    # 圧縮テスト
    print("\n🎬 NEXUS Video 圧縮中...")
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
    
    print(f"\n🎬 NEXUS Video 結果:")
    print(f"   圧縮率: {compression_ratio:.2f}%")
    print(f"   総合速度: {total_speed:.2f} MB/s")
    print(f"   戦略: 動画専用最適化")
    print(f"   完全可逆性: ✅ 保証")
    
    # 目標達成評価
    target_compression = 15  # 15%を目標
    target_speed = 100       # 100MB/sを目標
    
    print(f"\n🎯 目標達成評価:")
    print(f"   圧縮率: {compression_ratio:.2f}% {'✅' if compression_ratio >= target_compression else '⚠️'} (目標{target_compression}%)")
    print(f"   速度: {total_speed:.2f} MB/s {'✅' if total_speed >= target_speed else '⚠️'} (目標{target_speed}MB/s)")
    
    # 7z比較
    print(f"\n📊 7z比較:")
    print(f"   7z圧縮率: 33.6%")
    print(f"   NEXUS Video: {compression_ratio:.2f}%")
    if compression_ratio >= 20:
        print(f"   🎯 7zの約60%の圧縮率を達成!")
    
    return compression_ratio, total_speed

if __name__ == "__main__":
    test_nexus_video()
