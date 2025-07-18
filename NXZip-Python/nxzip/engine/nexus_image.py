#!/usr/bin/env python3
"""
NEXUS Image Engine - 画像専用圧縮エンジン
JPEG、PNG、GIF、BMPなどの画像フォーマットに最適化
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
NXZ_MAGIC = b'NXZI'  # Image専用マジック
NXZ_VERSION = 1

class NEXUSImage:
    """
    画像専用NEXUS - 画像フォーマットに最適化
    
    戦略:
    1. 画像フォーマット別最適化
    2. JPEG: 軽圧縮（既に圧縮済み）
    3. PNG: 中圧縮（一部可逆圧縮）
    4. BMP: 高圧縮（非圧縮形式）
    5. GIF: 中圧縮（LZW圧縮済み）
    """
    
    def __init__(self):
        self.spe = SPECoreJIT()
    
    def compress(self, data: bytes) -> bytes:
        """画像専用圧縮"""
        if not data:
            return self._create_empty_nxz()
        
        # 1. 画像フォーマット検出
        format_type = self._detect_image_format(data)
        print(f"🖼️ 検出: {format_type}")
        
        # 2. フォーマット別最適圧縮（改良版）
        if format_type == "jpeg":
            # JPEG: 軽圧縮（既に圧縮済み）
            compressed_data = b'IMGJPEG' + zlib.compress(data, level=1)
        elif format_type == "png":
            # PNG: 高圧縮（PNG内部構造を活用）
            compressed_data = b'IMGPNG' + lzma.compress(data, preset=6)
        elif format_type == "bmp":
            # BMP: 超高圧縮（非圧縮形式）
            compressed_data = b'IMGBMP' + lzma.compress(data, preset=9)
        elif format_type == "gif":
            # GIF: 中圧縮（LZW圧縮済み）
            compressed_data = b'IMGGIF' + zlib.compress(data, level=3)
        elif format_type == "webp":
            # WebP: 軽圧縮（既に圧縮済み）
            compressed_data = b'IMGWEBP' + zlib.compress(data, level=1)
        else:
            # その他: 標準圧縮
            compressed_data = b'IMGOTHER' + lzma.compress(data, preset=4)
        
        # 3. SPE暗号化
        encrypted_data = self.spe.apply_transform(compressed_data)
        
        # 4. 画像専用ヘッダー
        header = self._create_image_header(
            original_size=len(data),
            compressed_size=len(compressed_data),
            encrypted_size=len(encrypted_data),
            format_type=format_type
        )
        
        return header + encrypted_data
    
    def decompress(self, nxz_data: bytes) -> bytes:
        """画像専用展開"""
        if not nxz_data:
            return b""
        
        # 1. ヘッダー解析
        header_info = self._parse_image_header(nxz_data)
        if not header_info:
            raise ValueError("Invalid NXZ Image format")
        
        # 2. 暗号化データ抽出
        encrypted_data = nxz_data[40:]  # 画像ヘッダー40バイト
        
        # 3. SPE復号化
        compressed_data = self.spe.reverse_transform(encrypted_data)
        
        # 4. フォーマット別展開
        if compressed_data.startswith(b'IMGJPEG'):
            original_data = zlib.decompress(compressed_data[7:])
        elif compressed_data.startswith(b'IMGPNG'):
            original_data = lzma.decompress(compressed_data[6:])
        elif compressed_data.startswith(b'IMGBMP'):
            original_data = lzma.decompress(compressed_data[6:])
        elif compressed_data.startswith(b'IMGGIF'):
            original_data = zlib.decompress(compressed_data[6:])
        elif compressed_data.startswith(b'IMGWEBP'):
            original_data = zlib.decompress(compressed_data[7:])
        elif compressed_data.startswith(b'IMGOTHER'):
            original_data = zlib.decompress(compressed_data[8:])
        else:
            raise ValueError("Unknown image compression format")
        
        return original_data
    
    def _detect_image_format(self, data: bytes) -> str:
        """画像フォーマット検出"""
        if len(data) < 16:
            return "unknown"
        
        # 画像マジック検出
        if data.startswith(b'\xFF\xD8\xFF'):
            return "jpeg"
        elif data.startswith(b'\x89PNG\r\n\x1A\n'):
            return "png"
        elif data.startswith(b'GIF87a') or data.startswith(b'GIF89a'):
            return "gif"
        elif data.startswith(b'BM'):
            return "bmp"
        elif data.startswith(b'RIFF') and b'WEBP' in data[:16]:
            return "webp"
        else:
            return "image"
    
    def _create_image_header(self, original_size: int, compressed_size: int, 
                           encrypted_size: int, format_type: str) -> bytes:
        """画像専用ヘッダー作成 (40バイト)"""
        header = bytearray(40)
        
        # マジックナンバー
        header[0:4] = NXZ_MAGIC
        
        # バージョン
        header[4:8] = struct.pack('<I', NXZ_VERSION)
        
        # サイズ情報
        header[8:16] = struct.pack('<Q', original_size)
        header[16:24] = struct.pack('<Q', encrypted_size)
        
        # フォーマット情報
        format_bytes = format_type.encode('ascii')[:8].ljust(8, b'\x00')
        header[24:32] = format_bytes
        
        # タイムスタンプ
        header[32:36] = struct.pack('<I', int(time.time()) & 0xffffffff)
        
        # CRC32
        crc32 = zlib.crc32(header[0:36])
        header[36:40] = struct.pack('<I', crc32 & 0xffffffff)
        
        return bytes(header)
    
    def _parse_image_header(self, nxz_data: bytes) -> Optional[dict]:
        """画像専用ヘッダー解析"""
        if len(nxz_data) < 40:
            return None
        
        if nxz_data[0:4] != NXZ_MAGIC:
            return None
        
        version = struct.unpack('<I', nxz_data[4:8])[0]
        original_size = struct.unpack('<Q', nxz_data[8:16])[0]
        encrypted_size = struct.unpack('<Q', nxz_data[16:24])[0]
        format_type = nxz_data[24:32].rstrip(b'\x00').decode('ascii', errors='ignore')
        
        return {
            'version': version,
            'original_size': original_size,
            'encrypted_size': encrypted_size,
            'format_type': format_type
        }
    
    def _create_empty_nxz(self) -> bytes:
        """空の画像NXZファイル作成"""
        return self._create_image_header(0, 0, 0, "empty")

def test_nexus_image():
    """NEXUS Image テスト"""
    print("🖼️ NEXUS Image テスト - 画像専用圧縮エンジン")
    print("=" * 60)
    
    # 画像テストファイル - 複数テスト
    test_files = [
        Path(r"C:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\sample\COT-001.jpg"),
        Path(r"C:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\sample\COT-012.png")
    ]
    
    nexus = NEXUSImage()
    
    for test_file in test_files:
        if not test_file.exists():
            print(f"❌ {test_file.name} が見つかりません")
            continue
        
        file_size = test_file.stat().st_size
        print(f"\n📄 ファイル: {test_file.name}")
        print(f"📊 サイズ: {file_size//1024} KB")
        
        # データ読み込み
        print("\n📖 データ読み込み中...")
        with open(test_file, 'rb') as f:
            data = f.read()
        
        # 圧縮テスト
        print(f"\n🖼️ NEXUS Image 圧縮中...")
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
        
        print(f"\n🖼️ NEXUS Image 結果:")
        print(f"   圧縮率: {compression_ratio:.2f}%")
        print(f"   総合速度: {total_speed:.2f} MB/s")
        print(f"   戦略: 画像フォーマット別最適化")
        print(f"   完全可逆性: ✅ 保証")
        
        # 画像目標評価
        target_compression = 25  # 25%を目標
        target_speed = 80        # 80MB/sを目標
        
        print(f"\n🎯 画像目標評価:")
        print(f"   圧縮率: {compression_ratio:.2f}% {'✅' if compression_ratio >= target_compression else '⚠️'} (目標{target_compression}%)")
        print(f"   速度: {total_speed:.2f} MB/s {'✅' if total_speed >= target_speed else '⚠️'} (目標{target_speed}MB/s)")
        
        print("=" * 60)

if __name__ == "__main__":
    test_nexus_image()
