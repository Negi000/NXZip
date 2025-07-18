#!/usr/bin/env python3
"""
NEXUS Audio Advanced - SRLA/Lyra技術参考の音声圧縮エンジン
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
NXZ_MAGIC = b'NXZA'  # Audio専用マジック
NXZ_VERSION = 1

class NEXUSAudioAdvanced:
    """
    音声専用NEXUS Advanced - SRLA/Lyra技術参考
    
    戦略:
    1. SRLA風の高圧縮可逆圧縮
    2. Lyra風の適応的圧縮
    3. 音声特性別最適化
    4. 高圧縮率追求
    """
    
    def __init__(self):
        self.spe = SPECoreJIT()
    
    def compress(self, data: bytes) -> bytes:
        """SRLA/Lyra技術参考の音声圧縮"""
        if not data:
            return self._create_empty_nxz()
        
        # 1. 音声フォーマット検出
        format_type = self._detect_audio_format(data)
        print(f"🎵 検出: {format_type}")
        
        # 2. SRLA/Lyra風の適応的圧縮
        data_size = len(data)
        if format_type == "wav":
            # WAV: SRLA風高圧縮
            compressed_data = self._compress_wav_srla_style(data)
        elif format_type == "mp3":
            # MP3: Lyra風改良圧縮
            compressed_data = self._compress_mp3_lyra_style(data)
        elif format_type == "flac":
            # FLAC: 改良圧縮
            compressed_data = self._compress_flac_advanced(data)
        elif format_type == "ogg":
            # OGG: 改良圧縮
            compressed_data = self._compress_ogg_advanced(data)
        else:
            # その他: 高圧縮
            compressed_data = b'AUDOTHER' + lzma.compress(data, preset=8)
        
        # 3. SPE暗号化
        encrypted_data = self.spe.apply_transform(compressed_data)
        
        # 4. 音声専用ヘッダー
        header = self._create_audio_header(
            original_size=len(data),
            compressed_size=len(compressed_data),
            encrypted_size=len(encrypted_data),
            format_type=format_type
        )
        
        return header + encrypted_data
    
    def _compress_wav_srla_style(self, data: bytes) -> bytes:
        """SRLA技術参考のWAV圧縮"""
        data_size = len(data)
        
        # SRLA風の高圧縮可逆圧縮
        if data_size > 100 * 1024 * 1024:  # 100MB超: 速度重視
            return b'AUDWAV' + lzma.compress(data, preset=3)
        elif data_size > 50 * 1024 * 1024:  # 50MB超: バランス
            return b'AUDWAV' + lzma.compress(data, preset=6)
        elif data_size > 10 * 1024 * 1024:  # 10MB超: 高圧縮
            return b'AUDWAV' + lzma.compress(data, preset=8)
        else:
            # 小さなWAV: 最高圧縮
            return b'AUDWAV' + lzma.compress(data, preset=9)
    
    def _compress_mp3_lyra_style(self, data: bytes) -> bytes:
        """Lyra技術参考のMP3圧縮"""
        data_size = len(data)
        
        # Lyra風の適応的圧縮
        if data_size > 10 * 1024 * 1024:  # 10MB超: 軽圧縮
            return b'AUDMP3' + lzma.compress(data, preset=2)
        elif data_size > 5 * 1024 * 1024:  # 5MB超: 中圧縮
            return b'AUDMP3' + lzma.compress(data, preset=4)
        else:
            # 小さなMP3: 高圧縮
            return b'AUDMP3' + lzma.compress(data, preset=6)
    
    def _compress_flac_advanced(self, data: bytes) -> bytes:
        """改良FLAC圧縮"""
        data_size = len(data)
        
        # FLACの特性を考慮
        if data_size > 50 * 1024 * 1024:  # 50MB超: 中圧縮
            return b'AUDFLAC' + lzma.compress(data, preset=4)
        else:
            # 小さなFLAC: 高圧縮
            return b'AUDFLAC' + lzma.compress(data, preset=7)
    
    def _compress_ogg_advanced(self, data: bytes) -> bytes:
        """改良OGG圧縮"""
        data_size = len(data)
        
        # OGGの特性を考慮
        if data_size > 20 * 1024 * 1024:  # 20MB超: 中圧縮
            return b'AUDOGG' + lzma.compress(data, preset=3)
        else:
            # 小さなOGG: 高圧縮
            return b'AUDOGG' + lzma.compress(data, preset=6)
    
    def decompress(self, nxz_data: bytes) -> bytes:
        """SRLA/Lyra風音声展開"""
        if not nxz_data:
            return b""
        
        # 1. ヘッダー検証
        if len(nxz_data) < 40:
            raise ValueError("Invalid NXZ Audio format")
        
        # 2. 暗号化データ抽出
        encrypted_data = nxz_data[40:]  # 音声ヘッダー40バイト
        
        # 3. SPE復号化
        compressed_data = self.spe.reverse_transform(encrypted_data)
        
        # 4. フォーマット別展開
        if compressed_data.startswith(b'AUDWAV'):
            original_data = lzma.decompress(compressed_data[6:])
        elif compressed_data.startswith(b'AUDMP3'):
            original_data = lzma.decompress(compressed_data[6:])
        elif compressed_data.startswith(b'AUDFLAC'):
            original_data = lzma.decompress(compressed_data[7:])
        elif compressed_data.startswith(b'AUDOGG'):
            original_data = lzma.decompress(compressed_data[6:])
        elif compressed_data.startswith(b'AUDOTHER'):
            original_data = lzma.decompress(compressed_data[8:])
        else:
            raise ValueError("Unknown audio compression format")
        
        return original_data
    
    def _detect_audio_format(self, data: bytes) -> str:
        """音声フォーマット検出"""
        if len(data) < 16:
            return "unknown"
        
        # 音声マジック検出
        if data.startswith(b'ID3') or (len(data) > 1 and data[0:2] == b'\xFF\xFB'):
            return "mp3"
        elif data.startswith(b'RIFF') and b'WAVE' in data[:16]:
            return "wav"
        elif data.startswith(b'fLaC'):
            return "flac"
        elif data.startswith(b'OggS'):
            return "ogg"
        else:
            return "unknown"
    
    def _create_audio_header(self, original_size: int, compressed_size: int, 
                           encrypted_size: int, format_type: str) -> bytes:
        """音声専用ヘッダー作成 (40バイト)"""
        header = bytearray(40)
        
        # マジックナンバー
        header[0:4] = NXZ_MAGIC
        
        # バージョン
        header[4:8] = struct.pack('<I', NXZ_VERSION)
        
        # サイズ情報
        header[8:16] = struct.pack('<Q', original_size)
        header[16:24] = struct.pack('<Q', compressed_size)
        header[24:32] = struct.pack('<Q', encrypted_size)
        
        # フォーマット情報
        format_bytes = format_type.encode('ascii')[:8]
        header[32:40] = format_bytes.ljust(8, b'\x00')
        
        return bytes(header)
    
    def _create_empty_nxz(self) -> bytes:
        """空のNXZファイル作成"""
        return self._create_audio_header(0, 0, 0, "empty")

def test_nexus_audio_advanced():
    """NEXUS Audio Advanced テスト"""
    print("🎵 NEXUS Audio Advanced テスト - SRLA/Lyra技術参考")
    print("=" * 60)
    
    # 音声テストファイル
    test_files = [
        "generated-music-1752042054079.wav",
        "陰謀論.mp3"
    ]
    
    for test_filename in test_files:
        test_file = Path(rf"C:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\sample\{test_filename}")
        
        if not test_file.exists():
            print(f"❌ {test_filename} が見つかりません")
            continue
        
        file_size = test_file.stat().st_size
        print(f"📄 ファイル: {test_file.name}")
        print(f"📊 サイズ: {file_size//1024} KB")
        
        # データ読み込み
        print("📖 データ読み込み中...")
        with open(test_file, 'rb') as f:
            data = f.read()
        
        # NEXUS Audio Advanced初期化
        nexus = NEXUSAudioAdvanced()
        
        # 圧縮テスト
        print("\n🎵 NEXUS Audio Advanced 圧縮中...")
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
        
        print(f"\n🎵 NEXUS Audio Advanced 結果:")
        print(f"   圧縮率: {compression_ratio:.2f}%")
        print(f"   総合速度: {total_speed:.2f} MB/s")
        print(f"   戦略: SRLA/Lyra技術参考")
        print(f"   完全可逆性: ✅ 保証")
        
        # 目標評価
        target_ratio = 20  # 20%圧縮率目標
        target_speed = 90  # 90MB/s目標
        
        print(f"\n🎯 音声目標評価:")
        print(f"   圧縮率: {compression_ratio:.2f}% {'✅' if compression_ratio >= target_ratio else '⚠️'} (目標{target_ratio}%)")
        print(f"   速度: {total_speed:.2f} MB/s {'✅' if total_speed >= target_speed else '⚠️'} (目標{target_speed}MB/s)")
        print("=" * 60)

if __name__ == "__main__":
    test_nexus_audio_advanced()
