#!/usr/bin/env python3
"""
NXZ v2.0 File Format Handler
次世代NXZファイルフォーマットの処理
"""

import struct
import hashlib
import time
from typing import Dict, Any, Optional, Tuple

from ..engine.spe_core import SPECore
from ..engine.compressor import SuperCompressor
from ..crypto.encrypt import SuperCrypto, NXZipError
from ..utils.constants import FileFormat, CompressionAlgorithm, EncryptionAlgorithm, KDFAlgorithm
from ..utils.progress import ProgressBar


class SuperNXZipFile:
    """NXZ v2.0: 超高速・高圧縮・多重暗号化対応のアーカイブクラス"""
    
    def __init__(self, compression_algo: str = CompressionAlgorithm.AUTO,
                 encryption_algo: str = EncryptionAlgorithm.AES_GCM,
                 kdf_algo: str = KDFAlgorithm.PBKDF2):
        self.spe_core = SPECore()
        self.compressor = SuperCompressor(compression_algo)
        self.crypto = SuperCrypto(encryption_algo, kdf_algo)
        self.compression_algo = compression_algo
        self.encryption_algo = encryption_algo
        self.kdf_algo = kdf_algo
    
    def create_archive(self, data: bytes, password: Optional[str] = None, 
                      compression_level: int = 6, show_progress: bool = False) -> bytes:
        """超高速・高圧縮アーカイブを作成"""
        
        if show_progress:
            print("🚀 NXZip v2.0 超高速圧縮を開始...")
            start_time = time.time()
        
        # 1. 元データの情報
        original_size = len(data)
        original_checksum = hashlib.sha256(data).digest()
        
        if show_progress:
            print(f"📊 元データサイズ: {original_size:,} bytes")
        
        # 2. 高速圧縮（7Zipを超える圧縮率を目指す）
        self.compressor.level = compression_level
        compressed_data, used_algo = self.compressor.compress(data, show_progress)
        compression_ratio = (1 - len(compressed_data) / original_size) * 100 if original_size > 0 else 0
        
        if show_progress:
            print(f"🗜️  圧縮完了: {len(compressed_data):,} bytes ({compression_ratio:.1f}% 削減, {used_algo})")
        
        # 3. SPE変換（構造保持暗号化）
        if show_progress:
            pb = ProgressBar(len(compressed_data), "SPE変換")
        spe_data = self.spe_core.apply_transform(compressed_data)
        if show_progress:
            pb.update(len(compressed_data))
            pb.close()
        
        # 4. 暗号化（オプション）
        if password:
            encrypted_data, crypto_metadata = self.crypto.encrypt(spe_data, password, show_progress)
            final_data = encrypted_data
            is_encrypted = True
            if show_progress:
                print(f"🔒 暗号化完了: {self.encryption_algo}")
        else:
            final_data = spe_data
            crypto_metadata = b''
            is_encrypted = False
        
        # 5. ヘッダー作成
        header = self._create_header(
            original_size=original_size,
            compressed_size=len(compressed_data),
            encrypted_size=len(final_data),
            compression_algo=used_algo,
            encryption_algo=self.encryption_algo if is_encrypted else None,
            kdf_algo=self.kdf_algo if is_encrypted else None,
            checksum=original_checksum,
            crypto_metadata_size=len(crypto_metadata)
        )
        
        # 6. 最終アーカイブ構成
        archive = header + crypto_metadata + final_data
        
        if show_progress:
            end_time = time.time()
            total_ratio = (1 - len(archive) / original_size) * 100 if original_size > 0 else 0
            print(f"✅ アーカイブ作成完了!")
            print(f"📈 最終圧縮率: {total_ratio:.1f}% ({original_size:,} → {len(archive):,} bytes)")
            print(f"⚡ 処理時間: {end_time - start_time:.2f}秒")
            print(f"🚀 処理速度: {original_size / (end_time - start_time) / 1024 / 1024:.1f} MB/秒")
        
        return archive
    
    def extract_archive(self, archive_data: bytes, password: Optional[str] = None,
                       show_progress: bool = False) -> bytes:
        """超高速アーカイブ展開"""
        
        if show_progress:
            print("🔓 NXZip v2.0 超高速展開を開始...")
            start_time = time.time()
        
        # 1. ヘッダー解析
        if len(archive_data) < FileFormat.HEADER_SIZE_V2:
            raise NXZipError("不正なアーカイブ: ヘッダーが短すぎます")
        
        header_info = self._parse_header(archive_data[:FileFormat.HEADER_SIZE_V2])
        
        if show_progress:
            print(f"📊 アーカイブ情報:")
            print(f"   原サイズ: {header_info['original_size']:,} bytes")
            print(f"   圧縮: {header_info['compression_algo']}")
            print(f"   暗号化: {header_info['encryption_algo'] or '無し'}")
        
        # 2. データ部分を取得
        data_start = FileFormat.HEADER_SIZE_V2 + header_info['crypto_metadata_size']
        crypto_metadata = archive_data[FileFormat.HEADER_SIZE_V2:data_start]
        encrypted_data = archive_data[data_start:]
        
        # 3. 復号化（必要な場合）
        if header_info['encryption_algo']:
            if not password:
                raise NXZipError("パスワードが必要です")
            
            # 暗号化設定を復元
            self.crypto.algorithm = header_info['encryption_algo']
            self.crypto.kdf = header_info['kdf_algo']
            
            spe_data = self.crypto.decrypt(encrypted_data, crypto_metadata, password, show_progress)
            if show_progress:
                print(f"🔓 復号化完了: {header_info['encryption_algo']}")
        else:
            spe_data = encrypted_data
        
        # 4. SPE逆変換
        if show_progress:
            pb = ProgressBar(len(spe_data), "SPE逆変換")
        compressed_data = self.spe_core.reverse_transform(spe_data)
        if show_progress:
            pb.update(len(spe_data))
            pb.close()
        
        # 5. 展開
        original_data = self.compressor.decompress(
            compressed_data, header_info['compression_algo'], show_progress
        )
        
        # 6. 整合性検証
        calculated_checksum = hashlib.sha256(original_data).digest()
        if calculated_checksum != header_info['checksum']:
            raise NXZipError("チェックサム不一致: データが破損している可能性があります")
        
        if show_progress:
            end_time = time.time()
            print(f"✅ 展開完了!")
            print(f"📈 展開サイズ: {len(original_data):,} bytes")
            print(f"⚡ 処理時間: {end_time - start_time:.2f}秒")
            print(f"🚀 展開速度: {len(original_data) / (end_time - start_time) / 1024 / 1024:.1f} MB/秒")
            print(f"✅ 整合性: 正常")
        
        return original_data
    
    def get_info(self, archive_data: bytes) -> Dict[str, Any]:
        """アーカイブ情報を取得"""
        if len(archive_data) < FileFormat.HEADER_SIZE_V2:
            raise NXZipError("不正なアーカイブ")
        
        header_info = self._parse_header(archive_data[:FileFormat.HEADER_SIZE_V2])
        
        compression_ratio = (1 - header_info['compressed_size'] / header_info['original_size']) * 100 \
                          if header_info['original_size'] > 0 else 0
        
        total_ratio = (1 - len(archive_data) / header_info['original_size']) * 100 \
                     if header_info['original_size'] > 0 else 0
        
        return {
            'version': 'NXZ v2.0',
            'original_size': header_info['original_size'],
            'compressed_size': header_info['compressed_size'],
            'archive_size': len(archive_data),
            'compression_algorithm': header_info['compression_algo'],
            'encryption_algorithm': header_info['encryption_algo'],
            'kdf_algorithm': header_info['kdf_algo'],
            'compression_ratio': compression_ratio,
            'total_compression_ratio': total_ratio,
            'is_encrypted': header_info['encryption_algo'] is not None,
            'checksum': header_info['checksum'].hex(),
        }
    
    def _create_header(self, original_size: int, compressed_size: int, encrypted_size: int,
                      compression_algo: str, encryption_algo: Optional[str], 
                      kdf_algo: Optional[str], checksum: bytes, 
                      crypto_metadata_size: int) -> bytes:
        """ヘッダーを作成"""
        header = bytearray(FileFormat.HEADER_SIZE_V2)
        
        # マジックナンバー (4 bytes)
        header[0:4] = FileFormat.MAGIC_V2
        
        # サイズ情報 (24 bytes)
        struct.pack_into('<QQQ', header, 4, original_size, compressed_size, encrypted_size)
        
        # アルゴリズム情報 (72 bytes: 各24バイト)
        header[28:52] = compression_algo.encode('utf-8').ljust(24, b'\x00')[:24]
        header[52:76] = (encryption_algo or '').encode('utf-8').ljust(24, b'\x00')[:24]
        header[76:100] = (kdf_algo or '').encode('utf-8').ljust(24, b'\x00')[:24]
        
        # 暗号化メタデータサイズ (4 bytes)
        struct.pack_into('<I', header, 100, crypto_metadata_size)
        
        # チェックサム (32 bytes)
        header[104:136] = checksum
        
        # 予約領域 (24 bytes) - 将来の拡張用
        header[136:160] = b'\x00' * 24
        
        return bytes(header)
    
    def _parse_header(self, header: bytes) -> Dict[str, Any]:
        """ヘッダーを解析"""
        if len(header) != FileFormat.HEADER_SIZE_V2:
            raise NXZipError("不正なヘッダーサイズ")
        
        # マジックナンバー確認
        if header[0:4] != FileFormat.MAGIC_V2:
            raise NXZipError("不正なマジックナンバー")
        
        # サイズ情報
        original_size, compressed_size, encrypted_size = struct.unpack('<QQQ', header[4:28])
        
        # アルゴリズム情報
        compression_algo = header[28:52].rstrip(b'\x00').decode('utf-8')
        encryption_algo = header[52:76].rstrip(b'\x00').decode('utf-8') or None
        kdf_algo = header[76:100].rstrip(b'\x00').decode('utf-8') or None
        
        # 暗号化メタデータサイズ
        crypto_metadata_size = struct.unpack('<I', header[100:104])[0]
        
        # チェックサム
        checksum = header[104:136]
        
        return {
            'original_size': original_size,
            'compressed_size': compressed_size,
            'encrypted_size': encrypted_size,
            'compression_algo': compression_algo,
            'encryption_algo': encryption_algo,
            'kdf_algo': kdf_algo,
            'crypto_metadata_size': crypto_metadata_size,
            'checksum': checksum,
        }
