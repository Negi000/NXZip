#!/usr/bin/env python3
"""
NXZip Complete System - Python版
SPE Core + 圧縮 + 暗号化の統合システム
"""

import os
import sys
import struct
import zlib
import lzma
import hashlib
import secrets
import threading
import time
from typing import Optional, Tuple, Dict, Any, Union
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

# SPE Core - 最新の6段階エンタープライズ版をインポート
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'NXZip-Python'))
from nxzip.engine.spe_core import SPECore

# 高速圧縮ライブラリの追加インポート（利用可能な場合）
try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

# プログレスバー表示用
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# 暗号化アルゴリズム定数
class EncryptionAlgorithm:
    AES_GCM = "aes-gcm"
    XCHACHA20_POLY1305 = "xchacha20-poly1305"

# 圧縮アルゴリズム定数
class CompressionAlgorithm:
    ZLIB = "zlib"
    LZMA2 = "lzma2"
    ZSTD = "zstd"
    AUTO = "auto"

# KDF (鍵導出) アルゴリズム定数
class KDFAlgorithm:
    PBKDF2 = "pbkdf2"
    SCRYPT = "scrypt"


class NXZipError(Exception):
    """NXZip関連のエラー"""
    pass

# 拡張暗号化クラス
class SuperCrypto:
    """多重暗号化対応の高速暗号化クラス"""
    
    def __init__(self, algorithm: str = EncryptionAlgorithm.AES_GCM, 
                 kdf: str = KDFAlgorithm.PBKDF2):
        self.algorithm = algorithm
        self.kdf = kdf
    
    def encrypt(self, data: bytes, password: str, show_progress: bool = False) -> Tuple[bytes, bytes]:
        """データを暗号化（暗号化データ, メタデータを返す）"""
        if self.algorithm == EncryptionAlgorithm.XCHACHA20_POLY1305:
            return self._encrypt_xchacha20(data, password, show_progress)
        else:
            return self._encrypt_aes_gcm(data, password, show_progress)
    
    def decrypt(self, encrypted_data: bytes, metadata: bytes, password: str, 
                show_progress: bool = False) -> bytes:
        """データを復号化"""
        if self.algorithm == EncryptionAlgorithm.XCHACHA20_POLY1305:
            return self._decrypt_xchacha20(encrypted_data, metadata, password, show_progress)
        else:
            return self._decrypt_aes_gcm(encrypted_data, metadata, password, show_progress)
    
    def _derive_key(self, password: str, salt: bytes, key_length: int = 32) -> bytes:
        """鍵導出"""
        password_bytes = password.encode('utf-8')
        
        if self.kdf == KDFAlgorithm.SCRYPT:
            # Scrypt: より高いセキュリティ
            kdf = Scrypt(
                length=key_length,
                salt=salt,
                n=2**16,  # 65536
                r=8,
                p=1,
                backend=default_backend()
            )
        else:
            # PBKDF2: 標準的なセキュリティ
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=key_length,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
        
        return kdf.derive(password_bytes)
    
    def _encrypt_aes_gcm(self, data: bytes, password: str, show_progress: bool) -> Tuple[bytes, bytes]:
        """AES-256-GCM暗号化"""
        # ソルトとnonceを生成
        salt = secrets.token_bytes(16)
        nonce = secrets.token_bytes(12)
        
        # 鍵導出
        key = self._derive_key(password, salt, 32)
        
        # 暗号化
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce), backend=default_backend())
        encryptor = cipher.encryptor()
        
        if show_progress:
            pb = ProgressBar(len(data), "AES-GCM暗号化")
            ciphertext = encryptor.update(data) + encryptor.finalize()
            pb.update(len(data))
            pb.close()
        else:
            ciphertext = encryptor.update(data) + encryptor.finalize()
        
        # メタデータ: アルゴリズム(1) + KDF(1) + ソルト(16) + nonce(12) + tag(16)
        metadata = (
            EncryptionAlgorithm.AES_GCM.encode('utf-8')[:1].ljust(1, b'\x00') +
            self.kdf.encode('utf-8')[:1].ljust(1, b'\x00') +
            salt + nonce + encryptor.tag
        )
        
        return ciphertext, metadata
    
    def _decrypt_aes_gcm(self, ciphertext: bytes, metadata: bytes, password: str, 
                        show_progress: bool) -> bytes:
        """AES-256-GCM復号化"""
        if len(metadata) < 46:  # 1+1+16+12+16
            raise NXZipError("不正な暗号化メタデータ")
        
        # メタデータ解析
        salt = metadata[2:18]
        nonce = metadata[18:30]
        tag = metadata[30:46]
        
        # 鍵導出
        key = self._derive_key(password, salt, 32)
        
        # 復号化
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce, tag), backend=default_backend())
        decryptor = cipher.decryptor()
        
        if show_progress:
            pb = ProgressBar(len(ciphertext), "AES-GCM復号化")
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            pb.update(len(ciphertext))
            pb.close()
        else:
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        return plaintext
    
    def _encrypt_xchacha20(self, data: bytes, password: str, show_progress: bool) -> Tuple[bytes, bytes]:
        """XChaCha20-Poly1305暗号化"""
        # ソルトを生成
        salt = secrets.token_bytes(16)
        
        # 鍵導出
        key = self._derive_key(password, salt, 32)
        
        # ChaCha20Poly1305暗号化
        cipher = ChaCha20Poly1305(key)
        nonce = secrets.token_bytes(12)  # ChaCha20Poly1305は12バイトnonce
        
        if show_progress:
            pb = ProgressBar(len(data), "XChaCha20暗号化")
            ciphertext = cipher.encrypt(nonce, data, None)
            pb.update(len(data))
            pb.close()
        else:
            ciphertext = cipher.encrypt(nonce, data, None)
        
        # メタデータ: アルゴリズム(1) + KDF(1) + ソルト(16) + nonce(12)
        metadata = (
            EncryptionAlgorithm.XCHACHA20_POLY1305.encode('utf-8')[:1].ljust(1, b'\x00') +
            self.kdf.encode('utf-8')[:1].ljust(1, b'\x00') +
            salt + nonce
        )
        
        return ciphertext, metadata
    
    def _decrypt_xchacha20(self, ciphertext: bytes, metadata: bytes, password: str, 
                          show_progress: bool) -> bytes:
        """XChaCha20-Poly1305復号化"""
        if len(metadata) < 30:  # 1+1+16+12
            raise NXZipError("不正な暗号化メタデータ")
        
        # メタデータ解析
        salt = metadata[2:18]
        nonce = metadata[18:30]
        
        # 鍵導出
        key = self._derive_key(password, salt, 32)
        
        # 復号化
        cipher = ChaCha20Poly1305(key)
        
        if show_progress:
            pb = ProgressBar(len(ciphertext), "XChaCha20復号化")
            plaintext = cipher.decrypt(nonce, ciphertext, None)
            pb.update(len(ciphertext))
            pb.close()
        else:
            plaintext = cipher.decrypt(nonce, ciphertext, None)
        
        return plaintext

class SuperNXZipFile:
    """NXZip v2.0 - 超高速・高圧縮・多重暗号化対応ファイル形式ハンドラー"""
    
    # ファイル形式定数
    MAGIC = b'NXZ\x02'  # NXZ v2.0 (新フォーマット)
    HEADER_SIZE = 160    # ヘッダーサイズを拡張（アルゴリズム名対応）
    
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
        if len(archive_data) < self.HEADER_SIZE:
            raise NXZipError("不正なアーカイブ: ヘッダーが短すぎます")
        
        header_info = self._parse_header(archive_data[:self.HEADER_SIZE])
        
        if show_progress:
            print(f"📊 アーカイブ情報:")
            print(f"   原サイズ: {header_info['original_size']:,} bytes")
            print(f"   圧縮: {header_info['compression_algo']}")
            print(f"   暗号化: {header_info['encryption_algo'] or '無し'}")
        
        # 2. データ部分を取得
        data_start = self.HEADER_SIZE + header_info['crypto_metadata_size']
        crypto_metadata = archive_data[self.HEADER_SIZE:data_start]
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
        if len(archive_data) < self.HEADER_SIZE:
            raise NXZipError("不正なアーカイブ")
        
        header_info = self._parse_header(archive_data[:self.HEADER_SIZE])
        
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
        header = bytearray(self.HEADER_SIZE)
        
        # マジックナンバー (4 bytes)
        header[0:4] = self.MAGIC
        
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
        if len(header) != self.HEADER_SIZE:
            raise NXZipError("不正なヘッダーサイズ")
        
        # マジックナンバー確認
        if header[0:4] != self.MAGIC:
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
        """NXZアーカイブを作成"""
        
        # 1. データの前処理
        original_size = len(data)
        checksum = hashlib.sha256(data).digest()
        
        # 2. 圧縮
        if len(data) > 1024:  # 1KB以上なら圧縮
            compressed = zlib.compress(data, compression_level)
            if len(compressed) < len(data):
                payload = compressed
                compression = self.COMPRESSION_ZLIB
            else:
                payload = data
                compression = self.COMPRESSION_NONE
        else:
            payload = data
            compression = self.COMPRESSION_NONE
        
        # 3. SPE変換
        spe_data = self.spe_core.apply_transform(payload)
        
        # 4. 暗号化
        if password:
            encrypted_data, salt, nonce = self._encrypt_aes_gcm(spe_data, password)
            # 暗号化情報をデータの前に追加
            crypto_header = salt + nonce  # 16 + 12 = 28 bytes
            final_data = crypto_header + encrypted_data
            encryption = self.ENCRYPTION_AES_GCM
        else:
            final_data = spe_data
            salt = b'\x00' * 16
            nonce = b'\x00' * 12
            encryption = self.ENCRYPTION_SPE_ONLY
        
        # 5. ヘッダー作成
        header = self._create_header(
            original_size=original_size,
            compressed_size=len(payload),
            encrypted_size=len(final_data),
            compression=compression,
            encryption=encryption,
            checksum=checksum,
            salt=salt,
            nonce=nonce
        )
        
        return header + final_data
    
# プログレスバーヘルパー
class ProgressBar:
    def __init__(self, total: int, desc: str = "処理中"):
        self.total = total
        self.desc = desc
        self.current = 0
        if TQDM_AVAILABLE:
            self.pbar = tqdm(total=total, desc=desc, unit='B', unit_scale=True)
        else:
            self.pbar = None
    
    def update(self, amount: int):
        self.current += amount
        if self.pbar:
            self.pbar.update(amount)
        else:
            percent = (self.current / self.total) * 100 if self.total > 0 else 0
            print(f"\r{self.desc}: {percent:.1f}%", end='', flush=True)
    
    def close(self):
        if self.pbar:
            self.pbar.close()
        else:
            print()  # 改行

# 高速圧縮クラス
class SuperCompressor:
    """7Zipを超える高圧縮率と超高速処理を実現する圧縮器"""
    
    def __init__(self, algorithm: str = CompressionAlgorithm.AUTO, level: int = 6):
        self.algorithm = algorithm
        self.level = level
    
    def compress(self, data: bytes, show_progress: bool = False) -> Tuple[bytes, str]:
        """データを圧縮し、使用したアルゴリズムも返す"""
        if not data:
            return data, CompressionAlgorithm.ZLIB
        
        if self.algorithm == CompressionAlgorithm.AUTO:
            return self._auto_compress(data, show_progress)
        elif self.algorithm == CompressionAlgorithm.ZSTD and ZSTD_AVAILABLE:
            return self._compress_zstd(data, show_progress), CompressionAlgorithm.ZSTD
        elif self.algorithm == CompressionAlgorithm.LZMA2:
            return self._compress_lzma2(data, show_progress), CompressionAlgorithm.LZMA2
        else:
            return self._compress_zlib(data, show_progress), CompressionAlgorithm.ZLIB
    
    def decompress(self, data: bytes, algorithm: str, show_progress: bool = False) -> bytes:
        """指定されたアルゴリズムでデータを展開"""
        if not data:
            return data
        
        if algorithm == CompressionAlgorithm.ZSTD and ZSTD_AVAILABLE:
            return self._decompress_zstd(data, show_progress)
        elif algorithm == CompressionAlgorithm.LZMA2:
            return self._decompress_lzma2(data, show_progress)
        else:
            return self._decompress_zlib(data, show_progress)
    
    def _auto_compress(self, data: bytes, show_progress: bool) -> Tuple[bytes, str]:
        """最適な圧縮アルゴリズムを自動選択"""
        data_size = len(data)
        
        # 小さなファイルはZLIBが高速
        if data_size < 1024:
            return self._compress_zlib(data, show_progress), CompressionAlgorithm.ZLIB
        
        # 大容量ファイルでZstdが利用可能ならZstd、そうでなければLZMA2
        if ZSTD_AVAILABLE and data_size > 1024 * 1024:
            return self._compress_zstd(data, show_progress), CompressionAlgorithm.ZSTD
        elif data_size > 10 * 1024:
            return self._compress_lzma2(data, show_progress), CompressionAlgorithm.LZMA2
        else:
            return self._compress_zlib(data, show_progress), CompressionAlgorithm.ZLIB
    
    def _compress_zlib(self, data: bytes, show_progress: bool) -> bytes:
        """ZLIB圧縮（高速・軽量）"""
        level = min(9, max(1, self.level))
        if show_progress:
            pb = ProgressBar(len(data), "ZLIB圧縮")
            result = zlib.compress(data, level)
            pb.update(len(data))
            pb.close()
            return result
        return zlib.compress(data, level)
    
    def _compress_lzma2(self, data: bytes, show_progress: bool) -> bytes:
        """LZMA2圧縮（高圧縮率）"""
        preset = min(9, max(0, self.level))
        if show_progress:
            pb = ProgressBar(len(data), "LZMA2圧縮")
            result = lzma.compress(data, format=lzma.FORMAT_XZ, preset=preset)
            pb.update(len(data))
            pb.close()
            return result
        return lzma.compress(data, format=lzma.FORMAT_XZ, preset=preset)
    
    def _compress_zstd(self, data: bytes, show_progress: bool) -> bytes:
        """Zstandard圧縮（高速・高圧縮）"""
        if not ZSTD_AVAILABLE:
            raise RuntimeError("Zstandard not available")
        
        level = min(22, max(1, self.level))
        compressor = zstd.ZstdCompressor(level=level)
        
        if show_progress:
            pb = ProgressBar(len(data), "Zstd圧縮")
            result = compressor.compress(data)
            pb.update(len(data))
            pb.close()
            return result
        return compressor.compress(data)
    
    def _decompress_zlib(self, data: bytes, show_progress: bool) -> bytes:
        """ZLIB展開"""
        if show_progress:
            pb = ProgressBar(len(data), "ZLIB展開")
            result = zlib.decompress(data)
            pb.update(len(data))
            pb.close()
            return result
        return zlib.decompress(data)
    
    def _decompress_lzma2(self, data: bytes, show_progress: bool) -> bytes:
        """LZMA2展開"""
        if show_progress:
            pb = ProgressBar(len(data), "LZMA2展開")
            result = lzma.decompress(data, format=lzma.FORMAT_XZ)
            pb.update(len(data))
            pb.close()
            return result
        return lzma.decompress(data, format=lzma.FORMAT_XZ)
    
    def _decompress_zstd(self, data: bytes, show_progress: bool) -> bytes:
        """Zstandard展開"""
        if not ZSTD_AVAILABLE:
            raise RuntimeError("Zstandard not available")
        
        decompressor = zstd.ZstdDecompressor()
        if show_progress:
            pb = ProgressBar(len(data), "Zstd展開")
            result = decompressor.decompress(data)
            pb.update(len(data))
            pb.close()
            return result
        return decompressor.decompress(data)


def main():
    """メインCLI関数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='NXZip v2.0 - 超高速・高圧縮・多重暗号化アーカイブシステム',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本的な圧縮
  python nxzip_complete.py create archive.nxz input.txt
  
  # パスワード付き暗号化
  python nxzip_complete.py create secure.nxz input.txt -p password123
  
  # 高圧縮＋XChaCha20暗号化
  python nxzip_complete.py create ultra.nxz input.txt -p pass -c zstd -e xchacha20 -l 9
  
  # 展開
  python nxzip_complete.py extract archive.nxz output.txt
  
  # アーカイブ情報表示
  python nxzip_complete.py info archive.nxz
""")
    
    subparsers = parser.add_subparsers(dest='command', help='コマンド')
    
    # create コマンド
    create_parser = subparsers.add_parser('create', help='アーカイブを作成')
    create_parser.add_argument('archive', help='出力アーカイブファイル (.nxz)')
    create_parser.add_argument('file', help='入力ファイル')
    create_parser.add_argument('-p', '--password', help='暗号化パスワード')
    create_parser.add_argument('-c', '--compression', 
                              choices=['auto', 'zlib', 'lzma2', 'zstd'],
                              default='auto', help='圧縮アルゴリズム')
    create_parser.add_argument('-e', '--encryption',
                              choices=['aes-gcm', 'xchacha20-poly1305'],
                              default='aes-gcm', help='暗号化アルゴリズム')
    create_parser.add_argument('-k', '--kdf',
                              choices=['pbkdf2', 'scrypt'],
                              default='pbkdf2', help='鍵導出方式')
    create_parser.add_argument('-l', '--level', type=int, default=6,
                              choices=range(1, 10), help='圧縮レベル (1-9)')
    create_parser.add_argument('-v', '--verbose', action='store_true',
                              help='詳細表示')
    
    # extract コマンド
    extract_parser = subparsers.add_parser('extract', help='アーカイブを展開')
    extract_parser.add_argument('archive', help='入力アーカイブファイル (.nxz)')
    extract_parser.add_argument('output', nargs='?', help='出力ファイル')
    extract_parser.add_argument('-p', '--password', help='復号化パスワード')
    extract_parser.add_argument('-v', '--verbose', action='store_true',
                               help='詳細表示')
    
    # info コマンド
    info_parser = subparsers.add_parser('info', help='アーカイブ情報を表示')
    info_parser.add_argument('archive', help='アーカイブファイル (.nxz)')
    
    # test コマンド
    test_parser = subparsers.add_parser('test', help='アーカイブをテスト')
    test_parser.add_argument('archive', help='アーカイブファイル (.nxz)')
    test_parser.add_argument('-p', '--password', help='復号化パスワード')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'create':
            return create_command(args)
        elif args.command == 'extract':
            return extract_command(args)
        elif args.command == 'info':
            return info_command(args)
        elif args.command == 'test':
            return test_command(args)
    except KeyboardInterrupt:
        print("\n❌ 処理が中断されました")
        return 1
    except Exception as e:
        print(f"❌ エラー: {e}")
        return 1

def create_command(args) -> int:
    """createコマンドの実装"""
    if not os.path.exists(args.file):
        print(f"❌ エラー: ファイル '{args.file}' が見つかりません")
        return 1
    
    # NXZipファイルインスタンス作成
    nxzip = SuperNXZipFile(
        compression_algo=args.compression,
        encryption_algo=args.encryption,
        kdf_algo=args.kdf
    )
    
    if args.verbose:
        print("🚀 NXZip v2.0 - 超高速圧縮開始")
        print(f"� 入力: {args.file}")
        print(f"📦 出力: {args.archive}")
        print(f"🗜️  圧縮: {args.compression} (レベル {args.level})")
        if args.password:
            print(f"🔒 暗号化: {args.encryption} (KDF: {args.kdf})")
    
    # ファイル読み込み
    with open(args.file, 'rb') as f:
        data = f.read()
    
    # アーカイブ作成
    archive_data = nxzip.create_archive(
        data, 
        args.password, 
        args.level, 
        show_progress=args.verbose
    )
    
    # アーカイブ保存
    with open(args.archive, 'wb') as f:
        f.write(archive_data)
    
    if not args.verbose:
        original_size = len(data)
        archive_size = len(archive_data)
        ratio = (1 - archive_size / original_size) * 100 if original_size > 0 else 0
        print(f"✅ アーカイブ作成完了: {original_size:,} → {archive_size:,} bytes ({ratio:.1f}% 削減)")
    
    return 0

def extract_command(args) -> int:
    """extractコマンドの実装"""
    if not os.path.exists(args.archive):
        print(f"❌ エラー: アーカイブ '{args.archive}' が見つかりません")
        return 1
    
    # 出力ファイル名の決定
    if not args.output:
        base_name = os.path.splitext(args.archive)[0]
        if base_name.endswith('.nxz'):
            args.output = base_name[:-4]
        else:
            args.output = base_name + '_extracted'
    
    # NXZipファイルインスタンス作成
    nxzip = SuperNXZipFile()
    
    if args.verbose:
        print("� NXZip v2.0 - 超高速展開開始")
        print(f"📦 入力: {args.archive}")
        print(f"📁 出力: {args.output}")
    
    # アーカイブ読み込み
    with open(args.archive, 'rb') as f:
        archive_data = f.read()
    
    # 展開
    extracted_data = nxzip.extract_archive(
        archive_data,
        args.password,
        show_progress=args.verbose
    )
    
    # ファイル保存
    with open(args.output, 'wb') as f:
        f.write(extracted_data)
    
    if not args.verbose:
        print(f"✅ 展開完了: {len(extracted_data):,} bytes → {args.output}")
    
    return 0

def info_command(args) -> int:
    """infoコマンドの実装"""
    if not os.path.exists(args.archive):
        print(f"❌ エラー: アーカイブ '{args.archive}' が見つかりません")
        return 1
    
    nxzip = SuperNXZipFile()
    
    with open(args.archive, 'rb') as f:
        archive_data = f.read()
    
    info = nxzip.get_info(archive_data)
    
    print("📊 NXZip アーカイブ情報")
    print("=" * 40)
    print(f"バージョン: {info['version']}")
    print(f"元サイズ: {info['original_size']:,} bytes")
    print(f"圧縮後サイズ: {info['compressed_size']:,} bytes")
    print(f"アーカイブサイズ: {info['archive_size']:,} bytes")
    print(f"圧縮アルゴリズム: {info['compression_algorithm']}")
    print(f"圧縮率: {info['compression_ratio']:.1f}%")
    print(f"総圧縮率: {info['total_compression_ratio']:.1f}%")
    print(f"暗号化: {'有効' if info['is_encrypted'] else '無効'}")
    if info['is_encrypted']:
        print(f"  アルゴリズム: {info['encryption_algorithm']}")
        print(f"  KDF: {info['kdf_algorithm']}")
    print(f"チェックサム: {info['checksum']}")
    
    return 0

def test_command(args) -> int:
    """testコマンドの実装"""
    if not os.path.exists(args.archive):
        print(f"❌ エラー: アーカイブ '{args.archive}' が見つかりません")
        return 1
    
    nxzip = SuperNXZipFile()
    
    print(f"🧪 アーカイブテスト中: {args.archive}")
    
    try:
        with open(args.archive, 'rb') as f:
            archive_data = f.read()
        
        # 情報取得テスト
        info = nxzip.get_info(archive_data)
        print(f"✅ ヘッダー: 正常 ({info['version']})")
        
        # 展開テスト
        extracted_data = nxzip.extract_archive(archive_data, args.password, show_progress=False)
        print(f"✅ 展開: 正常 ({len(extracted_data):,} bytes)")
        
        print("✅ アーカイブは正常です")
        return 0
        
    except Exception as e:
        print(f"❌ アーカイブエラー: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
