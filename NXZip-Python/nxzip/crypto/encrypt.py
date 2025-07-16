#!/usr/bin/env python3
"""
NXZip Encryption System
多重暗号化対応の高速暗号化システム
"""

import secrets
import hashlib
from typing import Tuple
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

from ..utils.constants import EncryptionAlgorithm, KDFAlgorithm, SecurityConstants
from ..utils.progress import ProgressBar


class NXZipError(Exception):
    """NXZip関連のエラー"""
    pass


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
                n=SecurityConstants.SCRYPT_N,
                r=SecurityConstants.SCRYPT_R,
                p=SecurityConstants.SCRYPT_P,
                backend=default_backend()
            )
        else:
            # PBKDF2: 標準的なセキュリティ
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=key_length,
                salt=salt,
                iterations=SecurityConstants.PBKDF2_ITERATIONS,
                backend=default_backend()
            )
        
        return kdf.derive(password_bytes)
    
    def _encrypt_aes_gcm(self, data: bytes, password: str, show_progress: bool) -> Tuple[bytes, bytes]:
        """AES-256-GCM暗号化"""
        # ソルトとnonceを生成
        salt = secrets.token_bytes(SecurityConstants.SALT_SIZE)
        nonce = secrets.token_bytes(SecurityConstants.NONCE_SIZE_AES)
        
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
        salt = secrets.token_bytes(SecurityConstants.SALT_SIZE)
        
        # 鍵導出
        key = self._derive_key(password, salt, 32)
        
        # ChaCha20Poly1305暗号化
        cipher = ChaCha20Poly1305(key)
        nonce = secrets.token_bytes(SecurityConstants.NONCE_SIZE_XCHACHA)
        
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
