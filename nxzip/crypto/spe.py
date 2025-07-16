#!/usr/bin/env python3
"""
🔒 SPE (Structure-Preserving Encryption) System

構造保持暗号化システム - NXZip統合暗号化エンジン
Copyright (c) 2025 NXZip Project
"""

import os
import hashlib
import secrets
import struct
from typing import Dict, List, Tuple, Any, Optional
import time

# 暗号化ライブラリ
try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


class SPECrypto:
    """🔒 Structure-Preserving Encryption System"""
    
    def __init__(self):
        self.algorithm = "AES-GCM"
        self.key_length = 32  # 256-bit
        self.iv_length = 16   # 128-bit
        self.tag_length = 16  # 128-bit
        self.salt_length = 32 # 256-bit
        
        # SPE設定
        self.preserve_structure = True
        self.compress_before_encrypt = True
        
    def apply_transform(self, data: bytes, password: str) -> bytes:
        """SPE変換適用 (暗号化)"""
        return self.encrypt(data, password)
    
    def reverse_transform(self, data: bytes, password: str) -> bytes:
        """SPE変換逆適用 (復号化)"""
        return self.decrypt(data, password)
    
    def encrypt(self, data: bytes, password: str) -> bytes:
        """データ暗号化"""
        if not CRYPTO_AVAILABLE:
            return self._simple_encrypt(data, password)
        
        try:
            # 1. ソルト生成
            salt = secrets.token_bytes(self.salt_length)
            
            # 2. キー導出
            key = self._derive_key(password, salt)
            
            # 3. IV生成
            iv = secrets.token_bytes(self.iv_length)
            
            # 4. 構造保持前処理
            if self.preserve_structure:
                data = self._preserve_structure_pre(data)
            
            # 5. AES-GCM暗号化
            cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())
            encryptor = cipher.encryptor()
            
            ciphertext = encryptor.update(data) + encryptor.finalize()
            tag = encryptor.tag
            
            # 6. SPEヘッダー生成
            header = self._create_spe_header(salt, iv, tag)
            
            return header + ciphertext
            
        except Exception as e:
            print(f"暗号化エラー: {e}")
            return self._simple_encrypt(data, password)
    
    def decrypt(self, encrypted_data: bytes, password: str) -> bytes:
        """データ復号化"""
        if not CRYPTO_AVAILABLE:
            return self._simple_decrypt(encrypted_data, password)
        
        try:
            # 1. SPEヘッダー解析
            header_size = 4 + self.salt_length + self.iv_length + self.tag_length + 4
            if len(encrypted_data) < header_size:
                return self._simple_decrypt(encrypted_data, password)
            
            salt, iv, tag, ciphertext = self._parse_spe_data(encrypted_data)
            
            # 2. キー導出
            key = self._derive_key(password, salt)
            
            # 3. AES-GCM復号化
            cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=default_backend())
            decryptor = cipher.decryptor()
            
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            
            # 4. 構造保持後処理
            if self.preserve_structure:
                plaintext = self._preserve_structure_post(plaintext)
            
            return plaintext
            
        except Exception as e:
            print(f"復号化エラー: {e}")
            return self._simple_decrypt(encrypted_data, password)
    
    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """PBKDF2によるキー導出"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=self.key_length,
            salt=salt,
            iterations=100000,  # 100K iterations
            backend=default_backend()
        )
        return kdf.derive(password.encode('utf-8'))
    
    def _create_spe_header(self, salt: bytes, iv: bytes, tag: bytes) -> bytes:
        """SPEヘッダー生成"""
        # SPEマジックナンバー
        magic = b'SPE1'
        
        # バージョン情報
        version = struct.pack('<I', 1)
        
        # データ長
        data_length = struct.pack('<I', len(salt) + len(iv) + len(tag))
        
        return magic + version + salt + iv + tag + data_length
    
    def _parse_spe_data(self, data: bytes) -> Tuple[bytes, bytes, bytes, bytes]:
        """SPEデータ解析"""
        # マジックナンバー確認
        if not data.startswith(b'SPE1'):
            raise ValueError("Invalid SPE format")
        
        offset = 4  # magic
        
        # バージョン
        version = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        
        # ソルト
        salt = data[offset:offset+self.salt_length]
        offset += self.salt_length
        
        # IV
        iv = data[offset:offset+self.iv_length]
        offset += self.iv_length
        
        # タグ
        tag = data[offset:offset+self.tag_length]
        offset += self.tag_length
        
        # データ長
        data_length = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        
        # 暗号化データ
        ciphertext = data[offset:]
        
        return salt, iv, tag, ciphertext
    
    def _preserve_structure_pre(self, data: bytes) -> bytes:
        """構造保持前処理"""
        # データ構造の特徴を保持
        # 現在は基本実装
        return data
    
    def _preserve_structure_post(self, data: bytes) -> bytes:
        """構造保持後処理"""
        # 構造復元
        # 現在は基本実装
        return data
    
    def _simple_encrypt(self, data: bytes, password: str) -> bytes:
        """簡易暗号化 (cryptographyライブラリ未使用時)"""
        # XOR暗号化 (デモ用)
        key = hashlib.sha256(password.encode()).digest()
        
        result = bytearray()
        for i, byte in enumerate(data):
            result.append(byte ^ key[i % len(key)])
        
        # 簡易ヘッダー
        header = b'SPEX' + struct.pack('<I', len(data))
        return header + bytes(result)
    
    def _simple_decrypt(self, encrypted_data: bytes, password: str) -> bytes:
        """簡易復号化"""
        if not encrypted_data.startswith(b'SPEX'):
            return encrypted_data
        
        # ヘッダー解析
        original_length = struct.unpack('<I', encrypted_data[4:8])[0]
        ciphertext = encrypted_data[8:]
        
        # XOR復号化
        key = hashlib.sha256(password.encode()).digest()
        
        result = bytearray()
        for i, byte in enumerate(ciphertext):
            result.append(byte ^ key[i % len(key)])
        
        return bytes(result)
    
    def generate_key_info(self, password: str) -> Dict[str, Any]:
        """キー情報生成"""
        salt = secrets.token_bytes(self.salt_length)
        key = self._derive_key(password, salt)
        
        return {
            'algorithm': self.algorithm,
            'key_length': self.key_length,
            'salt': salt.hex(),
            'key_hash': hashlib.sha256(key).hexdigest(),
            'created_time': int(time.time())
        }
    
    def verify_password(self, password: str, key_info: Dict[str, Any]) -> bool:
        """パスワード検証"""
        try:
            salt = bytes.fromhex(key_info['salt'])
            key = self._derive_key(password, salt)
            key_hash = hashlib.sha256(key).hexdigest()
            
            return key_hash == key_info['key_hash']
        except:
            return False


class SPEManager:
    """SPE管理システム"""
    
    def __init__(self):
        self.crypto = SPECrypto()
        self.key_store = {}
    
    def register_key(self, key_id: str, password: str) -> bool:
        """キー登録"""
        try:
            key_info = self.crypto.generate_key_info(password)
            self.key_store[key_id] = key_info
            return True
        except:
            return False
    
    def encrypt_with_key(self, data: bytes, key_id: str, password: str) -> Optional[bytes]:
        """キーIDで暗号化"""
        if key_id in self.key_store:
            if self.crypto.verify_password(password, self.key_store[key_id]):
                return self.crypto.encrypt(data, password)
        return None
    
    def decrypt_with_key(self, encrypted_data: bytes, key_id: str, password: str) -> Optional[bytes]:
        """キーIDで復号化"""
        if key_id in self.key_store:
            if self.crypto.verify_password(password, self.key_store[key_id]):
                return self.crypto.decrypt(encrypted_data, password)
        return None
    
    def list_keys(self) -> List[str]:
        """登録済みキー一覧"""
        return list(self.key_store.keys())


# 公開API
__all__ = ['SPECrypto', 'SPEManager']
