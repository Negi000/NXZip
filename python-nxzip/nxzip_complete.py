#!/usr/bin/env python3
"""
NXZip Complete System - Python版
SPE Core + 圧縮 + 暗号化の統合システム
"""

import os
import sys
import struct
import zlib
import hashlib
import secrets
from typing import Optional, Tuple, Dict, Any
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend


class SPECore:
    """Structure-Preserving Encryption Core"""
    
    def __init__(self):
        self.xor_key = b"NXZip_SPE_2024_v3"
        self.block_size = 16
    
    def apply_transform(self, data: bytes) -> bytes:
        """完全可逆なSPE変換"""
        if not data:
            return data
            
        result = bytearray(data)
        original_len = len(result)
        
        # パディング
        padded_len = ((original_len + 15) // 16) * 16
        result.extend(b'\x00' * (padded_len - original_len))
        result.extend(struct.pack('<Q', original_len))
        
        # ブロック循環シフト
        if len(result) >= 32:
            self._apply_cyclic_shift(result)
        
        # バイトレベル変換
        self._apply_byte_transform(result)
        
        # XOR難読化
        self._apply_xor(result)
        
        return bytes(result)
    
    def reverse_transform(self, data: bytes) -> bytes:
        """SPE変換を完全に逆変換"""
        if not data:
            return data
            
        result = bytearray(data)
        
        # 逆変換の順序
        self._apply_xor(result)
        self._reverse_byte_transform(result)
        
        if len(result) >= 32:
            self._reverse_cyclic_shift(result)
        
        if len(result) >= 8:
            original_len = struct.unpack('<Q', result[-8:])[0]
            result = result[:-8]
            result = result[:original_len]
        
        return bytes(result)
    
    def _apply_xor(self, data: bytearray) -> None:
        for i in range(len(data)):
            data[i] ^= self.xor_key[i % len(self.xor_key)]
    
    def _apply_byte_transform(self, data: bytearray) -> None:
        for i in range(len(data)):
            data[i] = ((data[i] ^ 0xFF) + 0x5A) & 0xFF
    
    def _reverse_byte_transform(self, data: bytearray) -> None:
        for i in range(len(data)):
            data[i] = ((data[i] - 0x5A) & 0xFF) ^ 0xFF
    
    def _apply_cyclic_shift(self, data: bytearray) -> None:
        num_blocks = len(data) // self.block_size
        if num_blocks <= 1:
            return
        self._cyclic_shift_blocks(data, 1, num_blocks)
    
    def _reverse_cyclic_shift(self, data: bytearray) -> None:
        num_blocks = len(data) // self.block_size
        if num_blocks <= 1:
            return
        self._cyclic_shift_blocks(data, -1, num_blocks)
    
    def _cyclic_shift_blocks(self, data: bytearray, shift: int, num_blocks: int) -> None:
        if shift == 0 or num_blocks <= 1:
            return
        
        shift = shift % num_blocks
        if shift == 0:
            return
        
        temp_blocks = []
        for i in range(shift):
            start = i * self.block_size
            end = start + self.block_size
            temp_blocks.append(data[start:end])
        
        for i in range(shift, num_blocks):
            src_start = i * self.block_size
            dst_start = (i - shift) * self.block_size
            
            for j in range(self.block_size):
                if src_start + j < len(data) and dst_start + j < len(data):
                    data[dst_start + j] = data[src_start + j]
        
        for i, temp_block in enumerate(temp_blocks):
            dst_start = (num_blocks - shift + i) * self.block_size
            for j in range(len(temp_block)):
                if dst_start + j < len(data):
                    data[dst_start + j] = temp_block[j]


class NXZipError(Exception):
    """NXZip関連のエラー"""
    pass


class NXZipFile:
    """NXZip ファイル形式ハンドラー"""
    
    # ファイル形式定数
    MAGIC = b'NXZ\x01'  # NXZ v1.0
    HEADER_SIZE = 64
    
    # 圧縮方式
    COMPRESSION_NONE = 0
    COMPRESSION_ZLIB = 1
    
    # 暗号化方式
    ENCRYPTION_NONE = 0
    ENCRYPTION_AES_GCM = 1
    ENCRYPTION_SPE_ONLY = 2
    
    def __init__(self):
        self.spe_core = SPECore()
    
    def create_archive(self, data: bytes, password: Optional[str] = None, 
                      compression_level: int = 6) -> bytes:
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
    
    def extract_archive(self, archive_data: bytes, password: Optional[str] = None) -> bytes:
        """NXZアーカイブを展開"""
        
        if len(archive_data) < self.HEADER_SIZE:
            raise NXZipError("Invalid archive: too small")
        
        # 1. ヘッダー解析
        header = archive_data[:self.HEADER_SIZE]
        payload_data = archive_data[self.HEADER_SIZE:]
        
        header_info = self._parse_header(header)
        
        # 2. 復号化
        if header_info['encryption'] == self.ENCRYPTION_AES_GCM:
            if not password:
                raise NXZipError("Password required for encrypted archive")
            
            # 暗号化情報を抽出（最初の28バイト: salt 16 + nonce 12）
            if len(payload_data) < 28:
                raise NXZipError("Invalid encrypted archive: missing crypto header")
            
            salt = payload_data[:16]
            nonce = payload_data[16:28]
            encrypted_data = payload_data[28:]
            
            spe_data = self._decrypt_aes_gcm(encrypted_data, password, salt, nonce)
        elif header_info['encryption'] == self.ENCRYPTION_SPE_ONLY:
            spe_data = payload_data
        else:
            raise NXZipError(f"Unsupported encryption method: {header_info['encryption']}")
        
        # 3. SPE逆変換
        payload = self.spe_core.reverse_transform(spe_data)
        
        # 4. 展開
        if header_info['compression'] == self.COMPRESSION_ZLIB:
            data = zlib.decompress(payload)
        elif header_info['compression'] == self.COMPRESSION_NONE:
            data = payload
        else:
            raise NXZipError(f"Unsupported compression method: {header_info['compression']}")
        
        # 5. 整合性チェック
        if len(data) != header_info['original_size']:
            raise NXZipError("Size mismatch after decompression")
        
        actual_checksum = hashlib.sha256(data).digest()
        if actual_checksum != header_info['checksum']:
            raise NXZipError("Checksum verification failed")
        
        return data
    
    def _create_header(self, original_size: int, compressed_size: int, encrypted_size: int,
                      compression: int, encryption: int, checksum: bytes,
                      salt: bytes, nonce: bytes) -> bytes:
        """ヘッダーを作成"""
        header = bytearray(self.HEADER_SIZE)
        
        # マジックナンバー (4 bytes)
        header[0:4] = self.MAGIC
        
        # サイズ情報 (12 bytes)
        struct.pack_into('<III', header, 4, original_size, compressed_size, encrypted_size)
        
        # フラグ (2 bytes)
        flags = (compression & 0xFF) | ((encryption & 0xFF) << 8)
        struct.pack_into('<H', header, 16, flags)
        
        # 予約領域 (2 bytes)
        struct.pack_into('<H', header, 18, 0)
        
        # チェックサム (32 bytes)
        header[20:52] = checksum
        
        # 予約領域を縮小してsaltとnonceをフルサイズで保存
        # header[52:64] = 12 bytes available, split as salt(8) + nonce(4)はヘッダーサイズが不足
        # 暗号化情報はヘッダー後に別途保存する必要がある
        
        return bytes(header)
    
    def _parse_header(self, header: bytes) -> Dict[str, Any]:
        """ヘッダーを解析"""
        if len(header) != self.HEADER_SIZE:
            raise NXZipError("Invalid header size")
        
        if header[0:4] != self.MAGIC:
            raise NXZipError("Invalid magic number")
        
        original_size, compressed_size, encrypted_size = struct.unpack('<III', header[4:16])
        flags = struct.unpack('<H', header[16:18])[0]
        
        compression = flags & 0xFF
        encryption = (flags >> 8) & 0xFF
        
        checksum = header[20:52]
        
        return {
            'original_size': original_size,
            'compressed_size': compressed_size,
            'encrypted_size': encrypted_size,
            'compression': compression,
            'encryption': encryption,
            'checksum': checksum
        }
    
    def _encrypt_aes_gcm(self, data: bytes, password: str) -> Tuple[bytes, bytes, bytes]:
        """AES-GCM暗号化"""
        salt = secrets.token_bytes(16)
        nonce = secrets.token_bytes(12)
        
        # パスワードから鍵を導出
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = kdf.derive(password.encode('utf-8'))
        
        # 暗号化
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce), backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        # 認証タグを追加
        return ciphertext + encryptor.tag, salt, nonce
    
    def _decrypt_aes_gcm(self, encrypted_data: bytes, password: str, 
                        salt: bytes, nonce: bytes) -> bytes:
        """AES-GCM復号化"""
        if len(encrypted_data) < 16:
            raise NXZipError("Invalid encrypted data")
        
        ciphertext = encrypted_data[:-16]
        tag = encrypted_data[-16:]
        
        # パスワードから鍵を導出
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = kdf.derive(password.encode('utf-8'))
        
        # 復号化
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce, tag), backend=default_backend())
        decryptor = cipher.decryptor()
        
        return decryptor.update(ciphertext) + decryptor.finalize()


def main():
    """メイン関数 - CLI インターフェース"""
    import argparse
    
    parser = argparse.ArgumentParser(description='NXZip - Python版統合アーカイバー')
    parser.add_argument('command', choices=['create', 'extract', 'test'], 
                       help='実行するコマンド')
    parser.add_argument('archive', help='アーカイブファイル名')
    parser.add_argument('file', nargs='?', help='対象ファイル（createの場合）')
    parser.add_argument('-p', '--password', help='パスワード')
    parser.add_argument('-l', '--level', type=int, default=6, 
                       help='圧縮レベル (1-9, デフォルト: 6)')
    
    args = parser.parse_args()
    
    nxzip = NXZipFile()
    
    try:
        if args.command == 'create':
            if not args.file:
                print("エラー: 入力ファイルが指定されていません")
                return 1
            
            if not os.path.exists(args.file):
                print(f"エラー: ファイル '{args.file}' が見つかりません")
                return 1
            
            print(f"📦 アーカイブ作成中: {args.file} -> {args.archive}")
            
            with open(args.file, 'rb') as f:
                data = f.read()
            
            archive_data = nxzip.create_archive(data, args.password, args.level)
            
            with open(args.archive, 'wb') as f:
                f.write(archive_data)
            
            compression_ratio = len(archive_data) / len(data) * 100
            print(f"✅ 完了! 圧縮率: {compression_ratio:.1f}%")
            print(f"   元サイズ: {len(data):,} bytes")
            print(f"   圧縮後:   {len(archive_data):,} bytes")
        
        elif args.command == 'extract':
            if not os.path.exists(args.archive):
                print(f"エラー: アーカイブ '{args.archive}' が見つかりません")
                return 1
            
            output_file = args.file or args.archive.replace('.nxz', '_extracted')
            
            print(f"📂 アーカイブ展開中: {args.archive} -> {output_file}")
            
            with open(args.archive, 'rb') as f:
                archive_data = f.read()
            
            data = nxzip.extract_archive(archive_data, args.password)
            
            with open(output_file, 'wb') as f:
                f.write(data)
            
            print(f"✅ 完了! {len(data):,} bytes を展開しました")
        
        elif args.command == 'test':
            if not os.path.exists(args.archive):
                print(f"エラー: アーカイブ '{args.archive}' が見つかりません")
                return 1
            
            print(f"🔍 アーカイブテスト中: {args.archive}")
            
            with open(args.archive, 'rb') as f:
                archive_data = f.read()
            
            try:
                data = nxzip.extract_archive(archive_data, args.password)
                print(f"✅ OK - アーカイブは正常です ({len(data):,} bytes)")
            except Exception as e:
                print(f"❌ エラー: {e}")
                return 1
    
    except KeyboardInterrupt:
        print("\n⏹️ 中断されました")
        return 1
    except Exception as e:
        print(f"❌ エラー: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
