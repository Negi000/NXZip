#!/usr/bin/env python3
"""
📄 NXZip Format Specifications

.nxz アーカイブフォーマット定義
Copyright (c) 2025 NXZip Project
"""

import os
import struct
from typing import Dict, List, Tuple, Any
from enum import IntEnum


class NXZVersion(IntEnum):
    """NXZバージョン"""
    V1_0 = 100


class CompressionMethod(IntEnum):
    """圧縮方式"""
    NEXUS = 1
    LZMA = 2
    ZLIB = 3
    BZIP2 = 4
    ZSTD = 5


class EncryptionMethod(IntEnum):
    """暗号化方式"""
    NONE = 0
    SPE_AES_GCM = 1
    AES_256_GCM = 2
    XCHACHA20_POLY1305 = 3


class FileAttributes(IntEnum):
    """ファイル属性"""
    NONE = 0
    READONLY = 1
    HIDDEN = 2
    SYSTEM = 4
    DIRECTORY = 8
    ARCHIVE = 16
    COMPRESSED = 32
    ENCRYPTED = 64


class NXZFormat:
    """📄 NXZ フォーマット仕様"""
    
    # ファイルシグネチャ
    MAGIC_SIGNATURE = b'NXZ1.0\x00\x00'
    
    # ヘッダーサイズ
    HEADER_SIZE = 72
    ENTRY_BASE_SIZE = 84
    
    # 最大値
    MAX_FILENAME_LENGTH = 65535
    MAX_ENTRIES = 4294967295
    MAX_ARCHIVE_SIZE = 18446744073709551615  # 2^64-1
    
    # アルゴリズム
    SUPPORTED_COMPRESSION = [
        CompressionMethod.NEXUS,
        CompressionMethod.LZMA,
        CompressionMethod.ZLIB,
        CompressionMethod.BZIP2
    ]
    
    SUPPORTED_ENCRYPTION = [
        EncryptionMethod.NONE,
        EncryptionMethod.SPE_AES_GCM,
        EncryptionMethod.AES_256_GCM
    ]


class NXZHeader:
    """NXZ ファイルヘッダー (72 bytes)"""
    
    def __init__(self):
        self.magic = NXZFormat.MAGIC_SIGNATURE      # 8 bytes
        self.version = NXZVersion.V1_0              # 2 bytes
        self.flags = 0                              # 2 bytes
        self.entry_count = 0                        # 4 bytes
        self.directory_offset = 0                   # 8 bytes
        self.directory_size = 0                     # 8 bytes
        self.created_timestamp = 0                  # 4 bytes
        self.modified_timestamp = 0                 # 4 bytes
        self.archive_comment_offset = 0             # 8 bytes
        self.archive_comment_size = 0               # 4 bytes
        self.checksum = b'\x00' * 20               # 20 bytes (SHA-1)
    
    def pack(self) -> bytes:
        """ヘッダーをバイナリにシリアライズ"""
        return struct.pack('<8sHHIQQIIQI20s',
            self.magic,
            self.version,
            self.flags,
            self.entry_count,
            self.directory_offset,
            self.directory_size,
            self.created_timestamp,
            self.modified_timestamp,
            self.archive_comment_offset,
            self.archive_comment_size,
            self.checksum
        )
    
    @classmethod
    def unpack(cls, data: bytes) -> 'NXZHeader':
        """バイナリからヘッダーを復元"""
        if len(data) < NXZFormat.HEADER_SIZE:
            raise ValueError(f"ヘッダーサイズが不足: {len(data)} < {NXZFormat.HEADER_SIZE}")
        
        header = cls()
        unpacked = struct.unpack('<8sHHIQQIIQI20s', data[:NXZFormat.HEADER_SIZE])
        
        header.magic = unpacked[0]
        header.version = unpacked[1]
        header.flags = unpacked[2]
        header.entry_count = unpacked[3]
        header.directory_offset = unpacked[4]
        header.directory_size = unpacked[5]
        header.created_timestamp = unpacked[6]
        header.modified_timestamp = unpacked[7]
        header.archive_comment_offset = unpacked[8]
        header.archive_comment_size = unpacked[9]
        header.checksum = unpacked[10]
        
        return header
    
    def validate(self) -> bool:
        """ヘッダー検証"""
        return (self.magic == NXZFormat.MAGIC_SIGNATURE and
                self.version in [NXZVersion.V1_0] and
                self.entry_count <= NXZFormat.MAX_ENTRIES)


class NXZDirectoryEntry:
    """NXZ ディレクトリエントリ"""
    
    def __init__(self):
        self.filename_length = 0                    # 2 bytes
        self.original_size = 0                      # 8 bytes
        self.compressed_size = 0                    # 8 bytes
        self.data_offset = 0                        # 8 bytes
        self.modified_timestamp = 0                 # 4 bytes
        self.checksum = b'\x00' * 32               # 32 bytes (SHA-256)
        self.attributes = FileAttributes.NONE       # 4 bytes
        self.compression_method = CompressionMethod.NEXUS  # 2 bytes
        self.encryption_method = EncryptionMethod.NONE     # 2 bytes
        self.extra_field_length = 0                # 2 bytes
        self.reserved = b'\x00' * 12               # 12 bytes
        
        # 可変長フィールド
        self.filename = ""
        self.extra_field = b''
    
    def pack(self) -> bytes:
        """エントリをバイナリにシリアライズ"""
        filename_bytes = self.filename.encode('utf-8')
        self.filename_length = len(filename_bytes)
        self.extra_field_length = len(self.extra_field)
        
        fixed_part = struct.pack('<HQQQIII32sIHHH12s',
            self.filename_length,
            self.original_size,
            self.compressed_size,
            self.data_offset,
            self.modified_timestamp,
            self.attributes,
            self.compression_method,
            self.checksum,
            self.encryption_method,
            self.extra_field_length,
            0,  # padding
            self.reserved
        )
        
        return fixed_part + filename_bytes + self.extra_field
    
    @classmethod
    def unpack(cls, data: bytes, offset: int = 0) -> Tuple['NXZDirectoryEntry', int]:
        """バイナリからエントリを復元"""
        if len(data) < offset + NXZFormat.ENTRY_BASE_SIZE:
            raise ValueError("エントリデータが不足")
        
        entry = cls()
        
        # 固定部分
        fixed_data = struct.unpack('<HQQQIII32sIHHH12s', 
                                 data[offset:offset + NXZFormat.ENTRY_BASE_SIZE])
        
        entry.filename_length = fixed_data[0]
        entry.original_size = fixed_data[1]
        entry.compressed_size = fixed_data[2]
        entry.data_offset = fixed_data[3]
        entry.modified_timestamp = fixed_data[4]
        entry.attributes = fixed_data[5]
        entry.compression_method = fixed_data[6]
        entry.checksum = fixed_data[7]
        entry.encryption_method = fixed_data[8]
        entry.extra_field_length = fixed_data[9]
        entry.reserved = fixed_data[11]
        
        # 可変部分
        variable_offset = offset + NXZFormat.ENTRY_BASE_SIZE
        
        # ファイル名
        filename_end = variable_offset + entry.filename_length
        entry.filename = data[variable_offset:filename_end].decode('utf-8')
        
        # 拡張フィールド
        extra_start = filename_end
        extra_end = extra_start + entry.extra_field_length
        entry.extra_field = data[extra_start:extra_end]
        
        return entry, extra_end
    
    def get_compression_info(self) -> Dict[str, Any]:
        """圧縮情報取得"""
        ratio = (1 - self.compressed_size / self.original_size) * 100 if self.original_size > 0 else 0
        
        return {
            'method': CompressionMethod(self.compression_method).name,
            'original_size': self.original_size,
            'compressed_size': self.compressed_size,
            'compression_ratio': ratio,
            'is_encrypted': self.encryption_method != EncryptionMethod.NONE
        }


class NXZSecureHeader:
    """.nxz.sec ファイル用セキュアヘッダー"""
    
    def __init__(self):
        self.magic = b'NXZSEC10'                   # 8 bytes
        self.encryption_method = EncryptionMethod.SPE_AES_GCM  # 2 bytes
        self.key_derivation_iterations = 100000    # 4 bytes
        self.salt = b'\x00' * 32                  # 32 bytes
        self.iv = b'\x00' * 16                    # 16 bytes
        self.tag = b'\x00' * 16                   # 16 bytes
        self.encrypted_header_size = 0             # 4 bytes
        self.reserved = b'\x00' * 20              # 20 bytes
    
    def pack(self) -> bytes:
        """セキュアヘッダーをバイナリに"""
        return struct.pack('<8sHI32s16s16sI20s',
            self.magic,
            self.encryption_method,
            self.key_derivation_iterations,
            self.salt,
            self.iv,
            self.tag,
            self.encrypted_header_size,
            self.reserved
        )
    
    @classmethod
    def unpack(cls, data: bytes) -> 'NXZSecureHeader':
        """バイナリからセキュアヘッダーを復元"""
        if len(data) < 118:  # セキュアヘッダーサイズ
            raise ValueError("セキュアヘッダーが不足")
        
        header = cls()
        unpacked = struct.unpack('<8sHI32s16s16sI20s', data[:118])
        
        header.magic = unpacked[0]
        header.encryption_method = unpacked[1]
        header.key_derivation_iterations = unpacked[2]
        header.salt = unpacked[3]
        header.iv = unpacked[4]
        header.tag = unpacked[5]
        header.encrypted_header_size = unpacked[6]
        header.reserved = unpacked[7]
        
        return header


class NXZFormatValidator:
    """📋 NXZ フォーマット検証"""
    
    @staticmethod
    def validate_archive(filepath: str) -> Dict[str, Any]:
        """アーカイブファイル検証"""
        try:
            with open(filepath, 'rb') as f:
                # ヘッダー読み込み
                header_data = f.read(NXZFormat.HEADER_SIZE)
                if len(header_data) < NXZFormat.HEADER_SIZE:
                    return {'valid': False, 'error': 'ヘッダーサイズ不足'}
                
                # ヘッダー解析
                try:
                    header = NXZHeader.unpack(header_data)
                except:
                    return {'valid': False, 'error': 'ヘッダー解析失敗'}
                
                # 基本検証
                if not header.validate():
                    return {'valid': False, 'error': '無効なヘッダー'}
                
                # ディレクトリ検証
                if header.directory_offset > 0:
                    f.seek(header.directory_offset)
                    directory_data = f.read(header.directory_size)
                    
                    if len(directory_data) != header.directory_size:
                        return {'valid': False, 'error': 'ディレクトリサイズ不一致'}
                
                return {
                    'valid': True,
                    'version': header.version,
                    'entry_count': header.entry_count,
                    'created_timestamp': header.created_timestamp,
                    'archive_size': os.path.getsize(filepath)
                }
                
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    @staticmethod
    def get_format_info() -> Dict[str, Any]:
        """フォーマット情報取得"""
        return {
            'format_name': 'NXZip Archive',
            'extension': '.nxz',
            'secure_extension': '.nxz.sec',
            'magic_signature': NXZFormat.MAGIC_SIGNATURE,
            'version': NXZVersion.V1_0,
            'header_size': NXZFormat.HEADER_SIZE,
            'max_entries': NXZFormat.MAX_ENTRIES,
            'max_archive_size': NXZFormat.MAX_ARCHIVE_SIZE,
            'supported_compression': [method.name for method in NXZFormat.SUPPORTED_COMPRESSION],
            'supported_encryption': [method.name for method in NXZFormat.SUPPORTED_ENCRYPTION]
        }


# 公開API
__all__ = [
    'NXZFormat', 'NXZHeader', 'NXZDirectoryEntry', 'NXZSecureHeader',
    'NXZFormatValidator', 'CompressionMethod', 'EncryptionMethod', 
    'FileAttributes', 'NXZVersion'
]
