#!/usr/bin/env python3
"""
📊 NXZip Metadata Management System

メタデータ管理システム
Copyright (c) 2025 NXZip Project
"""

import json
import time
import hashlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class CompressionMetadata:
    """圧縮メタデータ"""
    algorithm: str
    original_size: int
    compressed_size: int
    compression_ratio: float
    processing_time: float
    format_type: str
    checksum: str


@dataclass
class EncryptionMetadata:
    """暗号化メタデータ"""
    algorithm: str
    key_derivation: str
    iv_length: int
    tag_length: int
    encrypted: bool


@dataclass
class FileMetadata:
    """ファイルメタデータ"""
    filepath: str
    filename: str
    original_size: int
    modified_time: int
    attributes: int
    mime_type: str


class MetadataManager:
    """📊 メタデータ管理システム"""
    
    def __init__(self):
        self.compression_data: Dict[str, CompressionMetadata] = {}
        self.encryption_data: Dict[str, EncryptionMetadata] = {}
        self.file_data: Dict[str, FileMetadata] = {}
        self.archive_metadata = {
            'created_time': int(time.time()),
            'version': '1.0.0',
            'creator': 'NXZip',
            'total_files': 0,
            'total_original_size': 0,
            'total_compressed_size': 0
        }
    
    def add_entry_metadata(self, entry, compression_info: Dict[str, Any]):
        """エントリメタデータ追加"""
        filepath = entry.filepath
        
        # 圧縮メタデータ
        compression_meta = CompressionMetadata(
            algorithm=compression_info.get('format', 'NEXUS'),
            original_size=compression_info.get('original_size', 0),
            compressed_size=compression_info.get('compressed_size', 0),
            compression_ratio=compression_info.get('ratio', 0),
            processing_time=compression_info.get('processing_time', 0),
            format_type=compression_info.get('format', 'UNKNOWN'),
            checksum=hashlib.sha256(entry.checksum).hexdigest()
        )
        self.compression_data[filepath] = compression_meta
        
        # ファイルメタデータ
        file_meta = FileMetadata(
            filepath=filepath,
            filename=entry.filename,
            original_size=entry.original_size,
            modified_time=entry.modified_time,
            attributes=entry.attributes,
            mime_type=self._detect_mime_type(entry.filename)
        )
        self.file_data[filepath] = file_meta
        
        # 暗号化メタデータ
        if entry.is_encrypted:
            encryption_meta = EncryptionMetadata(
                algorithm='AES-GCM',
                key_derivation='PBKDF2',
                iv_length=16,
                tag_length=16,
                encrypted=True
            )
            self.encryption_data[filepath] = encryption_meta
        
        # アーカイブ統計更新
        self._update_archive_stats()
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """圧縮統計取得"""
        if not self.compression_data:
            return {}
        
        ratios = [meta.compression_ratio for meta in self.compression_data.values()]
        sizes_original = [meta.original_size for meta in self.compression_data.values()]
        sizes_compressed = [meta.compressed_size for meta in self.compression_data.values()]
        times = [meta.processing_time for meta in self.compression_data.values()]
        
        return {
            'total_files': len(self.compression_data),
            'average_ratio': sum(ratios) / len(ratios),
            'best_ratio': max(ratios),
            'worst_ratio': min(ratios),
            'total_original_size': sum(sizes_original),
            'total_compressed_size': sum(sizes_compressed),
            'overall_ratio': (1 - sum(sizes_compressed) / sum(sizes_original)) * 100 if sum(sizes_original) > 0 else 0,
            'average_processing_time': sum(times) / len(times),
            'total_processing_time': sum(times)
        }
    
    def get_format_distribution(self) -> Dict[str, int]:
        """フォーマット分布取得"""
        formats = {}
        for meta in self.compression_data.values():
            format_type = meta.format_type
            formats[format_type] = formats.get(format_type, 0) + 1
        return formats
    
    def get_encryption_stats(self) -> Dict[str, Any]:
        """暗号化統計取得"""
        total_files = len(self.file_data)
        encrypted_files = len(self.encryption_data)
        
        return {
            'total_files': total_files,
            'encrypted_files': encrypted_files,
            'encryption_ratio': (encrypted_files / total_files) * 100 if total_files > 0 else 0,
            'encryption_algorithms': list(set(meta.algorithm for meta in self.encryption_data.values()))
        }
    
    def export_metadata(self) -> Dict[str, Any]:
        """メタデータエクスポート"""
        return {
            'archive_metadata': self.archive_metadata,
            'compression_data': {k: asdict(v) for k, v in self.compression_data.items()},
            'encryption_data': {k: asdict(v) for k, v in self.encryption_data.items()},
            'file_data': {k: asdict(v) for k, v in self.file_data.items()},
            'statistics': {
                'compression_stats': self.get_compression_stats(),
                'format_distribution': self.get_format_distribution(),
                'encryption_stats': self.get_encryption_stats()
            }
        }
    
    def import_metadata(self, metadata: Dict[str, Any]):
        """メタデータインポート"""
        try:
            self.archive_metadata = metadata.get('archive_metadata', {})
            
            # 圧縮データ復元
            compression_raw = metadata.get('compression_data', {})
            self.compression_data = {
                k: CompressionMetadata(**v) for k, v in compression_raw.items()
            }
            
            # 暗号化データ復元
            encryption_raw = metadata.get('encryption_data', {})
            self.encryption_data = {
                k: EncryptionMetadata(**v) for k, v in encryption_raw.items()
            }
            
            # ファイルデータ復元
            file_raw = metadata.get('file_data', {})
            self.file_data = {
                k: FileMetadata(**v) for k, v in file_raw.items()
            }
        except Exception as e:
            print(f"メタデータインポートエラー: {e}")
    
    def save_to_file(self, filepath: str):
        """メタデータをファイルに保存"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.export_metadata(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"メタデータ保存エラー: {e}")
    
    def load_from_file(self, filepath: str):
        """ファイルからメタデータを読み込み"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            self.import_metadata(metadata)
        except Exception as e:
            print(f"メタデータ読み込みエラー: {e}")
    
    def _update_archive_stats(self):
        """アーカイブ統計更新"""
        self.archive_metadata['total_files'] = len(self.file_data)
        self.archive_metadata['total_original_size'] = sum(
            meta.original_size for meta in self.compression_data.values()
        )
        self.archive_metadata['total_compressed_size'] = sum(
            meta.compressed_size for meta in self.compression_data.values()
        )
    
    def _detect_mime_type(self, filename: str) -> str:
        """MIMEタイプ検出"""
        import mimetypes
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type or 'application/octet-stream'


# 公開API
__all__ = ['MetadataManager', 'CompressionMetadata', 'EncryptionMetadata', 'FileMetadata']
