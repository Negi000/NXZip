#!/usr/bin/env python3
"""
🗂️ NXZip Archive System - Complete .nxz Archive Management

次世代アーカイブフォーマット with NEXUS + SPE
Copyright (c) 2025 NXZip Project
"""

import os
import struct
import time
import hashlib
import json
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import tempfile

from .nexus import NEXUSCompressor
from ..crypto.spe import SPECrypto
from ..utils.metadata import MetadataManager


class NXZFileHeader:
    """NXZ ファイルヘッダー"""
    
    def __init__(self):
        self.magic = b'NXZ1.0\x00\x00'  # 8バイト
        self.version = 1
        self.flags = 0  # 暗号化フラグなど
        self.entry_count = 0
        self.directory_offset = 0
        self.directory_size = 0
        self.created_time = int(time.time())
        self.checksum = b'\x00' * 32  # SHA256
    
    def pack(self) -> bytes:
        """ヘッダーをバイナリにパック"""
        return struct.pack('<8sHHIQQI32s',
            self.magic,
            self.version,
            self.flags,
            self.entry_count,
            self.directory_offset,
            self.directory_size,
            self.created_time,
            self.checksum
        )
    
    @classmethod
    def unpack(cls, data: bytes) -> 'NXZFileHeader':
        """バイナリからヘッダーを復元"""
        header = cls()
        unpacked = struct.unpack('<8sHHIQQI32s', data[:68])
        
        header.magic = unpacked[0]
        header.version = unpacked[1]
        header.flags = unpacked[2]
        header.entry_count = unpacked[3]
        header.directory_offset = unpacked[4]
        header.directory_size = unpacked[5]
        header.created_time = unpacked[6]
        header.checksum = unpacked[7]
        
        return header


class NXZEntry:
    """NXZ アーカイブエントリ"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.compressed_size = 0
        self.original_size = 0
        self.offset = 0
        self.checksum = b'\x00' * 32
        self.compression_method = 'NEXUS'
        self.modified_time = int(time.time())
        self.attributes = 0
        self.is_encrypted = False
    
    def pack(self) -> bytes:
        """エントリをバイナリにパック"""
        filepath_bytes = self.filepath.encode('utf-8')
        filepath_len = len(filepath_bytes)
        
        return struct.pack('<HQQQI32sI16sH',
            filepath_len,
            self.compressed_size,
            self.original_size,
            self.offset,
            self.modified_time,
            self.checksum,
            self.attributes,
            self.compression_method.encode('utf-8')[:16].ljust(16, b'\x00'),
            1 if self.is_encrypted else 0
        ) + filepath_bytes
    
    @classmethod
    def unpack(cls, data: bytes, offset: int = 0) -> Tuple['NXZEntry', int]:
        """バイナリからエントリを復元"""
        base_size = 2 + 8 + 8 + 8 + 4 + 32 + 4 + 16 + 2  # 84 bytes
        
        if len(data) < offset + base_size:
            raise ValueError("データが不足しています")
        
        # 基本情報解析
        filepath_len = struct.unpack('<H', data[offset:offset+2])[0]
        entry_data = struct.unpack('<HQQQI32sI16sH', 
                                 data[offset:offset+base_size])
        
        # ファイルパス取得
        filepath_start = offset + base_size
        filepath_end = filepath_start + filepath_len
        filepath = data[filepath_start:filepath_end].decode('utf-8')
        
        # エントリ作成
        entry = cls(filepath)
        entry.compressed_size = entry_data[1]
        entry.original_size = entry_data[2]
        entry.offset = entry_data[3]
        entry.modified_time = entry_data[4]
        entry.checksum = entry_data[5]
        entry.attributes = entry_data[6]
        entry.compression_method = entry_data[7].rstrip(b'\x00').decode('utf-8')
        entry.is_encrypted = bool(entry_data[8])
        
        return entry, filepath_end


class NXZipArchive:
    """🗂️ NXZip Archive Management System"""
    
    def __init__(self, archive_path: str, password: Optional[str] = None):
        self.archive_path = archive_path
        self.password = password
        self.nexus_compressor = NEXUSCompressor()
        self.spe_crypto = SPECrypto() if password else None
        self.metadata_manager = MetadataManager()
        
        self.header = NXZFileHeader()
        self.entries: List[NXZEntry] = []
        self.is_new = not os.path.exists(archive_path)
    
    def add_file(self, file_path: str, archive_path: Optional[str] = None) -> bool:
        """ファイルをアーカイブに追加"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
            
            # アーカイブ内パス決定
            if archive_path is None:
                archive_path = os.path.relpath(file_path)
            
            # ファイル読み込み
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # エントリ作成
            entry = NXZEntry(archive_path)
            entry.original_size = len(data)
            entry.checksum = hashlib.sha256(data).digest()
            entry.modified_time = int(os.path.getmtime(file_path))
            
            # 簡易圧縮メタデータ
            compression_info = {
                'format': 'TEXT',
                'original_size': len(data),
                'compressed_size': 0,  # 後で更新
                'ratio': 0,
                'processing_time': 0
            }
            
            # メタデータ保存 (try-catch で保護)
            try:
                self.metadata_manager.add_entry_metadata(entry, compression_info)
            except Exception as meta_error:
                print(f"メタデータ保存警告: {meta_error}")
            
            self.entries.append(entry)
            return True
            
        except Exception as e:
            print(f"ファイル追加エラー: {e}")
            return False
    
    def add_directory(self, dir_path: str, recursive: bool = True) -> int:
        """ディレクトリをアーカイブに追加"""
        added_count = 0
        
        try:
            path_obj = Path(dir_path)
            
            if recursive:
                # 再帰的に全ファイル追加
                for file_path in path_obj.rglob('*'):
                    if file_path.is_file():
                        relative_path = str(file_path.relative_to(path_obj.parent))
                        if self.add_file(str(file_path), relative_path):
                            added_count += 1
            else:
                # 直下のファイルのみ
                for file_path in path_obj.iterdir():
                    if file_path.is_file():
                        relative_path = str(file_path.relative_to(path_obj.parent))
                        if self.add_file(str(file_path), relative_path):
                            added_count += 1
            
            return added_count
            
        except Exception as e:
            print(f"ディレクトリ追加エラー: {e}")
            return added_count
    
    def extract_file(self, entry_name: str, output_path: str) -> bool:
        """特定ファイルを展開"""
        try:
            # エントリ検索
            entry = None
            for e in self.entries:
                if e.filepath == entry_name:
                    entry = e
                    break
            
            if not entry:
                raise ValueError(f"エントリが見つかりません: {entry_name}")
            
            # アーカイブから圧縮データ読み込み
            with open(self.archive_path, 'rb') as f:
                f.seek(entry.offset)
                compressed_data = f.read(entry.compressed_size)
            
            # SPE復号化
            if entry.is_encrypted and self.spe_crypto and self.password:
                decrypted_data = self.spe_crypto.decrypt(compressed_data, self.password)
                nexus_data = decrypted_data
            else:
                nexus_data = compressed_data
            
            # NEXUS展開 (要実装)
            original_data = self._decompress_nexus(nexus_data)
            
            # ファイル出力
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(original_data)
            
            # チェックサム検証
            actual_checksum = hashlib.sha256(original_data).digest()
            if actual_checksum != entry.checksum:
                raise ValueError("チェックサムが一致しません")
            
            return True
            
        except Exception as e:
            print(f"ファイル展開エラー: {e}")
            return False
    
    def extract_all(self, output_dir: str) -> int:
        """全ファイルを展開"""
        extracted_count = 0
        
        for entry in self.entries:
            output_path = os.path.join(output_dir, entry.filepath)
            if self.extract_file(entry.filepath, output_path):
                extracted_count += 1
        
        return extracted_count
    
    def save(self) -> bool:
        """アーカイブをファイルに保存"""
        try:
            with open(self.archive_path, 'wb') as f:
                # 1. ヘッダー領域を予約 (後で更新)
                header_pos = f.tell()
                f.write(b'\x00' * 68)  # ヘッダーサイズ (正確)
                
                # 2. 各エントリのデータを書き込み
                for entry in self.entries:
                    entry.offset = f.tell()
                    
                    # ファイルデータ取得・処理 (簡易実装)
                    if hasattr(entry, '_temp_data'):
                        f.write(entry._temp_data)
                    else:
                        # 既存ファイルから読み込み (暫定)
                        try:
                            with open(entry.filepath, 'rb') as src_file:
                                src_data = src_file.read()
                            
                            # NEXUS圧縮
                            compressed_data, _ = self.nexus_compressor.compress(src_data, entry.filepath)
                            
                            # SPE暗号化 (必要時)
                            if self.spe_crypto and self.password:
                                encrypted_data = self.spe_crypto.encrypt(compressed_data, self.password)
                                final_data = encrypted_data
                                entry.is_encrypted = True
                            else:
                                final_data = compressed_data
                                entry.is_encrypted = False
                            
                            f.write(final_data)
                            entry.compressed_size = len(final_data)
                            
                        except Exception as e:
                            print(f"エラー処理ファイル {entry.filepath}: {e}")
                            import traceback
                            traceback.print_exc()
                            continue
                
                # 3. ディレクトリ情報書き込み
                directory_offset = f.tell()
                directory_data = b''
                
                for entry in self.entries:
                    directory_data += entry.pack()
                
                f.write(directory_data)
                directory_size = len(directory_data)
                
                # 4. ヘッダー更新
                self.header.entry_count = len(self.entries)
                self.header.directory_offset = directory_offset
                self.header.directory_size = directory_size
                
                # 全体のチェックサム計算
                f.flush()  # バッファをフラッシュ
                
                # チェックサム計算のために一時的にファイルを読み直し
                with open(self.archive_path, 'rb') as read_f:
                    read_f.seek(68)  # ヘッダー後 (正確)
                    archive_data = read_f.read()
                    self.header.checksum = hashlib.sha256(archive_data).digest()[:20]  # 20バイトに切り詰め
                
                # ヘッダー書き込み
                f.seek(header_pos)
                f.write(self.header.pack())
            
            return True
            
        except Exception as e:
            print(f"アーカイブ保存エラー: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load(self) -> bool:
        """既存アーカイブを読み込み"""
        try:
            if not os.path.exists(self.archive_path):
                return False
            
            with open(self.archive_path, 'rb') as f:
                # ヘッダー読み込み
                header_data = f.read(68)
                self.header = NXZFileHeader.unpack(header_data)
                
                # マジックナンバー確認
                if self.header.magic != b'NXZ1.0\x00\x00':
                    raise ValueError("無効なNXZファイルです")
                
                # ディレクトリ読み込み
                f.seek(self.header.directory_offset)
                directory_data = f.read(self.header.directory_size)
                
                # エントリ解析
                self.entries = []
                offset = 0
                for _ in range(self.header.entry_count):
                    entry, next_offset = NXZEntry.unpack(directory_data, offset)
                    self.entries.append(entry)
                    offset = next_offset
            
            return True
            
        except Exception as e:
            print(f"アーカイブ読み込みエラー: {e}")
            return False
    
    def list_entries(self) -> List[Dict[str, Any]]:
        """エントリ一覧取得"""
        entries_info = []
        
        for entry in self.entries:
            compression_ratio = (1 - entry.compressed_size / entry.original_size) * 100 if entry.original_size > 0 else 0
            
            entries_info.append({
                'filepath': entry.filepath,
                'original_size': entry.original_size,
                'compressed_size': entry.compressed_size,
                'compression_ratio': compression_ratio,
                'modified_time': entry.modified_time,
                'is_encrypted': entry.is_encrypted,
                'compression_method': entry.compression_method
            })
        
        return entries_info
    
    def get_stats(self) -> Dict[str, Any]:
        """アーカイブ統計情報"""
        if not self.entries:
            return {'total_files': 0, 'total_original_size': 0, 'total_compressed_size': 0}
        
        total_original = sum(e.original_size for e in self.entries)
        total_compressed = sum(e.compressed_size for e in self.entries)
        overall_ratio = (1 - total_compressed / total_original) * 100 if total_original > 0 else 0
        
        return {
            'total_files': len(self.entries),
            'total_original_size': total_original,
            'total_compressed_size': total_compressed,
            'overall_compression_ratio': overall_ratio,
            'created_time': self.header.created_time,
            'has_encryption': any(e.is_encrypted for e in self.entries)
        }
    
    def _decompress_nexus(self, data: bytes) -> bytes:
        """NEXUS展開 (簡易実装)"""
        # TODO: 完全なNEXUS展開実装
        # 現在は基本的な展開のみ
        try:
            import lzma
            return lzma.decompress(data)
        except:
            return data


# 公開API
__all__ = ['NXZipArchive', 'NXZFileHeader', 'NXZEntry']
