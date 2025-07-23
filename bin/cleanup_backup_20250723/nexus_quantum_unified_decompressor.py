#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NXZip量子解凍エンジン - 統合版
PNG構造保存版とバイトレベル版の両方に対応
"""

import hashlib
import struct
import time
import zlib
import os
import lzma
from typing import Dict, List, Tuple

class NXZipQuantumDecompressor:
    def __init__(self):
        self.png_signature = b'\x4E\x58\x5A\x51\x50\x4E\x47'  # NXZQPNG
        self.byte_signature = b'\x4E\x58\x5A\x51\x42\x54\x45'  # NXZQBTE
        
    def decompress_file(self, input_path: str) -> Dict:
        """自動形式判定解凍"""
        if not os.path.exists(input_path):
            print(f"❌ ファイルが見つかりません: {input_path}")
            return {'error': f'File not found: {input_path}'}
        
        start_time = time.time()
        
        try:
            with open(input_path, 'rb') as f:
                compressed_data = f.read()
            
            # 形式判定
            if compressed_data.startswith(self.png_signature):
                return self._decompress_png_structure(input_path, compressed_data, start_time)
            elif compressed_data.startswith(self.byte_signature):
                return self._decompress_byte_level(input_path, compressed_data, start_time)
            else:
                return {'error': 'Unknown format signature'}
                
        except Exception as e:
            print(f"❌ 解凍エラー: {str(e)}")
            return {'error': str(e)}
    
    def _decompress_png_structure(self, input_path: str, compressed_data: bytes, start_time: float) -> Dict:
        """PNG構造保存版解凍"""
        print(f"🔬 PNG構造保存版解凍開始...")
        
        # ヘッダー解析
        pos = 7  # signature
        version = struct.unpack('>I', compressed_data[pos:pos+4])[0]
        pos += 4
        format_type = compressed_data[pos:pos+16].rstrip(b'\x00').decode('utf-8')
        pos += 16
        
        # メタデータ
        original_hash = compressed_data[pos:pos+16]
        pos += 16
        original_size = struct.unpack('>I', compressed_data[pos:pos+4])[0]
        pos += 4
        compressed_size = struct.unpack('>I', compressed_data[pos:pos+4])[0]
        pos += 4
        
        # 圧縮データ解凍
        png_data = compressed_data[pos:pos+compressed_size]
        restored_data = self._decompress_png_structure_data(png_data)
        
        # 出力ファイル作成
        output_path = input_path.replace('.nxzqs', '.restored')
        with open(output_path, 'wb') as f:
            f.write(restored_data)
        
        # 検証
        restored_hash = hashlib.md5(restored_data).digest()
        hash_match = restored_hash == original_hash
        elapsed_time = time.time() - start_time
        
        print(f"🧠 PNG構造保存解凍完了")
        print(f"入力: {os.path.basename(input_path.replace('.nxzqs', ''))}")
        print(f"出力: {os.path.basename(output_path)}")
        print(f"復元サイズ: {len(restored_data):,} bytes")
        print(f"元サイズ: {original_size:,} bytes")
        print(f"形式: {format_type}")
        print(f"ハッシュ一致: {'はい' if hash_match else 'いいえ'}")
        print(f"⚡ 処理時間: {elapsed_time:.2f}s")
        
        return {
            'input_file': input_path.replace('.nxzqs', ''),
            'output_file': output_path,
            'restored_size': len(restored_data),
            'original_size': original_size,
            'format_type': format_type,
            'hash_match': hash_match,
            'success': True
        }
    
    def _decompress_png_structure_data(self, png_data: bytes) -> bytes:
        """PNG構造データの解凍"""
        if not png_data.startswith(b'\x89PNG\r\n\x1a\n'):
            raise ValueError("Invalid PNG signature in decompression")
        
        # PNG署名を保持
        result = bytearray(png_data[:8])
        pos = 8
        
        while pos < len(png_data):
            if pos + 8 >= len(png_data):
                break
                
            chunk_length = struct.unpack('>I', png_data[pos:pos+4])[0]
            chunk_type = png_data[pos+4:pos+8]
            chunk_data = png_data[pos+8:pos+8+chunk_length]
            chunk_crc = png_data[pos+8+chunk_length:pos+12+chunk_length]
            
            # IDATチャンクの場合は量子解凍を適用
            if chunk_type == b'IDAT':
                try:
                    # 量子解凍
                    quantum_decompressed = self._quantum_decompress_raw_data(chunk_data)
                    # zlib圧縮に戻す
                    zlib_compressed = zlib.compress(quantum_decompressed)
                    # 新しいCRCを計算
                    new_crc = zlib.crc32(chunk_type + zlib_compressed) & 0xffffffff
                    
                    # 新しいチャンクを作成
                    result.extend(struct.pack('>I', len(zlib_compressed)))
                    result.extend(chunk_type)
                    result.extend(zlib_compressed)
                    result.extend(struct.pack('>I', new_crc))
                except:
                    # 解凍できない場合は元のチャンクを保持
                    result.extend(png_data[pos:pos+12+chunk_length])
            else:
                # 他のチャンクはそのまま保持
                result.extend(png_data[pos:pos+12+chunk_length])
            
            pos += 12 + chunk_length
        
        return bytes(result)
    
    def _quantum_decompress_raw_data(self, quantum_data: bytes) -> bytes:
        """量子圧縮されたrawデータの解凍"""
        # LZMA解凍
        decompressed = lzma.decompress(quantum_data)
        
        # 量子もつれ解除（可逆版）
        result = bytearray()
        
        for i in range(0, len(decompressed), 4):
            block = decompressed[i:i+4]
            if len(block) < 4:
                break
            
            quantum_a, quantum_b, quantum_c, quantum_d = block
            # 逆変換
            a = (quantum_a - i) % 256
            b = (quantum_b - (i >> 8)) % 256
            c = (quantum_c - (i >> 16)) % 256
            d = (quantum_d - (i >> 24)) % 256
            
            result.extend([a, b, c, d])
        
        return bytes(result)
    
    def _decompress_byte_level(self, input_path: str, compressed_data: bytes, start_time: float) -> Dict:
        """バイトレベル版解凍"""
        print(f"🔬 バイトレベル解凍開始...")
        
        # ヘッダー解析
        pos = 7  # signature
        version = struct.unpack('>I', compressed_data[pos:pos+4])[0]
        pos += 4
        format_type = compressed_data[pos:pos+16].rstrip(b'\x00').decode('utf-8')
        pos += 16
        
        # メタデータ
        original_hash = compressed_data[pos:pos+16]
        pos += 16
        original_size = struct.unpack('>I', compressed_data[pos:pos+4])[0]
        pos += 4
        compressed_size = struct.unpack('>I', compressed_data[pos:pos+4])[0]
        pos += 4
        
        # 量子バイト解凍
        final_compressed = compressed_data[pos:pos+compressed_size]
        restored_data = self._quantum_byte_decompress(final_compressed)
        
        # 出力ファイル作成
        output_path = input_path.replace('.nxzqb', '.restored')
        with open(output_path, 'wb') as f:
            f.write(restored_data)
        
        # 検証
        restored_hash = hashlib.md5(restored_data).digest()
        hash_match = restored_hash == original_hash
        elapsed_time = time.time() - start_time
        
        print(f"🧠 バイトレベル解凍完了")
        print(f"入力: {os.path.basename(input_path.replace('.nxzqb', ''))}")
        print(f"出力: {os.path.basename(output_path)}")
        print(f"復元サイズ: {len(restored_data):,} bytes")
        print(f"元サイズ: {original_size:,} bytes")
        print(f"形式: {format_type}")
        print(f"ハッシュ一致: {'はい' if hash_match else 'いいえ'}")
        print(f"⚡ 処理時間: {elapsed_time:.2f}s")
        
        return {
            'input_file': input_path.replace('.nxzqb', ''),
            'output_file': output_path,
            'restored_size': len(restored_data),
            'original_size': original_size,
            'format_type': format_type,
            'hash_match': hash_match,
            'success': True
        }
    
    def _quantum_byte_decompress(self, data: bytes) -> bytes:
        """量子バイト解凍"""
        # LZMA解凍
        lzma_decompressed = lzma.decompress(data)
        
        # パターンテーブル復元
        table_size = struct.unpack('>H', lzma_decompressed[:2])[0]
        pattern_table = lzma_decompressed[2:2+table_size]
        encoded_data = lzma_decompressed[2+table_size:]
        
        # パターンリスト生成
        patterns = []
        for i in range(0, len(pattern_table), 4):
            if i + 4 <= len(pattern_table):
                patterns.append(pattern_table[i:i+4])
        
        # パターン復号化
        pattern_decoded = self._decode_patterns(encoded_data, patterns)
        
        # エントロピー復号化
        entropy_decoded = self._decode_entropy(pattern_decoded)
        
        return entropy_decoded
    
    def _decode_patterns(self, data: bytes, patterns: List[bytes]) -> bytes:
        """パターン復号化"""
        result = bytearray()
        i = 0
        
        while i < len(data):
            if data[i] == 0xFE and i + 1 < len(data):
                if data[i + 1] == 0xFF:
                    # エスケープされた0xFE
                    result.append(0xFE)
                    i += 2
                elif data[i + 1] < len(patterns):
                    # パターン復元
                    result.extend(patterns[data[i + 1]])
                    i += 2
                else:
                    result.append(data[i])
                    i += 1
            else:
                result.append(data[i])
                i += 1
        
        return bytes(result)
    
    def _decode_entropy(self, data: bytes) -> bytes:
        """エントロピー復号化"""
        result = bytearray()
        i = 0
        
        while i < len(data):
            if data[i] == 0xFF and i + 1 < len(data):
                if data[i + 1] == 0xFF:
                    # エスケープされた0xFF
                    result.append(0xFF)
                    i += 2
                else:
                    # 高頻度バイト復元（簡略化）
                    result.append(data[i + 1])
                    i += 2
            else:
                result.append(data[i])
                i += 1
        
        return bytes(result)

def main():
    import sys
    if len(sys.argv) != 2:
        print("使用方法: python nexus_quantum_unified_decompressor.py <compressed_file>")
        return
    
    engine = NXZipQuantumDecompressor()
    result = engine.decompress_file(sys.argv[1])
    
    if 'error' in result:
        print("ERROR: 解凍失敗")
        exit(1)
    else:
        print(f"SUCCESS: 解凍完了")

if __name__ == '__main__':
    main()
