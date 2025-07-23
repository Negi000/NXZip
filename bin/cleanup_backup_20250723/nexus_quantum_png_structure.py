#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NXZip量子圧縮エンジン - PNG構造保存版
PNG構造を完全保持してzlib圧縮部分のみを量子圧縮で置換
"""

import hashlib
import struct
import time
import zlib
import os
import lzma
from typing import Dict, List, Tuple

class NXZipQuantumPNGStructure:
    def __init__(self):
        self.signature = b'\x4E\x58\x5A\x51\x50\x4E\x47'  # NXZQPNG
        
    def compress_file(self, input_path: str) -> Dict:
        """PNG構造保存型量子圧縮"""
        if not os.path.exists(input_path):
            print(f"❌ ファイルが見つかりません: {input_path}")
            return {'error': f'File not found: {input_path}'}
        
        start_time = time.time()
        
        try:
            with open(input_path, 'rb') as f:
                original_data = f.read()
            
            original_size = len(original_data)
            original_hash = hashlib.md5(original_data).digest()
            
            print(f"📁 処理: {os.path.basename(input_path)} ({original_size:,} bytes, PNG構造保存)")
            
            # PNG構造を解析して圧縮部分のみを処理
            compressed_data = self._compress_png_structure(original_data)
            
            # メタデータ付きで最終ファイル生成
            final_data = self._create_final_package(
                compressed_data, original_hash, original_size, 'PNG_STRUCT'
            )
            
            output_path = input_path + '.nxz'  # Standard NXZip format
            with open(output_path, 'wb') as f:
                f.write(final_data)
            
            compressed_size = len(final_data)
            compression_ratio = (1 - compressed_size / original_size) * 100
            elapsed_time = time.time() - start_time
            speed = original_size / 1024 / 1024 / elapsed_time
            
            print(f"✅ PNG構造保存圧縮完了: {compression_ratio:.1f}%")
            print(f"⚡ 処理時間: {elapsed_time:.2f}s ({speed:.1f} MB/s)")
            print(f"💾 保存: {os.path.basename(output_path)}")
            
            return {
                'success': True,
                'input_file': input_path,
                'output_file': output_path,
                'original_size': original_size,
                'compressed_size': compressed_size,
                'compression_ratio': compression_ratio,
                'processing_time': elapsed_time
            }
            
        except Exception as e:
            print(f"❌ 圧縮エラー: {str(e)}")
            return {'error': str(e)}
    
    def _compress_png_structure(self, png_data: bytes) -> bytes:
        """PNG構造を保持してzlib部分のみを量子圧縮"""
        if not png_data.startswith(b'\x89PNG\r\n\x1a\n'):
            raise ValueError("Invalid PNG signature")
        
        # PNG署名を保持
        result = bytearray(png_data[:8])
        pos = 8
        
        while pos < len(png_data):
            # チャンク長とタイプを読み取り
            if pos + 8 >= len(png_data):
                break
                
            chunk_length = struct.unpack('>I', png_data[pos:pos+4])[0]
            chunk_type = png_data[pos+4:pos+8]
            chunk_data = png_data[pos+8:pos+8+chunk_length]
            chunk_crc = png_data[pos+8+chunk_length:pos+12+chunk_length]
            
            # IDATチャンクの場合は量子圧縮を適用
            if chunk_type == b'IDAT':
                # zlibデータを解凍
                try:
                    raw_data = zlib.decompress(chunk_data)
                    # 量子圧縮を適用
                    quantum_compressed = self._quantum_compress_raw_data(raw_data)
                    # 新しいCRCを計算
                    new_crc = zlib.crc32(chunk_type + quantum_compressed) & 0xffffffff
                    
                    # 新しいチャンクを作成
                    result.extend(struct.pack('>I', len(quantum_compressed)))
                    result.extend(chunk_type)
                    result.extend(quantum_compressed)
                    result.extend(struct.pack('>I', new_crc))
                except:
                    # 解凍できない場合は元のチャンクを保持
                    result.extend(png_data[pos:pos+12+chunk_length])
            else:
                # 他のチャンクはそのまま保持
                result.extend(png_data[pos:pos+12+chunk_length])
            
            pos += 12 + chunk_length
        
        return bytes(result)
    
    def _quantum_compress_raw_data(self, raw_data: bytes) -> bytes:
        """生データに量子圧縮を適用（高速版・LZMA不使用）"""
        # 段階的圧縮アプローチ
        
        # ステップ1: 量子エンタングルメント（可逆版）
        quantum_data = bytearray()
        for i in range(0, len(raw_data), 4):
            block = raw_data[i:i+4]
            if len(block) < 4:
                block = block + b'\x00' * (4 - len(block))
            
            a, b, c, d = block
            # より強力な量子もつれ（多重変換）
            quantum_a = ((a + i) * 3 + (i % 7)) % 256
            quantum_b = ((b + (i >> 8)) * 5 + ((i >> 3) % 11)) % 256
            quantum_c = ((c + (i >> 16)) * 7 + ((i >> 6) % 13)) % 256
            quantum_d = ((d + (i >> 24)) * 11 + ((i >> 9) % 17)) % 256
            
            quantum_data.extend([quantum_a, quantum_b, quantum_c, quantum_d])
        
        # ステップ2: 差分予測（PNG画像データ向け）
        predicted_data = bytearray()
        for i in range(len(quantum_data)):
            if i == 0:
                predicted_data.append(quantum_data[i])
            else:
                # 複数予測の組み合わせ
                left = quantum_data[i-1]
                up = quantum_data[i-4] if i >= 4 else 0
                up_left = quantum_data[i-5] if i >= 5 else 0
                
                # Paeth予測
                p = left + up - up_left
                pa = abs(p - left)
                pb = abs(p - up)
                pc = abs(p - up_left)
                
                if pa <= pb and pa <= pc:
                    predictor = left
                elif pb <= pc:
                    predictor = up
                else:
                    predictor = up_left
                
                diff = (quantum_data[i] - predictor) % 256
                predicted_data.append(diff)
        
        # ステップ3: RLE（Run Length Encoding）
        rle_data = bytearray()
        i = 0
        while i < len(predicted_data):
            current = predicted_data[i]
            count = 1
            
            # 連続する同じ値をカウント
            while i + count < len(predicted_data) and predicted_data[i + count] == current and count < 255:
                count += 1
            
            if count >= 3:  # 3個以上なら圧縮
                rle_data.extend([0xFF, current, count])  # エスケープシーケンス
                i += count
            else:
                # エスケープ文字の処理
                if current == 0xFF:
                    rle_data.extend([0xFF, 0xFF, 1])  # エスケープ
                else:
                    rle_data.append(current)
                i += 1
        
        # ステップ4: 簡易Huffman符号化（頻度ベース）
        from collections import Counter
        freq = Counter(rle_data)
        
        if len(freq) <= 1:
            # 1種類のみの場合は特別処理
            return bytes([0x00]) + struct.pack('>I', len(rle_data)) + bytes([rle_data[0]]) if rle_data else b'\x00\x00\x00\x00\x00'
        
        # 頻度順ソート
        sorted_chars = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        
        # 簡易可変長符号（頻度に応じてビット長を決定）
        code_table = {}
        bit_lengths = [1, 2, 3, 4, 5, 6, 7, 8]  # 可能なビット長
        
        for i, (char, _) in enumerate(sorted_chars):
            if i < len(bit_lengths):
                bit_len = bit_lengths[i]
            else:
                bit_len = 8
            code_table[char] = (i % (2**bit_len), bit_len)
        
        # エンコード
        encoded_bits = []
        for byte_val in rle_data:
            code, bit_len = code_table[byte_val]
            encoded_bits.append((code, bit_len))
        
        # ビット列をバイトに変換
        result = bytearray()
        result.append(0x01)  # Huffman符号化フラグ
        result.extend(struct.pack('>H', len(code_table)))  # コードテーブルサイズ
        
        # コードテーブル保存
        for char, (code, bit_len) in code_table.items():
            result.extend(struct.pack('>BBB', char, code, bit_len))
        
        # エンコードデータサイズ
        result.extend(struct.pack('>I', len(encoded_bits)))
        
        # エンコードデータ
        bit_buffer = 0
        bit_count = 0
        
        for code, bit_len in encoded_bits:
            bit_buffer = (bit_buffer << bit_len) | code
            bit_count += bit_len
            
            while bit_count >= 8:
                result.append((bit_buffer >> (bit_count - 8)) & 0xFF)
                bit_count -= 8
        
        # 残りビット
        if bit_count > 0:
            result.append((bit_buffer << (8 - bit_count)) & 0xFF)
        
        # zlibでさらに圧縮（軽量設定）
        final_compressed = zlib.compress(bytes(result), level=6)
        
        return final_compressed
    
    def _create_final_package(self, compressed_data: bytes, original_hash: bytes, 
                            original_size: int, format_type: str) -> bytes:
        """最終パッケージ作成"""
        result = bytearray()
        
        # ヘッダー
        result.extend(self.signature)
        result.extend(struct.pack('>I', 2))  # Version 2
        result.extend(format_type.encode('utf-8').ljust(16, b'\x00'))
        
        # メタデータ
        result.extend(original_hash)
        result.extend(struct.pack('>I', original_size))
        result.extend(struct.pack('>I', len(compressed_data)))
        
        # 圧縮データ
        result.extend(compressed_data)
        
        return bytes(result)

def main():
    import sys
    if len(sys.argv) != 2:
        print("使用方法: python nexus_quantum_png_structure.py <png_file>")
        return
    
    engine = NXZipQuantumPNGStructure()
    result = engine.compress_file(sys.argv[1])
    
    if 'error' in result:
        print("ERROR: 圧縮失敗")
        exit(1)
    else:
        print(f"SUCCESS: 圧縮完了 - {result['output_file']}")

if __name__ == '__main__':
    main()
