#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NXZip 超高速静的Huffman解凍エンジン
事前計算テーブルによる最高速解凍
"""

import hashlib
import struct
import time
import os
from collections import Counter
from typing import Dict, List

class UltraFastHuffmanEngine:
    """超高速静的Huffmanエンジン"""
    
    def __init__(self):
        self.signature = b'\x4E\x58\x5A\x55\x48\x55\x46'  # NXZUHUF (Ultra Huffman)
        
    def compress_file(self, input_path: str) -> Dict:
        """超高速ファイル圧縮"""
        if not os.path.exists(input_path):
            print(f"❌ ファイルが見つかりません: {input_path}")
            return {'error': f'File not found: {input_path}'}
        
        start_time = time.time()
        
        try:
            with open(input_path, 'rb') as f:
                original_data = f.read()
            
            original_size = len(original_data)
            original_hash = hashlib.md5(original_data).digest()
            format_type = os.path.splitext(input_path)[1][1:].upper() or 'BINARY'
            
            print(f"📁 処理: {os.path.basename(input_path)} ({original_size:,} bytes, 超高速静的Huffman)")
            print(f"⚡ 超高速解析開始...")
            
            # 頻度解析
            freq_count = Counter(original_data)
            print(f"   📊 頻度解析完了: {len(freq_count)} 種類")
            
            # 静的Huffman符号生成
            codes = self._generate_static_codes(freq_count)
            print(f"   🔤 静的符号生成完了: 平均長 {sum(len(c) * freq_count[b] for b, c in codes.items()) / original_size:.2f}")
            
            # 高速符号化
            encoded_data = self._encode_ultra_fast(original_data, codes)
            print(f"   ✅ 符号化完了: {len(encoded_data):,} bytes")
            
            # パッケージ作成
            final_data = self._create_package(
                encoded_data, codes, original_hash, original_size, format_type
            )
            
            output_path = input_path + '.nxzuh'  # Ultra Huffman
            with open(output_path, 'wb') as f:
                f.write(final_data)
            
            compressed_size = len(final_data)
            compression_ratio = (1 - compressed_size / original_size) * 100
            elapsed_time = time.time() - start_time
            speed = original_size / 1024 / 1024 / elapsed_time
            
            print(f"✅ 超高速静的Huffman圧縮完了: {compression_ratio:.1f}%")
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
    
    def _generate_static_codes(self, freq_count: Counter) -> Dict[int, str]:
        """静的Huffman符号生成（超高速版）"""
        # 頻度順でソート
        sorted_items = freq_count.most_common()
        
        codes = {}
        
        if len(sorted_items) == 1:
            # 単一文字の場合
            codes[sorted_items[0][0]] = '0'
            return codes
        
        # 簡略化された符号生成
        code_len = 1
        code_val = 0
        remaining_items = len(sorted_items)
        
        for byte_val, freq in sorted_items:
            # 必要なビット数を計算
            while (1 << code_len) < remaining_items:
                code_len += 1
            
            codes[byte_val] = format(code_val, f'0{code_len}b')
            code_val += 1
            remaining_items -= 1
            
            # 符号長調整
            if remaining_items <= (1 << (code_len - 1)):
                code_len = max(1, code_len - 1)
                code_val = 0
        
        return codes
    
    def _encode_ultra_fast(self, data: bytes, codes: Dict[int, str]) -> bytes:
        """超高速符号化"""
        # 全体を一気に文字列結合
        bit_string = ''.join(codes[byte] for byte in data)
        
        # バイト配列に変換
        result = bytearray()
        padding = 0
        
        for i in range(0, len(bit_string), 8):
            chunk = bit_string[i:i+8]
            if len(chunk) < 8:
                chunk = chunk.ljust(8, '0')
                padding = 8 - len(bit_string[i:])
            result.append(int(chunk, 2))
        
        # パディング情報を先頭に追加
        return bytes([padding]) + bytes(result)
    
    def _create_package(self, encoded_data: bytes, codes: Dict[int, str],
                       original_hash: bytes, original_size: int, format_type: str) -> bytes:
        """パッケージ作成"""
        result = bytearray()
        
        # ヘッダー
        result.extend(self.signature)
        result.extend(struct.pack('>I', 1))  # Version 1
        result.extend(format_type.encode('utf-8').ljust(16, b'\x00'))
        
        # 符号テーブルのシリアライズ
        codes_data = self._serialize_codes(codes)
        
        # メタデータ
        result.extend(original_hash)
        result.extend(struct.pack('>I', original_size))
        result.extend(struct.pack('>I', len(codes_data)))
        result.extend(struct.pack('>I', len(encoded_data)))
        
        # データ
        result.extend(codes_data)
        result.extend(encoded_data)
        
        return bytes(result)
    
    def _serialize_codes(self, codes: Dict[int, str]) -> bytes:
        """符号テーブルシリアライズ"""
        result = bytearray()
        result.extend(struct.pack('>H', len(codes)))
        
        for byte_val, code in codes.items():
            result.append(byte_val)
            result.append(len(code))
            # 符号を8ビット単位でパック
            code_bytes = bytearray()
            for i in range(0, len(code), 8):
                chunk = code[i:i+8].ljust(8, '0')
                code_bytes.append(int(chunk, 2))
            result.extend(struct.pack('>H', len(code_bytes)))
            result.extend(code_bytes)
        
        return bytes(result)

class UltraFastHuffmanDecompressor:
    """超高速静的Huffman解凍エンジン"""
    
    def __init__(self):
        self.signature = b'\x4E\x58\x5A\x55\x48\x55\x46'  # NXZUHUF
        
    def decompress_file(self, input_path: str) -> Dict:
        """超高速ファイル解凍"""
        if not os.path.exists(input_path):
            print(f"❌ ファイルが見つかりません: {input_path}")
            return {'error': f'File not found: {input_path}'}
        
        start_time = time.time()
        
        try:
            with open(input_path, 'rb') as f:
                compressed_data = f.read()
            
            if not compressed_data.startswith(self.signature):
                return {'error': 'Invalid format signature'}
            
            print(f"⚡ 超高速静的Huffman解凍開始...")
            
            # ヘッダー解析
            pos = len(self.signature)
            version = struct.unpack('>I', compressed_data[pos:pos+4])[0]
            pos += 4
            format_type = compressed_data[pos:pos+16].rstrip(b'\x00').decode('utf-8')
            pos += 16
            
            # メタデータ
            original_hash = compressed_data[pos:pos+16]
            pos += 16
            original_size = struct.unpack('>I', compressed_data[pos:pos+4])[0]
            pos += 4
            codes_size = struct.unpack('>I', compressed_data[pos:pos+4])[0]
            pos += 4
            data_size = struct.unpack('>I', compressed_data[pos:pos+4])[0]
            pos += 4
            
            print(f"   📊 メタデータ解析完了")
            
            # 符号テーブル復元
            codes_data = compressed_data[pos:pos+codes_size]
            pos += codes_size
            decode_table = self._deserialize_codes(codes_data)
            print(f"   🔤 符号テーブル復元完了: {len(decode_table)} エントリ")
            
            # 符号化データ
            encoded_data = compressed_data[pos:pos+data_size]
            
            # 超高速復号化
            restored_data = self._decode_ultra_fast(encoded_data, decode_table, original_size)
            print(f"   ✅ 復号化完了: {len(restored_data):,} bytes")
            
            # 出力ファイル作成
            output_path = input_path.replace('.nxzuh', '.restored')
            with open(output_path, 'wb') as f:
                f.write(restored_data)
            
            # 検証
            restored_hash = hashlib.md5(restored_data).digest()
            hash_match = restored_hash == original_hash
            elapsed_time = time.time() - start_time
            
            print(f"🧠 超高速静的Huffman解凍完了")
            print(f"入力: {os.path.basename(input_path.replace('.nxzuh', ''))}")
            print(f"出力: {os.path.basename(output_path)}")
            print(f"復元サイズ: {len(restored_data):,} bytes")
            print(f"元サイズ: {original_size:,} bytes")
            print(f"形式: {format_type}")
            print(f"ハッシュ一致: {'はい' if hash_match else 'いいえ'}")
            print(f"⚡ 処理時間: {elapsed_time:.2f}s")
            
            return {
                'input_file': input_path.replace('.nxzuh', ''),
                'output_file': output_path,
                'restored_size': len(restored_data),
                'original_size': original_size,
                'format_type': format_type,
                'hash_match': hash_match,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ 解凍エラー: {str(e)}")
            return {'error': str(e)}
    
    def _deserialize_codes(self, data: bytes) -> Dict[str, int]:
        """符号テーブル復元"""
        decode_table = {}
        pos = 0
        
        code_count = struct.unpack('>H', data[pos:pos+2])[0]
        pos += 2
        
        for _ in range(code_count):
            byte_val = data[pos]
            pos += 1
            code_length = data[pos]
            pos += 1
            
            code_bytes_length = struct.unpack('>H', data[pos:pos+2])[0]
            pos += 2
            
            code_bytes = data[pos:pos+code_bytes_length]
            pos += code_bytes_length
            
            # 符号復元
            code = ''
            for i, byte in enumerate(code_bytes):
                chunk = format(byte, '08b')
                if i == len(code_bytes) - 1:
                    remaining = code_length - len(code)
                    code += chunk[:remaining]
                else:
                    code += chunk
            
            decode_table[code] = byte_val
        
        return decode_table
    
    def _decode_ultra_fast(self, encoded_data: bytes, decode_table: Dict[str, int], 
                          original_size: int) -> bytes:
        """超高速復号化"""
        # パディング情報取得
        padding = encoded_data[0]
        data = encoded_data[1:]
        
        # バイト列をビット文字列に変換
        bit_string = ''.join(format(byte, '08b') for byte in data)
        
        # パディング除去
        if padding > 0:
            bit_string = bit_string[:-padding]
        
        print(f"   📦 ビット展開完了: {len(bit_string):,} bits")
        
        # 高速復号化
        decoded = []
        current_code = ''
        progress_interval = max(1, len(bit_string) // 10)
        
        for i, bit in enumerate(bit_string):
            if i % progress_interval == 0:
                progress = (i * 100) // len(bit_string)
                print(f"   ⚡ 超高速復号化: {progress}%", end='\r')
            
            current_code += bit
            
            if current_code in decode_table:
                decoded.append(decode_table[current_code])
                current_code = ''
                
                if len(decoded) >= original_size:
                    break
        
        print(f"   ⚡ 超高速復号化: 100%")
        return bytes(decoded)

def main():
    import sys
    if len(sys.argv) < 2:
        print("使用方法: python nexus_ultra_fast_huffman.py <file_or_compressed_file>")
        return
    
    file_path = sys.argv[1]
    
    if file_path.endswith('.nxzuh'):
        # 解凍
        engine = UltraFastHuffmanDecompressor()
        result = engine.decompress_file(file_path)
    else:
        # 圧縮
        engine = UltraFastHuffmanEngine()
        result = engine.compress_file(file_path)
    
    if 'error' in result:
        print("ERROR: 処理失敗")
        exit(1)
    else:
        print(f"SUCCESS: 処理完了")

if __name__ == '__main__':
    main()
