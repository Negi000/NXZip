#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NXZip 超高速Huffman解凍エンジン
静的符号テーブルによる超高速解凍
"""

import hashlib
import struct
import time
import os
from typing import Dict, List

class NXZipFastHuffmanDecompressor:
    """NXZip超高速Huffman解凍エンジン"""
    
    def __init__(self):
        self.signature = b'\x4E\x58\x5A\x46\x48\x55\x46'  # NXZFHUF
        
    def decompress_file(self, input_path: str) -> Dict:
        """超高速ファイル解凍"""
        if not os.path.exists(input_path):
            print(f"❌ ファイルが見つかりません: {input_path}")
            return {'error': f'File not found: {input_path}'}
        
        start_time = time.time()
        
        try:
            with open(input_path, 'rb') as f:
                compressed_data = f.read()
            
            # 形式チェック
            if not compressed_data.startswith(self.signature):
                return {'error': 'Invalid format signature'}
            
            print(f"⚡ 超高速Huffman解凍開始...")
            
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
            bit_count = struct.unpack('>I', compressed_data[pos:pos+4])[0]
            pos += 4
            code_table_size = struct.unpack('>I', compressed_data[pos:pos+4])[0]
            pos += 4
            data_size = struct.unpack('>I', compressed_data[pos:pos+4])[0]
            pos += 4
            
            print(f"   📊 メタデータ解析完了")
            
            # 符号テーブル復元
            code_table_data = compressed_data[pos:pos+code_table_size]
            pos += code_table_size
            decode_table = self._deserialize_codes(code_table_data)
            print(f"   🔤 符号テーブル復元完了: {len(decode_table)} エントリ")
            
            # 圧縮データ抽出
            payload = compressed_data[pos:pos+data_size]
            print(f"   📦 データ抽出完了: {len(payload):,} bytes")
            
            # ビット列に変換
            encoded_bits = self._bytes_to_bits(payload, bit_count)
            print(f"   🔤 ビット展開完了: {len(encoded_bits):,} bits")
            
            # 超高速復号化
            restored_data = self._decode_fast(encoded_bits, decode_table, original_size)
            print(f"   ✅ 復号化完了: {len(restored_data):,} bytes")
            
            # 出力ファイル作成
            output_path = input_path.replace('.nxzfh', '.restored')
            with open(output_path, 'wb') as f:
                f.write(restored_data)
            
            # 検証
            restored_hash = hashlib.md5(restored_data).digest()
            hash_match = restored_hash == original_hash
            elapsed_time = time.time() - start_time
            
            print(f"🧠 超高速Huffman解凍完了")
            print(f"入力: {os.path.basename(input_path.replace('.nxzfh', ''))}")
            print(f"出力: {os.path.basename(output_path)}")
            print(f"復元サイズ: {len(restored_data):,} bytes")
            print(f"元サイズ: {original_size:,} bytes")
            print(f"形式: {format_type}")
            print(f"ハッシュ一致: {'はい' if hash_match else 'いいえ'}")
            print(f"⚡ 処理時間: {elapsed_time:.2f}s")
            
            return {
                'input_file': input_path.replace('.nxzfh', ''),
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
            
            # バイト列から符号復元
            code = ''
            for i, byte in enumerate(code_bytes):
                chunk = format(byte, '08b')
                if i == len(code_bytes) - 1:
                    # 最後のバイトは実際の長さまで
                    remaining = code_length - len(code)
                    code += chunk[:remaining]
                else:
                    code += chunk
            
            decode_table[code] = byte_val
        
        return decode_table
    
    def _bytes_to_bits(self, data: bytes, bit_count: int) -> List[int]:
        """バイト列からビット列変換（指定長まで）"""
        bits = []
        for byte in data:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
                if len(bits) >= bit_count:
                    return bits
        return bits
    
    def _decode_fast(self, encoded_bits: List[int], decode_table: Dict[str, int], 
                    original_length: int) -> bytes:
        """超高速復号化"""
        decoded = []
        current_code = ''
        progress_interval = max(1, len(encoded_bits) // 10)
        
        for i, bit in enumerate(encoded_bits):
            if i % progress_interval == 0:
                progress = (i * 100) // len(encoded_bits)
                print(f"   ⚡ 高速復号化: {progress}%", end='\r')
            
            current_code += str(bit)
            
            if current_code in decode_table:
                decoded.append(decode_table[current_code])
                current_code = ''
                
                if len(decoded) >= original_length:
                    break
        
        print(f"   ⚡ 高速復号化: 100%")
        return bytes(decoded)

def main():
    import sys
    if len(sys.argv) != 2:
        print("使用方法: python nexus_lightning_fast_huffman_decompressor.py <compressed_file>")
        return
    
    engine = NXZipFastHuffmanDecompressor()
    result = engine.decompress_file(sys.argv[1])
    
    if 'error' in result:
        print("ERROR: 解凍失敗")
        exit(1)
    else:
        print(f"SUCCESS: 解凍完了")

if __name__ == '__main__':
    main()
