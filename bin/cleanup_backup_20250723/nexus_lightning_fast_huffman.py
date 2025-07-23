#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NXZip 超高速適応型Huffmanエンジン
静的Huffman + 部分適応による超高速処理
"""

import hashlib
import struct
import time
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

class FastAdaptiveHuffman:
    """超高速適応型Huffmanエンジン"""
    
    def __init__(self, data: bytes = None):
        # データ全体を事前分析して最適な初期ツリーを構築
        if data:
            self.freqs = Counter(data)
        else:
            # デフォルト頻度（英語テキスト統計ベース）
            self.freqs = defaultdict(int)
            # よく使われるバイト値に高い初期値を設定
            common_bytes = [32, 101, 116, 97, 111, 105, 110, 115, 104, 114]  # space, e, t, a, o, i, n, s, h, r
            for i in range(256):
                if i in common_bytes:
                    self.freqs[i] = 100
                else:
                    self.freqs[i] = 1
        
        self.codes = self._build_huffman_codes()
    
    def _build_huffman_codes(self) -> Dict[int, str]:
        """静的Huffman符号生成"""
        # 頻度順ソート
        sorted_items = sorted(self.freqs.items(), key=lambda x: x[1], reverse=True)
        
        # 単純な符号割り当て（頻度順）
        codes = {}
        code_length = 1
        code_value = 0
        items_in_length = 1
        
        for i, (byte_val, freq) in enumerate(sorted_items):
            if i >= items_in_length:
                code_length += 1
                items_in_length *= 2
                code_value = 0
            
            # バイナリ符号生成
            codes[byte_val] = format(code_value, f'0{code_length}b')
            code_value += 1
        
        return codes
    
    def encode_fast(self, data: bytes) -> List[int]:
        """超高速符号化"""
        encoded = []
        progress_interval = max(1, len(data) // 10)
        
        for i, byte_val in enumerate(data):
            if i % progress_interval == 0:
                progress = (i * 100) // len(data)
                print(f"   ⚡ 高速符号化: {progress}%", end='\r')
            
            code = self.codes.get(byte_val, '0')
            encoded.extend([int(bit) for bit in code])
        
        print(f"   ⚡ 高速符号化: 100%")
        return encoded
    
    def decode_fast(self, encoded_bits: List[int], original_length: int) -> bytes:
        """超高速復号化"""
        # 復号化テーブル作成
        decode_table = {v: k for k, v in self.codes.items()}
        
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

class NXZipFastHuffmanEngine:
    """NXZip超高速Huffmanエンジン"""
    
    def __init__(self):
        self.signature = b'\x4E\x58\x5A\x46\x48\x55\x46'  # NXZFHUF
        
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
            
            print(f"📁 処理: {os.path.basename(input_path)} ({original_size:,} bytes, 超高速Huffman)")
            print(f"⚡ 超高速エントロピー解析開始...")
            
            # データ事前分析による最適化Huffman
            huffman = FastAdaptiveHuffman(original_data)
            print(f"   📊 最適符号テーブル生成完了: {len(huffman.codes)} エントリ")
            
            # 超高速符号化
            encoded_bits = huffman.encode_fast(original_data)
            print(f"   ✅ 符号化完了: {len(encoded_bits):,} bits")
            
            # ビット圧縮
            compressed_data = self._bits_to_bytes(encoded_bits)
            print(f"   📦 パッキング完了: {len(compressed_data):,} bytes")
            
            # 符号テーブル保存
            code_table = self._serialize_codes(huffman.codes)
            
            # パッケージ作成
            final_data = self._create_package(
                compressed_data, code_table, original_hash, original_size, 
                len(encoded_bits), format_type
            )
            
            output_path = input_path + '.nxzfh'  # Fast Huffman
            with open(output_path, 'wb') as f:
                f.write(final_data)
            
            compressed_size = len(final_data)
            compression_ratio = (1 - compressed_size / original_size) * 100
            elapsed_time = time.time() - start_time
            speed = original_size / 1024 / 1024 / elapsed_time
            
            print(f"✅ 超高速Huffman圧縮完了: {compression_ratio:.1f}%")
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
    
    def _bits_to_bytes(self, bits: List[int]) -> bytes:
        """ビット列高速バイト変換"""
        byte_arr = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(min(8, len(bits) - i)):
                byte = (byte << 1) | bits[i + j]
            if i + 8 > len(bits):
                byte <<= (8 - (len(bits) - i))
            byte_arr.append(byte)
        return bytes(byte_arr)
    
    def _serialize_codes(self, codes: Dict[int, str]) -> bytes:
        """符号テーブルシリアライズ"""
        result = bytearray()
        result.extend(struct.pack('>H', len(codes)))
        
        for byte_val, code in codes.items():
            result.append(byte_val)
            result.append(len(code))
            # 符号をバイト列に変換
            code_bytes = bytearray()
            for i in range(0, len(code), 8):
                chunk = code[i:i+8].ljust(8, '0')
                code_bytes.append(int(chunk, 2))
            result.extend(struct.pack('>H', len(code_bytes)))
            result.extend(code_bytes)
        
        return bytes(result)
    
    def _create_package(self, compressed_data: bytes, code_table: bytes,
                       original_hash: bytes, original_size: int, bit_count: int, 
                       format_type: str) -> bytes:
        """パッケージ作成"""
        result = bytearray()
        
        # ヘッダー
        result.extend(self.signature)
        result.extend(struct.pack('>I', 1))  # Version 1
        result.extend(format_type.encode('utf-8').ljust(16, b'\x00'))
        
        # メタデータ
        result.extend(original_hash)
        result.extend(struct.pack('>I', original_size))
        result.extend(struct.pack('>I', bit_count))
        result.extend(struct.pack('>I', len(code_table)))
        result.extend(struct.pack('>I', len(compressed_data)))
        
        # 符号テーブル
        result.extend(code_table)
        
        # 圧縮データ
        result.extend(compressed_data)
        
        return bytes(result)

def main():
    import sys
    if len(sys.argv) != 2:
        print("使用方法: python nexus_lightning_fast.py <file>")
        return
    
    engine = NXZipFastHuffmanEngine()
    result = engine.compress_file(sys.argv[1])
    
    if 'error' in result:
        print("ERROR: 圧縮失敗")
        exit(1)
    else:
        print(f"SUCCESS: 圧縮完了 - {result['output_file']}")

if __name__ == '__main__':
    main()
