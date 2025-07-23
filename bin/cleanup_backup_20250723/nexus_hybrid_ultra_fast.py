#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NXZip 超高速ハイブリッド圧縮エンジン
LZ77 + 静的Huffmanによる軽量高速圧縮システム
"""

import hashlib
import struct
import time
import os
import heapq
from typing import Dict, List, Tuple
from collections import Counter

def simple_lz77(data: bytes, window: int = 512, max_len: int = 32) -> List[Tuple]:
    """超軽量LZ77"""
    i = 0
    output = []
    
    while i < len(data):
        best_len = 0
        best_offset = 0
        
        start = max(0, i - window)
        for j in range(start, i):
            length = 0
            while (i + length < len(data) and 
                   j + length < i and 
                   data[j + length] == data[i + length] and 
                   length < max_len):
                length += 1
            
            if length > best_len and length >= 3:
                best_len = length
                best_offset = i - j
        
        if best_len >= 3:
            output.append((0, best_offset, best_len))
            i += best_len
        else:
            output.append((1, data[i], 0))
            i += 1
    
    return output

def simple_huffman_encode(data: bytes) -> Tuple[bytes, Dict]:
    """超軽量Huffman符号化"""
    if not data:
        return b'', {}
    
    freq = Counter(data)
    
    # 単一文字の場合
    if len(freq) == 1:
        symbol = list(freq.keys())[0]
        return bytes([len(data) & 0xFF, len(data) >> 8]), {symbol: '0'}
    
    # 2文字の場合
    if len(freq) == 2:
        symbols = list(freq.keys())
        codes = {symbols[0]: '0', symbols[1]: '1'}
    else:
        # 通常のHuffman
        heap = [[f, s] for s, f in freq.items()]
        heapq.heapify(heap)
        
        counter = 0
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            heapq.heappush(heap, [lo[0] + hi[0], counter])
            counter += 1
        
        # 簡易符号割り当て
        codes = {}
        for i, (symbol, _) in enumerate(freq.items()):
            codes[symbol] = format(i, f'0{min(8, len(freq).bit_length())}b')
    
    # 符号化
    encoded_str = ''.join(codes.get(b, '0') for b in data)
    
    # パディング
    padding = 8 - (len(encoded_str) % 8) if len(encoded_str) % 8 else 0
    encoded_str += '0' * padding
    
    # バイト変換
    encoded_bytes = bytearray()
    for i in range(0, len(encoded_str), 8):
        byte = int(encoded_str[i:i+8], 2)
        encoded_bytes.append(byte)
    
    return bytes(encoded_bytes), codes

def pack_lz77(tokens: List[Tuple]) -> bytes:
    """LZ77トークンパッキング"""
    result = bytearray()
    
    for flag, val1, val2 in tokens:
        if flag == 0:  # マッチ
            result.extend([0, val1 & 0xFF, val1 >> 8, val2])
        else:  # リテラル
            result.extend([1, val1])
    
    return bytes(result)

def serialize_huffman_dict(codes: Dict) -> bytes:
    """Huffman辞書シリアライズ"""
    result = bytearray()
    result.append(len(codes))
    
    for symbol, code in codes.items():
        result.append(symbol)
        result.append(len(code))
        # 符号をバイト単位でパック
        padded = code.ljust((len(code) + 7) // 8 * 8, '0')
        for i in range(0, len(padded), 8):
            result.append(int(padded[i:i+8], 2))
    
    return bytes(result)

class NXZipUltraFast:
    """NXZip超高速圧縮エンジン"""
    
    def __init__(self):
        self.signature = b'NXZUF'  # NXZip Ultra Fast
    
    def compress_file(self, input_path: str) -> Dict:
        """超高速圧縮"""
        if not os.path.exists(input_path):
            return {'error': f'File not found: {input_path}'}
        
        start_time = time.time()
        
        try:
            with open(input_path, 'rb') as f:
                data = f.read()
            
            original_size = len(data)
            original_hash = hashlib.md5(data).digest()
            
            # 小さなファイルは直接圧縮
            if original_size <= 32768:  # 32KB以下
                tokens = simple_lz77(data, window=256, max_len=16)
                packed = pack_lz77(tokens)
                compressed, codes = simple_huffman_encode(packed)
                dict_data = serialize_huffman_dict(codes)
                
                chunks = [(compressed, dict_data, len(packed))]
            else:
                # チャンク処理（64KB）
                chunk_size = 65536
                chunks = []
                
                for i in range(0, original_size, chunk_size):
                    chunk = data[i:i + chunk_size]
                    tokens = simple_lz77(chunk, window=512, max_len=32)
                    packed = pack_lz77(tokens)
                    compressed, codes = simple_huffman_encode(packed)
                    dict_data = serialize_huffman_dict(codes)
                    chunks.append((compressed, dict_data, len(packed)))
            
            # パッケージ作成
            final_data = self._create_package(chunks, original_hash, original_size)
            
            output_path = input_path + '.nxz'
            with open(output_path, 'wb') as f:
                f.write(final_data)
            
            compressed_size = len(final_data)
            ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
            elapsed = time.time() - start_time
            speed = original_size / 1024 / 1024 / elapsed if elapsed > 0 else 0
            
            return {
                'success': True,
                'input_file': input_path,
                'output_file': output_path,
                'original_size': original_size,
                'compressed_size': compressed_size,
                'compression_ratio': ratio,
                'processing_time': elapsed,
                'compression_speed': speed
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _create_package(self, chunks: List, original_hash: bytes, original_size: int) -> bytes:
        """パッケージ作成"""
        result = bytearray()
        
        # ヘッダー
        result.extend(self.signature)
        result.extend(struct.pack('>I', 1))  # バージョン
        result.extend(original_hash)
        result.extend(struct.pack('>I', original_size))
        result.extend(struct.pack('>I', len(chunks)))
        
        # チャンクデータ
        for compressed, dict_data, original_chunk_size in chunks:
            result.extend(struct.pack('>I', original_chunk_size))
            result.extend(struct.pack('>I', len(dict_data)))
            result.extend(struct.pack('>I', len(compressed)))
            result.extend(dict_data)
            result.extend(compressed)
        
        return bytes(result)

def main():
    import sys
    if len(sys.argv) != 2:
        print("使用方法: python nexus_hybrid_ultra_fast.py <file>")
        return
    
    engine = NXZipUltraFast()
    result = engine.compress_file(sys.argv[1])
    
    if 'error' in result:
        print(f"ERROR: {result['error']}")
        exit(1)
    else:
        print(f"SUCCESS: {result['output_file']}")
        print(f"Original: {result['original_size']:,} bytes → {result['compressed_size']:,} bytes")
        print(f"Ratio: {result['compression_ratio']:.1f}%, Speed: {result['compression_speed']:.1f} MB/s")

if __name__ == '__main__':
    main()
