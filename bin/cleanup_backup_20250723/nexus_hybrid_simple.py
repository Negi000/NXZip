#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NXZip シンプルハイブリッド圧縮エンジン
LZ77 + 静的Huffmanによる高速圧縮システム
"""

import hashlib
import struct
import time
import os
import heapq
from typing import Dict, List, Tuple
from collections import Counter, defaultdict

def lz77_compress_simple(data: bytes, window_size: int = 1024, max_match: int = 32, min_match: int = 3) -> List[Tuple]:
    """軽量LZ77圧縮"""
    i = 0
    output = []
    
    while i < len(data):
        match_len = 0
        match_offset = 0
        
        # 単純な線形検索（高速化のため範囲制限）
        start = max(0, i - window_size)
        for j in range(start, i):
            k = 0
            while (i + k < len(data) and 
                   j + k < i and 
                   data[j + k] == data[i + k] and 
                   k < max_match):
                k += 1
            
            if k >= min_match and k > match_len:
                match_len = k
                match_offset = i - j
        
        if match_len >= min_match:
            output.append((0, match_offset, match_len))
            i += match_len
        else:
            output.append((1, data[i], 0))
            i += 1
    
    return output

def lz77_decompress_simple(output: List[Tuple]) -> bytes:
    """軽量LZ77解凍"""
    data = bytearray()
    for flag, val1, val2 in output:
        if flag == 0:  # マッチ
            offset, length = val1, val2
            start = len(data) - offset
            for _ in range(length):
                data.append(data[start])
                start += 1
        else:  # リテラル
            data.append(val1)
    return bytes(data)

class SimpleHuffman:
    """シンプルHuffman符号化器"""
    
    def __init__(self):
        self.codes = {}
        self.tree = None
    
    def build_tree(self, data: bytes):
        """Huffmanツリー構築"""
        if not data:
            return
        
        # 頻度計算
        freq = Counter(data)
        
        # 単一文字の場合
        if len(freq) == 1:
            symbol = list(freq.keys())[0]
            self.codes[symbol] = '0'
            return
        
        # ヒープ作成
        heap = [[weight, [[symbol, ""]]] for symbol, weight in freq.items()]
        heapq.heapify(heap)
        
        # ツリー構築
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            
            for pair in lo[1:]:
                if len(pair) >= 2:  # 安全性チェック
                    pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                if len(pair) >= 2:  # 安全性チェック
                    pair[1] = '1' + pair[1]
            
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        
        # 符号辞書作成
        if heap and len(heap[0]) > 1:
            for pair in heap[0][1:]:
                if len(pair) >= 2:  # 安全性チェック
                    self.codes[pair[0]] = pair[1] if pair[1] else '0'
    
    def encode(self, data: bytes) -> str:
        """データ符号化"""
        if not self.codes:
            self.build_tree(data)
        
        encoded = ''
        for byte in data:
            encoded += self.codes.get(byte, '0')
        
        return encoded
    
    def decode(self, encoded: str) -> bytes:
        """データ復号化"""
        if not self.codes:
            return b''
        
        # 逆引き辞書作成
        reverse_codes = {code: symbol for symbol, code in self.codes.items()}
        
        decoded = bytearray()
        i = 0
        
        while i < len(encoded):
            for length in range(1, 33):  # 最大32bit
                if i + length > len(encoded):
                    break
                
                code = encoded[i:i+length]
                if code in reverse_codes:
                    decoded.append(reverse_codes[code])
                    i += length
                    break
            else:
                break
        
        return bytes(decoded)

def serialize_simple(output: List[Tuple]) -> bytes:
    """シンプルシリアライズ"""
    result = bytearray()
    
    for flag, val1, val2 in output:
        if flag == 0:  # マッチ
            result.extend([0, val1 & 0xFF, val1 >> 8, val2])
        else:  # リテラル
            result.extend([1, val1])
    
    return bytes(result)

def deserialize_simple(data: bytes) -> List[Tuple]:
    """シンプル復元"""
    output = []
    i = 0
    
    while i < len(data):
        flag = data[i]
        i += 1
        
        if flag == 0 and i + 2 < len(data):  # マッチ
            offset = data[i] | (data[i+1] << 8)
            length = data[i+2]
            output.append((0, offset, length))
            i += 3
        elif flag == 1 and i < len(data):  # リテラル
            literal = data[i]
            output.append((1, literal, 0))
            i += 1
        else:
            break
    
    return output

class NXZipHybridSimple:
    """NXZipシンプルハイブリッド圧縮エンジン"""
    
    def __init__(self, progress: bool = False):
        self.signature = b'\x4E\x58\x5A\x48\x53\x4D\x50'  # NXZHSMP
        self.progress = progress
    
    def compress_file(self, input_path: str) -> Dict:
        """シンプルハイブリッド圧縮"""
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
            
            if self.progress:
                print(f"📁 処理: {os.path.basename(input_path)} ({original_size:,} bytes)")
                print(f"🚀 シンプルハイブリッド圧縮開始...")
            
            # フェーズ1: LZ77圧縮
            if self.progress:
                print("   📦 LZ77圧縮中...")
            lz_output = lz77_compress_simple(original_data)
            
            # フェーズ2: シリアライズ
            if self.progress:
                print("   📦 シリアライズ中...")
            serialized = serialize_simple(lz_output)
            
            # フェーズ3: Huffman圧縮
            if self.progress:
                print("   🔤 Huffman符号化中...")
            huffman = SimpleHuffman()
            encoded_str = huffman.encode(serialized)
            
            # フェーズ4: ビットパッキング
            if self.progress:
                print("   📦 パッキング中...")
            
            # 8bit単位に変換
            padding = 8 - (len(encoded_str) % 8) if len(encoded_str) % 8 != 0 else 0
            encoded_str += '0' * padding
            
            compressed = bytearray()
            for i in range(0, len(encoded_str), 8):
                byte = int(encoded_str[i:i+8], 2)
                compressed.append(byte)
            
            # Huffman辞書をシリアライズ
            dict_data = self._serialize_huffman_dict(huffman.codes)
            
            # パッケージ作成
            final_data = self._create_package(
                bytes(compressed), dict_data, original_hash, 
                original_size, len(encoded_str), padding, format_type
            )
            
            output_path = input_path + '.nxz'  # Standard NXZip format
            with open(output_path, 'wb') as f:
                f.write(final_data)
            
            compressed_size = len(final_data)
            compression_ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
            elapsed_time = time.time() - start_time
            speed = original_size / 1024 / 1024 / elapsed_time if elapsed_time > 0 else 0
            
            if self.progress:
                print(f"✅ 圧縮完了: {compression_ratio:.1f}%")
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
    
    def _serialize_huffman_dict(self, codes: Dict) -> bytes:
        """Huffman辞書シリアライズ"""
        result = bytearray()
        result.extend(struct.pack('>H', len(codes)))  # 辞書サイズ
        
        for symbol, code in codes.items():
            result.append(symbol)  # シンボル
            result.append(len(code))  # 符号長
            # 符号をバイト単位でパック
            padded_code = code.ljust((len(code) + 7) // 8 * 8, '0')
            for i in range(0, len(padded_code), 8):
                result.append(int(padded_code[i:i+8], 2))
        
        return bytes(result)
    
    def _create_package(self, compressed_data: bytes, dict_data: bytes, 
                       original_hash: bytes, original_size: int, 
                       bit_count: int, padding: int, format_type: str) -> bytes:
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
        result.extend(struct.pack('>B', padding))
        result.extend(struct.pack('>I', len(dict_data)))
        result.extend(struct.pack('>I', len(compressed_data)))
        
        # 辞書データ
        result.extend(dict_data)
        
        # 圧縮データ
        result.extend(compressed_data)
        
        return bytes(result)

def main():
    import sys
    if len(sys.argv) != 2:
        print("使用方法: python nexus_hybrid_simple.py <file>")
        return
    
    engine = NXZipHybridSimple(progress=True)
    result = engine.compress_file(sys.argv[1])
    
    if 'error' in result:
        print("ERROR: 圧縮失敗")
        exit(1)
    else:
        print(f"SUCCESS: 圧縮完了 - {result['output_file']}")

if __name__ == '__main__':
    main()
