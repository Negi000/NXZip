#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NXZip ハイブリッド圧縮エンジン (超高速・高圧縮版)
LZ77 + FGK適応型Huffmanによる超高効率圧縮システム
"""

import hashlib
import struct
import time
import os
from typing import Dict, List, Tuple, Union
from collections import defaultdict

class Node:
    """FGK適応型Huffmanツリーノード"""
    def __init__(self, symbol=None, weight=0, left=None, right=None, parent=None):
        self.symbol = symbol
        self.weight = weight
        self.left = left
        self.right = right
        self.parent = parent

class FGKAdaptiveHuffman:
    """FGK適応型Huffman符号化器（最適化版）"""
    
    def __init__(self, initial_freqs=None):
        self.nyt = Node('NYT', 0)
        self.root = self.nyt
        self.nodes = [self.nyt]
        self.symbol_to_node = {}
        self.weight_to_nodes = defaultdict(list)
        self.weight_to_nodes[0].append(self.nyt)
        if initial_freqs:
            for symbol, freq in initial_freqs.items():
                self.insert_symbol(symbol)
                if symbol in self.symbol_to_node:
                    self.weight_to_nodes[0].remove(self.symbol_to_node[symbol])
                    self.symbol_to_node[symbol].weight = freq
                    self.weight_to_nodes[freq].append(self.symbol_to_node[symbol])

    def get_code(self, symbol):
        if symbol == 'NYT':
            node = self.nyt
        else:
            node = self.symbol_to_node.get(symbol)
            if node is None:
                return None
        
        code = bytearray()
        while node.parent is not None:
            if node == node.parent.left:
                code.append(0)
            else:
                code.append(1)
            node = node.parent
        return code

    def find_highest_node_with_weight(self, weight):
        if weight in self.weight_to_nodes and self.weight_to_nodes[weight]:
            return self.weight_to_nodes[weight][-1]
        return None

    def swap_nodes(self, n1, n2):
        if n1 == n2 or n1 is None or n2 is None:
            return
        
        if n1 in self.weight_to_nodes[n1.weight]:
            self.weight_to_nodes[n1.weight].remove(n1)
        if n2 in self.weight_to_nodes[n2.weight]:
            self.weight_to_nodes[n2.weight].remove(n2)
        
        idx1 = self.nodes.index(n1)
        idx2 = self.nodes.index(n2)
        self.nodes[idx1], self.nodes[idx2] = n2, n1
        
        p1 = n1.parent
        p2 = n2.parent
        
        if p1:
            if p1.left == n1:
                p1.left = n2
            else:
                p1.right = n2
            
        if p2:
            if p2.left == n2:
                p2.left = n1
            else:
                p2.right = n1
            
        n1.parent, n2.parent = p2, p1
        
        self.weight_to_nodes[n1.weight].append(n1)
        self.weight_to_nodes[n2.weight].append(n2)

    def update_tree(self, node):
        while node is not None:
            if node in self.weight_to_nodes[node.weight]:
                self.weight_to_nodes[node.weight].remove(node)
            
            block_leader = self.find_highest_node_with_weight(node.weight)
            if (block_leader and block_leader != node and 
                block_leader.parent != node and 
                node.parent != block_leader):
                self.swap_nodes(node, block_leader)
            
            node.weight += 1
            self.weight_to_nodes[node.weight].append(node)
            node = node.parent

    def insert_symbol(self, symbol):
        if symbol in self.symbol_to_node:
            leaf = self.symbol_to_node[symbol]
            self.update_tree(leaf)
        else:
            new_leaf = Node(symbol, 0)
            new_internal = Node(None, 0, self.nyt, new_leaf)
            new_leaf.parent = new_internal
            
            parent = self.nyt.parent
            self.nyt.parent = new_internal
            
            if self.root == self.nyt:
                self.root = new_internal
            else:
                if parent and parent.left == self.nyt:
                    parent.left = new_internal
                elif parent:
                    parent.right = new_internal
                new_internal.parent = parent
            
            self.nodes.insert(0, new_internal)
            self.nodes.insert(0, new_leaf)
            self.weight_to_nodes[new_internal.weight].append(new_internal)
            self.weight_to_nodes[new_leaf.weight].append(new_leaf)
            self.symbol_to_node[symbol] = new_leaf
            self.nyt = new_internal.left
            self.update_tree(new_internal)

    def encode(self, data: bytes) -> List[int]:
        encoded = bytearray()
        update_count = 0
        max_updates = 1000  # 適応更新を制限
        
        for byte_val in data:
            code = self.get_code(byte_val)
            if code is None:
                nyt_code = self.get_code('NYT')
                encoded.extend(nyt_code)
                encoded.extend(int(b) for b in format(byte_val, '08b'))
            else:
                encoded.extend(code)
            if update_count < max_updates:
                self.insert_symbol(byte_val)
                update_count += 1
        
        encoded.extend(self.get_code('NYT'))
        return list(encoded)

    def decode(self, encoded_bits: List[int]) -> bytes:
        decoded = bytearray()
        current_node = self.root
        i = 0
        update_count = 0
        max_updates = 1000
        
        while i < len(encoded_bits):
            while (current_node.left is not None or 
                   current_node.right is not None):
                if i >= len(encoded_bits):
                    break
                bit = encoded_bits[i]
                i += 1
                if bit == 0:
                    current_node = current_node.left
                else:
                    current_node = current_node.right
            
            if current_node.symbol == 'NYT':
                if i + 8 > len(encoded_bits):
                    break
                symbol_bits = encoded_bits[i:i+8]
                symbol = sum(bit << (7-j) for j, bit in enumerate(symbol_bits))
                i += 8
            else:
                symbol = current_node.symbol
            
            if symbol != 'NYT':
                decoded.append(symbol)
                if update_count < max_updates:
                    self.insert_symbol(symbol)
                    update_count += 1
            else:
                break
            
            current_node = self.root
        
        return bytes(decoded)

def lz77_compress(data: bytes, window_size: int = 2048, max_match_len: int = 258, min_match: int = 4) -> List[Tuple]:
    """LZ77圧縮（超高速ハッシュ版）"""
    i = 0
    output = []
    hash_table = [[] for _ in range(65536)]  # 固定サイズハッシュ
    
    while i < len(data):
        match_len = 0
        match_offset = 0
        
        if i + 2 < len(data):
            h = ((data[i] << 8) | data[i+1]) & 0xFFFF
            candidates = [p for p in hash_table[h] if i - window_size <= p < i]
            for j in sorted(candidates, reverse=True)[:10]:  # 上限10
                k = 2
                while i + k < len(data) and data[j + k] == data[i + k] and k < max_match_len:
                    k += 1
                if k > match_len:
                    match_len = k
                    match_offset = i - j
                if match_len >= 16:
                    break
        
        if match_len >= min_match:
            output.append((0, match_offset, match_len))
            for offset in range(match_len):
                if i + offset + 2 < len(data):
                    h = ((data[i+offset] << 8) | data[i+offset+1]) & 0xFFFF
                    hash_table[h].append(i + offset)
                    if len(hash_table[h]) > 10:
                        hash_table[h].pop(0)
            i += match_len
        else:
            output.append((1, data[i], 0))
            if i + 2 < len(data):
                h = ((data[i] << 8) | data[i+1]) & 0xFFFF
                hash_table[h].append(i)
                if len(hash_table[h]) > 10:
                    hash_table[h].pop(0)
            i += 1
    
    return output

def lz77_decompress(output: List[Tuple]) -> bytes:
    """LZ77解凍（超高速版）"""
    data = bytearray()
    for flag, val1, val2 in output:
        if flag == 0:
            offset, length = val1, val2
            start = len(data) - offset
            data.extend(data[start:start + length])
        else:
            data.append(val1)
    return bytes(data)

def serialize_lz77(output: List[Tuple]) -> bytes:
    """LZ77出力シリアライズ（超高速版）"""
    serialized = bytearray(len(output) * 4)
    pos = 0
    
    for flag, val1, val2 in output:
        serialized[pos] = flag
        pos += 1
        if flag == 0:
            serialized[pos] = val1 >> 8
            serialized[pos+1] = val1 & 255
            serialized[pos+2] = val2
            pos += 3
        else:
            serialized[pos] = val1
            pos += 1
    
    return serialized[:pos]

def deserialize_lz77(serialized: bytes) -> List[Tuple]:
    """LZ77シリアライズ復元"""
    output = []
    i = 0
    
    while i < len(serialized):
        flag = serialized[i]
        i += 1
        
        if flag == 0:
            offset = (serialized[i] << 8) | serialized[i+1]
            length = serialized[i+2]
            output.append((0, offset, length))
            i += 3
        else:
            literal = serialized[i]
            output.append((1, literal, 0))
            i += 1
    
    return output

def bits_to_bytes(bits: List[int]) -> bytes:
    """ビットリストをバイト配列に変換（バッファリング）"""
    byte_arr = bytearray((len(bits) + 7) // 8)
    pos = 0
    byte = 0
    bit_count = 0
    
    for bit in bits:
        byte = (byte << 1) | bit
        bit_count += 1
        if bit_count == 8:
            byte_arr[pos] = byte
            pos += 1
            byte = 0
            bit_count = 0
    
    if bit_count:
        byte <<= (8 - bit_count)
        byte_arr[pos] = byte
    
    return bytes(byte_arr)

def bytes_to_bits(byte_data: bytes) -> List[int]:
    """バイト配列からビットリストに変換"""
    bits = []
    for byte in byte_data:
        for j in range(7, -1, -1):
            bits.append((byte >> j) & 1)
    return bits

class NXZipHybridEngine:
    """NXZipハイブリッド圧縮エンジン"""
    
    def __init__(self):
        self.signature = b'\x4E\x58\x5A\x48\x59\x42\x52'
    
    def compress_file(self, input_path: str, chunk_size: int = 1024*1024) -> Dict:
        """ハイブリッド圧縮（チャンク処理）"""
        if not os.path.exists(input_path):
            return {'error': f'File not found: {input_path}'}
        
        start_time = time.time()
        
        try:
            with open(input_path, 'rb') as f:
                original_data = f.read()
            
            original_size = len(original_data)
            original_hash = hashlib.md5(original_data).digest()
            format_type = os.path.splitext(input_path)[1][1:].upper() or 'BINARY'
            
            # PNG向け初期頻度
            initial_freqs = {i: 10 if i in range(128, 256) else 1 for i in range(256)} if format_type == 'PNG' else {i: 1 for i in range(256)}
            window_size = 4096 if format_type == 'PNG' else 2048
            
            # チャンク処理
            compressed_chunks = []
            chunk_count = (original_size + chunk_size - 1) // chunk_size
            for i in range(0, original_size, chunk_size):
                chunk = original_data[i:i + chunk_size]
                lz_output = lz77_compress(chunk, window_size=window_size)
                serialized = serialize_lz77(lz_output)
                fgk = FGKAdaptiveHuffman(initial_freqs)
                encoded_bits = fgk.encode(serialized)
                compressed = bits_to_bytes(encoded_bits)
                compressed_chunks.append((compressed, len(encoded_bits)))
            
            # パッケージ作成
            final_data = self._create_package(
                compressed_chunks, original_hash, original_size, 
                format_type, chunk_count
            )
            
            output_path = input_path + '.nxz'
            with open(output_path, 'wb') as f:
                f.write(final_data)
            
            compressed_size = len(final_data)
            compression_ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
            elapsed_time = time.time() - start_time
            speed = original_size / 1024 / 1024 / elapsed_time if elapsed_time > 0 else 0
            
            return {
                'success': True,
                'input_file': input_path,
                'output_file': output_path,
                'original_size': original_size,
                'compressed_size': compressed_size,
                'compression_ratio': compression_ratio,
                'processing_time': elapsed_time,
                'compression_speed': speed
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _create_package(self, compressed_chunks: List[Tuple[bytes, int]], 
                       original_hash: bytes, original_size: int, 
                       format_type: str, chunk_count: int) -> bytes:
        result = bytearray()
        
        result.extend(self.signature)
        result.extend(struct.pack('>I', 1))
        result.extend(format_type.encode('utf-8').ljust(16, b'\x00'))
        
        result.extend(original_hash)
        result.extend(struct.pack('>I', original_size))
        result.extend(struct.pack('>I', chunk_count))
        
        for compressed, bit_count in compressed_chunks:
            result.extend(struct.pack('>I', bit_count))
            result.extend(struct.pack('>I', len(compressed)))
            result.extend(compressed)
        
        return bytes(result)

def main():
    import sys
    if len(sys.argv) != 2:
        print("使用方法: python nexus_hybrid_engine.py <file>")
        return
    
    engine = NXZipHybridEngine()
    result = engine.compress_file(sys.argv[1])
    
    if 'error' in result:
        print(f"ERROR: 圧縮失敗 - {result['error']}")
        exit(1)
    else:
        print(f"SUCCESS: 圧縮完了 - {result['output_file']}")
        print(f"Original: {result['original_size']:,} bytes, Compressed: {result['compressed_size']:,} bytes")
        print(f"Ratio: {result['compression_ratio']:.1f}%, Speed: {result['compression_speed']:.1f} MB/s")

if __name__ == '__main__':
    main()