#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NXZip 適応型Huffman符号化エンジン
完全可逆性保証の適応型エントロピー圧縮システム
"""

import heapq
import hashlib
import struct
import time
import os
from typing import Dict, List, Tuple

class Node:
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

def build_tree(freqs):
    """Huffmanツリー構築"""
    priority_queue = [Node(symbol, freq) for symbol, freq in freqs.items() if freq > 0]
    heapq.heapify(priority_queue)
    while len(priority_queue) > 1:
        right = heapq.heappop(priority_queue)
        left = heapq.heappop(priority_queue)
        merged = Node(None, left.freq + right.freq, left, right)
        heapq.heappush(priority_queue, merged)
    return priority_queue[0] if priority_queue else None

class AdaptiveHuffman:
    """適応型Huffman符号化器（最適化版）"""
    
    def __init__(self):
        # 初期頻度（全バイト値に1を設定）
        self.freqs = {i: 1 for i in range(256)}
        self.tree = build_tree(self.freqs)
        self.codes = {}
        self.update_counter = 0
        self._build_codes()

    def _build_codes(self):
        """符号テーブル構築"""
        self.codes = {}
        if not self.tree:
            return
        
        def traverse(node, code=''):
            if node.symbol is not None:
                self.codes[node.symbol] = code if code else '0'
            else:
                if node.left:
                    traverse(node.left, code + '0')
                if node.right:
                    traverse(node.right, code + '1')
        
        traverse(self.tree)

    def update_tree(self):
        """ツリー更新（頻度削減版）"""
        self.tree = build_tree(self.freqs)
        self._build_codes()

    def encode(self, data: bytes) -> List[int]:
        """データ符号化（最適化版）"""
        encoded = []
        progress_interval = max(1, len(data) // 20)  # 進捗表示を5%刻みに削減
        update_interval = max(1, len(data) // 1000)  # ツリー更新を1000分の1に削減
        
        for i, byte_val in enumerate(data):
            # 進捗表示（削減）
            if i % progress_interval == 0:
                progress = (i * 100) // len(data)
                print(f"   📊 符号化進捗: {progress}%", end='\r')
            
            # 符号取得
            code = self.codes.get(byte_val, '0')
            for bit in code:
                encoded.append(int(bit))
            
            # 頻度更新
            self.freqs[byte_val] += 1
            self.update_counter += 1
            
            # ツリー再構築（頻度を大幅削減）
            if self.update_counter % update_interval == 0:
                self.update_tree()
        
        print(f"   📊 符号化進捗: 100%")
        return encoded

    def decode(self, encoded_bits: List[int], length: int) -> bytes:
        """データ復号化（最適化版）"""
        decoded = []
        freqs_copy = {i: 1 for i in range(256)}
        current_tree = build_tree(freqs_copy)
        
        if not current_tree:
            return b''
        
        current_node = current_tree
        progress_interval = max(1, len(encoded_bits) // 20)  # 進捗表示削減
        update_counter = 0
        update_interval = max(1, length // 1000)  # ツリー更新頻度削減
        
        bit_index = 0
        while bit_index < len(encoded_bits) and len(decoded) < length:
            if bit_index % progress_interval == 0:
                progress = (bit_index * 100) // len(encoded_bits)
                print(f"   📊 復号化進捗: {progress}%", end='\r')
            
            # 単一ノードの場合
            if current_node.symbol is not None:
                symbol = current_node.symbol
                decoded.append(symbol)
                freqs_copy[symbol] += 1
                update_counter += 1
                
                # ツリー更新（頻度削減）
                if update_counter % update_interval == 0:
                    current_tree = build_tree(freqs_copy)
                
                current_node = current_tree
                continue
            
            bit = encoded_bits[bit_index]
            bit_index += 1
            
            # ツリー移動
            if bit == 0:
                current_node = current_node.left if current_node.left else current_tree
            else:
                current_node = current_node.right if current_node.right else current_tree
            
            # リーフノードに到達
            if current_node and current_node.symbol is not None:
                symbol = current_node.symbol
                decoded.append(symbol)
                freqs_copy[symbol] += 1
                update_counter += 1
                
                # ツリー更新（頻度削減）
                if update_counter % update_interval == 0:
                    current_tree = build_tree(freqs_copy)
                
                current_node = current_tree
        
        print(f"   📊 復号化進捗: 100%")
        return bytes(decoded)

def bits_to_bytes(bits: List[int]) -> bytes:
    """ビットリストをバイト配列に変換"""
    byte_arr = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            if i + j < len(bits):
                byte = (byte << 1) | bits[i + j]
            else:
                byte = (byte << 1)
        byte_arr.append(byte)
    return bytes(byte_arr)

def bytes_to_bits(byte_data: bytes) -> List[int]:
    """バイト配列からビットリストに変換"""
    bits = []
    for b in byte_data:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits

class NXZipAdaptiveHuffmanEngine:
    """NXZip適応型Huffmanエンジン"""
    
    def __init__(self):
        self.signature = b'\x4E\x58\x5A\x41\x48\x55\x46'  # NXZAHUF
        
    def compress_file(self, input_path: str) -> Dict:
        """ファイル圧縮"""
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
            
            print(f"📁 処理: {os.path.basename(input_path)} ({original_size:,} bytes, 適応型Huffman)")
            print(f"🔬 適応型エントロピー解析開始...")
            
            # 適応型Huffman符号化
            huffman = AdaptiveHuffman()
            encoded_bits = huffman.encode(original_data)
            print(f"   ✅ 符号化完了: {len(encoded_bits):,} bits")
            
            # ビットをバイトに変換
            compressed_data = bits_to_bytes(encoded_bits)
            print(f"   📦 パッキング完了: {len(compressed_data):,} bytes")
            
            # パッケージ作成
            final_data = self._create_package(
                compressed_data, original_hash, original_size, 
                len(encoded_bits), format_type
            )
            
            output_path = input_path + '.nxzah'  # Adaptive Huffman
            with open(output_path, 'wb') as f:
                f.write(final_data)
            
            compressed_size = len(final_data)
            compression_ratio = (1 - compressed_size / original_size) * 100
            elapsed_time = time.time() - start_time
            speed = original_size / 1024 / 1024 / elapsed_time
            
            print(f"✅ 適応型Huffman圧縮完了: {compression_ratio:.1f}%")
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
    
    def _create_package(self, compressed_data: bytes, original_hash: bytes, 
                       original_size: int, bit_count: int, format_type: str) -> bytes:
        """パッケージ作成"""
        result = bytearray()
        
        # ヘッダー
        result.extend(self.signature)
        result.extend(struct.pack('>I', 1))  # Version 1
        result.extend(format_type.encode('utf-8').ljust(16, b'\x00'))
        
        # メタデータ
        result.extend(original_hash)
        result.extend(struct.pack('>I', original_size))
        result.extend(struct.pack('>I', bit_count))  # 元のビット数
        result.extend(struct.pack('>I', len(compressed_data)))
        
        # 圧縮データ
        result.extend(compressed_data)
        
        return bytes(result)

def main():
    import sys
    if len(sys.argv) != 2:
        print("使用方法: python nexus_adaptive_huffman.py <file>")
        return
    
    engine = NXZipAdaptiveHuffmanEngine()
    result = engine.compress_file(sys.argv[1])
    
    if 'error' in result:
        print("ERROR: 圧縮失敗")
        exit(1)
    else:
        print(f"SUCCESS: 圧縮完了 - {result['output_file']}")

if __name__ == '__main__':
    main()
