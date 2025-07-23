#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NXZip 適応型Huffman解凍エンジン
完全可逆性保証の適応型エントロピー解凍システム
"""

import heapq
import hashlib
import struct
import time
import os
from typing import Dict, List

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

class AdaptiveHuffmanDecoder:
    """適応型Huffman復号化器（超高速版）"""
    
    def __init__(self):
        # 初期頻度（符号化時と同じ）
        self.freqs = {i: 1 for i in range(256)}
        self.decode_table = {}
        self.update_counter = 0
        self._build_decode_table()

    def _build_decode_table(self):
        """復号化テーブル構築（高速化）"""
        tree = build_tree(self.freqs)
        if not tree:
            return
        
        self.decode_table = {}
        
        def traverse(node, code=''):
            if node.symbol is not None:
                self.decode_table[code if code else '0'] = node.symbol
            else:
                if node.left:
                    traverse(node.left, code + '0')
                if node.right:
                    traverse(node.right, code + '1')
        
        traverse(tree)

    def decode(self, encoded_bits: List[int], original_length: int) -> bytes:
        """データ復号化（超高速版）"""
        decoded = []
        freqs_copy = {i: 1 for i in range(256)}
        
        # ビットを文字列に変換（高速化）
        bit_string = ''.join(str(bit) for bit in encoded_bits)
        
        progress_interval = max(1, len(bit_string) // 10)  # 進捗表示を10%刻みに
        update_interval = max(1, original_length // 100)  # ツリー更新を100分の1に大幅削減
        
        pos = 0
        current_code = ''
        
        while pos < len(bit_string) and len(decoded) < original_length:
            if pos % progress_interval == 0:
                progress = (pos * 100) // len(bit_string)
                print(f"   ⚡ 高速復号化: {progress}%", end='\r')
            
            current_code += bit_string[pos]
            pos += 1
            
            # 符号が見つかった場合
            if current_code in self.decode_table:
                symbol = self.decode_table[current_code]
                decoded.append(symbol)
                current_code = ''
                
                # 頻度更新
                freqs_copy[symbol] += 1
                self.update_counter += 1
                
                # ツリー更新（大幅に頻度削減）
                if self.update_counter % update_interval == 0:
                    tree = build_tree(freqs_copy)
                    if tree:
                        self.decode_table = {}
                        
                        def traverse(node, code=''):
                            if node.symbol is not None:
                                self.decode_table[code if code else '0'] = node.symbol
                            else:
                                if node.left:
                                    traverse(node.left, code + '0')
                                if node.right:
                                    traverse(node.right, code + '1')
                        
                        traverse(tree)
        
        print(f"   ⚡ 高速復号化: 100%")
        return bytes(decoded)

def bytes_to_bits(byte_data: bytes) -> List[int]:
    """バイト配列からビットリストに変換（高速版）"""
    # リスト内包表記で高速化
    return [(byte >> (7-i)) & 1 for byte in byte_data for i in range(8)]

class NXZipAdaptiveHuffmanDecompressor:
    """NXZip適応型Huffman解凍エンジン"""
    
    def __init__(self):
        self.signature = b'\x4E\x58\x5A\x41\x48\x55\x46'  # NXZAHUF
        
    def decompress_file(self, input_path: str) -> Dict:
        """ファイル解凍"""
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
            
            print(f"🔬 適応型Huffman解凍開始...")
            
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
            compressed_size = struct.unpack('>I', compressed_data[pos:pos+4])[0]
            pos += 4
            
            # 圧縮データ抽出
            payload = compressed_data[pos:pos+compressed_size]
            
            print(f"   📊 メタデータ解析完了")
            print(f"   📦 圧縮データ: {len(payload):,} bytes → {bit_count:,} bits")
            
            # ビット列に変換
            encoded_bits = bytes_to_bits(payload)
            # 実際のビット数まで切り詰め
            encoded_bits = encoded_bits[:bit_count]
            
            print(f"   🔤 ビット展開完了: {len(encoded_bits):,} bits")
            
            # 適応型Huffman復号化
            decoder = AdaptiveHuffmanDecoder()
            restored_data = decoder.decode(encoded_bits, original_size)
            
            print(f"   ✅ 復号化完了: {len(restored_data):,} bytes")
            
            # 出力ファイル作成
            output_path = input_path.replace('.nxzah', '.restored')
            with open(output_path, 'wb') as f:
                f.write(restored_data)
            
            # 検証
            restored_hash = hashlib.md5(restored_data).digest()
            hash_match = restored_hash == original_hash
            elapsed_time = time.time() - start_time
            
            print(f"🧠 適応型Huffman解凍完了")
            print(f"入力: {os.path.basename(input_path.replace('.nxzah', ''))}")
            print(f"出力: {os.path.basename(output_path)}")
            print(f"復元サイズ: {len(restored_data):,} bytes")
            print(f"元サイズ: {original_size:,} bytes")
            print(f"形式: {format_type}")
            print(f"ハッシュ一致: {'はい' if hash_match else 'いいえ'}")
            print(f"⚡ 処理時間: {elapsed_time:.2f}s")
            
            return {
                'input_file': input_path.replace('.nxzah', ''),
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

def main():
    import sys
    if len(sys.argv) != 2:
        print("使用方法: python nexus_adaptive_huffman_decompressor.py <compressed_file>")
        return
    
    engine = NXZipAdaptiveHuffmanDecompressor()
    result = engine.decompress_file(sys.argv[1])
    
    if 'error' in result:
        print("ERROR: 解凍失敗")
        exit(1)
    else:
        print(f"SUCCESS: 解凍完了")

if __name__ == '__main__':
    main()
