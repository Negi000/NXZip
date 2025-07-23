#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NXZip ハイブリッド解凍エンジン
LZ77 + FGK適応型Huffmanの解凍システム
"""

import hashlib
import struct
import time
import os
from typing import Dict, List, Tuple

# FGK適応型Huffmanクラスをインポート（同じ実装）
class Node:
    def __init__(self, symbol=None, weight=0, left=None, right=None, parent=None):
        self.symbol = symbol
        self.weight = weight
        self.left = left
        self.right = right
        self.parent = parent

class FGKAdaptiveHuffman:
    def __init__(self):
        self.nyt = Node('NYT', 0)
        self.root = self.nyt
        self.nodes = [self.nyt]
        self.symbol_to_node = {}

    def get_code(self, symbol):
        if symbol == 'NYT':
            node = self.nyt
        else:
            node = self.symbol_to_node.get(symbol)
            if node is None:
                return None
        
        code = ''
        while node.parent is not None:
            if node == node.parent.left:
                code = '0' + code
            else:
                code = '1' + code
            node = node.parent
        return code

    def find_highest_node_with_weight(self, weight):
        for n in reversed(self.nodes):
            if n.weight == weight:
                return n
        return None

    def swap_nodes(self, n1, n2):
        if n1 == n2:
            return
        
        idx1 = self.nodes.index(n1)
        idx2 = self.nodes.index(n2)
        self.nodes[idx1], self.nodes[idx2] = n2, n1
        
        p1 = n1.parent
        p2 = n2.parent
        
        if p1 and p1.left == n1:
            p1.left = n2
        elif p1:
            p1.right = n2
            
        if p2 and p2.left == n2:
            p2.left = n1
        elif p2:
            p2.right = n1
            
        n1.parent, n2.parent = p2, p1

    def update_tree(self, node):
        while node is not None:
            block_leader = self.find_highest_node_with_weight(node.weight)
            if (block_leader != node and 
                block_leader.parent != node and 
                node.parent != block_leader):
                self.swap_nodes(node, block_leader)
            node.weight += 1
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
                if parent.left == self.nyt:
                    parent.left = new_internal
                else:
                    parent.right = new_internal
                new_internal.parent = parent
            
            self.nodes.insert(0, new_internal)
            self.nodes.insert(0, new_leaf)
            self.symbol_to_node[symbol] = new_leaf
            self.nyt = new_internal.left
            self.update_tree(new_internal)

    def decode(self, encoded_bits: List[int]) -> bytes:
        decoded = []
        current_node = self.root
        i = 0
        progress_interval = max(1, len(encoded_bits) // 20)
        
        while i < len(encoded_bits):
            if i % progress_interval == 0:
                progress = (i * 100) // len(encoded_bits)
                print(f"   🔓 FGK復号化: {progress}%", end='\r')
            
            # ツリー走査
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
            
            # シンボル処理
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
                self.insert_symbol(symbol)
            else:
                break
            
            current_node = self.root
        
        print(f"   🔓 FGK復号化: 100%")
        return bytes(decoded)

def deserialize_lz77(serialized: bytes) -> List[Tuple]:
    """LZ77シリアライズ復元"""
    output = []
    i = 0
    progress_interval = max(1, len(serialized) // 20)
    
    while i < len(serialized):
        if i % progress_interval == 0:
            progress = (i * 100) // len(serialized)
            print(f"   📋 LZ77復元: {progress}%", end='\r')
        
        flag = serialized[i]
        i += 1
        
        if flag == 0:  # マッチ
            if i + 2 < len(serialized):
                offset = (serialized[i] << 8) | serialized[i+1]
                length = serialized[i+2]
                output.append((0, offset, length))
                i += 3
            else:
                break
        else:  # リテラル
            if i < len(serialized):
                literal = serialized[i]
                output.append((1, literal, 0))
                i += 1
            else:
                break
    
    print(f"   📋 LZ77復元: 100%")
    return output

def lz77_decompress(output: List[Tuple]) -> bytes:
    """LZ77解凍"""
    data = bytearray()
    progress_interval = max(1, len(output) // 20)
    
    for i, (flag, val1, val2) in enumerate(output):
        if i % progress_interval == 0:
            progress = (i * 100) // len(output)
            print(f"   📖 LZ77解凍: {progress}%", end='\r')
        
        if flag == 0:  # マッチ
            offset, length = val1, val2
            start = len(data) - offset
            for _ in range(length):
                if start < len(data):
                    data.append(data[start])
                    start += 1
                else:
                    break
        else:  # リテラル
            data.append(val1)
    
    print(f"   📖 LZ77解凍: 100%")
    return bytes(data)

def bytes_to_bits(byte_data: bytes) -> List[int]:
    """バイト配列からビットリストに変換"""
    return [(byte >> (7-i)) & 1 for byte in byte_data for i in range(8)]

class NXZipHybridDecompressor:
    """NXZipハイブリッド解凍エンジン"""
    
    def __init__(self):
        self.signature = b'\x4E\x58\x5A\x48\x59\x42\x52'  # NXZHYBR
        
    def decompress_file(self, input_path: str) -> Dict:
        """ハイブリッド解凍"""
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
            
            print(f"🔬 ハイブリッド解凍開始...")
            
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
            data_size = struct.unpack('>I', compressed_data[pos:pos+4])[0]
            pos += 4
            
            print(f"   📊 メタデータ解析完了")
            
            # 圧縮データ抽出
            payload = compressed_data[pos:pos+data_size]
            print(f"   📦 データ抽出完了: {len(payload):,} bytes")
            
            # フェーズ1: ビット列に変換
            encoded_bits = bytes_to_bits(payload)
            # 実際のビット数まで切り詰め
            encoded_bits = encoded_bits[:bit_count]
            print(f"   🔤 ビット展開完了: {len(encoded_bits):,} bits")
            
            # フェーズ2: FGK適応型Huffman復号化
            fgk_decoder = FGKAdaptiveHuffman()
            serialized = fgk_decoder.decode(encoded_bits)
            print(f"   ✅ FGK復号化完了: {len(serialized):,} bytes")
            
            # フェーズ3: LZ77復元
            lz_output = deserialize_lz77(serialized)
            print(f"   ✅ LZ77復元完了: {len(lz_output):,} トークン")
            
            # フェーズ4: LZ77解凍
            restored_data = lz77_decompress(lz_output)
            print(f"   ✅ 解凍完了: {len(restored_data):,} bytes")
            
            # 出力ファイル作成
            output_path = input_path.replace('.nxzhb', '.restored')
            with open(output_path, 'wb') as f:
                f.write(restored_data)
            
            # 検証
            restored_hash = hashlib.md5(restored_data).digest()
            hash_match = restored_hash == original_hash
            elapsed_time = time.time() - start_time
            
            print(f"🧠 ハイブリッド解凍完了")
            print(f"入力: {os.path.basename(input_path.replace('.nxzhb', ''))}")
            print(f"出力: {os.path.basename(output_path)}")
            print(f"復元サイズ: {len(restored_data):,} bytes")
            print(f"元サイズ: {original_size:,} bytes")
            print(f"形式: {format_type}")
            print(f"ハッシュ一致: {'はい' if hash_match else 'いいえ'}")
            print(f"⚡ 処理時間: {elapsed_time:.2f}s")
            
            return {
                'input_file': input_path.replace('.nxzhb', ''),
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
        print("使用方法: python nexus_hybrid_decompressor.py <compressed_file>")
        return
    
    engine = NXZipHybridDecompressor()
    result = engine.decompress_file(sys.argv[1])
    
    if 'error' in result:
        print("ERROR: 解凍失敗")
        exit(1)
    else:
        print(f"SUCCESS: 解凍完了")

if __name__ == '__main__':
    main()
