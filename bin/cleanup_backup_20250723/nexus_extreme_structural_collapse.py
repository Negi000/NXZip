#!/usr/bin/env python3
"""
Nexus Extreme Structural Collapse Compressor
極限構造崩壊圧縮エンジン - 完全可逆性保証

特徴:
- PNG/MP4/PDF等の内部構造を完全に崩壊
- バイト順序の完全再構築による極限圧縮
- 多次元ソーティングと相関解析
- 量子エンタングルメント風バイト関係構築
- 完全可逆性保証（MD5検証）
"""

import struct
import time
import hashlib
import os
import sys
import zlib
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import math

class StructuralCollapseEngine:
    def __init__(self):
        self.magic = b'NXSC'  # Nexus Structural Collapse
        self.version = 1
        
    def analyze_byte_correlations(self, data: bytes) -> Dict[str, any]:
        """バイト間相関の高度解析"""
        correlations = {}
        
        # 隣接バイト相関
        adjacent_pairs = defaultdict(int)
        for i in range(len(data) - 1):
            pair = (data[i], data[i+1])
            adjacent_pairs[pair] += 1
        
        # 距離別相関（2,3,4,8,16バイト間隔）
        distance_correlations = {}
        for dist in [2, 3, 4, 8, 16]:
            dist_pairs = defaultdict(int)
            for i in range(len(data) - dist):
                pair = (data[i], data[i+dist])
                dist_pairs[pair] += 1
            distance_correlations[dist] = dist_pairs
        
        # 周期性検出
        periodic_patterns = {}
        for period in [3, 4, 8, 16, 24, 32]:
            if len(data) >= period * 3:
                pattern_matches = 0
                for i in range(period, len(data) - period):
                    if data[i] == data[i-period] and data[i] == data[i+period]:
                        pattern_matches += 1
                periodic_patterns[period] = pattern_matches / max(1, len(data) - 2*period)
        
        return {
            'adjacent_pairs': dict(adjacent_pairs),
            'distance_correlations': distance_correlations,
            'periodic_patterns': periodic_patterns,
            'unique_bytes': len(set(data)),
            'entropy': self.calculate_entropy(data)
        }
    
    def calculate_entropy(self, data: bytes) -> float:
        """データエントロピー計算"""
        if not data:
            return 0.0
        
        byte_counts = defaultdict(int)
        for byte in data:
            byte_counts[byte] += 1
        
        entropy = 0.0
        total = len(data)
        for count in byte_counts.values():
            if count > 0:
                prob = count / total
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    def multi_dimensional_sort(self, data: bytes) -> Tuple[bytes, List[int]]:
        """多次元ソーティングによるバイト再配置"""
        if len(data) == 0:
            return b'', []
        
        # バイト値とその位置をペアにする
        byte_positions = [(data[i], i) for i in range(len(data))]
        
        # 複数のソート基準を適用
        # 1. バイト値
        # 2. 位置のmod値（周期性活用）
        # 3. 隣接バイトとの差分
        
        def sort_key(item):
            byte_val, pos = item
            
            # 隣接バイトとの差分計算
            prev_diff = abs(byte_val - data[pos-1]) if pos > 0 else 0
            next_diff = abs(byte_val - data[pos+1]) if pos < len(data)-1 else 0
            
            # 位置の周期性
            pos_mod8 = pos % 8
            pos_mod16 = pos % 16
            
            return (byte_val, prev_diff + next_diff, pos_mod8, pos_mod16, pos)
        
        sorted_pairs = sorted(byte_positions, key=sort_key)
        
        # ソート後のバイト列と元位置のインデックス
        sorted_bytes = bytes([pair[0] for pair in sorted_pairs])
        position_indices = [pair[1] for pair in sorted_pairs]
        
        return sorted_bytes, position_indices
    
    def quantum_entanglement_compression(self, data: bytes) -> Tuple[bytes, Dict]:
        """量子エンタングルメント風バイト関係圧縮"""
        if len(data) == 0:
            return b'', {}
        
        # バイト間の「エンタングルメント」関係を構築
        entangled_groups = defaultdict(list)
        
        # XOR関係でのグループ化
        for i in range(len(data)):
            xor_signature = 0
            
            # 周辺バイトとのXOR関係
            for offset in [-2, -1, 1, 2]:
                if 0 <= i + offset < len(data):
                    xor_signature ^= data[i + offset]
            
            # XOR署名でグループ化
            entangled_groups[xor_signature % 64].append((data[i], i))
        
        # 各グループ内でバイト値による再ソート
        compressed_data = bytearray()
        group_info = {}
        
        for group_id, group_bytes in entangled_groups.items():
            if group_bytes:
                # グループ内ソート
                sorted_group = sorted(group_bytes, key=lambda x: x[0])
                group_values = [item[0] for item in sorted_group]
                group_positions = [item[1] for item in sorted_group]
                
                # 差分圧縮
                if len(group_values) > 1:
                    first_val = group_values[0]
                    diffs = [first_val] + [(group_values[i] - group_values[i-1]) & 0xFF 
                                          for i in range(1, len(group_values))]
                    compressed_data.extend(diffs)
                else:
                    compressed_data.extend(group_values)
                
                group_info[group_id] = {
                    'size': len(group_values),
                    'positions': group_positions
                }
        
        return bytes(compressed_data), group_info
    
    def advanced_pattern_elimination(self, data: bytes) -> Tuple[bytes, Dict]:
        """高度なパターン除去と復元情報記録"""
        if len(data) == 0:
            return b'', {}
        
        # 反復パターンの検出と除去
        patterns = {}
        eliminated_data = bytearray(data)
        
        # 2-8バイトのパターンを検出
        for pattern_len in range(2, min(9, len(data) // 3)):
            pattern_positions = {}
            
            for i in range(len(data) - pattern_len + 1):
                pattern = data[i:i+pattern_len]
                pattern_key = pattern.hex()
                
                if pattern_key not in pattern_positions:
                    pattern_positions[pattern_key] = []
                pattern_positions[pattern_key].append(i)
            
            # 3回以上出現するパターンを圧縮対象とする
            for pattern_hex, positions in pattern_positions.items():
                if len(positions) >= 3:
                    pattern_bytes = bytes.fromhex(pattern_hex)
                    patterns[pattern_hex] = {
                        'length': pattern_len,
                        'positions': positions,
                        'data': pattern_bytes
                    }
        
        # パターン除去（長いパターンを優先）
        eliminated_positions = set()
        for pattern_hex, pattern_info in sorted(patterns.items(), 
                                              key=lambda x: x[1]['length'], reverse=True):
            valid_positions = [pos for pos in pattern_info['positions'] 
                             if not any(pos + i in eliminated_positions 
                                      for i in range(pattern_info['length']))]
            
            if len(valid_positions) >= 3:
                # パターンを一度だけ残し、他の位置は除去
                for pos in valid_positions[1:]:
                    for i in range(pattern_info['length']):
                        eliminated_positions.add(pos + i)
                
                patterns[pattern_hex]['eliminated_positions'] = valid_positions[1:]
        
        # 除去後のデータ生成
        remaining_data = bytearray()
        for i in range(len(data)):
            if i not in eliminated_positions:
                remaining_data.append(data[i])
        
        return bytes(remaining_data), patterns
    
    def extreme_compress(self, data: bytes) -> bytes:
        """極限構造崩壊圧縮"""
        if not data:
            return self.magic + struct.pack('>I', 0)
        
        original_md5 = hashlib.md5(data).hexdigest()
        
        # ステップ1: 相関解析
        correlations = self.analyze_byte_correlations(data)
        
        # ステップ2: 多次元ソート
        sorted_data, sort_indices = self.multi_dimensional_sort(data)
        
        # ステップ3: 量子エンタングルメント圧縮
        entangled_data, entanglement_info = self.quantum_entanglement_compression(sorted_data)
        
        # ステップ4: パターン除去
        pattern_eliminated_data, pattern_info = self.advanced_pattern_elimination(entangled_data)
        
        # ステップ5: 最終zlib圧縮
        final_compressed = zlib.compress(pattern_eliminated_data, level=9)
        
        # 復元情報のパッケージング
        restoration_info = {
            'original_md5': original_md5,
            'original_size': len(data),
            'correlations': correlations,
            'sort_indices': sort_indices,
            'entanglement_info': entanglement_info,
            'pattern_info': pattern_info,
            'entropy_reduction': correlations['entropy'] - self.calculate_entropy(pattern_eliminated_data)
        }
        
        # 復元情報をバイナリ化
        import pickle
        restoration_bytes = pickle.dumps(restoration_info)
        restoration_compressed = zlib.compress(restoration_bytes, level=9)
        
        # 最終パッケージ
        header = self.magic + struct.pack('>I', len(data))
        header += struct.pack('>I', len(restoration_compressed))
        header += struct.pack('>I', len(final_compressed))
        
        result = header + restoration_compressed + final_compressed
        
        # サイズ増加回避
        if len(result) >= len(data):
            return b'RAW_EXTREME' + struct.pack('>I', len(data)) + data
        
        return result
    
    def extreme_decompress(self, compressed: bytes) -> bytes:
        """極限構造崩壊展開"""
        if not compressed:
            return b''
        
        # RAW形式チェック
        if compressed.startswith(b'RAW_EXTREME'):
            original_size = struct.unpack('>I', compressed[11:15])[0]
            return compressed[15:15+original_size]
        
        # フォーマットチェック
        if not compressed.startswith(self.magic):
            raise ValueError("Invalid NXSC format")
        
        pos = len(self.magic)
        
        # ヘッダー解析
        original_size = struct.unpack('>I', compressed[pos:pos+4])[0]
        pos += 4
        
        restoration_size = struct.unpack('>I', compressed[pos:pos+4])[0]
        pos += 4
        
        compressed_data_size = struct.unpack('>I', compressed[pos:pos+4])[0]
        pos += 4
        
        # 復元情報展開
        restoration_compressed = compressed[pos:pos+restoration_size]
        pos += restoration_size
        
        import pickle
        restoration_bytes = zlib.decompress(restoration_compressed)
        restoration_info = pickle.loads(restoration_bytes)
        
        # 圧縮データ展開
        final_compressed = compressed[pos:pos+compressed_data_size]
        pattern_eliminated_data = zlib.decompress(final_compressed)
        
        # 逆処理チェーン
        # ステップ1: パターン復元
        entangled_data = self.restore_patterns(pattern_eliminated_data, restoration_info['pattern_info'])
        
        # ステップ2: 量子エンタングルメント復元
        sorted_data = self.restore_entanglement(entangled_data, restoration_info['entanglement_info'])
        
        # ステップ3: ソート復元
        restored_data = self.restore_sort(sorted_data, restoration_info['sort_indices'])
        
        # 完全性検証
        restored_md5 = hashlib.md5(restored_data).hexdigest()
        if restored_md5 != restoration_info['original_md5']:
            raise ValueError(f"Integrity check failed: {restored_md5} != {restoration_info['original_md5']}")
        
        return restored_data
    
    def restore_patterns(self, data: bytes, pattern_info: Dict) -> bytes:
        """パターン復元"""
        if not pattern_info:
            return data
        
        result = bytearray(data)
        
        # パターンを挿入位置でソート（後ろから処理）
        all_insertions = []
        for pattern_hex, info in pattern_info.items():
            if 'eliminated_positions' in info:
                pattern_bytes = info['data']
                for pos in info['eliminated_positions']:
                    all_insertions.append((pos, pattern_bytes))
        
        # 位置でソート（後ろから処理）
        all_insertions.sort(key=lambda x: x[0], reverse=True)
        
        # パターン挿入
        for pos, pattern_bytes in all_insertions:
            result[pos:pos] = pattern_bytes
        
        return bytes(result)
    
    def restore_entanglement(self, data: bytes, entanglement_info: Dict) -> bytes:
        """量子エンタングルメント復元"""
        if not entanglement_info:
            return data
        
        # 各グループからバイトを復元
        result = bytearray(len(data))  # 暫定サイズ
        data_pos = 0
        
        # グループIDでソート
        for group_id in sorted(entanglement_info.keys()):
            group_info = entanglement_info[group_id]
            group_size = group_info['size']
            group_positions = group_info['positions']
            
            # 圧縮データからグループデータを取得
            group_compressed = data[data_pos:data_pos+group_size]
            data_pos += group_size
            
            # 差分展開
            if group_size > 1:
                first_val = group_compressed[0]
                group_values = [first_val]
                
                for i in range(1, group_size):
                    val = (group_values[-1] + group_compressed[i]) & 0xFF
                    group_values.append(val)
            else:
                group_values = list(group_compressed)
            
            # 元の位置に復元
            for val, pos in zip(group_values, group_positions):
                if pos < len(result):
                    result[pos] = val
        
        return bytes(result)
    
    def restore_sort(self, data: bytes, sort_indices: List[int]) -> bytes:
        """ソート復元"""
        if not sort_indices or len(sort_indices) != len(data):
            return data
        
        result = bytearray(len(data))
        
        for i, original_pos in enumerate(sort_indices):
            if original_pos < len(result):
                result[original_pos] = data[i]
        
        return bytes(result)
    
    def compress_file(self, input_path: str):
        """ファイル極限圧縮"""
        if not os.path.exists(input_path):
            print(f"❌ ファイルが見つかりません: {input_path}")
            return None
        
        print(f"🚀 極限構造崩壊圧縮開始: {os.path.basename(input_path)}")
        start_time = time.time()
        
        # ファイル読み込み
        with open(input_path, 'rb') as f:
            original_data = f.read()
        
        original_size = len(original_data)
        original_md5 = hashlib.md5(original_data).hexdigest()
        
        print(f"📁 元ファイル: {original_size:,} bytes")
        print(f"🔒 元MD5: {original_md5}")
        
        # 極限圧縮
        compressed_data = self.extreme_compress(original_data)
        compressed_size = len(compressed_data)
        
        # 圧縮率計算
        compression_ratio = ((original_size - compressed_size) / original_size) * 100 if original_size > 0 else 0
        
        # 処理時間・速度
        processing_time = time.time() - start_time
        throughput = original_size / (1024 * 1024) / processing_time if processing_time > 0 else 0
        
        # 結果表示
        print(f"🔹 極限圧縮完了: {compression_ratio:.1f}%")
        print(f"⚡ 処理時間: {processing_time:.3f}s ({throughput:.1f} MB/s)")
        
        # 保存
        output_path = input_path + '.nxsc'
        with open(output_path, 'wb') as f:
            f.write(compressed_data)
        
        print(f"💾 保存: {os.path.basename(output_path)}")
        
        # 可逆性テスト
        try:
            decompressed_data = self.extreme_decompress(compressed_data)
            decompressed_md5 = hashlib.md5(decompressed_data).hexdigest()
            
            if decompressed_md5 == original_md5:
                print(f"✅ 完全可逆性確認: MD5一致")
                print(f"🎯 SUCCESS: 極限構造崩壊圧縮完了 - {output_path}")
                
                return {
                    'input_file': input_path,
                    'output_file': output_path,
                    'original_size': original_size,
                    'compressed_size': compressed_size,
                    'compression_ratio': compression_ratio,
                    'processing_time': processing_time,
                    'throughput': throughput,
                    'lossless': True,
                    'method': 'Extreme Structural Collapse'
                }
            else:
                print(f"❌ エラー: MD5不一致")
                print(f"   元: {original_md5}")
                print(f"   復元: {decompressed_md5}")
                return None
                
        except Exception as e:
            print(f"❌ 展開エラー: {e}")
            return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用法: python nexus_extreme_structural_collapse.py <ファイルパス>")
        print("\n🎯 極限構造崩壊圧縮エンジン")
        print("📋 特徴:")
        print("  ✅ 完全可逆性保証（MD5検証）")
        print("  🧬 多次元バイトソーティング")
        print("  ⚡ 量子エンタングルメント風圧縮")
        print("  🎨 高度なパターン除去・復元")
        print("  💥 データ構造完全崩壊→復元")
        sys.exit(1)
    
    input_file = sys.argv[1]
    engine = StructuralCollapseEngine()
    result = engine.compress_file(input_file)
    
    if result:
        print(f"\n{'='*60}")
        print(f"🏆 ULTIMATE SUCCESS: {result['compression_ratio']:.1f}% compression")
        print(f"⚡ {result['throughput']:.1f} MB/s processing speed")
        print(f"✅ 100% lossless with complete structural collapse & restoration")
        print(f"{'='*60}")
