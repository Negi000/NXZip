#!/usr/bin/env python3
"""
NXZip Binary-Level Structural Collapse Engine
バイナリレベル構造崩壊エンジン - 完全可逆性保証

特徴:
- バイナリレベル詳細解析
- 元状態の完全保存
- データ構造崩壊による極限圧縮
- 元状態ベース完全復元
- 100%可逆性保証
"""

import struct
import time
import hashlib
import os
import sys
import zlib
import pickle
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import math

class BinaryStructuralCollapseEngine:
    def __init__(self):
        self.magic = b'NXBSC'  # NXZip Binary Structural Collapse
        self.version = 1
        
    def deep_binary_analysis(self, data: bytes) -> Dict:
        """バイナリレベル深層解析"""
        analysis = {
            'size': len(data),
            'md5_hash': hashlib.md5(data).hexdigest(),
            'byte_frequency': [0] * 256,
            'entropy_regions': [],
            'pattern_map': {},
            'structural_markers': [],
            'correlation_matrix': {},
            'sequence_patterns': {}
        }
        
        # バイト頻度解析
        for byte in data:
            analysis['byte_frequency'][byte] += 1
        
        # エントロピー地域解析（1KB単位）
        chunk_size = 1024
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i+chunk_size]
            if chunk:
                entropy = self.calculate_entropy(chunk)
                analysis['entropy_regions'].append({
                    'offset': i,
                    'size': len(chunk),
                    'entropy': entropy,
                    'dominant_bytes': self.get_dominant_bytes(chunk)
                })
        
        # 構造マーカー検出
        analysis['structural_markers'] = self.detect_structural_markers(data)
        
        # パターンマップ構築
        analysis['pattern_map'] = self.build_pattern_map(data)
        
        # バイト相関解析
        analysis['correlation_matrix'] = self.analyze_byte_correlations(data)
        
        # シーケンスパターン解析
        analysis['sequence_patterns'] = self.analyze_sequence_patterns(data)
        
        print(f"🔬 バイナリ深層解析完了: {len(data):,} bytes analyzed")
        print(f"📊 エントロピー地域: {len(analysis['entropy_regions'])} regions")
        print(f"🏗️  構造マーカー: {len(analysis['structural_markers'])} markers")
        print(f"🧩 パターン: {len(analysis['pattern_map'])} unique patterns")
        
        return analysis
    
    def calculate_entropy(self, data: bytes) -> float:
        """データエントロピー計算"""
        if not data:
            return 0.0
        
        freq = defaultdict(int)
        for byte in data:
            freq[byte] += 1
        
        entropy = 0.0
        total = len(data)
        for count in freq.values():
            if count > 0:
                prob = count / total
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    def get_dominant_bytes(self, data: bytes) -> List[int]:
        """支配的バイト検出"""
        freq = defaultdict(int)
        for byte in data:
            freq[byte] += 1
        
        # 頻度順にソート
        sorted_bytes = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [byte for byte, count in sorted_bytes[:5]]  # 上位5つ
    
    def detect_structural_markers(self, data: bytes) -> List[Dict]:
        """構造マーカー検出"""
        markers = []
        
        # ファイル形式署名検出
        signatures = [
            (b'\x89PNG\r\n\x1a\n', 'PNG_SIGNATURE'),
            (b'\xff\xd8\xff', 'JPEG_SOI'),
            (b'PK\x03\x04', 'ZIP_LOCAL_HEADER'),
            (b'%PDF', 'PDF_HEADER'),
            (b'\x1f\x8b', 'GZIP_HEADER'),
            (b'RIFF', 'RIFF_HEADER'),
            (b'\x00\x00\x01', 'MPEG_START_CODE'),
        ]
        
        for sig, name in signatures:
            pos = data.find(sig)
            if pos != -1:
                markers.append({
                    'type': name,
                    'offset': pos,
                    'size': len(sig),
                    'data': sig
                })
        
        # 繰り返しパターン検出
        for pattern_len in [2, 3, 4, 8, 16]:
            self.detect_repeating_patterns(data, pattern_len, markers)
        
        # ゼロ填充領域検出
        self.detect_zero_regions(data, markers)
        
        return markers
    
    def detect_repeating_patterns(self, data: bytes, pattern_len: int, markers: List[Dict]):
        """繰り返しパターン検出"""
        pattern_positions = defaultdict(list)
        
        for i in range(len(data) - pattern_len + 1):
            pattern = data[i:i+pattern_len]
            pattern_positions[pattern].append(i)
        
        # 3回以上繰り返されるパターンをマーカーとして追加
        for pattern, positions in pattern_positions.items():
            if len(positions) >= 3:
                markers.append({
                    'type': f'REPEAT_PATTERN_{pattern_len}',
                    'pattern': pattern,
                    'positions': positions,
                    'count': len(positions)
                })
    
    def detect_zero_regions(self, data: bytes, markers: List[Dict]):
        """ゼロ填充領域検出"""
        in_zero_region = False
        start_pos = 0
        
        for i, byte in enumerate(data):
            if byte == 0:
                if not in_zero_region:
                    in_zero_region = True
                    start_pos = i
            else:
                if in_zero_region:
                    length = i - start_pos
                    if length >= 8:  # 8バイト以上のゼロ領域
                        markers.append({
                            'type': 'ZERO_REGION',
                            'offset': start_pos,
                            'size': length
                        })
                    in_zero_region = False
        
        # ファイル末尾のゼロ領域
        if in_zero_region:
            length = len(data) - start_pos
            if length >= 8:
                markers.append({
                    'type': 'ZERO_REGION',
                    'offset': start_pos,
                    'size': length
                })
    
    def build_pattern_map(self, data: bytes) -> Dict:
        """パターンマップ構築"""
        pattern_map = {}
        
        # 2-8バイトのパターンを解析
        for pattern_len in range(2, 9):
            if len(data) < pattern_len:
                continue
                
            patterns = defaultdict(int)
            for i in range(len(data) - pattern_len + 1):
                pattern = data[i:i+pattern_len]
                patterns[pattern] += 1
            
            # 頻度の高いパターンを保存
            frequent_patterns = {p: count for p, count in patterns.items() if count >= 3}
            if frequent_patterns:
                pattern_map[pattern_len] = frequent_patterns
        
        return pattern_map
    
    def analyze_byte_correlations(self, data: bytes) -> Dict:
        """バイト相関解析"""
        correlations = {}
        
        # 隣接バイト相関
        if len(data) > 1:
            adjacent_pairs = defaultdict(int)
            for i in range(len(data) - 1):
                pair = (data[i], data[i+1])
                adjacent_pairs[pair] += 1
            correlations['adjacent'] = dict(adjacent_pairs)
        
        # 距離別相関
        for distance in [2, 4, 8, 16]:
            if len(data) > distance:
                distant_pairs = defaultdict(int)
                for i in range(len(data) - distance):
                    pair = (data[i], data[i+distance])
                    distant_pairs[pair] += 1
                correlations[f'distance_{distance}'] = dict(distant_pairs)
        
        return correlations
    
    def analyze_sequence_patterns(self, data: bytes) -> Dict:
        """シーケンスパターン解析"""
        patterns = {}
        
        # 増加シーケンス検出
        inc_sequences = []
        current_seq = [data[0]] if data else []
        
        for i in range(1, len(data)):
            if data[i] == current_seq[-1] + 1:
                current_seq.append(data[i])
            else:
                if len(current_seq) >= 4:
                    inc_sequences.append({
                        'start': i - len(current_seq),
                        'length': len(current_seq),
                        'type': 'INCREASING'
                    })
                current_seq = [data[i]]
        
        patterns['increasing_sequences'] = inc_sequences
        
        # 減少シーケンス検出
        dec_sequences = []
        current_seq = [data[0]] if data else []
        
        for i in range(1, len(data)):
            if data[i] == current_seq[-1] - 1:
                current_seq.append(data[i])
            else:
                if len(current_seq) >= 4:
                    dec_sequences.append({
                        'start': i - len(current_seq),
                        'length': len(current_seq),
                        'type': 'DECREASING'
                    })
                current_seq = [data[i]]
        
        patterns['decreasing_sequences'] = dec_sequences
        
        return patterns
    
    def structural_collapse(self, data: bytes, analysis: Dict) -> Tuple[bytes, Dict]:
        """データ構造崩壊処理"""
        print(f"💥 構造崩壊開始: {len(data):,} bytes")
        
        # ステップ1: 頻度順バイト再マッピング
        freq_sorted = sorted(range(256), key=lambda x: analysis['byte_frequency'][x], reverse=True)
        byte_remap = {}
        reverse_remap = {}
        
        for new_val, original_val in enumerate(freq_sorted):
            if analysis['byte_frequency'][original_val] > 0:
                byte_remap[original_val] = new_val
                reverse_remap[new_val] = original_val
        
        remapped_data = bytearray()
        for byte in data:
            remapped_data.append(byte_remap[byte])
        
        print(f"🔄 バイト再マッピング: {len(data):,} → {len(remapped_data):,} bytes")
        
        # ステップ2: パターン除去
        pattern_removed, pattern_info = self.remove_patterns(bytes(remapped_data), analysis['pattern_map'])
        print(f"🧩 パターン除去: {len(remapped_data):,} → {len(pattern_removed):,} bytes")
        
        # ステップ3: シーケンス圧縮
        sequence_compressed, sequence_info = self.compress_sequences(pattern_removed, analysis['sequence_patterns'])
        print(f"📈 シーケンス圧縮: {len(pattern_removed):,} → {len(sequence_compressed):,} bytes")
        
        # ステップ4: ゼロ領域圧縮
        zero_compressed, zero_info = self.compress_zero_regions(sequence_compressed)
        print(f"⚫ ゼロ圧縮: {len(sequence_compressed):,} → {len(zero_compressed):,} bytes")
        
        # ステップ5: 最終差分変換
        final_data = self.final_differential_transform(zero_compressed)
        print(f"🔧 差分変換: {len(zero_compressed):,} → {len(final_data):,} bytes")
        
        collapse_info = {
            'byte_remap': reverse_remap,
            'pattern_info': pattern_info,
            'sequence_info': sequence_info,
            'zero_info': zero_info,
            'original_analysis': analysis
        }
        
        print(f"💥 構造崩壊完了: {len(data):,} → {len(final_data):,} bytes ({(1-len(final_data)/len(data))*100:.1f}%減少)")
        return final_data, collapse_info
    
    def remove_patterns(self, data: bytes, pattern_map: Dict) -> Tuple[bytes, Dict]:
        """パターン除去"""
        removed_data = bytearray(data)
        pattern_info = {}
        removed_positions = set()
        
        # 長いパターンから処理
        for pattern_len in sorted(pattern_map.keys(), reverse=True):
            patterns = pattern_map[pattern_len]
            
            for pattern, count in patterns.items():
                if count < 3:
                    continue
                
                positions = []
                pos = 0
                while pos <= len(removed_data) - pattern_len:
                    if pos not in removed_positions:
                        segment = bytes(removed_data[pos:pos+pattern_len])
                        if segment == pattern:
                            positions.append(pos)
                            # 最初の出現以外を除去
                            if len(positions) > 1:
                                for i in range(pattern_len):
                                    removed_positions.add(pos + i)
                    pos += 1
                
                if len(positions) > 1:
                    pattern_info[pattern.hex()] = {
                        'pattern': pattern,
                        'positions': positions,
                        'length': pattern_len
                    }
        
        # 除去されていないバイトのみ残す
        final_data = bytearray()
        for i, byte in enumerate(removed_data):
            if i not in removed_positions:
                final_data.append(byte)
        
        return bytes(final_data), pattern_info
    
    def compress_sequences(self, data: bytes, sequence_patterns: Dict) -> Tuple[bytes, Dict]:
        """シーケンス圧縮"""
        compressed = bytearray()
        sequence_info = {'compressed_sequences': []}
        i = 0
        
        while i < len(data):
            # 増加シーケンス検出
            if i < len(data) - 3:
                seq_len = 1
                while (i + seq_len < len(data) and 
                       seq_len < 255 and
                       data[i + seq_len] == (data[i] + seq_len) & 0xFF):
                    seq_len += 1
                
                if seq_len >= 4:
                    # シーケンス圧縮: [0xFE, length, start_value]
                    compressed.extend([0xFE, seq_len, data[i]])
                    sequence_info['compressed_sequences'].append({
                        'type': 'increasing',
                        'start': data[i],
                        'length': seq_len,
                        'original_pos': i
                    })
                    i += seq_len
                    continue
            
            # 通常のバイト
            if data[i] == 0xFE:
                compressed.extend([0xFD, 0xFE])  # エスケープ
            else:
                compressed.append(data[i])
            i += 1
        
        return bytes(compressed), sequence_info
    
    def compress_zero_regions(self, data: bytes) -> Tuple[bytes, Dict]:
        """ゼロ領域圧縮"""
        compressed = bytearray()
        zero_info = {'zero_regions': []}
        i = 0
        
        while i < len(data):
            if data[i] == 0:
                # ゼロの連続をカウント
                zero_count = 0
                j = i
                while j < len(data) and data[j] == 0 and zero_count < 255:
                    zero_count += 1
                    j += 1
                
                if zero_count >= 3:
                    # ゼロ圧縮: [0xFF, count]
                    compressed.extend([0xFF, zero_count])
                    zero_info['zero_regions'].append({
                        'start': i,
                        'length': zero_count
                    })
                    i += zero_count
                else:
                    compressed.append(data[i])
                    i += 1
            else:
                if data[i] == 0xFF:
                    compressed.extend([0xFC, 0xFF])  # エスケープ
                else:
                    compressed.append(data[i])
                i += 1
        
        return bytes(compressed), zero_info
    
    def final_differential_transform(self, data: bytes) -> bytes:
        """最終差分変換"""
        if len(data) < 2:
            return data
        
        result = bytearray([data[0]])
        for i in range(1, len(data)):
            diff = (data[i] - data[i-1]) & 0xFF
            result.append(diff)
        
        return bytes(result)
    
    def structural_restore(self, collapsed_data: bytes, collapse_info: Dict, original_analysis: Dict) -> bytes:
        """構造復元処理"""
        print(f"🔄 構造復元開始: {len(collapsed_data):,} bytes")
        
        # ステップ1: 差分変換復元
        diff_restored = self.restore_differential_transform(collapsed_data)
        print(f"🔧 差分復元: {len(collapsed_data):,} → {len(diff_restored):,} bytes")
        
        # ステップ2: ゼロ領域復元
        zero_restored = self.restore_zero_regions(diff_restored, collapse_info['zero_info'])
        print(f"⚫ ゼロ復元: {len(diff_restored):,} → {len(zero_restored):,} bytes")
        
        # ステップ3: シーケンス復元
        sequence_restored = self.restore_sequences(zero_restored, collapse_info['sequence_info'])
        print(f"📈 シーケンス復元: {len(zero_restored):,} → {len(sequence_restored):,} bytes")
        
        # ステップ4: パターン復元
        pattern_restored = self.restore_patterns(sequence_restored, collapse_info['pattern_info'])
        print(f"🧩 パターン復元: {len(sequence_restored):,} → {len(pattern_restored):,} bytes")
        
        # ステップ5: バイト再マッピング復元
        final_data = self.restore_byte_remapping(pattern_restored, collapse_info['byte_remap'])
        print(f"🔄 バイト復元: {len(pattern_restored):,} → {len(final_data):,} bytes")
        
        print(f"🔄 構造復元完了: {len(collapsed_data):,} → {len(final_data):,} bytes")
        return final_data
    
    def restore_differential_transform(self, data: bytes) -> bytes:
        """差分変換復元"""
        if len(data) < 2:
            return data
        
        result = bytearray([data[0]])
        for i in range(1, len(data)):
            value = (result[i-1] + data[i]) & 0xFF
            result.append(value)
        
        return bytes(result)
    
    def restore_zero_regions(self, data: bytes, zero_info: Dict) -> bytes:
        """ゼロ領域復元"""
        result = bytearray()
        i = 0
        
        while i < len(data):
            if i < len(data) - 1 and data[i] == 0xFF:
                count = data[i + 1]
                result.extend([0] * count)
                i += 2
            elif i < len(data) - 1 and data[i] == 0xFC and data[i + 1] == 0xFF:
                result.append(0xFF)
                i += 2
            else:
                result.append(data[i])
                i += 1
        
        return bytes(result)
    
    def restore_sequences(self, data: bytes, sequence_info: Dict) -> bytes:
        """シーケンス復元"""
        result = bytearray()
        i = 0
        
        while i < len(data):
            if i < len(data) - 2 and data[i] == 0xFE:
                length = data[i + 1]
                start_val = data[i + 2]
                
                # シーケンス展開
                for j in range(length):
                    result.append((start_val + j) & 0xFF)
                i += 3
            elif i < len(data) - 1 and data[i] == 0xFD and data[i + 1] == 0xFE:
                result.append(0xFE)
                i += 2
            else:
                result.append(data[i])
                i += 1
        
        return bytes(result)
    
    def restore_patterns(self, data: bytes, pattern_info: Dict) -> bytes:
        """パターン復元"""
        result = bytearray(data)
        
        # パターンを挿入位置でソート（後ろから処理）
        insertions = []
        for pattern_hex, info in pattern_info.items():
            pattern = info['pattern']
            positions = info['positions']
            
            # 最初の出現以外の位置に挿入
            for pos in positions[1:]:
                insertions.append((pos, pattern))
        
        # 位置でソート（後ろから処理）
        insertions.sort(key=lambda x: x[0], reverse=True)
        
        for pos, pattern in insertions:
            # 適切な位置にパターンを挿入
            if pos <= len(result):
                result[pos:pos] = pattern
        
        return bytes(result)
    
    def restore_byte_remapping(self, data: bytes, reverse_remap: Dict) -> bytes:
        """バイト再マッピング復元"""
        result = bytearray()
        for byte in data:
            if byte in reverse_remap:
                result.append(reverse_remap[byte])
            else:
                result.append(byte)
        
        return bytes(result)
    
    def compress_with_complete_reversibility(self, data: bytes) -> bytes:
        """完全可逆性保証圧縮"""
        if not data:
            return self.magic + struct.pack('>I', 0)
        
        original_md5 = hashlib.md5(data).hexdigest()
        print(f"🔒 元データMD5: {original_md5}")
        
        # バイナリレベル深層解析
        analysis = self.deep_binary_analysis(data)
        
        # データ構造崩壊
        collapsed_data, collapse_info = self.structural_collapse(data, analysis)
        
        # 最終zlib圧縮
        final_compressed = zlib.compress(collapsed_data, level=9)
        print(f"📦 最終圧縮: {len(collapsed_data):,} → {len(final_compressed):,} bytes")
        
        # 復元情報パッケージング
        restoration_package = {
            'original_md5': original_md5,
            'original_size': len(data),
            'collapse_info': collapse_info,
            'analysis': analysis
        }
        
        restoration_bytes = pickle.dumps(restoration_package)
        restoration_compressed = zlib.compress(restoration_bytes, level=9)
        
        # 最終パッケージ
        header = self.magic + struct.pack('>I', len(data))
        header += struct.pack('>I', len(restoration_compressed))
        header += struct.pack('>I', len(final_compressed))
        
        result = header + restoration_compressed + final_compressed
        
        # サイズ増加回避
        if len(result) >= len(data):
            print("⚠️  圧縮効果なし - RAW保存")
            return b'RAW_BSC' + struct.pack('>I', len(data)) + data
        
        total_ratio = ((len(data) - len(result)) / len(data)) * 100
        print(f"🏆 総圧縮率: {total_ratio:.1f}% ({len(data):,} → {len(result):,} bytes)")
        
        return result
    
    def decompress_with_complete_restoration(self, compressed: bytes) -> bytes:
        """完全復元展開"""
        if not compressed:
            return b''
        
        # RAW形式チェック
        if compressed.startswith(b'RAW_BSC'):
            original_size = struct.unpack('>I', compressed[7:11])[0]
            return compressed[11:11+original_size]
        
        if not compressed.startswith(self.magic):
            raise ValueError("Invalid NXBSC format")
        
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
        
        restoration_bytes = zlib.decompress(restoration_compressed)
        restoration_package = pickle.loads(restoration_bytes)
        
        # 圧縮データ展開
        final_compressed = compressed[pos:pos+compressed_data_size]
        collapsed_data = zlib.decompress(final_compressed)
        
        # 構造復元
        restored_data = self.structural_restore(
            collapsed_data, 
            restoration_package['collapse_info'],
            restoration_package['analysis']
        )
        
        # 完全性検証
        restored_md5 = hashlib.md5(restored_data).hexdigest()
        original_md5 = restoration_package['original_md5']
        
        if restored_md5 != original_md5:
            raise ValueError(f"Integrity check failed: {restored_md5} != {original_md5}")
        
        print(f"✅ 完全復元確認: MD5一致 ({original_md5})")
        return restored_data
    
    def compress_file(self, input_path: str):
        """ファイル圧縮（完全可逆性保証）"""
        if not os.path.exists(input_path):
            print(f"❌ ファイルが見つかりません: {input_path}")
            return None
        
        print(f"🚀 バイナリ構造崩壊圧縮開始: {os.path.basename(input_path)}")
        start_time = time.time()
        
        # ファイル読み込み
        with open(input_path, 'rb') as f:
            original_data = f.read()
        
        original_size = len(original_data)
        original_md5 = hashlib.md5(original_data).hexdigest()
        
        print(f"📁 元ファイル: {original_size:,} bytes")
        print(f"🔒 元MD5: {original_md5}")
        
        # 完全可逆性保証圧縮
        compressed_data = self.compress_with_complete_reversibility(original_data)
        compressed_size = len(compressed_data)
        
        # 圧縮率計算
        compression_ratio = ((original_size - compressed_size) / original_size) * 100 if original_size > 0 else 0
        
        # 処理時間・速度
        processing_time = time.time() - start_time
        throughput = original_size / (1024 * 1024) / processing_time if processing_time > 0 else 0
        
        # 結果表示
        print(f"🔹 構造崩壊圧縮完了: {compression_ratio:.1f}%")
        print(f"⚡ 処理時間: {processing_time:.3f}s ({throughput:.1f} MB/s)")
        
        # 保存
        output_path = input_path + '.nxbsc'
        with open(output_path, 'wb') as f:
            f.write(compressed_data)
        
        print(f"💾 保存: {os.path.basename(output_path)}")
        
        # 完全可逆性テスト
        try:
            decompressed_data = self.decompress_with_complete_restoration(compressed_data)
            decompressed_md5 = hashlib.md5(decompressed_data).hexdigest()
            
            if decompressed_md5 == original_md5:
                print(f"✅ 完全可逆性確認: MD5一致")
                print(f"🎯 SUCCESS: バイナリ構造崩壊圧縮完了 - {output_path}")
                
                return {
                    'input_file': input_path,
                    'output_file': output_path,
                    'original_size': original_size,
                    'compressed_size': compressed_size,
                    'compression_ratio': compression_ratio,
                    'processing_time': processing_time,
                    'throughput': throughput,
                    'lossless': True,
                    'method': 'Binary Structural Collapse'
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
        print("使用法: python nxzip_binary_structural_collapse.py <ファイルパス>")
        print("\n🎯 NXZip バイナリレベル構造崩壊エンジン")
        print("📋 特徴:")
        print("  🔬 バイナリレベル深層解析")
        print("  💾 元状態完全保存")
        print("  💥 データ構造崩壊による極限圧縮")
        print("  🔄 元状態ベース完全復元")
        print("  ✅ 100% 完全可逆性保証")
        sys.exit(1)
    
    input_file = sys.argv[1]
    engine = BinaryStructuralCollapseEngine()
    result = engine.compress_file(input_file)
    
    if result:
        print(f"\n{'='*60}")
        print(f"🏆 ULTIMATE SUCCESS: {result['compression_ratio']:.1f}% compression")
        print(f"📊 {result['original_size']:,} → {result['compressed_size']:,} bytes")
        print(f"⚡ {result['throughput']:.1f} MB/s processing speed")
        print(f"✅ Perfect reversibility with binary structural collapse")
        print(f"{'='*60}")
