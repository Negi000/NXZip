#!/usr/bin/env python3
"""
NXZip Binary Structural Dictionary Compressor
バイナリ構造辞書圧縮エンジン - 16進数辞書による極限圧縮

革新的アプローチ:
- バイナリレベル徹底構造解析
- 構造情報の完全保存
- 16進数パターン辞書圧縮
- 構造ベース完全復元
- 極限圧縮率の実現
"""

import struct
import time
import hashlib
import os
import sys
import zlib
import pickle
from typing import List, Tuple, Dict, Set
from collections import defaultdict, Counter
import re

class BinaryStructuralDictionaryCompressor:
    def __init__(self):
        self.magic = b'NXBSD'  # NXZip Binary Structural Dictionary
        self.version = 1
        
    def deep_binary_structural_analysis(self, data: bytes) -> Dict:
        """バイナリレベル徹底構造解析"""
        print(f"🔬 バイナリ構造徹底解析開始: {len(data):,} bytes")
        
        analysis = {
            'total_size': len(data),
            'md5_hash': hashlib.md5(data).hexdigest(),
            'hex_patterns': {},
            'structural_markers': {},
            'repetition_map': {},
            'offset_correlation': {},
            'byte_distribution': [0] * 256,
            'entropy_regions': [],
            'compression_zones': []
        }
        
        # 16進数データ変換
        hex_data = data.hex()
        print(f"📊 16進数変換完了: {len(hex_data)} hex chars")
        
        # バイト分布解析
        for byte in data:
            analysis['byte_distribution'][byte] += 1
        
        # 16進数パターン解析（2-16文字）
        print("🧩 16進数パターン解析中...")
        analysis['hex_patterns'] = self.analyze_hex_patterns(hex_data)
        
        # 構造マーカー検出
        print("🏗️ 構造マーカー検出中...")
        analysis['structural_markers'] = self.detect_structural_markers(data)
        
        # 繰り返しマップ構築
        print("🔄 繰り返しマップ構築中...")
        analysis['repetition_map'] = self.build_repetition_map(data)
        
        # オフセット相関解析
        print("📐 オフセット相関解析中...")
        analysis['offset_correlation'] = self.analyze_offset_correlation(data)
        
        # エントロピー地域分析
        print("📈 エントロピー地域分析中...")
        analysis['entropy_regions'] = self.analyze_entropy_regions(data)
        
        # 圧縮ゾーン識別
        print("🎯 圧縮ゾーン識別中...")
        analysis['compression_zones'] = self.identify_compression_zones(analysis)
        
        print(f"✅ 構造解析完了:")
        print(f"   🧩 16進パターン: {len(analysis['hex_patterns'])} patterns")
        print(f"   🏗️ 構造マーカー: {len(analysis['structural_markers'])} markers")
        print(f"   🔄 繰り返し領域: {len(analysis['repetition_map'])} regions")
        print(f"   🎯 圧縮ゾーン: {len(analysis['compression_zones'])} zones")
        
        return analysis
    
    def analyze_hex_patterns(self, hex_data: str) -> Dict:
        """16進数パターン解析"""
        patterns = {}
        
        # 2-16文字の16進数パターンを解析
        for pattern_len in range(2, 17, 2):  # 2, 4, 6, 8, 10, 12, 14, 16
            if len(hex_data) < pattern_len:
                continue
                
            pattern_count = defaultdict(int)
            positions = defaultdict(list)
            
            # パターン検出
            for i in range(len(hex_data) - pattern_len + 1):
                pattern = hex_data[i:i+pattern_len]
                pattern_count[pattern] += 1
                positions[pattern].append(i)
            
            # 高頻度パターンのみ保存（3回以上出現）
            frequent_patterns = {
                pattern: {
                    'count': count,
                    'positions': positions[pattern],
                    'savings': (count - 1) * pattern_len  # 圧縮効果推定
                }
                for pattern, count in pattern_count.items()
                if count >= 3
            }
            
            if frequent_patterns:
                patterns[pattern_len] = frequent_patterns
        
        return patterns
    
    def detect_structural_markers(self, data: bytes) -> Dict:
        """構造マーカー検出（詳細版）"""
        markers = {
            'file_signatures': [],
            'alignment_patterns': [],
            'padding_regions': [],
            'checksum_positions': []
        }
        
        # ファイル署名検出
        signatures = [
            (b'\x89PNG\r\n\x1a\n', 'PNG_SIGNATURE'),
            (b'\xff\xd8\xff', 'JPEG_SOI'),
            (b'\xff\xd9', 'JPEG_EOI'),
            (b'PK\x03\x04', 'ZIP_LOCAL_HEADER'),
            (b'PK\x01\x02', 'ZIP_CENTRAL_HEADER'),
            (b'%PDF', 'PDF_HEADER'),
            (b'\x1f\x8b', 'GZIP_HEADER'),
            (b'RIFF', 'RIFF_HEADER'),
            (b'\x00\x00\x01', 'MPEG_START_CODE'),
            (b'ftyp', 'MP4_FTYP'),
            (b'moov', 'MP4_MOOV'),
            (b'mdat', 'MP4_MDAT'),
        ]
        
        for sig, name in signatures:
            pos = 0
            while True:
                pos = data.find(sig, pos)
                if pos == -1:
                    break
                markers['file_signatures'].append({
                    'type': name,
                    'offset': pos,
                    'signature': sig,
                    'hex': sig.hex()
                })
                pos += len(sig)
        
        # アライメントパターン（4, 8, 16バイト境界）
        for alignment in [4, 8, 16]:
            aligned_positions = []
            for i in range(0, len(data), alignment):
                if i + alignment <= len(data):
                    block = data[i:i+alignment]
                    if len(set(block)) == 1:  # 同じバイトの繰り返し
                        aligned_positions.append({
                            'offset': i,
                            'size': alignment,
                            'value': block[0],
                            'hex': block.hex()
                        })
            if aligned_positions:
                markers['alignment_patterns'].extend(aligned_positions)
        
        # パディング領域検出（連続するゼロ、0xFF）
        padding_values = [0x00, 0xFF]
        for pad_val in padding_values:
            in_padding = False
            start_pos = 0
            
            for i, byte in enumerate(data):
                if byte == pad_val:
                    if not in_padding:
                        in_padding = True
                        start_pos = i
                else:
                    if in_padding:
                        length = i - start_pos
                        if length >= 8:  # 8バイト以上
                            markers['padding_regions'].append({
                                'offset': start_pos,
                                'size': length,
                                'value': pad_val,
                                'hex': format(pad_val, '02x') * length
                            })
                        in_padding = False
        
        return markers
    
    def build_repetition_map(self, data: bytes) -> Dict:
        """繰り返しマップ構築"""
        repetition_map = {}
        
        # 2-64バイトの繰り返しパターンを検出
        for pattern_len in [2, 4, 8, 16, 32, 64]:
            if len(data) < pattern_len * 2:
                continue
                
            pattern_positions = defaultdict(list)
            
            for i in range(len(data) - pattern_len + 1):
                pattern = data[i:i+pattern_len]
                pattern_positions[pattern].append(i)
            
            # 2回以上出現するパターン
            repeated_patterns = {
                pattern.hex(): {
                    'binary': pattern,
                    'positions': positions,
                    'count': len(positions),
                    'total_bytes': len(positions) * pattern_len
                }
                for pattern, positions in pattern_positions.items()
                if len(positions) >= 2
            }
            
            if repeated_patterns:
                repetition_map[pattern_len] = repeated_patterns
        
        return repetition_map
    
    def analyze_offset_correlation(self, data: bytes) -> Dict:
        """オフセット相関解析"""
        correlations = {}
        
        # 固定距離での値相関
        for distance in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            if len(data) <= distance:
                continue
                
            correlation_count = 0
            total_comparisons = 0
            value_pairs = defaultdict(int)
            
            for i in range(len(data) - distance):
                val1 = data[i]
                val2 = data[i + distance]
                
                if val1 == val2:
                    correlation_count += 1
                
                value_pairs[(val1, val2)] += 1
                total_comparisons += 1
            
            if total_comparisons > 0:
                correlation_ratio = correlation_count / total_comparisons
                
                if correlation_ratio > 0.1:  # 10%以上の相関
                    correlations[distance] = {
                        'correlation_ratio': correlation_ratio,
                        'exact_matches': correlation_count,
                        'total_comparisons': total_comparisons,
                        'top_pairs': sorted(value_pairs.items(), 
                                          key=lambda x: x[1], reverse=True)[:10]
                    }
        
        return correlations
    
    def analyze_entropy_regions(self, data: bytes) -> List:
        """エントロピー地域分析"""
        regions = []
        chunk_size = 1024  # 1KB単位で解析
        
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i+chunk_size]
            if len(chunk) == 0:
                continue
                
            # エントロピー計算
            byte_count = Counter(chunk)
            entropy = 0.0
            total = len(chunk)
            
            for count in byte_count.values():
                if count > 0:
                    prob = count / total
                    entropy -= prob * (count.bit_length() - 1)  # 簡易エントロピー
            
            # 圧縮可能性評価
            unique_bytes = len(byte_count)
            repetition_factor = max(byte_count.values()) / total
            
            regions.append({
                'offset': i,
                'size': len(chunk),
                'entropy': entropy,
                'unique_bytes': unique_bytes,
                'repetition_factor': repetition_factor,
                'compressibility': 'HIGH' if repetition_factor > 0.5 else 
                                 'MEDIUM' if repetition_factor > 0.2 else 'LOW'
            })
        
        return regions
    
    def identify_compression_zones(self, analysis: Dict) -> List:
        """圧縮ゾーン識別"""
        zones = []
        
        # 高圧縮可能領域の特定
        for region in analysis['entropy_regions']:
            if region['compressibility'] in ['HIGH', 'MEDIUM']:
                zones.append({
                    'type': 'HIGH_ENTROPY',
                    'offset': region['offset'],
                    'size': region['size'],
                    'method': 'DICTIONARY_COMPRESSION',
                    'priority': 'HIGH' if region['compressibility'] == 'HIGH' else 'MEDIUM'
                })
        
        # 繰り返しパターン領域
        for pattern_len, patterns in analysis['repetition_map'].items():
            for hex_pattern, info in patterns.items():
                if info['count'] >= 3:
                    zones.append({
                        'type': 'REPETITION_PATTERN',
                        'pattern_length': pattern_len,
                        'pattern_hex': hex_pattern,
                        'positions': info['positions'],
                        'method': 'PATTERN_REPLACEMENT',
                        'priority': 'HIGH'
                    })
        
        return zones
    
    def create_hex_dictionary(self, analysis: Dict) -> Dict:
        """16進数辞書作成"""
        print("📚 16進数辞書作成中...")
        
        dictionary = {
            'patterns': {},
            'replacements': {},
            'metadata': {
                'total_patterns': 0,
                'estimated_savings': 0
            }
        }
        
        dict_id = 0
        total_savings = 0
        
        # 16進数パターンから辞書作成
        for pattern_len, patterns in analysis['hex_patterns'].items():
            for hex_pattern, info in patterns.items():
                if info['count'] >= 3 and info['savings'] > pattern_len:
                    # 辞書エントリ作成
                    dict_key = f"D{dict_id:04X}"  # D0000, D0001, ...
                    
                    dictionary['patterns'][dict_key] = {
                        'hex_pattern': hex_pattern,
                        'binary_pattern': bytes.fromhex(hex_pattern),
                        'original_length': pattern_len,
                        'occurrences': info['count'],
                        'positions': info['positions'],
                        'savings': info['savings']
                    }
                    
                    dictionary['replacements'][hex_pattern] = dict_key
                    total_savings += info['savings']
                    dict_id += 1
        
        dictionary['metadata']['total_patterns'] = dict_id
        dictionary['metadata']['estimated_savings'] = total_savings
        
        print(f"📚 辞書作成完了: {dict_id} patterns, 推定節約: {total_savings} chars")
        return dictionary
    
    def apply_dictionary_compression(self, data: bytes, dictionary: Dict) -> Tuple[bytes, Dict]:
        """辞書圧縮適用"""
        print("🗜️ 辞書圧縮適用中...")
        
        hex_data = data.hex()
        compressed_hex = hex_data
        replacement_log = []
        
        # パターンを長い順にソート（長いパターンを優先）
        patterns = sorted(dictionary['replacements'].items(), 
                         key=lambda x: len(x[0]), reverse=True)
        
        # 辞書置換適用
        for hex_pattern, dict_key in patterns:
            if hex_pattern in compressed_hex:
                occurrences = compressed_hex.count(hex_pattern)
                compressed_hex = compressed_hex.replace(hex_pattern, dict_key)
                replacement_log.append({
                    'pattern': hex_pattern,
                    'dict_key': dict_key,
                    'occurrences': occurrences,
                    'original_length': len(hex_pattern),
                    'compressed_length': len(dict_key)
                })
        
        # 16進数文字列をバイト列に変換（辞書キーは特別処理）
        compressed_bytes = self.hex_with_dict_to_bytes(compressed_hex, dictionary)
        
        compression_info = {
            'original_hex_length': len(hex_data),
            'compressed_hex_length': len(compressed_hex),
            'replacement_log': replacement_log,
            'final_bytes_length': len(compressed_bytes)
        }
        
        hex_reduction = (len(hex_data) - len(compressed_hex)) / len(hex_data) * 100
        print(f"🗜️ 辞書圧縮完了: {hex_reduction:.1f}% hex reduction")
        
        return compressed_bytes, compression_info
    
    def hex_with_dict_to_bytes(self, hex_string: str, dictionary: Dict) -> bytes:
        """辞書キー付き16進数文字列をバイト列に変換"""
        result = bytearray()
        i = 0
        
        while i < len(hex_string):
            # 辞書キーチェック
            if hex_string[i:i+1] == 'D' and i + 4 < len(hex_string):
                dict_key = hex_string[i:i+5]  # D0000形式
                if dict_key in dictionary['patterns']:
                    # 辞書キーマーカー + ID
                    result.append(0xFD)  # 辞書マーカー
                    key_id = int(dict_key[1:], 16)
                    result.extend(struct.pack('>H', key_id))
                    i += 5
                    continue
            
            # 通常の16進数
            if i + 1 < len(hex_string):
                try:
                    byte_val = int(hex_string[i:i+2], 16)
                    result.append(byte_val)
                    i += 2
                except ValueError:
                    # 無効な16進数の場合はスキップ
                    i += 1
            else:
                i += 1
        
        return bytes(result)
    
    def compress_with_structural_dictionary(self, data: bytes) -> bytes:
        """構造辞書による完全圧縮"""
        if not data:
            return self.magic + struct.pack('>I', 0)
        
        print(f"🚀 構造辞書圧縮開始: {len(data):,} bytes")
        start_time = time.time()
        
        original_md5 = hashlib.md5(data).hexdigest()
        print(f"🔒 元データMD5: {original_md5}")
        
        # 1. バイナリ構造徹底解析
        analysis = self.deep_binary_structural_analysis(data)
        
        # 2. 16進数辞書作成
        dictionary = self.create_hex_dictionary(analysis)
        
        # 3. 辞書圧縮適用
        compressed_data, compression_info = self.apply_dictionary_compression(data, dictionary)
        
        # 4. 最終zlib圧縮
        final_compressed = zlib.compress(compressed_data, level=9)
        
        # 5. 構造情報パッケージング
        structure_package = {
            'original_md5': original_md5,
            'original_size': len(data),
            'analysis': analysis,
            'dictionary': dictionary,
            'compression_info': compression_info
        }
        
        structure_bytes = pickle.dumps(structure_package, protocol=pickle.HIGHEST_PROTOCOL)
        structure_compressed = zlib.compress(structure_bytes, level=9)
        
        # 6. 最終パッケージ構築
        header = self.magic + struct.pack('>I', len(data))
        header += struct.pack('>I', len(structure_compressed))
        header += struct.pack('>I', len(final_compressed))
        
        result = header + structure_compressed + final_compressed
        
        processing_time = time.time() - start_time
        compression_ratio = ((len(data) - len(result)) / len(data)) * 100
        
        print(f"🏆 構造辞書圧縮完了:")
        print(f"   💥 圧縮率: {compression_ratio:.1f}%")
        print(f"   📊 {len(data):,} → {len(result):,} bytes")
        print(f"   ⚡ 処理時間: {processing_time:.3f}s")
        
        # RAW保存判定
        if len(result) >= len(data) * 0.95:
            print("⚠️ 圧縮効果限定 - RAW保存")
            return b'RAW_BSD' + struct.pack('>I', len(data)) + data
        
        return result
    
    def restore_from_structural_dictionary(self, compressed: bytes) -> bytes:
        """構造辞書による完全復元"""
        if not compressed:
            return b''
        
        # RAW形式チェック
        if compressed.startswith(b'RAW_BSD'):
            size = struct.unpack('>I', compressed[7:11])[0]
            return compressed[11:11+size]
        
        if not compressed.startswith(self.magic):
            raise ValueError("Invalid NXBSD format")
        
        pos = len(self.magic)
        
        # ヘッダー解析
        original_size = struct.unpack('>I', compressed[pos:pos+4])[0]
        pos += 4
        
        structure_size = struct.unpack('>I', compressed[pos:pos+4])[0]
        pos += 4
        
        data_size = struct.unpack('>I', compressed[pos:pos+4])[0]
        pos += 4
        
        # 構造情報復元
        structure_compressed = compressed[pos:pos+structure_size]
        pos += structure_size
        
        structure_bytes = zlib.decompress(structure_compressed)
        structure_package = pickle.loads(structure_bytes)
        
        # データ復元
        data_compressed = compressed[pos:pos+data_size]
        compressed_data = zlib.decompress(data_compressed)
        
        # 辞書復元適用
        restored_data = self.restore_dictionary_compression(
            compressed_data, 
            structure_package['dictionary']
        )
        
        # 完全性検証
        restored_md5 = hashlib.md5(restored_data).hexdigest()
        if restored_md5 != structure_package['original_md5']:
            raise ValueError(f"Integrity check failed: {restored_md5} != {structure_package['original_md5']}")
        
        print(f"✅ 構造辞書復元成功: MD5一致 ({restored_md5})")
        return restored_data
    
    def restore_dictionary_compression(self, compressed_data: bytes, dictionary: Dict) -> bytes:
        """辞書圧縮復元"""
        result = bytearray()
        i = 0
        
        while i < len(compressed_data):
            if compressed_data[i] == 0xFD and i + 2 < len(compressed_data):
                # 辞書キーマーカー
                key_id = struct.unpack('>H', compressed_data[i+1:i+3])[0]
                dict_key = f"D{key_id:04X}"
                
                if dict_key in dictionary['patterns']:
                    # 辞書パターン復元
                    pattern_info = dictionary['patterns'][dict_key]
                    result.extend(pattern_info['binary_pattern'])
                
                i += 3
            else:
                # 通常のバイト
                result.append(compressed_data[i])
                i += 1
        
        return bytes(result)
    
    def compress_file(self, input_path: str):
        """ファイル圧縮"""
        if not os.path.exists(input_path):
            print(f"❌ ファイルが見つかりません: {input_path}")
            return None
        
        print(f"🚀 構造辞書圧縮開始: {os.path.basename(input_path)}")
        
        with open(input_path, 'rb') as f:
            data = f.read()
        
        original_size = len(data)
        original_md5 = hashlib.md5(data).hexdigest()
        
        print(f"📁 元ファイル: {original_size:,} bytes")
        print(f"🔒 元MD5: {original_md5}")
        
        start_time = time.time()
        compressed = self.compress_with_structural_dictionary(data)
        processing_time = time.time() - start_time
        
        compression_ratio = ((original_size - len(compressed)) / original_size) * 100
        throughput = original_size / (1024 * 1024) / processing_time if processing_time > 0 else 0
        
        # 保存
        output_path = input_path + '.nxbsd'
        with open(output_path, 'wb') as f:
            f.write(compressed)
        
        print(f"💾 保存: {os.path.basename(output_path)}")
        
        # 完全性テスト
        try:
            restored = self.restore_from_structural_dictionary(compressed)
            restored_md5 = hashlib.md5(restored).hexdigest()
            
            if restored_md5 == original_md5:
                print(f"✅ 完全可逆性確認: MD5一致")
                print(f"🎯 SUCCESS: 構造辞書圧縮完了")
                print(f"⚡ 処理速度: {throughput:.1f} MB/s")
                return True
            else:
                print(f"❌ MD5不一致: {original_md5} != {restored_md5}")
                return False
        except Exception as e:
            print(f"❌ 復元エラー: {e}")
            return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用法: python nxzip_binary_structural_dictionary.py <ファイルパス>")
        print("\n🎯 NXZip バイナリ構造辞書圧縮エンジン")
        print("📋 革新的特徴:")
        print("  🔬 バイナリレベル徹底構造解析")
        print("  💾 構造情報完全保存")
        print("  📚 16進数パターン辞書圧縮")
        print("  🔄 構造ベース完全復元")
        print("  🏆 極限圧縮率実現")
        sys.exit(1)
    
    input_file = sys.argv[1]
    compressor = BinaryStructuralDictionaryCompressor()
    compressor.compress_file(input_file)
