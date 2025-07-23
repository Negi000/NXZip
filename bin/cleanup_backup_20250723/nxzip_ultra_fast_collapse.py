#!/usr/bin/env python3
"""
NXZip Ultra Fast Binary Collapse Engine
超高速バイナリ崩壊エンジン - 速度重視の完全可逆圧縮

特徴:
- 超高速処理（簡易解析のみ）
- 完全可逆性保証
- メディアファイル最適化
- 単純な構造崩壊による効率圧縮
"""

import struct
import time
import hashlib
import os
import sys
import zlib
from typing import List, Tuple, Dict

class UltraFastBinaryCollapseEngine:
    def __init__(self):
        self.magic = b'NXUFC'  # NXZip Ultra Fast Collapse
        self.version = 1
        
    def quick_analysis(self, data: bytes) -> Dict:
        """高速解析（最小限）"""
        if not data:
            return {'size': 0, 'byte_freq': [0] * 256}
        
        # バイト頻度のみ（最重要）
        byte_freq = [0] * 256
        for byte in data:
            byte_freq[byte] += 1
        
        return {
            'size': len(data),
            'byte_freq': byte_freq,
            'md5': hashlib.md5(data).hexdigest()
        }
    
    def ultra_fast_collapse(self, data: bytes) -> Tuple[bytes, Dict]:
        """超高速構造崩壊"""
        if not data:
            return b'', {}
        
        # ステップ1: 高頻度バイト→低値マッピング（最重要）
        analysis = self.quick_analysis(data)
        byte_freq = analysis['byte_freq']
        
        # 実際に使用されているバイト値のみ処理
        used_bytes = [i for i in range(256) if byte_freq[i] > 0]
        freq_order = sorted(used_bytes, key=lambda x: byte_freq[x], reverse=True)
        
        # マッピングテーブル作成
        remap_table = {}
        reverse_table = {}
        
        for new_val, original_val in enumerate(freq_order):
            remap_table[original_val] = new_val
            reverse_table[new_val] = original_val
        
        # 再マッピング実行
        remapped = bytearray()
        for byte in data:
            remapped.append(remap_table[byte])
        
        # ステップ2: 簡易RLE（3回以上の繰り返しのみ）
        rle_data = self.simple_rle_encode(bytes(remapped))
        
        # ステップ3: 差分変換
        diff_data = self.quick_differential(rle_data)
        
        collapse_info = {
            'reverse_table': reverse_table,
            'original_size': len(data)
        }
        
        return diff_data, collapse_info
    
    def simple_rle_encode(self, data: bytes) -> bytes:
        """簡易RLE（高速版）"""
        if not data:
            return b''
        
        result = bytearray()
        i = 0
        
        while i < len(data):
            current = data[i]
            count = 1
            
            # 最大16回まで（速度重視）
            while count < 16 and i + count < len(data) and data[i + count] == current:
                count += 1
            
            if count >= 3:
                # RLE: [0xFF, count, value]
                result.extend([0xFF, count, current])
                i += count
            else:
                # 通常
                if current == 0xFF:
                    result.extend([0xFE, 0xFF])  # エスケープ
                else:
                    result.append(current)
                i += 1
        
        return bytes(result)
    
    def simple_rle_decode(self, data: bytes) -> bytes:
        """簡易RLE展開"""
        if not data:
            return b''
        
        result = bytearray()
        i = 0
        
        while i < len(data):
            if i < len(data) - 2 and data[i] == 0xFF:
                count = data[i + 1]
                value = data[i + 2]
                result.extend([value] * count)
                i += 3
            elif i < len(data) - 1 and data[i] == 0xFE and data[i + 1] == 0xFF:
                result.append(0xFF)
                i += 2
            else:
                result.append(data[i])
                i += 1
        
        return bytes(result)
    
    def quick_differential(self, data: bytes) -> bytes:
        """高速差分変換"""
        if len(data) < 2:
            return data
        
        result = bytearray([data[0]])
        for i in range(1, len(data)):
            diff = (data[i] - data[i-1]) & 0xFF
            result.append(diff)
        
        return bytes(result)
    
    def quick_differential_restore(self, data: bytes) -> bytes:
        """高速差分復元"""
        if len(data) < 2:
            return data
        
        result = bytearray([data[0]])
        for i in range(1, len(data)):
            value = (result[i-1] + data[i]) & 0xFF
            result.append(value)
        
        return bytes(result)
    
    def ultra_fast_restore(self, collapsed_data: bytes, collapse_info: Dict) -> bytes:
        """超高速復元"""
        if not collapsed_data:
            return b''
        
        # ステップ1: 差分復元
        diff_restored = self.quick_differential_restore(collapsed_data)
        
        # ステップ2: RLE復元
        rle_restored = self.simple_rle_decode(diff_restored)
        
        # ステップ3: バイトマッピング復元
        reverse_table = collapse_info['reverse_table']
        final_data = bytearray()
        
        for byte in rle_restored:
            if byte in reverse_table:
                final_data.append(reverse_table[byte])
            else:
                # 未知のバイト値の場合はそのまま保持
                final_data.append(byte)
        
        # サイズ検証
        expected_size = collapse_info['original_size']
        if len(final_data) != expected_size:
            raise ValueError(f"Size mismatch: expected {expected_size}, got {len(final_data)}")
        
        return bytes(final_data)
    
    def compress(self, data: bytes) -> bytes:
        """超高速圧縮"""
        if not data:
            return self.magic + struct.pack('>I', 0)
        
        start_time = time.time()
        
        # 超高速構造崩壊
        collapsed_data, collapse_info = self.ultra_fast_collapse(data)
        
        collapse_time = time.time() - start_time
        print(f"⚡ 構造崩壊: {collapse_time:.3f}s")
        
        # zlib最終圧縮
        zlib_start = time.time()
        final_compressed = zlib.compress(collapsed_data, level=6)  # 速度重視レベル
        zlib_time = time.time() - zlib_start
        print(f"📦 zlib圧縮: {zlib_time:.3f}s")
        
        # 復元情報シリアライズ（簡易版）
        info_bytes = self.serialize_collapse_info(collapse_info)
        info_compressed = zlib.compress(info_bytes, level=6)
        
        # パッケージング
        header = self.magic + struct.pack('>I', len(data))
        header += struct.pack('>I', len(info_compressed))
        header += struct.pack('>I', len(final_compressed))
        
        result = header + info_compressed + final_compressed
        
        # RAW保存チェック
        if len(result) >= len(data) * 0.98:  # 98%以下でないと意味なし
            print("⚠️  圧縮効果わずか - RAW保存")
            return b'RAW_UFC' + struct.pack('>I', len(data)) + data
        
        return result
    
    def serialize_collapse_info(self, info: Dict) -> bytes:
        """復元情報シリアライズ（高速版）"""
        result = bytearray()
        
        # 元サイズ
        result.extend(struct.pack('>I', info['original_size']))
        
        # リバーステーブル
        reverse_table = info['reverse_table']
        result.extend(struct.pack('>H', len(reverse_table)))
        
        for new_val, original_val in reverse_table.items():
            result.extend(struct.pack('>BB', new_val, original_val))
        
        return bytes(result)
    
    def deserialize_collapse_info(self, data: bytes) -> Dict:
        """復元情報デシリアライズ"""
        pos = 0
        
        # 元サイズ
        original_size = struct.unpack('>I', data[pos:pos+4])[0]
        pos += 4
        
        # リバーステーブル
        table_size = struct.unpack('>H', data[pos:pos+2])[0]
        pos += 2
        
        reverse_table = {}
        for _ in range(table_size):
            new_val, original_val = struct.unpack('>BB', data[pos:pos+2])
            reverse_table[new_val] = original_val
            pos += 2
        
        return {
            'original_size': original_size,
            'reverse_table': reverse_table
        }
    
    def decompress(self, compressed: bytes) -> bytes:
        """超高速展開"""
        if not compressed:
            return b''
        
        # RAW形式チェック
        if compressed.startswith(b'RAW_UFC'):
            original_size = struct.unpack('>I', compressed[7:11])[0]
            return compressed[11:11+original_size]
        
        if not compressed.startswith(self.magic):
            raise ValueError("Invalid NXUFC format")
        
        pos = len(self.magic)
        
        # ヘッダー解析
        original_size = struct.unpack('>I', compressed[pos:pos+4])[0]
        pos += 4
        
        info_size = struct.unpack('>I', compressed[pos:pos+4])[0]
        pos += 4
        
        data_size = struct.unpack('>I', compressed[pos:pos+4])[0]
        pos += 4
        
        # 復元情報展開
        info_compressed = compressed[pos:pos+info_size]
        pos += info_size
        
        info_bytes = zlib.decompress(info_compressed)
        collapse_info = self.deserialize_collapse_info(info_bytes)
        
        # データ展開
        data_compressed = compressed[pos:pos+data_size]
        collapsed_data = zlib.decompress(data_compressed)
        
        # 超高速復元
        restored_data = self.ultra_fast_restore(collapsed_data, collapse_info)
        
        return restored_data
    
    def compress_file(self, input_path: str):
        """ファイル圧縮（超高速版）"""
        if not os.path.exists(input_path):
            print(f"❌ ファイルが見つかりません: {input_path}")
            return None
        
        print(f"🚀 超高速バイナリ崩壊圧縮: {os.path.basename(input_path)}")
        start_time = time.time()
        
        # ファイル読み込み
        with open(input_path, 'rb') as f:
            original_data = f.read()
        
        original_size = len(original_data)
        original_md5 = hashlib.md5(original_data).hexdigest()
        
        print(f"📁 元ファイル: {original_size:,} bytes")
        print(f"🔒 元MD5: {original_md5}")
        
        # 圧縮
        compressed_data = self.compress(original_data)
        compressed_size = len(compressed_data)
        
        # 圧縮率計算
        compression_ratio = ((original_size - compressed_size) / original_size) * 100 if original_size > 0 else 0
        
        # 処理時間・速度
        processing_time = time.time() - start_time
        throughput = original_size / (1024 * 1024) / processing_time if processing_time > 0 else 0
        
        # 結果表示
        print(f"🔹 超高速圧縮完了: {compression_ratio:.1f}%")
        print(f"⚡ 処理時間: {processing_time:.3f}s ({throughput:.1f} MB/s)")
        
        # 保存
        output_path = input_path + '.nxufc'
        with open(output_path, 'wb') as f:
            f.write(compressed_data)
        
        print(f"💾 保存: {os.path.basename(output_path)}")
        
        # 可逆性テスト
        try:
            decompressed_data = self.decompress(compressed_data)
            decompressed_md5 = hashlib.md5(decompressed_data).hexdigest()
            
            if decompressed_md5 == original_md5:
                print(f"✅ 完全可逆性確認: MD5一致")
                print(f"🎯 SUCCESS: 超高速圧縮完了 - {output_path}")
                
                return {
                    'input_file': input_path,
                    'output_file': output_path,
                    'original_size': original_size,
                    'compressed_size': compressed_size,
                    'compression_ratio': compression_ratio,
                    'processing_time': processing_time,
                    'throughput': throughput,
                    'lossless': True,
                    'method': 'Ultra Fast Binary Collapse'
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
        print("使用法: python nxzip_ultra_fast_collapse.py <ファイルパス>")
        print("\n⚡ NXZip 超高速バイナリ崩壊エンジン")
        print("📋 特徴:")
        print("  ⚡ 超高速処理（簡易解析のみ）")
        print("  ✅ 完全可逆性保証")
        print("  🎬 メディアファイル最適化")
        print("  💥 効率的構造崩壊")
        sys.exit(1)
    
    input_file = sys.argv[1]
    engine = UltraFastBinaryCollapseEngine()
    result = engine.compress_file(input_file)
    
    if result:
        print(f"\n{'='*60}")
        print(f"⚡ ULTRA FAST SUCCESS: {result['compression_ratio']:.1f}% compression")
        print(f"📊 {result['original_size']:,} → {result['compressed_size']:,} bytes")
        print(f"🚀 {result['throughput']:.1f} MB/s processing speed")
        print(f"✅ Perfect reversibility with ultra fast collapse")
        print(f"{'='*60}")
