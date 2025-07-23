#!/usr/bin/env python3
"""
CABLC PNG Decoder Engine
PNGファイルを内部展開してから再圧縮するエンジン

アプローチ:
1. PNGのIDATチャンクからzlib圧縮データを抽出
2. zlibで展開してピクセルデータを取得
3. ピクセルデータにCABLC圧縮を適用
4. より高い圧縮率を目指す
"""

import struct
import time
import hashlib
import os
import sys
import zlib
from typing import Dict, List, Tuple, Optional

class CABLCPNGDecoder:
    def __init__(self):
        self.magic = b'CABLCPNG'
        self.version = 1
    
    def parse_png_structure(self, png_data: bytes) -> Dict:
        """PNG構造を解析"""
        if not png_data.startswith(b'\x89PNG\r\n\x1a\n'):
            raise ValueError("Invalid PNG signature")
        
        chunks = []
        pos = 8  # PNG signature
        
        while pos < len(png_data):
            if pos + 8 >= len(png_data):
                break
            
            # チャンク情報読み取り
            chunk_length = struct.unpack('>I', png_data[pos:pos+4])[0]
            chunk_type = png_data[pos+4:pos+8]
            chunk_data = png_data[pos+8:pos+8+chunk_length]
            chunk_crc = png_data[pos+8+chunk_length:pos+12+chunk_length]
            
            chunks.append({
                'type': chunk_type,
                'length': chunk_length,
                'data': chunk_data,
                'crc': chunk_crc,
                'position': pos
            })
            
            pos += 12 + chunk_length
            
            if chunk_type == b'IEND':
                break
        
        return {
            'signature': png_data[:8],
            'chunks': chunks,
            'total_size': len(png_data)
        }
    
    def extract_pixel_data(self, png_structure: Dict) -> Tuple[bytes, Dict]:
        """PNGからピクセルデータを抽出"""
        ihdr_chunk = None
        idat_chunks = []
        other_chunks = []
        
        # チャンクを分類
        for chunk in png_structure['chunks']:
            if chunk['type'] == b'IHDR':
                ihdr_chunk = chunk
            elif chunk['type'] == b'IDAT':
                idat_chunks.append(chunk)
            else:
                other_chunks.append(chunk)
        
        if not ihdr_chunk:
            raise ValueError("IHDR chunk not found")
        
        if not idat_chunks:
            raise ValueError("IDAT chunks not found")
        
        # IHDR情報解析
        ihdr_data = ihdr_chunk['data']
        width = struct.unpack('>I', ihdr_data[0:4])[0]
        height = struct.unpack('>I', ihdr_data[4:8])[0]
        bit_depth = ihdr_data[8]
        color_type = ihdr_data[9]
        compression_method = ihdr_data[10]
        filter_method = ihdr_data[11]
        interlace_method = ihdr_data[12]
        
        print(f"📊 PNG情報: {width}x{height}, {bit_depth}bit, カラータイプ{color_type}")
        
        # IDATデータを結合
        combined_idat = b''.join(chunk['data'] for chunk in idat_chunks)
        
        # zlibで展開
        try:
            pixel_data = zlib.decompress(combined_idat)
            print(f"🔓 ピクセルデータ展開: {len(combined_idat):,} → {len(pixel_data):,} bytes")
        except Exception as e:
            raise ValueError(f"Failed to decompress IDAT data: {e}")
        
        metadata = {
            'width': width,
            'height': height,
            'bit_depth': bit_depth,
            'color_type': color_type,
            'compression_method': compression_method,
            'filter_method': filter_method,
            'interlace_method': interlace_method,
            'ihdr_chunk': ihdr_chunk,
            'other_chunks': other_chunks,
            'original_idat_size': len(combined_idat)
        }
        
        return pixel_data, metadata
    
    def cablc_compress_pixels(self, pixel_data: bytes) -> bytes:
        """ピクセルデータをCABLC圧縮（シンプル高効率版）"""
        print(f"🚀 ピクセルデータCABLC圧縮開始...")
        
        # シンプルな差分予測圧縮
        compressed_data = self.simple_differential_compress(pixel_data)
        
        # RLE圧縮
        rle_data = self.safe_rle_compress(compressed_data)
        
        print(f"📦 CABLC圧縮完了: {len(pixel_data):,} → {len(rle_data):,} bytes")
        return rle_data
    
    def simple_differential_compress(self, data: bytes) -> bytes:
        """シンプル差分圧縮（安全版）"""
        if not data:
            return b''
        
        result = bytearray()
        result.append(data[0])  # 最初のバイトはそのまま
        
        for i in range(1, len(data)):
            current = data[i]
            previous = data[i-1]
            
            # 安全な差分計算
            diff = (current - previous) % 256
            result.append(diff)
        
        return bytes(result)
    
    def safe_rle_compress(self, data: bytes) -> bytes:
        """安全なRLE圧縮"""
        if not data:
            return b''
        
        result = bytearray()
        i = 0
        
        while i < len(data):
            current_byte = data[i]
            count = 1
            
            # 連続する同じバイトをカウント
            while (i + count < len(data) and 
                   data[i + count] == current_byte and 
                   count < 255):
                count += 1
            
            if count >= 3:  # 3回以上で圧縮
                # エスケープシーケンス: [253, count, value]
                result.extend([253, count, current_byte])
                i += count
            else:
                # 通常のバイト（エスケープ処理）
                if current_byte == 253:
                    result.extend([253, 0])  # エスケープ
                else:
                    result.append(current_byte)
                i += 1
        
        return bytes(result)
    
    def rgb_correlation_compress(self, pixel_data: bytes) -> bytes:
        """RGB相関圧縮（カラーチャンネル間の予測）"""
        if len(pixel_data) < 3:
            return pixel_data
        
        result = bytearray()
        pos = 0
        
        # フィルタータイプを考慮しながら処理
        while pos < len(pixel_data):
            if pos >= len(pixel_data):
                break
            
            # フィルタータイプ保持
            if pos < len(pixel_data) and pixel_data[pos] < 5:  # Valid PNG filter type
                filter_type = pixel_data[pos]
                result.append(filter_type)
                pos += 1
                
                # RGB予測処理（3バイトずつ、安全な範囲で）
                line_start = pos
                processed = 0
                while pos < len(pixel_data) and processed < 10500:  # 安全な行長制限
                    if pos + 2 < len(pixel_data):
                        r, g, b = pixel_data[pos], pixel_data[pos+1], pixel_data[pos+2]
                        
                        # RGB相関予測（安全なバイト演算）
                        r_pred = min(255, max(0, g))  # 範囲制限
                        b_pred = min(255, max(0, g))
                        
                        r_diff = (r - r_pred) % 256  # 安全な差分計算
                        g_val = g
                        b_diff = (b - b_pred) % 256
                        
                        result.extend([r_diff, g_val, b_diff])
                        pos += 3
                        processed += 3
                    else:
                        # 端数処理
                        if pos < len(pixel_data):
                            result.append(pixel_data[pos])
                            pos += 1
                        break
            else:
                # 通常のバイト
                if pos < len(pixel_data):
                    result.append(pixel_data[pos])
                    pos += 1
        
        return bytes(result)
    
    def advanced_prediction(self, data: bytes) -> bytes:
        """高度差分予測"""
        if len(data) < 2:
            return data
        
        result = bytearray([data[0]])
        
        for i in range(1, len(data)):
            # 適応的予測
            if i < 4:
                pred = data[i-1]
            else:
                # 複数前の値から予測
                pred1 = data[i-1]
                pred2 = data[i-2]
                pred3 = data[i-3]
                pred4 = data[i-4]
                
                # 加重平均予測
                pred = (pred1 * 4 + pred2 * 2 + pred3 + pred4) // 8
            
            diff = (data[i] - pred) & 0xFF
            result.append(diff)
        
        return bytes(result)
    
    def enhanced_rle_compress(self, data: bytes) -> bytes:
        """強化RLE圧縮"""
        if not data:
            return b''
        
        result = bytearray()
        i = 0
        
        while i < len(data):
            val = data[i]
            count = 1
            
            # 連続検出（より積極的）
            while i + count < len(data) and data[i + count] == val and count < 255:
                count += 1
            
            if count >= 2:  # 2回以上の繰り返しで圧縮
                result.extend([254, count, val])  # エスケープシーケンス
                i += count
            else:
                # エスケープ処理
                if val == 254:
                    result.extend([254, 0])  # エスケープ
                else:
                    result.append(val)
                i += 1
        
        return bytes(result)
    
    def byte_level_optimize(self, data: bytes) -> bytes:
        """バイトレベル最適化"""
        if len(data) < 4:
            return data
        
        # 4バイトパターンの検出と置換
        patterns = {}
        result = bytearray()
        
        # 頻出4バイトパターンを検出
        for i in range(len(data) - 3):
            pattern = data[i:i+4]
            patterns[pattern] = patterns.get(pattern, 0) + 1
        
        # 頻出パターン（10回以上）を特別符号で置換
        frequent_patterns = {p: idx for idx, (p, count) in enumerate(patterns.items()) if count >= 10}
        
        if frequent_patterns:
            # パターンテーブル追加
            result.append(253)  # パターンテーブルマーカー
            result.append(len(frequent_patterns))
            
            for pattern, idx in frequent_patterns.items():
                result.append(idx)
                result.extend(pattern)
            
            # データ部分でパターン置換
            i = 0
            while i < len(data):
                if i + 3 < len(data):
                    pattern = data[i:i+4]
                    if pattern in frequent_patterns:
                        result.extend([252, frequent_patterns[pattern]])  # パターン参照
                        i += 4
                        continue
                
                result.append(data[i])
                i += 1
        else:
            # パターンなしマーカー
            result.append(251)
            result.extend(data)
        
        return bytes(result)
    
    def predict_and_compress(self, line_data: bytes, filter_type: int) -> bytes:
        """行データの予測圧縮"""
        if not line_data:
            return b''
        
        # フィルタタイプに応じた予測
        if filter_type == 0:  # None filter
            return self.simple_prediction(line_data)
        elif filter_type == 1:  # Sub filter
            return self.sub_prediction(line_data)
        elif filter_type == 2:  # Up filter
            return self.up_prediction(line_data)
        elif filter_type == 3:  # Average filter
            return self.average_prediction(line_data)
        elif filter_type == 4:  # Paeth filter
            return self.paeth_prediction(line_data)
        else:
            return self.simple_prediction(line_data)
    
    def simple_prediction(self, data: bytes) -> bytes:
        """シンプル差分予測"""
        if not data:
            return b''
        
        result = bytearray([data[0]])
        for i in range(1, len(data)):
            diff = (data[i] - data[i-1]) & 0xFF
            result.append(diff)
        
        return bytes(result)
    
    def sub_prediction(self, data: bytes) -> bytes:
        """Sub予測（PNG Sub filter逆処理）"""
        return self.simple_prediction(data)
    
    def up_prediction(self, data: bytes) -> bytes:
        """Up予測"""
        # 簡易的に前の値からの差分
        return self.simple_prediction(data)
    
    def average_prediction(self, data: bytes) -> bytes:
        """Average予測"""
        if len(data) < 2:
            return data
        
        result = bytearray([data[0]])
        for i in range(1, len(data)):
            if i == 1:
                pred = data[0]
            else:
                pred = (data[i-1] + data[i-2]) // 2
            diff = (data[i] - pred) & 0xFF
            result.append(diff)
        
        return bytes(result)
    
    def paeth_prediction(self, data: bytes) -> bytes:
        """Paeth予測"""
        if len(data) < 3:
            return self.simple_prediction(data)
        
        result = bytearray([data[0], data[1]])
        
        for i in range(2, len(data)):
            left = data[i-1]
            up = data[i-2]
            up_left = data[i-3] if i >= 3 else 0
            
            # Paeth predictor
            p = left + up - up_left
            pa = abs(p - left)
            pb = abs(p - up)
            pc = abs(p - up_left)
            
            if pa <= pb and pa <= pc:
                pred = left
            elif pb <= pc:
                pred = up
            else:
                pred = up_left
            
            diff = (data[i] - pred) & 0xFF
            result.append(diff)
        
        return bytes(result)
    
    def rle_compress(self, data: bytes) -> bytes:
        """RLE圧縮"""
        if not data:
            return b''
        
        result = bytearray()
        i = 0
        
        while i < len(data):
            val = data[i]
            count = 1
            
            # 連続する同じ値をカウント
            while i + count < len(data) and data[i + count] == val and count < 255:
                count += 1
            
            if count >= 3:
                # RLE: [255, count, value]
                result.extend([255, count, val])
                i += count
            else:
                # 通常値（255のエスケープ処理）
                if val == 255:
                    result.extend([255, 0])  # エスケープ
                else:
                    result.append(val)
                i += 1
        
        return bytes(result)
    
    def compress_png_file(self, input_path: str):
        """PNGファイル圧縮"""
        if not os.path.exists(input_path):
            print(f"❌ ファイルが見つかりません: {input_path}")
            return None
        
        print(f"🎨 PNG内部展開圧縮開始: {os.path.basename(input_path)}")
        start_time = time.time()
        
        # PNG読み込み
        with open(input_path, 'rb') as f:
            png_data = f.read()
        
        original_size = len(png_data)
        original_md5 = hashlib.md5(png_data).hexdigest()
        
        print(f"📁 元PNG: {original_size:,} bytes")
        print(f"🔒 元MD5: {original_md5}")
        
        try:
            # PNG構造解析
            png_structure = self.parse_png_structure(png_data)
            
            # ピクセルデータ抽出
            pixel_data, metadata = self.extract_pixel_data(png_structure)
            
            # CABLC圧縮
            compressed_pixels = self.cablc_compress_pixels(pixel_data)
            
            # パッケージング
            final_data = self.create_package(
                compressed_pixels, metadata, original_md5, original_size
            )
            
            # 保存
            output_path = input_path + '.cablcpng'
            with open(output_path, 'wb') as f:
                f.write(final_data)
            
            # 結果計算
            compressed_size = len(final_data)
            compression_ratio = ((original_size - compressed_size) / original_size) * 100
            processing_time = time.time() - start_time
            throughput = original_size / (1024 * 1024) / processing_time
            
            print(f"🔹 PNG内部圧縮完了: {compression_ratio:.1f}%")
            print(f"⚡ 処理時間: {processing_time:.3f}s ({throughput:.1f} MB/s)")
            print(f"💾 保存: {os.path.basename(output_path)}")
            print(f"✅ SUCCESS: PNG内部展開圧縮完了")
            
            return {
                'input_file': input_path,
                'output_file': output_path,
                'original_size': original_size,
                'compressed_size': compressed_size,
                'compression_ratio': compression_ratio,
                'processing_time': processing_time,
                'pixel_data_size': len(pixel_data),
                'compressed_pixel_size': len(compressed_pixels)
            }
            
        except Exception as e:
            print(f"❌ PNG処理エラー: {str(e)}")
            return None
    
    def create_package(self, compressed_pixels: bytes, metadata: Dict, 
                      original_md5: str, original_size: int) -> bytes:
        """最終パッケージ作成"""
        result = bytearray()
        
        # ヘッダー
        result.extend(self.magic)
        result.extend(struct.pack('>I', self.version))
        result.extend(original_md5.encode()[:32].ljust(32, b'\x00'))
        result.extend(struct.pack('>I', original_size))
        
        # メタデータ
        result.extend(struct.pack('>I', metadata['width']))
        result.extend(struct.pack('>I', metadata['height']))
        result.extend(struct.pack('B', metadata['bit_depth']))
        result.extend(struct.pack('B', metadata['color_type']))
        result.extend(struct.pack('B', metadata['compression_method']))
        result.extend(struct.pack('B', metadata['filter_method']))
        result.extend(struct.pack('B', metadata['interlace_method']))
        
        # 圧縮ピクセルデータ
        result.extend(struct.pack('>I', len(compressed_pixels)))
        result.extend(compressed_pixels)
        
        return bytes(result)

def main():
    if len(sys.argv) != 2:
        print("使用法: python nexus_cablc_png_decoder.py <png_file>")
        return
    
    input_file = sys.argv[1]
    
    if not input_file.lower().endswith('.png'):
        print("❌ PNGファイルを指定してください")
        return
    
    engine = CABLCPNGDecoder()
    result = engine.compress_png_file(input_file)
    
    if result:
        print(f"\n📊 詳細結果:")
        print(f"   元PNG: {result['original_size']:,} bytes")
        print(f"   ピクセルデータ: {result['pixel_data_size']:,} bytes")
        print(f"   圧縮後ピクセル: {result['compressed_pixel_size']:,} bytes")
        print(f"   最終ファイル: {result['compressed_size']:,} bytes")
        print(f"   圧縮率: {result['compression_ratio']:.1f}%")

if __name__ == "__main__":
    main()
