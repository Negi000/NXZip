#!/usr/bin/env python3
"""
NXZip Smart Image Compressor
スマート画像圧縮エンジン - 実用的な画像最適化圧縮

特徴:
- PNG/JPEGの特性に応じた最適化
- オーバーヘッド最小限のメタデータ
- ピクセルデータの効率的な再配置
- 実用的な圧縮率向上
"""

import struct
import time
import hashlib
import os
import sys
import zlib
from typing import Tuple, Dict
from collections import defaultdict

class SmartImageCompressor:
    def __init__(self):
        self.magic = b'NXSIC'  # NXZip Smart Image Compressor
        self.version = 1
        
    def detect_image_format(self, data: bytes) -> str:
        """画像形式検出"""
        if data.startswith(b'\x89PNG\r\n\x1a\n'):
            return 'PNG'
        elif data.startswith(b'\xff\xd8\xff'):
            return 'JPEG'
        else:
            return 'UNKNOWN'
    
    def smart_png_compression(self, data: bytes) -> Tuple[bytes, Dict]:
        """スマートPNG圧縮"""
        print("🔬 PNG最適化分析中...")
        
        # PNG構造解析（軽量版）
        pos = 8  # PNG署名スキップ
        idat_chunks = []
        other_chunks = []
        
        while pos < len(data):
            if pos + 8 > len(data):
                break
                
            chunk_length = struct.unpack('>I', data[pos:pos+4])[0]
            chunk_type = data[pos+4:pos+8]
            chunk_data = data[pos+8:pos+8+chunk_length]
            
            if chunk_type == b'IDAT':
                idat_chunks.append(chunk_data)
            else:
                other_chunks.append(data[pos:pos+12+chunk_length])
            
            pos += 12 + chunk_length
            if chunk_type == b'IEND':
                break
        
        # IDAT結合
        combined_idat = b''.join(idat_chunks)
        print(f"📦 IDAT結合: {len(idat_chunks)} chunks → {len(combined_idat):,} bytes")
        
        # ピクセルデータ展開
        try:
            pixel_data = zlib.decompress(combined_idat)
            print(f"🖼️  ピクセル展開: {len(pixel_data):,} bytes")
            
            # ピクセル最適化
            optimized_pixels = self.optimize_pixel_data(pixel_data)
            print(f"🎯 ピクセル最適化: {len(pixel_data):,} → {len(optimized_pixels):,}")
            
            # 最適IDAT再圧縮
            new_idat = zlib.compress(optimized_pixels, level=9)
            print(f"📦 IDAT再圧縮: {len(optimized_pixels):,} → {len(new_idat):,}")
            
            # 他チャンク結合
            other_data = b''.join(other_chunks)
            
            info = {
                'original_idat_size': len(combined_idat),
                'optimized_idat_size': len(new_idat),
                'other_chunks_size': len(other_data),
                'pixel_data_size': len(pixel_data)
            }
            
            return new_idat + other_data, info
            
        except Exception as e:
            print(f"⚠️  ピクセル展開失敗: {e}")
            return combined_idat, {'original_idat_size': len(combined_idat)}
    
    def smart_jpeg_compression(self, data: bytes) -> Tuple[bytes, Dict]:
        """スマートJPEG圧縮"""
        print("🔬 JPEG最適化分析中...")
        
        # JPEG構造を保持したまま最適化
        # エントロピー部分のみ抽出して再圧縮
        
        # SOSマーカー検索
        sos_pos = data.find(b'\xff\xda')
        if sos_pos == -1:
            print("⚠️  SOS見つからず - 元データ返却")
            return data, {}
        
        # ヘッダー部分（SOS含む）
        header_part = data[:sos_pos+2]
        
        # SOS長さ取得
        if sos_pos + 4 < len(data):
            sos_length = struct.unpack('>H', data[sos_pos+2:sos_pos+4])[0]
            sos_data = data[sos_pos+2:sos_pos+2+sos_length]
            entropy_start = sos_pos + 2 + sos_length
        else:
            sos_data = b''
            entropy_start = sos_pos + 2
        
        # エントロピーデータ
        entropy_data = data[entropy_start:]
        print(f"🗜️  エントロピーデータ: {len(entropy_data):,} bytes")
        
        # エントロピー最適化
        optimized_entropy = self.optimize_entropy_data(entropy_data)
        print(f"🎯 エントロピー最適化: {len(entropy_data):,} → {len(optimized_entropy):,}")
        
        info = {
            'header_size': len(header_part) + len(sos_data),
            'original_entropy_size': len(entropy_data),
            'optimized_entropy_size': len(optimized_entropy)
        }
        
        return header_part + sos_data + optimized_entropy, info
    
    def optimize_pixel_data(self, data: bytes) -> bytes:
        """ピクセルデータ最適化"""
        if len(data) < 2:
            return data
        
        # 1. バイト頻度順再配置
        freq = defaultdict(int)
        for byte in data:
            freq[byte] += 1
        
        # 頻度順マッピング
        sorted_bytes = sorted(freq.keys(), key=lambda x: freq[x], reverse=True)
        remap = {old: new for new, old in enumerate(sorted_bytes)}
        
        remapped = bytearray()
        for byte in data:
            remapped.append(remap[byte])
        
        # 2. 差分変換
        diff_data = bytearray([remapped[0]])
        for i in range(1, len(remapped)):
            diff = (remapped[i] - remapped[i-1]) & 0xFF
            diff_data.append(diff)
        
        # 3. 簡易RLE
        rle_data = self.simple_rle(bytes(diff_data))
        
        # リバースマッピングテーブル保存用
        reverse_map = bytes([sorted_bytes[i] if i < len(sorted_bytes) else 0 for i in range(256)])
        
        return reverse_map + rle_data
    
    def optimize_entropy_data(self, data: bytes) -> bytes:
        """エントロピーデータ最適化"""
        return self.optimize_pixel_data(data)  # 同じ最適化手法
    
    def simple_rle(self, data: bytes) -> bytes:
        """シンプルRLE"""
        if not data:
            return b''
        
        result = bytearray()
        i = 0
        
        while i < len(data):
            current = data[i]
            count = 1
            
            while count < 255 and i + count < len(data) and data[i + count] == current:
                count += 1
            
            if count >= 4:
                result.extend([0xFF, count, current])
                i += count
            else:
                if current == 0xFF:
                    result.extend([0xFE, current])
                else:
                    result.append(current)
                i += 1
        
        return bytes(result)
    
    def decompress_optimized_data(self, data: bytes) -> bytes:
        """最適化データ展開"""
        if len(data) < 256:
            return data
        
        # リバースマッピングテーブル
        reverse_map = data[:256]
        rle_data = data[256:]
        
        # RLE展開
        expanded = self.simple_rle_decode(rle_data)
        
        # 差分復元
        if len(expanded) > 0:
            restored = bytearray([expanded[0]])
            for i in range(1, len(expanded)):
                value = (restored[i-1] + expanded[i]) & 0xFF
                restored.append(value)
        else:
            restored = bytearray()
        
        # マッピング復元
        final_data = bytearray()
        for byte in restored:
            final_data.append(reverse_map[byte])
        
        return bytes(final_data)
    
    def simple_rle_decode(self, data: bytes) -> bytes:
        """シンプルRLE展開"""
        result = bytearray()
        i = 0
        
        while i < len(data):
            if i < len(data) - 2 and data[i] == 0xFF:
                count = data[i + 1]
                value = data[i + 2]
                result.extend([value] * count)
                i += 3
            elif i < len(data) - 1 and data[i] == 0xFE:
                result.append(data[i + 1])
                i += 2
            else:
                result.append(data[i])
                i += 1
        
        return bytes(result)
    
    def compress_image(self, data: bytes) -> bytes:
        """画像圧縮"""
        if not data:
            return self.magic + struct.pack('>I', 0)
        
        print(f"🚀 スマート画像圧縮開始: {len(data):,} bytes")
        original_md5 = hashlib.md5(data).hexdigest()
        
        # 形式検出
        image_format = self.detect_image_format(data)
        print(f"📷 検出形式: {image_format}")
        
        if image_format == 'PNG':
            optimized_data, info = self.smart_png_compression(data)
        elif image_format == 'JPEG':
            optimized_data, info = self.smart_jpeg_compression(data)
        else:
            print(f"⚠️  未対応形式 - RAW保存")
            return b'RAW_SIC' + struct.pack('>I', len(data)) + data
        
        # 最終zlib圧縮
        final_compressed = zlib.compress(optimized_data, level=9)
        
        # 軽量メタデータ
        metadata = {
            'format': image_format,
            'original_md5': original_md5,
            'info': info
        }
        
        meta_bytes = self.pack_metadata(metadata)
        
        # パッケージング
        header = self.magic + struct.pack('>I', len(data))
        header += struct.pack('>H', len(meta_bytes))
        header += struct.pack('>I', len(final_compressed))
        
        result = header + meta_bytes + final_compressed
        
        compression_ratio = ((len(data) - len(result)) / len(data)) * 100
        print(f"🏆 圧縮完了: {compression_ratio:.1f}% ({len(data):,} → {len(result):,})")
        
        if len(result) >= len(data) * 0.98:
            print("⚠️  圧縮効果わずか - RAW保存")
            return b'RAW_SIC' + struct.pack('>I', len(data)) + data
        
        return result
    
    def pack_metadata(self, metadata: Dict) -> bytes:
        """軽量メタデータパック"""
        result = bytearray()
        
        # 形式（1バイト）
        format_code = 1 if metadata['format'] == 'PNG' else 2
        result.append(format_code)
        
        # MD5（32バイト）
        result.extend(metadata['original_md5'].encode('utf-8')[:32])
        
        # 追加情報は最小限
        info_bytes = str(metadata['info']).encode('utf-8')
        result.extend(struct.pack('>H', len(info_bytes)))
        result.extend(info_bytes)
        
        return bytes(result)
    
    def compress_file(self, input_path: str):
        """ファイル圧縮"""
        if not os.path.exists(input_path):
            print(f"❌ ファイルが見つかりません: {input_path}")
            return None
        
        start_time = time.time()
        
        with open(input_path, 'rb') as f:
            data = f.read()
        
        original_md5 = hashlib.md5(data).hexdigest()
        print(f"📁 元ファイル: {len(data):,} bytes")
        print(f"🔒 元MD5: {original_md5}")
        
        compressed = self.compress_image(data)
        
        processing_time = time.time() - start_time
        compression_ratio = ((len(data) - len(compressed)) / len(data)) * 100
        throughput = len(data) / (1024 * 1024) / processing_time if processing_time > 0 else 0
        
        print(f"⚡ 処理時間: {processing_time:.3f}s ({throughput:.1f} MB/s)")
        
        # 保存
        output_path = input_path + '.nxsic'
        with open(output_path, 'wb') as f:
            f.write(compressed)
        
        print(f"💾 保存: {os.path.basename(output_path)}")
        print(f"🎯 SUCCESS: スマート画像圧縮完了")
        
        return {
            'compression_ratio': compression_ratio,
            'throughput': throughput,
            'output_size': len(compressed)
        }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用法: python nxzip_smart_image_compressor.py <画像ファイル>")
        print("\n🎯 NXZip スマート画像圧縮エンジン")
        print("📋 特徴:")
        print("  🔬 PNG/JPEG特性に応じた最適化")
        print("  📦 軽量メタデータ")
        print("  🎨 効率的ピクセル再配置")
        print("  ⚡ 実用的な圧縮率向上")
        sys.exit(1)
    
    input_file = sys.argv[1]
    compressor = SmartImageCompressor()
    compressor.compress_file(input_file)
