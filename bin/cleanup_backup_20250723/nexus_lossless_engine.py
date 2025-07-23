#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NXZip完全可逆圧縮エンジン - AV1/AVIFロスレス技術応用
AV1/AVIFのロスレスモード（予測 + エントロピーコーディング）をシミュレート
完全可逆性（ロスレス）を確保した高性能圧縮システム

🎯 目標: 完全可逆性を保ちつつ最大圧縮率達成
- PNG: 予測フィルタ + LZMA（AVIFロスレス予測応用）
- MP4: フレーム差分予測 + LZMA（AV1インタ予測応用）
- 汎用: バイトレベル予測 + LZMA（エントロピーコーディング）
"""

import os
import time
import struct
import hashlib
import lzma
import zlib
import numpy as np
from typing import Dict, List, Tuple

class NXZipLosslessEngine:
    """NXZip完全可逆圧縮エンジン（AV1/AVIFロスレス技術応用）"""
    
    def __init__(self):
        self.signature = b'NXLSLS'  # NXZip Lossless
        self.version = 1
        
    def detect_format(self, data: bytes) -> str:
        """フォーマット検出"""
        if data.startswith(b'\x89PNG\r\n\x1a\n'):
            return 'PNG'
        elif data.startswith(b'\xFF\xD8\xFF'):
            return 'JPEG'
        elif len(data) > 8 and data[4:8] == b'ftyp':
            return 'MP4'
        elif data.startswith(b'RIFF') and len(data) > 12 and data[8:12] == b'WAVE':
            return 'WAV'
        elif data.startswith(b'ID3') or data.startswith(b'\xFF\xFB'):
            return 'MP3'
        else:
            return 'BINARY'
    
    def simple_predict_diff(self, data: bytes) -> bytes:
        """シンプル差分予測（numpy不要版）"""
        if len(data) == 0:
            return b''
        
        result = bytearray()
        prev_byte = 0
        
        for byte in data:
            diff = (byte - prev_byte) % 256
            result.append(diff)
            prev_byte = byte
        
        return bytes(result)
    
    def inverse_predict_diff(self, predicted: bytes) -> bytes:
        """差分予測逆変換"""
        if len(predicted) == 0:
            return b''
        
        result = bytearray()
        current_byte = 0
        
        for diff in predicted:
            current_byte = (current_byte + diff) % 256
            result.append(current_byte)
        
        return bytes(result)
    
    def paeth_predictor(self, a: int, b: int, c: int) -> int:
        """Paethフィルタ（PNG標準予測器）"""
        p = a + b - c
        pa = abs(p - a)
        pb = abs(p - b)
        pc = abs(p - c)
        
        if pa <= pb and pa <= pc:
            return a
        elif pb <= pc:
            return b
        else:
            return c
    
    def png_advanced_predict(self, data: bytes, width: int = 64) -> bytes:
        """PNG向け高度予測（Paethフィルタ応用）"""
        if len(data) < width:
            return self.simple_predict_diff(data)
        
        result = bytearray()
        
        for i in range(len(data)):
            current = data[i]
            
            # 左の画素
            left = data[i-1] if i > 0 else 0
            
            # 上の画素
            up = data[i-width] if i >= width else 0
            
            # 左上の画素
            up_left = data[i-width-1] if i >= width and i % width > 0 else 0
            
            # Paeth予測
            predicted = self.paeth_predictor(left, up, up_left)
            diff = (current - predicted) % 256
            result.append(diff)
        
        return bytes(result)
    
    def png_advanced_inverse(self, predicted: bytes, width: int = 64) -> bytes:
        """PNG高度予測逆変換"""
        if len(predicted) < width:
            return self.inverse_predict_diff(predicted)
        
        result = bytearray(len(predicted))
        
        for i in range(len(predicted)):
            diff = predicted[i]
            
            # 左の画素
            left = result[i-1] if i > 0 else 0
            
            # 上の画素
            up = result[i-width] if i >= width else 0
            
            # 左上の画素
            up_left = result[i-width-1] if i >= width and i % width > 0 else 0
            
            # Paeth予測
            predicted_val = self.paeth_predictor(left, up, up_left)
            current = (predicted_val + diff) % 256
            result[i] = current
        
        return bytes(result)
    
    def mp4_frame_predict(self, data: bytes, frame_size: int = 1024) -> bytes:
        """MP4向けフレーム差分予測（AV1インタ予測応用）"""
        if len(data) < frame_size * 2:
            return self.simple_predict_diff(data)
        
        result = bytearray()
        
        # 最初のフレームはそのまま
        result.extend(data[:frame_size])
        
        # 後続フレームは差分
        for i in range(frame_size, len(data), frame_size):
            frame_end = min(i + frame_size, len(data))
            current_frame = data[i:frame_end]
            prev_frame = data[i-frame_size:i]
            
            # フレーム間差分
            for j in range(len(current_frame)):
                if j < len(prev_frame):
                    diff = (current_frame[j] - prev_frame[j]) % 256
                else:
                    diff = current_frame[j]
                result.append(diff)
        
        return bytes(result)
    
    def mp4_frame_inverse(self, predicted: bytes, frame_size: int = 1024) -> bytes:
        """MP4フレーム差分逆変換"""
        if len(predicted) < frame_size * 2:
            return self.inverse_predict_diff(predicted)
        
        result = bytearray()
        
        # 最初のフレーム復元
        result.extend(predicted[:frame_size])
        
        # 後続フレーム復元
        for i in range(frame_size, len(predicted), frame_size):
            frame_end = min(i + frame_size, len(predicted))
            diff_frame = predicted[i:frame_end]
            prev_frame = result[i-frame_size:i]
            
            # 差分から元フレーム復元
            current_frame = bytearray()
            for j in range(len(diff_frame)):
                if j < len(prev_frame):
                    current_byte = (prev_frame[j] + diff_frame[j]) % 256
                else:
                    current_byte = diff_frame[j]
                current_frame.append(current_byte)
            
            result.extend(current_frame)
        
        return bytes(result)
    
    def lossless_compress_data(self, data: bytes, format_type: str) -> bytes:
        """フォーマット別ロスレス圧縮"""
        print(f"   🔮 {format_type}向け予測処理...")
        
        # フォーマット別予測
        if format_type == 'PNG':
            # PNG向け高度予測（Paethフィルタ）
            predicted = self.png_advanced_predict(data, width=64)
            method = 2  # PNG高度予測
        elif format_type == 'MP4':
            # MP4向けフレーム差分予測
            predicted = self.mp4_frame_predict(data, frame_size=1024)
            method = 3  # MP4フレーム予測
        else:
            # 汎用バイト差分予測
            predicted = self.simple_predict_diff(data)
            method = 1  # シンプル差分
        
        print(f"   🗜️ LZMA高圧縮（preset=9）...")
        
        # LZMA2圧縮（最高圧縮率）
        compressed = lzma.compress(
            predicted, 
            format=lzma.FORMAT_RAW,
            filters=[{"id": lzma.FILTER_LZMA2, "preset": 9, "dict_size": 16777216}]  # 16MB辞書
        )
        
        return struct.pack('>B', method) + compressed
    
    def lossless_decompress_data(self, compressed_data: bytes, original_size: int) -> bytes:
        """ロスレス復元"""
        method = compressed_data[0]
        compressed = compressed_data[1:]
        
        print(f"   🔄 LZMA解凍...")
        
        # LZMA2解凍
        predicted = lzma.decompress(
            compressed,
            format=lzma.FORMAT_RAW,
            filters=[{"id": lzma.FILTER_LZMA2}]
        )
        
        print(f"   🔮 予測逆変換（方式: {method}）...")
        
        # 予測逆変換
        if method == 2:  # PNG高度予測
            decompressed = self.png_advanced_inverse(predicted, width=64)
        elif method == 3:  # MP4フレーム予測
            decompressed = self.mp4_frame_inverse(predicted, frame_size=1024)
        else:  # シンプル差分
            decompressed = self.inverse_predict_diff(predicted)
        
        # サイズチェック
        if len(decompressed) != original_size:
            # 必要に応じてトランケート
            decompressed = decompressed[:original_size]
        
        return decompressed
    
    def compress_file(self, input_path: str) -> Dict:
        """完全可逆ファイル圧縮"""
        if not os.path.exists(input_path):
            return {'error': f'File not found: {input_path}'}
        
        start_time = time.time()
        
        try:
            with open(input_path, 'rb') as f:
                original_data = f.read()
            
            original_size = len(original_data)
            original_hash = hashlib.md5(original_data).digest()
            format_type = self.detect_format(original_data)
            
            print(f"📁 処理: {os.path.basename(input_path)} ({original_size:,} bytes, {format_type})")
            print(f"🔒 完全可逆圧縮開始（AV1/AVIFロスレス技術応用）...")
            
            # ロスレス圧縮
            compressed_data = self.lossless_compress_data(original_data, format_type)
            
            # 最終パッケージ作成
            final_data = self._create_package(compressed_data, original_hash, original_size, format_type)
            
            # 保存
            output_path = input_path + '.nxz'
            with open(output_path, 'wb') as f:
                f.write(final_data)
            
            # 統計
            compressed_size = len(final_data)
            compression_ratio = (1 - compressed_size / original_size) * 100
            elapsed_time = time.time() - start_time
            speed = original_size / 1024 / 1024 / elapsed_time if elapsed_time > 0 else 0
            
            # 80%目標達成率
            target_80 = 80.0
            achievement = (compression_ratio / target_80) * 100 if target_80 > 0 else 0
            
            achievement_icon = "🏆" if compression_ratio >= 70 else "✅" if compression_ratio >= 50 else "⚠️" if compression_ratio >= 30 else "🔹"
            
            print(f"{achievement_icon} ロスレス圧縮完了: {compression_ratio:.1f}% (目標: 80%, 達成率: {achievement:.1f}%)")
            print(f"⚡ 処理時間: {elapsed_time:.2f}s ({speed:.1f} MB/s)")
            print(f"💾 保存: {os.path.basename(output_path)}")
            print(f"🔒 完全可逆性: 保証済み")
            
            return {
                'success': True,
                'input_file': input_path,
                'output_file': output_path,
                'original_size': original_size,
                'compressed_size': compressed_size,
                'compression_ratio': compression_ratio,
                'processing_time': elapsed_time,
                'lossless': True
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def decompress_file(self, input_path: str) -> Dict:
        """完全可逆復元"""
        if not os.path.exists(input_path):
            return {'error': f'File not found: {input_path}'}
        
        start_time = time.time()
        
        try:
            with open(input_path, 'rb') as f:
                compressed_data = f.read()
            
            # シグネチャチェック
            if not compressed_data.startswith(self.signature):
                return {'error': 'Invalid NXZ Lossless file signature'}
            
            print(f"📁 復元: {os.path.basename(input_path)}")
            print(f"🔓 完全可逆復元開始...")
            
            # メタデータ解析
            pos = len(self.signature)
            version = struct.unpack('>I', compressed_data[pos:pos+4])[0]
            pos += 4
            
            format_type = compressed_data[pos:pos+16].decode('utf-8').rstrip('\x00')
            pos += 16
            
            original_hash = compressed_data[pos:pos+16]
            pos += 16
            
            original_size = struct.unpack('>I', compressed_data[pos:pos+4])[0]
            pos += 4
            
            compressed_size = struct.unpack('>I', compressed_data[pos:pos+4])[0]
            pos += 4
            
            # データ復元
            payload = compressed_data[pos:pos+compressed_size]
            decompressed_data = self.lossless_decompress_data(payload, original_size)
            
            # ハッシュ検証
            recovered_hash = hashlib.md5(decompressed_data).digest()
            if recovered_hash != original_hash:
                return {'error': 'Hash verification failed - data corruption detected'}
            
            # 保存
            output_path = input_path.replace('.nxz', '.restored')
            with open(output_path, 'wb') as f:
                f.write(decompressed_data)
            
            # 統計
            elapsed_time = time.time() - start_time
            speed = len(decompressed_data) / 1024 / 1024 / elapsed_time if elapsed_time > 0 else 0
            
            print(f"✅ 完全可逆復元完了: {len(decompressed_data):,} bytes")
            print(f"⚡ 処理時間: {elapsed_time:.2f}s ({speed:.1f} MB/s)")
            print(f"💾 保存: {os.path.basename(output_path)}")
            print(f"🔐 ハッシュ検証: ✅ 完全一致")
            
            return {
                'success': True,
                'input_file': input_path,
                'output_file': output_path,
                'decompressed_size': len(decompressed_data),
                'processing_time': elapsed_time,
                'hash_verified': True
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _create_package(self, compressed_data: bytes, original_hash: bytes,
                       original_size: int, format_type: str) -> bytes:
        """最終パッケージ作成"""
        result = bytearray()
        
        # ヘッダー
        result.extend(self.signature)  # 6 bytes
        result.extend(struct.pack('>I', self.version))  # 4 bytes
        result.extend(format_type.encode('utf-8').ljust(16, b'\x00'))  # 16 bytes
        
        # メタデータ
        result.extend(original_hash)  # 16 bytes
        result.extend(struct.pack('>I', original_size))  # 4 bytes
        result.extend(struct.pack('>I', len(compressed_data)))  # 4 bytes
        
        # 圧縮データ
        result.extend(compressed_data)
        
        return bytes(result)

def main():
    import sys
    
    if len(sys.argv) != 2:
        print("🔒 NXZip完全可逆圧縮エンジン (AV1/AVIFロスレス技術応用)")
        print("=" * 70)
        print("使用方法: python nexus_lossless_engine.py <file>")
        print("復元: python nexus_lossless_engine.py <file.nxz>")
        print("")
        print("🚀 革新技術:")
        print("  • PNG: Paethフィルタ予測 + LZMA（AVIF応用）")
        print("  • MP4: フレーム差分予測 + LZMA（AV1応用）")
        print("  • 汎用: バイト差分予測 + LZMA（エントロピー最適化）")
        print("  • 完全可逆性: MD5ハッシュ検証による保証")
        print("  • 目標: ロスレスで最大圧縮率達成")
        return
    
    engine = NXZipLosslessEngine()
    
    # 復元処理
    if sys.argv[1].endswith('.nxz'):
        result = engine.decompress_file(sys.argv[1])
        if 'error' in result:
            print(f"❌ DECOMPRESS ERROR: {result['error']}")
            exit(1)
        else:
            print(f"✅ DECOMPRESS SUCCESS: 完全復元完了 - {result['output_file']}")
    else:
        # 圧縮処理
        result = engine.compress_file(sys.argv[1])
        if 'error' in result:
            print(f"❌ COMPRESS ERROR: {result['error']}")
            exit(1)
        else:
            print(f"✅ COMPRESS SUCCESS: ロスレス圧縮完了 - {result['output_file']}")

if __name__ == '__main__':
    main()
