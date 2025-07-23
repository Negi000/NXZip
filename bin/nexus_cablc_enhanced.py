#!/usr/bin/env python3
"""
CABLC Enhanced (Custom AV-Inspired Binary Lossless Compressor Enhanced)
強化版CABLC - ブロック分割と複数予測モードで圧縮率向上

新機能:
- ブロック分割（256/1024バイト単位）
- 複数予測モード（平均予測、Paeth予測）
- 適応的予測選択
- より効果的なRLE
"""

import struct
import time
import hashlib
import os
import sys
from typing import List, Tuple

class CABLCEnhanced:
    def __init__(self, block_size: int = 1024):
        self.magic = b'CABLC2'  # Enhanced version
        self.version = 2
        self.block_size = min(block_size, 2048)  # 最大ブロックサイズ制限で高速化
    
    def predict_simple(self, data: bytes) -> bytes:
        """シンプル予測（前のバイト）"""
        if not data:
            return b''
        
        residuals = bytearray([data[0]])
        for i in range(1, len(data)):
            pred = data[i-1]
            residual = (data[i] - pred) & 0xFF
            residuals.append(residual)
        
        return bytes(residuals)
    
    def predict_average(self, data: bytes) -> bytes:
        """平均予測（AV1インスパイア）"""
        if not data:
            return b''
        
        residuals = bytearray([data[0]])
        for i in range(1, len(data)):
            if i == 1:
                pred = data[0]
            else:
                # 前の2バイトの平均
                pred = (data[i-1] + data[i-2]) // 2
            residual = (data[i] - pred) & 0xFF
            residuals.append(residual)
        
        return bytes(residuals)
    
    def predict_paeth(self, data: bytes, width: int = 16) -> bytes:
        """Paeth予測（PNG風・2D構造想定）"""
        if not data:
            return b''
        
        residuals = bytearray()
        
        for i in range(len(data)):
            if i < width:  # 最初の行
                pred = data[i-1] if i > 0 else 0
            else:  # 2行目以降
                left = data[i-1] if (i % width) > 0 else 0
                up = data[i-width]
                up_left = data[i-width-1] if (i % width) > 0 and i >= width+1 else 0
                
                # Paeth予測
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
            
            residual = (data[i] - pred) & 0xFF
            residuals.append(residual)
        
        return bytes(residuals)
    
    def inverse_predict_simple(self, residuals: bytes) -> bytes:
        """シンプル予測の逆処理"""
        if not residuals:
            return b''
        
        data = bytearray([residuals[0]])
        for i in range(1, len(residuals)):
            pred = data[i-1]
            value = (residuals[i] + pred) & 0xFF
            data.append(value)
        
        return bytes(data)
    
    def inverse_predict_average(self, residuals: bytes) -> bytes:
        """平均予測の逆処理"""
        if not residuals:
            return b''
        
        data = bytearray([residuals[0]])
        for i in range(1, len(residuals)):
            if i == 1:
                pred = data[0]
            else:
                pred = (data[i-1] + data[i-2]) // 2
            value = (residuals[i] + pred) & 0xFF
            data.append(value)
        
        return bytes(data)
    
    def inverse_predict_paeth(self, residuals: bytes, width: int = 16) -> bytes:
        """Paeth予測の逆処理"""
        if not residuals:
            return b''
        
        data = bytearray()
        
        for i in range(len(residuals)):
            if i < width:  # 最初の行
                pred = data[i-1] if i > 0 else 0
            else:  # 2行目以降
                left = data[i-1] if (i % width) > 0 else 0
                up = data[i-width]
                up_left = data[i-width-1] if (i % width) > 0 and i >= width+1 else 0
                
                # Paeth予測
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
            
            value = (residuals[i] + pred) & 0xFF
            data.append(value)
        
        return bytes(data)
    
    def enhanced_rle_encode(self, data: bytes) -> bytes:
        """強化RLE: より効率的な繰り返し圧縮（高速化版）"""
        if not data:
            return b''
        
        encoded = bytearray()
        i = 0
        
        while i < len(data):
            val = data[i]
            count = 1
            
            # 繰り返し検出（最大32個まで制限で高速化）
            max_count = min(32, len(data) - i)
            while count < max_count and data[i + count] == val:
                count += 1
            
            if count >= 3:  # 3回以上の繰り返しで圧縮
                encoded.append(0xFF)  # エスケープ
                encoded.append(count)
                encoded.append(val)
                i += count
            else:
                # 単独値（エスケープ処理）
                if val == 0xFF:
                    encoded.extend([0xFF, 0x00])  # エスケープ
                else:
                    encoded.append(val)
                i += 1
        
        return bytes(encoded)
    
    def enhanced_rle_decode(self, encoded: bytes) -> bytes:
        """強化RLEデコード"""
        if not encoded:
            return b''
        
        decoded = bytearray()
        i = 0
        
        while i < len(encoded):
            if encoded[i] == 0xFF and i + 1 < len(encoded):
                if encoded[i + 1] == 0x00:
                    # エスケープされた0xFF
                    decoded.append(0xFF)
                    i += 2
                else:
                    # RLE: カウント + 値
                    count = encoded[i + 1]
                    val = encoded[i + 2] if i + 2 < len(encoded) else 0
                    decoded.extend([val] * count)
                    i += 3
            else:
                # 通常の値
                decoded.append(encoded[i])
                i += 1
        
        return bytes(decoded)
    
    def find_best_prediction(self, block: bytes) -> Tuple[int, bytes]:
        """最適な予測モードを選択（高速化版）"""
        if len(block) == 0:
            return 0, b''
        
        # 小さなブロックは単純予測のみ
        if len(block) < 64:
            return 0, self.predict_simple(block)
        
        # サンプリングによる高速評価（最初の32バイトのみ評価）
        sample = block[:32]
        
        predictions = [
            (0, self.predict_simple(sample)),
            (1, self.predict_average(sample)),
            (2, self.predict_paeth(sample, min(16, len(sample))))
        ]
        
        # 最も圧縮効果の高い予測を選択（簡易版）
        best_mode = 0
        best_score = float('inf')
        
        for mode, residuals in predictions:
            # 簡易スコア（0の数をカウント）
            score = sum(1 for b in residuals if b == 0)
            if score > best_score:  # 0が多いほど良い
                best_score = score
                best_mode = mode
        
        # 選択されたモードで全体を処理
        if best_mode == 0:
            return 0, self.predict_simple(block)
        elif best_mode == 1:
            return 1, self.predict_average(block)
        else:
            return 2, self.predict_paeth(block, min(16, len(block)))
    
    def calculate_entropy_score(self, data: bytes) -> float:
        """データのエントロピースコア（簡易版）"""
        if not data:
            return 0.0
        
        # バイト値の分散を計算（低い方が圧縮しやすい）
        if len(data) == 1:
            return 0.0
        
        mean = sum(data) / len(data)
        variance = sum((b - mean) ** 2 for b in data) / len(data)
        return variance
    
    def compress(self, data: bytes) -> bytes:
        """強化CABLC圧縮"""
        if not data:
            return self.magic + struct.pack('>I', 0) + b''
        
        # ブロック分割
        blocks = []
        for i in range(0, len(data), self.block_size):
            block = data[i:i+self.block_size]
            blocks.append(block)
        
        # 各ブロックを最適予測で圧縮
        compressed_blocks = []
        modes = []
        
        for block in blocks:
            mode, residuals = self.find_best_prediction(block)
            rle_data = self.enhanced_rle_encode(residuals)
            
            modes.append(mode)
            compressed_blocks.append(rle_data)
        
        # パッケージング
        header = self.magic + struct.pack('>I', len(data))
        header += struct.pack('>H', self.block_size)  # ブロックサイズ
        header += struct.pack('>H', len(blocks))      # ブロック数
        
        # 予測モード配列
        mode_data = bytes(modes)
        header += struct.pack('>H', len(mode_data)) + mode_data
        
        # 圧縮ブロックサイズ配列
        block_sizes = [len(block) for block in compressed_blocks]
        size_data = b''.join(struct.pack('>H', size) for size in block_sizes)
        header += size_data
        
        # 圧縮データ
        compressed_data = header + b''.join(compressed_blocks)
        
        # サイズ増加回避
        if len(compressed_data) >= len(data) + len(self.magic) + 4:
            return b'RAW2' + struct.pack('>I', len(data)) + data
        
        return compressed_data
    
    def decompress(self, compressed: bytes) -> bytes:
        """強化CABLC展開"""
        if not compressed:
            return b''
        
        # RAW形式チェック
        if compressed.startswith(b'RAW2'):
            original_size = struct.unpack('>I', compressed[4:8])[0]
            return compressed[8:8+original_size]
        
        # CABLC2形式チェック
        if not compressed.startswith(self.magic):
            raise ValueError("Invalid CABLC2 format")
        
        pos = len(self.magic)
        
        # ヘッダー解析
        original_size = struct.unpack('>I', compressed[pos:pos+4])[0]
        pos += 4
        
        block_size = struct.unpack('>H', compressed[pos:pos+2])[0]
        pos += 2
        
        num_blocks = struct.unpack('>H', compressed[pos:pos+2])[0]
        pos += 2
        
        # 予測モード配列
        mode_data_size = struct.unpack('>H', compressed[pos:pos+2])[0]
        pos += 2
        
        modes = list(compressed[pos:pos+mode_data_size])
        pos += mode_data_size
        
        # ブロックサイズ配列
        block_sizes = []
        for _ in range(num_blocks):
            size = struct.unpack('>H', compressed[pos:pos+2])[0]
            block_sizes.append(size)
            pos += 2
        
        # 各ブロックを展開
        result = bytearray()
        
        for i in range(num_blocks):
            mode = modes[i]
            block_size = block_sizes[i]
            rle_data = compressed[pos:pos+block_size]
            pos += block_size
            
            # RLE展開
            residuals = self.enhanced_rle_decode(rle_data)
            
            # 予測逆処理
            if mode == 0:
                block_data = self.inverse_predict_simple(residuals)
            elif mode == 1:
                block_data = self.inverse_predict_average(residuals)
            elif mode == 2:
                block_data = self.inverse_predict_paeth(residuals, 16)
            else:
                raise ValueError(f"Unknown prediction mode: {mode}")
            
            result.extend(block_data)
        
        # サイズ検証
        if len(result) != original_size:
            raise ValueError(f"Decompression size mismatch: expected {original_size}, got {len(result)}")
        
        return bytes(result)
    
    def compress_file(self, input_path: str):
        """ファイル圧縮（強化版）"""
        if not os.path.exists(input_path):
            print(f"❌ ファイルが見つかりません: {input_path}")
            return None
        
        print(f"🚀 CABLC Enhanced圧縮開始: {os.path.basename(input_path)}")
        print(f"🔧 ブロックサイズ: {self.block_size} bytes")
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
        print(f"🔹 CABLC Enhanced完了: {compression_ratio:.1f}%")
        print(f"⚡ 処理時間: {processing_time:.3f}s ({throughput:.1f} MB/s)")
        
        # 保存
        output_path = input_path + '.cablc2'
        with open(output_path, 'wb') as f:
            f.write(compressed_data)
        
        print(f"💾 保存: {os.path.basename(output_path)}")
        
        # 可逆性テスト
        decompressed_data = self.decompress(compressed_data)
        decompressed_md5 = hashlib.md5(decompressed_data).hexdigest()
        
        if decompressed_md5 == original_md5:
            print(f"✅ 完全可逆性確認: MD5一致")
        else:
            print(f"❌ エラー: MD5不一致")
            print(f"   元: {original_md5}")
            print(f"   復元: {decompressed_md5}")
            return None
        
        print(f"🎯 SUCCESS: CABLC Enhanced完了 - {output_path}")
        
        return {
            'input_file': input_path,
            'output_file': output_path,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
            'processing_time': processing_time,
            'throughput': throughput,
            'lossless': True
        }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用法: python nexus_cablc_enhanced.py <ファイルパス>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # 高速化：1つのブロックサイズのみでテスト
    print(f"\n{'='*60}")
    print(f"CABLC Enhanced - 高速化版")
    print(f"{'='*60}")
    
    engine = CABLCEnhanced(block_size=1024)  # 固定サイズで高速化
    engine.compress_file(input_file)
