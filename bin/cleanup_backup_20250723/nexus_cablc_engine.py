#!/usr/bin/env python3
"""
CABLC (Custom AV-Inspired Binary Lossless Compressor)
AV1/AVIFの予測技術をバイナリデータに適用した高速ロスレス圧縮エンジン

特徴:
- AV1予測インスパイア（バイト差分予測）
- QOI風高速RLE（Run-Length Encoding）
- Python標準ライブラリのみ使用
- 完全ロスレス保証
- LZMA比で大幅高速化
"""

import struct
import time
import hashlib
import os
import sys

class CABLCEngine:
    def __init__(self):
        self.magic = b'CABLC'
        self.version = 1
    
    def predict(self, data: bytes) -> bytes:
        """AV1インスパイア予測: バイト差分予測"""
        if not data:
            return b''
        
        residuals = bytearray([data[0]])  # 最初のバイトはそのまま
        
        for i in range(1, len(data)):
            # 前のバイトから現在のバイトを予測
            pred = data[i-1]
            residual = (data[i] - pred) & 0xFF  # 8bit差分
            residuals.append(residual)
        
        return bytes(residuals)
    
    def inverse_predict(self, residuals: bytes) -> bytes:
        """逆予測: 残差から元データを復元"""
        if not residuals:
            return b''
        
        data = bytearray([residuals[0]])  # 最初のバイトはそのまま
        
        for i in range(1, len(residuals)):
            pred = data[i-1]  # 前のバイトが予測値
            value = (residuals[i] + pred) & 0xFF  # 残差 + 予測値
            data.append(value)
        
        return bytes(data)
    
    def rle_encode(self, data: bytes) -> bytes:
        """QOI風高速RLE: 繰り返し圧縮"""
        if not data:
            return b''
        
        encoded = bytearray()
        i = 0
        
        while i < len(data):
            count = 1
            val = data[i]
            
            # 同じ値が続く数をカウント（最大255）
            while i + count < len(data) and data[i + count] == val and count < 255:
                count += 1
            
            # カウント + 値の形式でエンコード
            encoded.append(count)
            encoded.append(val)
            i += count
        
        return bytes(encoded)
    
    def rle_decode(self, encoded: bytes) -> bytes:
        """RLEデコード"""
        if not encoded or len(encoded) % 2 != 0:
            return b''
        
        decoded = bytearray()
        i = 0
        
        while i < len(encoded):
            count = encoded[i]
            val = encoded[i + 1]
            decoded.extend([val] * count)
            i += 2
        
        return bytes(decoded)
    
    def compress(self, data: bytes) -> bytes:
        """CABLC圧縮（改良版：サイズ増加回避）"""
        if not data:
            return self.magic + struct.pack('>I', 0) + b''
        
        # ステップ1: 予測（残差計算）
        residuals = self.predict(data)
        
        # ステップ2: RLE圧縮
        rle_data = self.rle_encode(residuals)
        
        # ヘッダー + 圧縮データ
        header = self.magic + struct.pack('>I', len(data))
        compressed = header + rle_data
        
        # サイズ増加回避: 圧縮効果がない場合は生データ
        if len(compressed) >= len(data) + len(header):
            return b'RAW' + struct.pack('>I', len(data)) + data
        
        return compressed
    
    def decompress(self, compressed: bytes) -> bytes:
        """CABLC展開"""
        if not compressed:
            return b''
        
        # 生データ形式チェック
        if compressed.startswith(b'RAW'):
            original_size = struct.unpack('>I', compressed[3:7])[0]
            return compressed[7:7+original_size]
        
        # CABLC形式チェック
        if not compressed.startswith(self.magic):
            raise ValueError("Invalid CABLC format")
        
        # ヘッダー解析
        original_size = struct.unpack('>I', compressed[5:9])[0]
        rle_data = compressed[9:]
        
        # ステップ1: RLE展開
        residuals = self.rle_decode(rle_data)
        
        # ステップ2: 逆予測
        data = self.inverse_predict(residuals)
        
        # サイズ検証
        if len(data) != original_size:
            raise ValueError(f"Decompression size mismatch: expected {original_size}, got {len(data)}")
        
        return data
    
    def compress_file(self, input_path: str):
        """ファイル圧縮"""
        if not os.path.exists(input_path):
            print(f"❌ ファイルが見つかりません: {input_path}")
            return None
        
        print(f"🚀 CABLC圧縮開始: {os.path.basename(input_path)}")
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
        print(f"🔹 CABLC圧縮完了: {compression_ratio:.1f}%")
        print(f"⚡ 処理時間: {processing_time:.3f}s ({throughput:.1f} MB/s)")
        
        # 保存
        output_path = input_path + '.cablc'
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
        
        print(f"🎯 SUCCESS: CABLC圧縮完了 - {output_path}")
        
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
    
    def decompress_file(self, compressed_path: str):
        """ファイル展開"""
        if not os.path.exists(compressed_path):
            print(f"❌ ファイルが見つかりません: {compressed_path}")
            return None
        
        print(f"📦 CABLC展開開始: {os.path.basename(compressed_path)}")
        start_time = time.time()
        
        # 圧縮ファイル読み込み
        with open(compressed_path, 'rb') as f:
            compressed_data = f.read()
        
        # 展開
        decompressed_data = self.decompress(compressed_data)
        
        # 出力ファイル保存
        output_path = compressed_path.replace('.cablc', '.restored')
        with open(output_path, 'wb') as f:
            f.write(decompressed_data)
        
        processing_time = time.time() - start_time
        throughput = len(decompressed_data) / (1024 * 1024) / processing_time if processing_time > 0 else 0
        
        print(f"⚡ 展開時間: {processing_time:.3f}s ({throughput:.1f} MB/s)")
        print(f"💾 展開ファイル: {os.path.basename(output_path)}")
        print(f"✅ SUCCESS: CABLC展開完了")
        
        return output_path

def demo_test():
    """デモンストレーション"""
    print("🧪 CABLC デモンストレーション")
    
    # テストデータ（繰り返しパターン）
    sample_data = bytes([0] * 100 + [1] * 50 + [255] * 200 + list(range(256)) * 2)
    
    engine = CABLCEngine()
    
    print(f"📊 テストデータ: {len(sample_data)} bytes")
    
    # 圧縮
    start_time = time.time()
    compressed = engine.compress(sample_data)
    compress_time = time.time() - start_time
    
    # 展開
    start_time = time.time()
    decompressed = engine.decompress(compressed)
    decompress_time = time.time() - start_time
    
    # 結果
    compression_ratio = ((len(sample_data) - len(compressed)) / len(sample_data)) * 100
    lossless = decompressed == sample_data
    
    print(f"🔹 圧縮率: {compression_ratio:.2f}%")
    print(f"⚡ 圧縮時間: {compress_time*1000:.2f}ms")
    print(f"⚡ 展開時間: {decompress_time*1000:.2f}ms")
    print(f"✅ 完全可逆性: {lossless}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # デモ実行
        demo_test()
    elif len(sys.argv) == 2:
        input_file = sys.argv[1]
        engine = CABLCEngine()
        
        if input_file.endswith('.cablc'):
            # 展開
            engine.decompress_file(input_file)
        else:
            # 圧縮
            engine.compress_file(input_file)
    else:
        print("使用法:")
        print("  python nexus_cablc_engine.py                    # デモ実行")
        print("  python nexus_cablc_engine.py <ファイルパス>      # 圧縮")
        print("  python nexus_cablc_engine.py <ファイル.cablc>   # 展開")
