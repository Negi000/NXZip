#!/usr/bin/env python3
"""
Nexus Ultra Fast Optimized Compressor
超高速最適化圧縮エンジン - 効率重視の極限圧縮

特徴:
- 単一パス処理による高速化
- 最適化されたバイト予測アルゴリズム
- インライン処理によるオーバーヘッド削減
- 並列処理風の効率的なブロック処理
- 完全可逆性保証
"""

import struct
import time
import hashlib
import os
import sys
from typing import List, Tuple

class UltraFastOptimized:
    def __init__(self):
        self.magic = b'NXUF'  # Nexus Ultra Fast
        self.version = 1
        
    def fast_differential_compress(self, data: bytes) -> bytes:
        """超高速差分圧縮（単一パス）"""
        if not data:
            return b''
        
        # 単一パスで差分+RLE処理
        result = bytearray()
        prev_byte = 0
        i = 0
        
        while i < len(data):
            current = data[i]
            diff = (current - prev_byte) & 0xFF
            
            # 高速RLE検出（最大3バイト先読み）
            count = 1
            max_check = min(i + 64, len(data))  # 64バイト制限で高速化
            
            while i + count < max_check and data[i + count] == current:
                count += 1
            
            if count >= 4:  # 4回以上で圧縮
                result.extend([0xFF, count & 0xFF, diff])
                i += count
            else:
                if diff == 0xFF:
                    result.extend([0xFF, 0x00])  # エスケープ
                else:
                    result.append(diff)
                i += 1
            
            prev_byte = current
        
        return bytes(result)
    
    def fast_differential_decompress(self, compressed: bytes) -> bytes:
        """超高速差分展開"""
        if not compressed:
            return b''
        
        result = bytearray()
        prev_byte = 0
        i = 0
        
        while i < len(compressed):
            if compressed[i] == 0xFF and i + 1 < len(compressed):
                if compressed[i + 1] == 0x00:
                    # エスケープされた0xFF差分
                    current = (prev_byte + 0xFF) & 0xFF
                    result.append(current)
                    prev_byte = current
                    i += 2
                else:
                    # RLE展開
                    count = compressed[i + 1]
                    diff = compressed[i + 2] if i + 2 < len(compressed) else 0
                    current = (prev_byte + diff) & 0xFF
                    result.extend([current] * count)
                    prev_byte = current
                    i += 3
            else:
                # 通常の差分
                diff = compressed[i]
                current = (prev_byte + diff) & 0xFF
                result.append(current)
                prev_byte = current
                i += 1
        
        return bytes(result)
    
    def adaptive_prediction_fast(self, data: bytes) -> bytes:
        """高速適応予測（簡略化）"""
        if len(data) < 2:
            return data
        
        residuals = bytearray([data[0]])
        
        # 簡略化された予測ロジック
        for i in range(1, len(data)):
            if i == 1:
                pred = data[0]
            elif i == 2:
                pred = data[i-1]
            else:
                # 3点平均予測（高速）
                pred = (data[i-1] + data[i-2] + data[i-3]) // 3
            
            residual = (data[i] - pred) & 0xFF
            residuals.append(residual)
        
        return bytes(residuals)
    
    def inverse_adaptive_prediction_fast(self, residuals: bytes) -> bytes:
        """高速適応予測の逆処理"""
        if len(residuals) < 2:
            return residuals
        
        data = bytearray([residuals[0]])
        
        for i in range(1, len(residuals)):
            if i == 1:
                pred = data[0]
            elif i == 2:
                pred = data[i-1]
            else:
                # 3点平均予測（高速）
                pred = (data[i-1] + data[i-2] + data[i-3]) // 3
            
            value = (residuals[i] + pred) & 0xFF
            data.append(value)
        
        return bytes(data)
    
    def ultra_fast_compress(self, data: bytes) -> bytes:
        """超高速圧縮メイン処理"""
        if not data:
            return self.magic + struct.pack('>I', 0)
        
        # ステップ1: 高速適応予測
        predicted = self.adaptive_prediction_fast(data)
        
        # ステップ2: 高速差分+RLE圧縮
        compressed = self.fast_differential_compress(predicted)
        
        # ヘッダー作成
        header = self.magic + struct.pack('>I', len(data))
        result = header + compressed
        
        # サイズ増加回避
        if len(result) >= len(data) + 8:
            return b'RAWUF' + struct.pack('>I', len(data)) + data
        
        return result
    
    def ultra_fast_decompress(self, compressed: bytes) -> bytes:
        """超高速展開メイン処理"""
        if not compressed:
            return b''
        
        # RAW形式チェック
        if compressed.startswith(b'RAWUF'):
            original_size = struct.unpack('>I', compressed[5:9])[0]
            return compressed[9:9+original_size]
        
        # フォーマットチェック
        if not compressed.startswith(self.magic):
            raise ValueError("Invalid NXUF format")
        
        # ヘッダー解析
        original_size = struct.unpack('>I', compressed[4:8])[0]
        compressed_data = compressed[8:]
        
        # ステップ1: 高速差分展開
        differential_data = self.fast_differential_decompress(compressed_data)
        
        # ステップ2: 高速適応予測逆処理
        result = self.inverse_adaptive_prediction_fast(differential_data)
        
        if len(result) != original_size:
            raise ValueError(f"Size mismatch: expected {original_size}, got {len(result)}")
        
        return result
    
    def compress_file(self, input_path: str):
        """超高速ファイル圧縮"""
        if not os.path.exists(input_path):
            print(f"❌ ファイルが見つかりません: {input_path}")
            return None
        
        print(f"⚡ 超高速圧縮開始: {os.path.basename(input_path)}")
        start_time = time.time()
        
        # ファイル読み込み
        with open(input_path, 'rb') as f:
            original_data = f.read()
        
        original_size = len(original_data)
        original_md5 = hashlib.md5(original_data).hexdigest()
        
        print(f"📁 元ファイル: {original_size:,} bytes")
        
        # 超高速圧縮
        compressed_data = self.ultra_fast_compress(original_data)
        compressed_size = len(compressed_data)
        
        # 圧縮率計算
        compression_ratio = ((original_size - compressed_size) / original_size) * 100 if original_size > 0 else 0
        
        # 処理時間・速度
        processing_time = time.time() - start_time
        throughput = original_size / (1024 * 1024) / processing_time if processing_time > 0 else 0
        
        # 結果表示
        print(f"🚀 圧縮完了: {compression_ratio:.1f}%")
        print(f"⚡ 処理時間: {processing_time:.3f}s ({throughput:.1f} MB/s)")
        
        # 保存
        output_path = input_path + '.nxuf'
        with open(output_path, 'wb') as f:
            f.write(compressed_data)
        
        # 超高速可逆性テスト
        test_start = time.time()
        decompressed_data = self.ultra_fast_decompress(compressed_data)
        test_time = time.time() - test_start
        
        decompressed_md5 = hashlib.md5(decompressed_data).hexdigest()
        
        if decompressed_md5 == original_md5:
            print(f"✅ 完全可逆性確認: MD5一致 ({test_time:.3f}s)")
            print(f"🎯 SUCCESS: 超高速圧縮完了 - {output_path}")
            
            return {
                'input_file': input_path,
                'output_file': output_path,
                'original_size': original_size,
                'compressed_size': compressed_size,
                'compression_ratio': compression_ratio,
                'processing_time': processing_time,
                'throughput': throughput,
                'decompression_time': test_time,
                'lossless': True,
                'method': 'Ultra Fast Optimized'
            }
        else:
            print(f"❌ エラー: MD5不一致")
            return None

class BatchCompressor:
    """バッチ処理用の超高速圧縮"""
    
    def __init__(self):
        self.engine = UltraFastOptimized()
    
    def compress_multiple_files(self, file_paths: List[str]):
        """複数ファイルの超高速バッチ圧縮"""
        results = []
        total_start = time.time()
        
        print(f"🚀 バッチ圧縮開始: {len(file_paths)} ファイル")
        print("=" * 60)
        
        for i, file_path in enumerate(file_paths, 1):
            print(f"\n[{i}/{len(file_paths)}] {os.path.basename(file_path)}")
            result = self.engine.compress_file(file_path)
            if result:
                results.append(result)
        
        total_time = time.time() - total_start
        
        if results:
            total_original = sum(r['original_size'] for r in results)
            total_compressed = sum(r['compressed_size'] for r in results)
            avg_compression = ((total_original - total_compressed) / total_original) * 100
            total_throughput = total_original / (1024 * 1024) / total_time
            
            print(f"\n{'='*60}")
            print(f"🏆 バッチ圧縮完了!")
            print(f"📊 総圧縮率: {avg_compression:.1f}%")
            print(f"⚡ 総処理時間: {total_time:.3f}s")
            print(f"🚀 総スループット: {total_throughput:.1f} MB/s")
            print(f"✅ 成功率: {len(results)}/{len(file_paths)} ({len(results)/len(file_paths)*100:.1f}%)")
        
        return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用法: python nexus_ultra_fast_optimized.py <ファイルパス> [ファイルパス2] ...")
        print("\n⚡ 超高速最適化圧縮エンジン")
        print("📋 特徴:")
        print("  🚀 単一パス処理による高速化")
        print("  🧠 最適化された予測アルゴリズム")
        print("  💨 インライン処理でオーバーヘッド削減")
        print("  ✅ 完全可逆性保証")
        print("  📦 バッチ処理対応")
        sys.exit(1)
    
    input_files = sys.argv[1:]
    
    if len(input_files) == 1:
        # 単一ファイル処理
        engine = UltraFastOptimized()
        result = engine.compress_file(input_files[0])
    else:
        # バッチ処理
        batch = BatchCompressor()
        results = batch.compress_multiple_files(input_files)
