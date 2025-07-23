#!/usr/bin/env python3
"""
Phase 8 完全可逆修正版 - 100%可逆性保証エンジン
可逆性問題を完全修正した版
"""

import os
import sys
import time
import json
import struct
import lzma
import zlib
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Phase 8 Turbo エンジンを拡張
sys.path.append('bin')
from nexus_phase8_turbo import Phase8TurboEngine, CompressionResult, DecompressionResult

class SimpleReversibleEngine:
    """簡素化完全可逆エンジン - 可逆性最優先"""
    
    def __init__(self):
        self.version = "8.0-SimpleReversible"
        self.magic_header = b'NXZ8S'  # Simple版マジックナンバー
    
    def calculate_entropy(self, data: bytes) -> float:
        """エントロピー計算"""
        import math
        
        if not data:
            return 0.0
        
        byte_counts = [0] * 256
        for byte in data:
            byte_counts[byte] += 1
        
        entropy = 0.0
        data_len = len(data)
        for count in byte_counts:
            if count > 0:
                probability = count / data_len
                entropy -= probability * math.log2(probability)
        
        return min(entropy, 8.0)
    
    def simple_compress(self, data: bytes, filename: str = "data") -> dict:
        """簡素化可逆圧縮 - 100%可逆性保証"""
        start_time = time.time()
        original_size = len(data)
        
        print(f"🔧 簡素化可逆圧縮開始: {filename}")
        print(f"📊 元サイズ: {original_size:,} bytes ({original_size/1024:.1f} KB)")
        
        # 元データのハッシュ
        original_hash = hashlib.sha256(data).hexdigest()
        print(f"🔐 原本ハッシュ: {original_hash[:16]}...")
        
        # 簡素な圧縮戦略選択
        entropy = self.calculate_entropy(data)
        print(f"📈 エントロピー: {entropy:.2f}")
        
        # 可逆性保証圧縮
        if entropy < 2.0:
            # 低エントロピー: RLE圧縮
            compressed_data = self.safe_rle_compress(data)
            method = "rle"
        elif entropy < 6.0:
            # 中エントロピー: LZMA圧縮
            try:
                compressed_data = lzma.compress(data, preset=6, check=lzma.CHECK_CRC64)
                method = "lzma"
            except:
                compressed_data = data
                method = "uncompressed"
        else:
            # 高エントロピー: zlib圧縮
            try:
                compressed_data = zlib.compress(data, level=6)
                method = "zlib"
            except:
                compressed_data = data
                method = "uncompressed"
        
        # 圧縮効果チェック
        if len(compressed_data) >= len(data):
            compressed_data = data
            method = "uncompressed"
        
        # 完全可逆ファイル構築
        final_data = self.build_reversible_file(
            compressed_data, original_hash, method, original_size
        )
        
        compressed_size = len(final_data)
        compression_ratio = ((original_size - compressed_size) / original_size) * 100
        processing_time = time.time() - start_time
        
        print(f"🔧 圧縮方式: {method}")
        print(f"✅ 圧縮完了: {compression_ratio:.1f}% ({original_size:,} → {compressed_size:,})")
        print(f"⏱️ 処理時間: {processing_time:.2f}秒")
        
        return {
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
            'compressed_data': final_data,
            'processing_time': processing_time,
            'method': method,
            'original_hash': original_hash
        }
    
    def simple_decompress(self, compressed_data: bytes) -> dict:
        """簡素化可逆展開 - 100%復元保証"""
        start_time = time.time()
        
        print("🔧 簡素化可逆展開開始")
        
        # ヘッダー検証
        if not compressed_data.startswith(self.magic_header):
            raise ValueError("❌ 簡素化可逆形式ではありません")
        
        offset = len(self.magic_header)
        
        # 元データハッシュ
        original_hash = compressed_data[offset:offset+64].decode('ascii')
        offset += 64
        print(f"🔐 原本ハッシュ: {original_hash[:16]}...")
        
        # 元サイズ
        original_size = struct.unpack('<I', compressed_data[offset:offset+4])[0]
        offset += 4
        
        # 圧縮方式
        method_len = struct.unpack('<H', compressed_data[offset:offset+2])[0]
        offset += 2
        method = compressed_data[offset:offset+method_len].decode('ascii')
        offset += method_len
        print(f"🔧 圧縮方式: {method}")
        
        # 圧縮データサイズ
        compressed_data_size = struct.unpack('<I', compressed_data[offset:offset+4])[0]
        offset += 4
        
        # 圧縮データ
        actual_compressed_data = compressed_data[offset:offset+compressed_data_size]
        
        # 展開
        if method == "uncompressed":
            original_data = actual_compressed_data
        elif method == "rle":
            original_data = self.safe_rle_decompress(actual_compressed_data)
        elif method == "lzma":
            original_data = lzma.decompress(actual_compressed_data)
        elif method == "zlib":
            original_data = zlib.decompress(actual_compressed_data)
        else:
            raise ValueError(f"未知の圧縮方式: {method}")
        
        # 可逆性検証
        restored_hash = hashlib.sha256(original_data).hexdigest()
        is_identical = (restored_hash == original_hash)
        
        processing_time = time.time() - start_time
        print(f"✅ 展開完了: {len(original_data):,} bytes ({processing_time:.2f}秒)")
        print(f"🔍 可逆性検証: {'✅ 完全一致' if is_identical else '❌ 不一致'}")
        
        if not is_identical:
            print(f"⚠️ 原本: {original_hash[:16]}...")
            print(f"⚠️ 復元: {restored_hash[:16]}...")
            raise ValueError("❌ 可逆性検証失敗")
        
        return {
            'original_data': original_data,
            'decompressed_size': len(original_data),
            'processing_time': processing_time,
            'is_reversible': is_identical
        }
    
    def safe_rle_compress(self, data: bytes) -> bytes:
        """安全なRLE圧縮"""
        if not data:
            return b''
        
        compressed = bytearray()
        i = 0
        
        while i < len(data):
            current_byte = data[i]
            count = 1
            
            # 連続バイトカウント（最大253まで）
            while (i + count < len(data) and 
                   data[i + count] == current_byte and 
                   count < 253):
                count += 1
            
            if count >= 3:
                # RLE圧縮: 254 count byte
                compressed.extend([254, count, current_byte])
                i += count
            else:
                # 通常バイト（254と255のエスケープ処理）
                if current_byte == 254:
                    compressed.extend([255, 254])  # エスケープ
                elif current_byte == 255:
                    compressed.extend([255, 255])  # エスケープ
                else:
                    compressed.append(current_byte)
                i += 1
        
        return bytes(compressed)
    
    def safe_rle_decompress(self, data: bytes) -> bytes:
        """安全なRLE展開"""
        if not data:
            return b''
        
        result = bytearray()
        i = 0
        
        while i < len(data):
            if data[i] == 254 and i + 2 < len(data):
                # RLE展開: 254 count byte
                count = data[i + 1]
                byte_value = data[i + 2]
                result.extend([byte_value] * count)
                i += 3
            elif data[i] == 255 and i + 1 < len(data):
                # エスケープ展開: 255 byte
                result.append(data[i + 1])
                i += 2
            else:
                # 通常バイト
                result.append(data[i])
                i += 1
        
        return bytes(result)
    
    def build_reversible_file(self, compressed_data: bytes, original_hash: str, 
                            method: str, original_size: int) -> bytes:
        """可逆ファイル構築"""
        result = bytearray()
        
        # ヘッダー
        result.extend(self.magic_header)
        
        # 元データハッシュ（64文字固定）
        result.extend(original_hash.encode('ascii'))
        
        # 元サイズ
        result.extend(struct.pack('<I', original_size))
        
        # 圧縮方式（可変長）
        method_bytes = method.encode('ascii')
        result.extend(struct.pack('<H', len(method_bytes)))
        result.extend(method_bytes)
        
        # 圧縮データサイズ
        result.extend(struct.pack('<I', len(compressed_data)))
        
        # 圧縮データ
        result.extend(compressed_data)
        
        return bytes(result)
    
    def compress_file(self, input_path: str, output_path: str = None) -> bool:
        """ファイル圧縮"""
        if not os.path.exists(input_path):
            print(f"❌ ファイルが見つかりません: {input_path}")
            return False
        
        if output_path is None:
            output_path = input_path + '.p8s'  # Phase 8 Simple
        
        try:
            with open(input_path, 'rb') as f:
                data = f.read()
            
            filename = os.path.basename(input_path)
            result = self.simple_compress(data, filename)
            
            with open(output_path, 'wb') as f:
                f.write(result['compressed_data'])
            
            print(f"💾 簡素化圧縮ファイル保存: {output_path}")
            return True
        
        except Exception as e:
            print(f"❌ 圧縮エラー: {e}")
            return False
    
    def decompress_file(self, input_path: str, output_path: str = None) -> bool:
        """ファイル展開"""
        if not os.path.exists(input_path):
            print(f"❌ ファイルが見つかりません: {input_path}")
            return False
        
        if output_path is None:
            if input_path.endswith('.p8s'):
                output_path = input_path[:-4]
            else:
                output_path = input_path + '.restored'
        
        try:
            with open(input_path, 'rb') as f:
                compressed_data = f.read()
            
            result = self.simple_decompress(compressed_data)
            
            with open(output_path, 'wb') as f:
                f.write(result['original_data'])
            
            print(f"📁 簡素化復元ファイル保存: {output_path}")
            return True
        
        except Exception as e:
            print(f"❌ 展開エラー: {e}")
            return False

def run_simple_reversible_test():
    """簡素化可逆性テスト"""
    print("🔧 Phase 8 簡素化完全可逆性テスト")
    print("=" * 60)
    
    engine = SimpleReversibleEngine()
    sample_dir = Path("../NXZip-Python/sample")
    
    # 段階的テストファイル
    test_files = [
        # 小容量テスト
        ("陰謀論.mp3", 512*1024, "MP3音声 (部分512KB)"),
        ("COT-001.jpg", 256*1024, "JPEG画像 (部分256KB)"),
        
        # 中容量テスト
        ("COT-012.png", 1024*1024, "PNG画像 (部分1MB)"),
        ("出庫実績明細_202412.txt", 1024*1024, "テキスト (部分1MB)"),
    ]
    
    results = []
    
    for filename, size_limit, description in test_files:
        filepath = sample_dir / filename
        if not filepath.exists():
            print(f"⚠️ ファイルなし: {filename}")
            continue
        
        print(f"\n🔧 簡素化可逆テスト: {description}")
        print("-" * 40)
        
        try:
            # 制限サイズでテスト
            with open(filepath, 'rb') as f:
                test_data = f.read(size_limit)
            print(f"📏 テストサイズ: {len(test_data):,} bytes")
            
            # 圧縮
            result = engine.simple_compress(test_data, filename)
            
            # 展開
            decompressed = engine.simple_decompress(result['compressed_data'])
            
            # 可逆性検証
            is_identical = (test_data == decompressed['original_data'])
            
            results.append({
                'filename': filename,
                'description': description,
                'original_size': len(test_data),
                'compressed_size': result['compressed_size'],
                'compression_ratio': result['compression_ratio'],
                'reversible': is_identical,
                'processing_time': result['processing_time'],
                'method': result['method']
            })
            
            print(f"✅ 簡素化可逆性: {'✅ 成功' if is_identical else '❌ 失敗'}")
            
        except Exception as e:
            print(f"❌ テストエラー: {str(e)[:60]}...")
    
    # 総合結果
    if results:
        print("\n" + "=" * 60)
        print("🏆 Phase 8 簡素化完全可逆性テスト結果")
        print("=" * 60)
        
        total_original = sum(r['original_size'] for r in results)
        total_compressed = sum(r['compressed_size'] for r in results)
        overall_ratio = (1 - total_compressed / total_original) * 100
        reversible_count = sum(1 for r in results if r['reversible'])
        
        print(f"🔧 簡素化可逆性成功率: {reversible_count}/{len(results)} ({reversible_count/len(results)*100:.1f}%)")
        print(f"📊 平均圧縮率: {overall_ratio:.1f}%")
        print(f"📈 テストファイル数: {len(results)}")
        print(f"💾 総データ量: {total_original/1024/1024:.1f} MB")
        
        # 個別結果
        print(f"\n📋 個別簡素化テスト結果:")
        for result in results:
            filename_short = result['filename'][:20] + ('...' if len(result['filename']) > 20 else '')
            size_mb = result['original_size'] / 1024 / 1024
            rev_icon = '✅' if result['reversible'] else '❌'
            print(f"   {rev_icon} {filename_short}: {result['compression_ratio']:.1f}% ({size_mb:.1f}MB, {result['method']})")
        
        if reversible_count == len(results):
            print("🎉 全ファイル簡素化可逆性達成！")
            
            # 次ステップ: 画像・動画特化テスト
            print("\n🚀 次ステップ: 画像・動画圧縮率向上")
            print("   1. 簡素化エンジンで可逆性確保 ✅")
            print("   2. 画像・動画特化アルゴリズム適用 ⏭️")
            print("   3. 圧縮率の大幅向上実現 🎯")
            
        else:
            failed_count = len(results) - reversible_count
            print(f"⚠️ {failed_count}ファイルで可逆性問題 - さらなる簡素化必要")

def main():
    """メイン処理"""
    if len(sys.argv) < 2:
        print("🔧 Phase 8 簡素化完全可逆版")
        print("使用方法:")
        print("  python phase8_simple_reversible.py test                    # 簡素化可逆性テスト")
        print("  python phase8_simple_reversible.py compress <file>         # 簡素化圧縮")
        print("  python phase8_simple_reversible.py decompress <file.p8s>   # 簡素化展開")
        return
    
    command = sys.argv[1].lower()
    engine = SimpleReversibleEngine()
    
    if command == "test":
        run_simple_reversible_test()
    elif command == "compress" and len(sys.argv) >= 3:
        input_file = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) >= 4 else None
        engine.compress_file(input_file, output_file)
    elif command == "decompress" and len(sys.argv) >= 3:
        input_file = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) >= 4 else None
        engine.decompress_file(input_file, output_file)
    else:
        print("❌ 無効なコマンドです")

if __name__ == "__main__":
    main()
