#!/usr/bin/env python3
"""
NEXUS Simple Image Compressor (NSIC)
シンプル高速画像圧縮エンジン

特徴:
1. 確実な可逆圧縮 - Delta + RLE のみ
2. 高速処理 - 複雑なHuffmanを避ける
3. 独自バイナリ形式 (.nxs) - Simple Archive
4. 確実な動作 - シンプル構造
"""

import os
import sys
import time
import struct
import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class CompressionMetadata:
    """圧縮メタデータ"""
    original_size: int
    compressed_size: int
    file_type: str
    width: int
    height: int
    channels: int
    checksum: str
    compression_time: float

class SimpleImageCompressor:
    """シンプル画像圧縮エンジン"""
    
    def __init__(self):
        self.version = "1.0-Simple"
        self.magic = b'NSIC2025'  # NEXUS Simple Image Compressor
        
        print(f"🚀 NEXUS Simple Image Compressor v{self.version}")
        print("⚡ シンプル高速圧縮エンジン初期化完了")
    
    def detect_image_format(self, data: bytes) -> Tuple[str, int, int, int]:
        """画像形式検出"""
        if len(data) < 50:
            return "UNKNOWN", 0, 0, 0
        
        # PNG検出
        if data.startswith(b'\x89PNG\r\n\x1a\n'):
            try:
                ihdr_pos = data.find(b'IHDR')
                if ihdr_pos != -1:
                    ihdr_start = ihdr_pos + 4
                    width = struct.unpack('>I', data[ihdr_start:ihdr_start+4])[0]
                    height = struct.unpack('>I', data[ihdr_start+4:ihdr_start+8])[0]
                    color_type = data[ihdr_start+9]
                    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type, 3)
                    return "PNG", width, height, channels
            except:
                pass
            return "PNG", 0, 0, 3
        
        # JPEG検出
        elif data.startswith(b'\xff\xd8\xff'):
            try:
                for marker in [b'\xff\xc0', b'\xff\xc1', b'\xff\xc2']:
                    pos = data.find(marker)
                    if pos != -1:
                        sof_start = pos + 5
                        height = struct.unpack('>H', data[sof_start+1:sof_start+3])[0]
                        width = struct.unpack('>H', data[sof_start+3:sof_start+5])[0]
                        channels = data[sof_start+5]
                        return "JPEG", width, height, channels
            except:
                pass
            return "JPEG", 0, 0, 3
        
        # BMP検出
        elif data.startswith(b'BM'):
            try:
                if len(data) >= 54:
                    width = struct.unpack('<I', data[18:22])[0]
                    height = struct.unpack('<I', data[22:26])[0]
                    bit_count = struct.unpack('<H', data[28:30])[0]
                    channels = max(1, bit_count // 8)
                    return "BMP", width, height, channels
            except:
                pass
            return "BMP", 0, 0, 3
        
        return "BINARY", 0, 0, 1
    
    def compress_simple(self, data: bytes) -> bytes:
        """シンプル圧縮"""
        if len(data) == 0:
            return data
        
        print(f"📦 シンプル圧縮開始: {len(data)} bytes")
        start_time = time.time()
        
        # 画像形式検出
        file_type, width, height, channels = self.detect_image_format(data)
        print(f"🔍 検出: {file_type} ({width}x{height}, {channels}ch)")
        
        # チェックサム計算
        checksum = hashlib.sha256(data).hexdigest()[:16]
        
        # 段階的圧縮
        compressed_data = data
        
        # ステージ1: Delta圧縮 (画像に効果的)
        if file_type in ["PNG", "JPEG", "BMP"]:
            compressed_data = self._delta_compress(compressed_data)
            print(f"  📈 Delta圧縮: {len(data)} → {len(compressed_data)} bytes")
        
        # ステージ2: Run-Length圧縮
        compressed_data = self._rle_compress(compressed_data)
        print(f"  🔄 RLE圧縮: → {len(compressed_data)} bytes")
        
        # メタデータ構築
        compression_time = time.time() - start_time
        metadata = CompressionMetadata(
            original_size=len(data),
            compressed_size=len(compressed_data),
            file_type=file_type,
            width=width,
            height=height,
            channels=channels,
            checksum=checksum,
            compression_time=compression_time
        )
        
        # アーカイブパッケージング
        archive = self._package_archive(compressed_data, metadata)
        
        compression_ratio = (1 - len(archive) / len(data)) * 100
        print(f"✅ 圧縮完了: {len(data)} → {len(archive)} bytes ({compression_ratio:.1f}%, {compression_time:.3f}s)")
        
        return archive
    
    def _delta_compress(self, data: bytes) -> bytes:
        """Delta圧縮"""
        if len(data) < 2:
            return data
        
        result = bytearray()
        result.append(data[0])  # 最初のバイト
        
        # 差分計算
        for i in range(1, len(data)):
            delta = (data[i] - data[i-1]) % 256
            result.append(delta)
        
        return bytes(result)
    
    def _rle_compress(self, data: bytes) -> bytes:
        """Run-Length圧縮（シンプル版）"""
        if len(data) == 0:
            return data
        
        result = bytearray()
        i = 0
        
        while i < len(data):
            current_byte = data[i]
            count = 1
            
            # 連続する同じバイトをカウント
            while i + count < len(data) and data[i + count] == current_byte and count < 255:
                count += 1
            
            if count >= 3:  # 3回以上の繰り返しでRLE適用
                result.append(0xFF)  # RLEマーカー
                result.append(count)
                result.append(current_byte)
                i += count
            else:
                # 単発またはマーカー回避
                if current_byte == 0xFF:
                    result.append(0xFF)  # エスケープ
                    result.append(0x00)  # エスケープ識別子
                result.append(current_byte)
                i += 1
        
        return bytes(result)
    
    def _package_archive(self, compressed_data: bytes, metadata: CompressionMetadata) -> bytes:
        """アーカイブパッケージング"""
        archive = bytearray()
        
        # マジックヘッダー
        archive.extend(self.magic)
        
        # バージョン
        archive.append(1)
        
        # メタデータサイズとデータ
        metadata_bytes = self._serialize_metadata(metadata)
        archive.extend(struct.pack('<I', len(metadata_bytes)))
        archive.extend(metadata_bytes)
        
        # 圧縮データサイズとデータ
        archive.extend(struct.pack('<I', len(compressed_data)))
        archive.extend(compressed_data)
        
        return bytes(archive)
    
    def _serialize_metadata(self, metadata: CompressionMetadata) -> bytes:
        """メタデータシリアライズ"""
        data = bytearray()
        
        # 基本情報
        data.extend(struct.pack('<I', metadata.original_size))
        data.extend(struct.pack('<I', metadata.compressed_size))
        data.extend(struct.pack('<I', metadata.width))
        data.extend(struct.pack('<I', metadata.height))
        data.extend(struct.pack('<I', metadata.channels))
        data.extend(struct.pack('<f', metadata.compression_time))
        
        # 文字列データ
        file_type_bytes = metadata.file_type.encode('utf-8')
        data.append(len(file_type_bytes))
        data.extend(file_type_bytes)
        
        checksum_bytes = metadata.checksum.encode('utf-8')
        data.append(len(checksum_bytes))
        data.extend(checksum_bytes)
        
        return bytes(data)
    
    def decompress_simple(self, archive_data: bytes) -> bytes:
        """シンプル解凍"""
        if len(archive_data) < len(self.magic) + 10:
            raise ValueError("無効なアーカイブ形式")
        
        print("📂 シンプル解凍開始...")
        start_time = time.time()
        
        # ヘッダー検証
        if not archive_data.startswith(self.magic):
            raise ValueError("無効なマジックヘッダー")
        
        pos = len(self.magic)
        version = archive_data[pos]
        pos += 1
        
        # メタデータ読み込み
        metadata_size = struct.unpack('<I', archive_data[pos:pos+4])[0]
        pos += 4
        metadata_bytes = archive_data[pos:pos+metadata_size]
        pos += metadata_size
        
        metadata = self._deserialize_metadata(metadata_bytes)
        print(f"🔍 メタデータ: {metadata.file_type} {metadata.width}x{metadata.height}")
        
        # 圧縮データ読み込み
        compressed_size = struct.unpack('<I', archive_data[pos:pos+4])[0]
        pos += 4
        compressed_data = archive_data[pos:pos+compressed_size]
        
        # 段階的解凍（逆順）
        decompressed_data = compressed_data
        
        # RLE解凍
        decompressed_data = self._rle_decompress(decompressed_data)
        print(f"  🔄 RLE解凍: → {len(decompressed_data)} bytes")
        
        # Delta解凍
        if metadata.file_type in ["PNG", "JPEG", "BMP"]:
            decompressed_data = self._delta_decompress(decompressed_data)
            print(f"  📈 Delta解凍: → {len(decompressed_data)} bytes")
        
        # チェックサム検証
        actual_checksum = hashlib.sha256(decompressed_data).hexdigest()[:16]
        if actual_checksum != metadata.checksum:
            raise ValueError(f"チェックサム不一致: {actual_checksum} != {metadata.checksum}")
        
        decomp_time = time.time() - start_time
        print(f"✅ 解凍完了: {len(compressed_data)} → {len(decompressed_data)} bytes ({decomp_time:.3f}s)")
        
        return decompressed_data
    
    def _deserialize_metadata(self, data: bytes) -> CompressionMetadata:
        """メタデータデシリアライズ"""
        pos = 0
        
        # 基本情報
        original_size = struct.unpack('<I', data[pos:pos+4])[0]
        pos += 4
        compressed_size = struct.unpack('<I', data[pos:pos+4])[0]
        pos += 4
        width = struct.unpack('<I', data[pos:pos+4])[0]
        pos += 4
        height = struct.unpack('<I', data[pos:pos+4])[0]
        pos += 4
        channels = struct.unpack('<I', data[pos:pos+4])[0]
        pos += 4
        compression_time = struct.unpack('<f', data[pos:pos+4])[0]
        pos += 4
        
        # 文字列データ
        file_type_len = data[pos]
        pos += 1
        file_type = data[pos:pos+file_type_len].decode('utf-8')
        pos += file_type_len
        
        checksum_len = data[pos]
        pos += 1
        checksum = data[pos:pos+checksum_len].decode('utf-8')
        pos += checksum_len
        
        return CompressionMetadata(
            original_size=original_size,
            compressed_size=compressed_size,
            file_type=file_type,
            width=width,
            height=height,
            channels=channels,
            checksum=checksum,
            compression_time=compression_time
        )
    
    def _rle_decompress(self, data: bytes) -> bytes:
        """RLE解凍"""
        result = bytearray()
        i = 0
        
        while i < len(data):
            if data[i] == 0xFF and i + 1 < len(data):
                if data[i + 1] == 0x00:  # エスケープシーケンス
                    result.append(0xFF)
                    i += 2
                else:  # RLEシーケンス
                    if i + 2 < len(data):
                        count = data[i + 1]
                        value = data[i + 2]
                        result.extend([value] * count)
                        i += 3
                    else:
                        result.append(data[i])
                        i += 1
            else:
                result.append(data[i])
                i += 1
        
        return bytes(result)
    
    def _delta_decompress(self, data: bytes) -> bytes:
        """Delta解凍"""
        if len(data) < 2:
            return data
        
        result = bytearray()
        result.append(data[0])  # 最初のバイト
        
        # 累積和で復元
        for i in range(1, len(data)):
            restored = (result[-1] + data[i]) % 256
            result.append(restored)
        
        return bytes(result)
    
    def compress_file(self, file_path: str, output_path: str = None) -> dict:
        """ファイル圧縮"""
        try:
            if not os.path.exists(file_path):
                return {'success': False, 'error': f'ファイルが見つかりません: {file_path}'}
            
            print(f"📁 ファイル圧縮開始: {file_path}")
            
            with open(file_path, 'rb') as f:
                data = f.read()
            
            if len(data) == 0:
                return {'success': False, 'error': 'ファイルが空です'}
            
            # 圧縮実行
            compressed = self.compress_simple(data)
            
            # 出力ファイル決定
            if output_path is None:
                base_name = os.path.splitext(file_path)[0]
                output_path = f"{base_name}.nxs"
            
            # ファイル出力
            with open(output_path, 'wb') as f:
                f.write(compressed)
            
            compression_ratio = (1 - len(compressed) / len(data)) * 100
            
            return {
                'success': True,
                'input_file': file_path,
                'output_file': output_path,
                'original_size': len(data),
                'compressed_size': len(compressed),
                'compression_ratio': compression_ratio,
                'algorithm': 'Simple Image Compressor'
            }
            
        except Exception as e:
            return {'success': False, 'error': f'圧縮エラー: {str(e)}'}
    
    def decompress_file(self, archive_path: str, output_path: str = None) -> dict:
        """ファイル解凍"""
        try:
            if not os.path.exists(archive_path):
                return {'success': False, 'error': f'アーカイブが見つかりません: {archive_path}'}
            
            print(f"📂 ファイル解凍開始: {archive_path}")
            
            with open(archive_path, 'rb') as f:
                archive_data = f.read()
            
            # 解凍実行
            decompressed = self.decompress_simple(archive_data)
            
            # 出力ファイル決定
            if output_path is None:
                base_name = os.path.splitext(archive_path)[0]
                
                # 元の拡張子を推定
                file_type, _, _, _ = self.detect_image_format(decompressed)
                if file_type == "PNG":
                    output_path = f"{base_name}_restored.png"
                elif file_type == "JPEG":
                    output_path = f"{base_name}_restored.jpg"
                elif file_type == "BMP":
                    output_path = f"{base_name}_restored.bmp"
                else:
                    output_path = f"{base_name}_restored.bin"
            
            # ファイル出力
            with open(output_path, 'wb') as f:
                f.write(decompressed)
            
            return {
                'success': True,
                'input_file': archive_path,
                'output_file': output_path,
                'decompressed_size': len(decompressed),
                'algorithm': 'Simple Image Compressor'
            }
            
        except Exception as e:
            return {'success': False, 'error': f'解凍エラー: {str(e)}'}

def main():
    """メイン関数"""
    if len(sys.argv) < 2:
        print("🚀 NEXUS Simple Image Compressor")
        print("シンプル高速画像圧縮エンジン")
        print()
        print("使用方法:")
        print("  python nexus_simple_image_compressor.py compress <ファイル>")
        print("  python nexus_simple_image_compressor.py decompress <アーカイブ>")
        print("  python nexus_simple_image_compressor.py test")
        print()
        print("特徴:")
        print("  ⚡ 高速処理 - シンプル構造")
        print("  🔄 確実動作 - Delta + RLE")
        print("  📦 独自形式 - .nxs アーカイブ")
        print("  🖼️  画像最適化 - PNG/JPEG/BMP対応")
        return
    
    command = sys.argv[1].lower()
    
    if command == "test":
        # テストモード
        print("🧪 Simple Image Compressor テスト実行")
        compressor = SimpleImageCompressor()
        
        # テストデータ生成（画像パターン）
        test_data = bytearray()
        test_data.extend(b'\x89PNG\r\n\x1a\n')  # PNG署名
        test_data.extend(b'\x00\x00\x00\rIHDR')  # IHDR
        test_data.extend(struct.pack('>II', 64, 64))  # 64x64
        test_data.extend(b'\x08\x02\x00\x00\x00')  # 8bit RGB
        
        # パターンデータ（圧縮しやすい）
        for i in range(1000):
            # 単純なパターン
            test_data.extend([100, 150, 200])
            if i % 10 == 0:
                test_data.extend([255, 255, 255])  # 白のパターン
        
        original_data = bytes(test_data)
        print(f"テストデータ: {len(original_data)} bytes")
        
        # 圧縮テスト
        compressed = compressor.compress_simple(original_data)
        
        # 解凍テスト
        decompressed = compressor.decompress_simple(compressed)
        
        # 検証
        if original_data == decompressed:
            compression_ratio = (1 - len(compressed) / len(original_data)) * 100
            print(f"✅ テスト成功!")
            print(f"📊 圧縮率: {compression_ratio:.1f}%")
            print(f"📏 サイズ: {len(original_data)} → {len(compressed)} → {len(decompressed)}")
        else:
            print(f"❌ テスト失敗: データが一致しません")
            print(f"原サイズ: {len(original_data)}, 復元サイズ: {len(decompressed)}")
    
    elif command == "compress" and len(sys.argv) >= 3:
        file_path = sys.argv[2]
        compressor = SimpleImageCompressor()
        
        result = compressor.compress_file(file_path)
        
        if result['success']:
            print(f"✅ 圧縮成功!")
            print(f"📁 出力: {result['output_file']}")
            print(f"📊 圧縮率: {result['compression_ratio']:.1f}%")
            print(f"📏 サイズ: {result['original_size']} → {result['compressed_size']} bytes")
        else:
            print(f"❌ 圧縮失敗: {result['error']}")
    
    elif command == "decompress" and len(sys.argv) >= 3:
        archive_path = sys.argv[2]
        compressor = SimpleImageCompressor()
        
        result = compressor.decompress_file(archive_path)
        
        if result['success']:
            print(f"✅ 解凍成功!")
            print(f"📁 出力: {result['output_file']}")
            print(f"📏 サイズ: {result['decompressed_size']} bytes")
        else:
            print(f"❌ 解凍失敗: {result['error']}")
    
    else:
        print("❌ 無効なコマンドです。'test', 'compress', 'decompress' を使用してください。")

if __name__ == "__main__":
    main()
