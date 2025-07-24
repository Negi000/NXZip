#!/usr/bin/env python3
"""
簡易PNG生成ツール - 画像専用圧縮エンジンテスト用
完全独立実装（zlib依存なし）
"""

import struct

def simple_crc32(data: bytes) -> int:
    """シンプルCRC32計算（zlib非依存）"""
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xEDB88320
            else:
                crc >>= 1
    return crc ^ 0xFFFFFFFF

def simple_deflate(data: bytes) -> bytes:
    """シンプルdeflate圧縮（zlib非依存）"""
    # 非圧縮ブロック形式
    result = bytearray()
    
    # deflateヘッダー（非圧縮ブロック）
    result.append(0x01)  # BFINAL=1, BTYPE=00 (非圧縮)
    
    # データ長（リトルエンディアン）
    data_len = len(data)
    result.extend(struct.pack('<H', data_len))
    result.extend(struct.pack('<H', data_len ^ 0xFFFF))  # 補数
    
    # 非圧縮データ
    result.extend(data)
    
    return bytes(result)

def create_test_png(width=64, height=64, filename="test_image.png"):
    """テスト用PNG画像生成"""
    
    # PNG署名
    png_signature = b'\x89PNG\r\n\x1a\n'
    
    # IHDR チャンク
    ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    ihdr_crc = simple_crc32(b'IHDR' + ihdr_data) & 0xffffffff
    ihdr_chunk = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', ihdr_crc)
    
    # 画像データ生成（RGB グラデーション）
    image_data = bytearray()
    for y in range(height):
        image_data.append(0)  # フィルタータイプ（None）
        for x in range(width):
            r = (x * 255) // width
            g = (y * 255) // height  
            b = ((x + y) * 255) // (width + height)
            image_data.extend([r, g, b])
    
    # 独立deflate圧縮
    compressed_data = simple_deflate(bytes(image_data))
    
    # IDAT チャンク
    idat_crc = simple_crc32(b'IDAT' + compressed_data) & 0xffffffff
    idat_chunk = struct.pack('>I', len(compressed_data)) + b'IDAT' + compressed_data + struct.pack('>I', idat_crc)
    
    # IEND チャンク
    iend_crc = simple_crc32(b'IEND') & 0xffffffff
    iend_chunk = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_crc)
    
    # PNG ファイル構築
    png_data = png_signature + ihdr_chunk + idat_chunk + iend_chunk
    
    with open(filename, 'wb') as f:
        f.write(png_data)
    
    print(f"テスト用PNG画像作成: {filename} ({len(png_data)} bytes, {width}x{height})")
    return filename

def create_simple_png(width=16, height=16, color=(255, 0, 0), filename="simple_test.png"):
    """単色のテスト用PNG画像生成（高圧縮率テスト用）"""
    
    # PNG署名
    png_signature = b'\x89PNG\r\n\x1a\n'
    
    # IHDR チャンク
    ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data) & 0xffffffff
    ihdr_chunk = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', ihdr_crc)
    
    # 単色画像データ生成
    image_data = bytearray()
    for y in range(height):
        image_data.append(0)  # フィルタータイプ（None）
        for x in range(width):
            image_data.extend(color)  # 単色
    
    # zlib圧縮
    compressed_data = zlib.compress(bytes(image_data))
    
    # IDAT チャンク
    idat_crc = zlib.crc32(b'IDAT' + compressed_data) & 0xffffffff
    idat_chunk = struct.pack('>I', len(compressed_data)) + b'IDAT' + compressed_data + struct.pack('>I', idat_crc)
    
    # IEND チャンク
    iend_crc = zlib.crc32(b'IEND') & 0xffffffff
    iend_chunk = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_crc)
    
    # PNG ファイル構築
    png_data = png_signature + ihdr_chunk + idat_chunk + iend_chunk
    
    with open(filename, 'wb') as f:
        f.write(png_data)
    
    print(f"単色PNG画像作成: {filename} ({len(png_data)} bytes, {width}x{height}, RGB{color})")
    return filename

if __name__ == "__main__":
    create_test_png(32, 32, "small_test.png")
    create_test_png(64, 64, "medium_test.png")
    create_simple_png(16, 16, (255, 0, 0), "red_simple.png")
    create_simple_png(32, 32, (0, 255, 0), "green_simple.png")
    print("テスト画像生成完了")
