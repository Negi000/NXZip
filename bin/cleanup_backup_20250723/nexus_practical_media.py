#!/usr/bin/env python3
"""
NXZip実用的メディア最適化エンジン - 現実的なアプローチ
既存の圧縮ファイルに対する実用的な最適化手法

理論的アルゴリズムから実用的最適化へのシフト：
- メタデータ除去・最適化
- コンテナ最適化
- 冗長データ削除
- ファイル構造最適化
"""

import os
import sys
import time
import hashlib
import zipfile
import io
import struct
from pathlib import Path

def md5_hash(data):
    """MD5ハッシュ計算"""
    return hashlib.md5(data).hexdigest()

def remove_metadata_png(data):
    """PNG メタデータ除去・最適化"""
    if not data.startswith(b'\x89PNG\r\n\x1a\n'):
        return data
    
    result = bytearray()
    pos = 8  # PNG シグネチャをスキップ
    
    result.extend(data[:8])  # PNG シグネチャを保持
    
    while pos < len(data):
        if pos + 8 > len(data):
            break
            
        # チャンク長さとタイプを読み取り
        chunk_length = struct.unpack('>I', data[pos:pos+4])[0]
        chunk_type = data[pos+4:pos+8]
        
        # 必要なチャンクのみを保持
        essential_chunks = {b'IHDR', b'PLTE', b'IDAT', b'IEND'}
        optimization_chunks = {b'tRNS', b'gAMA', b'cHRM', b'sRGB'}
        
        if chunk_type in essential_chunks or chunk_type in optimization_chunks:
            # チャンク全体をコピー（長さ + タイプ + データ + CRC）
            chunk_end = pos + 12 + chunk_length
            if chunk_end <= len(data):
                result.extend(data[pos:chunk_end])
        
        pos += 12 + chunk_length
        
        if chunk_type == b'IEND':
            break
    
    return bytes(result)

def optimize_mp4_container(data):
    """MP4 コンテナ最適化"""
    if len(data) < 8:
        return data
    
    # MP4 ftyp ボックスをチェック
    if data[4:8] not in [b'ftyp', b'styp']:
        return data
    
    # 基本的なMP4構造最適化
    # より高度な最適化には専門的なMP4パーサーが必要
    
    # 単純な冗長データ除去
    result = bytearray(data)
    
    # 連続する同じバイトのパターンを短縮
    compressed = bytearray()
    i = 0
    while i < len(result):
        current_byte = result[i]
        count = 1
        
        # 同じバイトが続く数をカウント
        while i + count < len(result) and result[i + count] == current_byte and count < 255:
            count += 1
        
        if count > 3:
            # 4回以上同じバイトが続く場合は短縮表現
            compressed.extend([0xFF, current_byte, count])
            i += count
        else:
            # 通常のバイトをそのまま追加
            compressed.extend(result[i:i+count])
            i += count
    
    # 圧縮効果があれば適用
    if len(compressed) < len(result):
        return bytes(compressed)
    
    return data

def optimize_file_structure(data, file_ext):
    """ファイル構造最適化"""
    # 現在は安全のため、構造最適化を無効化
    # 将来的により安全な最適化手法を実装
    return data

def simple_compression(data):
    """シンプルな追加圧縮"""
    # ZIP圧縮を使用した追加圧縮
    try:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            zf.writestr('data', data)
        
        compressed = buffer.getvalue()
        
        # ZIP圧縮が効果的な場合のみ適用
        if len(compressed) < len(data) * 0.95:  # 5%以上の圧縮効果
            return compressed, True
        
    except Exception:
        pass
    
    return data, False

def practical_media_compress(file_path):
    """実用的メディア圧縮"""
    print(f"🛠️  実用的メディア最適化: {file_path}")
    
    start_time = time.time()
    
    # ファイル読み込み
    with open(file_path, 'rb') as f:
        original_data = f.read()
    
    original_size = len(original_data)
    original_md5 = md5_hash(original_data)
    
    print(f"📁 元ファイル: {original_size:,} bytes")
    print(f"🔒 元MD5: {original_md5}")
    
    # ファイル拡張子を取得
    file_ext = Path(file_path).suffix
    
    # ステップ1: メタデータ最適化・構造最適化
    print("🔧 メタデータ最適化...")
    optimized_data = optimize_file_structure(original_data, file_ext)
    
    # ステップ2: 追加圧縮
    print("📦 追加圧縮...")
    compressed_data, was_compressed = simple_compression(optimized_data)
    
    final_size = len(compressed_data)
    compression_ratio = ((original_size - final_size) / original_size) * 100
    
    # 圧縮データにヘッダーを追加
    header = struct.pack('<QQ32s?', original_size, final_size, original_md5.encode()[:32], was_compressed)
    final_data = header + compressed_data
    
    # 結果出力
    processing_time = time.time() - start_time
    throughput = original_size / (1024 * 1024) / processing_time
    
    print(f"🔹 最適化完了: {compression_ratio:.1f}%")
    print(f"⚡ 処理時間: {processing_time:.2f}s ({throughput:.1f} MB/s)")
    
    # 保存
    output_path = file_path + '.practical.nxz'
    with open(output_path, 'wb') as f:
        f.write(final_data)
    
    print(f"💾 保存: {output_path}")
    print(f"✅ SUCCESS: 実用的最適化完了")
    
    return output_path

def practical_media_decompress(compressed_file):
    """実用的メディア展開"""
    print(f"📦 実用的メディア展開: {compressed_file}")
    
    with open(compressed_file, 'rb') as f:
        data = f.read()
    
    # ヘッダー解析
    header_size = struct.calcsize('<QQ32s?')
    header = data[:header_size]
    compressed_data = data[header_size:]
    
    original_size, final_size, original_md5_bytes, was_compressed = struct.unpack('<QQ32s?', header)
    original_md5 = original_md5_bytes.decode().rstrip('\x00')
    
    # 展開
    if was_compressed:
        # ZIP展開
        buffer = io.BytesIO(compressed_data)
        with zipfile.ZipFile(buffer, 'r') as zf:
            decompressed_data = zf.read('data')
    else:
        decompressed_data = compressed_data
    
    # 検証
    restored_md5 = md5_hash(decompressed_data)
    
    if restored_md5 == original_md5:
        print(f"✅ 完全可逆性確認: MD5一致")
        
        # 展開ファイル保存
        output_path = compressed_file.replace('.practical.nxz', '.restored')
        with open(output_path, 'wb') as f:
            f.write(decompressed_data)
        
        print(f"💾 展開ファイル: {output_path}")
        return output_path
    else:
        print(f"❌ エラー: MD5不一致")
        print(f"   元: {original_md5}")
        print(f"   復元: {restored_md5}")
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用法: python nexus_practical_media.py <ファイルパス>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"エラー: ファイルが見つかりません: {input_file}")
        sys.exit(1)
    
    if input_file.endswith('.practical.nxz'):
        # 展開
        practical_media_decompress(input_file)
    else:
        # 圧縮
        practical_media_compress(input_file)
