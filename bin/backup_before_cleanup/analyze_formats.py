#!/usr/bin/env python3
"""
フォーマット分析ツール
各ファイル形式の内部構造と圧縮可能性を詳細分析
"""

import os
import sys
from pathlib import Path
import struct
import hashlib

# プロジェクトパス追加
current_dir = Path(__file__).parent
project_root = current_dir.parent / "NXZip-Python"
sys.path.insert(0, str(project_root))

from nxzip.engine.nexus_unified import NEXUSUnified

def analyze_file_structure(file_path):
    """ファイル構造の詳細分析"""
    print(f"\n🔬 詳細分析: {Path(file_path).name}")
    print("=" * 60)
    
    with open(file_path, 'rb') as f:
        data = f.read()
    
    size = len(data)
    print(f"📊 ファイルサイズ: {size:,} bytes ({size/1024/1024:.2f} MB)")
    
    # ヘッダー分析
    if size >= 16:
        header = data[:16]
        print(f"🔍 ヘッダー: {header.hex()}")
        
        # 具体的なフォーマット判定
        if header.startswith(b'\xFF\xD8\xFF'):
            analyze_jpeg(data)
        elif header.startswith(b'\x89PNG'):
            analyze_png(data)
        elif data[4:8] == b'ftyp':
            analyze_mp4(data)
        elif header.startswith(b'ID3') or header.startswith(b'\xFF\xFB'):
            analyze_mp3(data)
        elif header.startswith(b'RIFF'):
            analyze_wav(data)
        else:
            analyze_generic(data)
    
    # エントロピー分析
    entropy = calculate_entropy(data)
    print(f"📈 エントロピー: {entropy:.3f} (理論最大: 8.000)")
    
    # 圧縮テスト
    test_compression_methods(data)

def analyze_jpeg(data):
    """JPEG分析"""
    print("🖼️  フォーマット: JPEG")
    
    # JPEG セグメント解析
    pos = 2  # FF D8 の後
    segments = []
    
    while pos < len(data) - 1:
        if data[pos] == 0xFF:
            marker = data[pos:pos+2]
            if marker == b'\xFF\xD9':  # EOI
                break
            
            # セグメント長取得
            if pos + 3 < len(data):
                length = struct.unpack('>H', data[pos+2:pos+4])[0]
                segments.append({
                    'marker': marker.hex(),
                    'length': length,
                    'position': pos
                })
                pos += length + 2
            else:
                break
        else:
            pos += 1
    
    print(f"📝 JPEGセグメント数: {len(segments)}")
    for seg in segments[:5]:  # 最初の5個だけ表示
        print(f"   {seg['marker']}: {seg['length']} bytes at {seg['position']}")
    
    # 圧縮済みデータの特徴
    print("⚠️  問題: JPEGは既にDCT圧縮済み")
    print("💡 解決策: DCT係数の再配置、量子化テーブル最適化が必要")

def analyze_png(data):
    """PNG分析"""
    print("🖼️  フォーマット: PNG")
    
    # PNG チャンク解析
    pos = 8  # PNG署名の後
    chunks = []
    
    while pos < len(data):
        if pos + 8 > len(data):
            break
            
        length = struct.unpack('>I', data[pos:pos+4])[0]
        chunk_type = data[pos+4:pos+8]
        chunks.append({
            'type': chunk_type.decode('ascii', errors='ignore'),
            'length': length,
            'position': pos
        })
        pos += length + 12  # length + type + data + CRC
    
    print(f"📝 PNGチャンク数: {len(chunks)}")
    for chunk in chunks[:5]:
        print(f"   {chunk['type']}: {chunk['length']} bytes")
    
    print("⚠️  問題: PNGは既にzlib圧縮済み")
    print("💡 解決策: 生ピクセルデータへの変換後、専用圧縮が必要")

def analyze_mp4(data):
    """MP4分析"""
    print("🎬 フォーマット: MP4")
    
    # MP4 ボックス解析
    pos = 0
    boxes = []
    
    while pos < len(data) and len(boxes) < 10:  # 最初の10個
        if pos + 8 > len(data):
            break
            
        size = struct.unpack('>I', data[pos:pos+4])[0]
        box_type = data[pos+4:pos+8]
        
        boxes.append({
            'type': box_type.decode('ascii', errors='ignore'),
            'size': size,
            'position': pos
        })
        
        if size == 0:
            break
        pos += size
    
    print(f"📝 MP4ボックス数: {len(boxes)}")
    for box in boxes:
        print(f"   {box['type']}: {box['size']} bytes")
    
    print("⚠️  問題: MP4は既にH.264/H.265圧縮済み")
    print("💡 解決策: 動きベクトル再配置、残差信号最適化が必要")

def analyze_mp3(data):
    """MP3分析"""
    print("🎵 フォーマット: MP3")
    
    # ID3タグ解析
    if data.startswith(b'ID3'):
        id3_size = struct.unpack('>I', data[6:10])[0]
        print(f"📝 ID3タグサイズ: {id3_size} bytes")
        
        # フレーム開始位置
        frame_start = 10 + id3_size
    else:
        frame_start = 0
    
    # MP3フレーム数推定
    frame_count = 0
    pos = frame_start
    
    while pos < len(data) - 4 and frame_count < 100:  # 最初の100フレーム
        if data[pos] == 0xFF and (data[pos+1] & 0xE0) == 0xE0:
            # MPEGフレームヘッダー
            frame_count += 1
            # フレーム長計算（簡易）
            pos += 144  # 平均的なフレーム長
        else:
            pos += 1
    
    print(f"📝 推定フレーム数: {frame_count}")
    print("⚠️  問題: MP3は既に心理音響モデル圧縮済み")
    print("💡 解決策: フレーム構造最適化、ビットリザーバ再配置が必要")

def analyze_wav(data):
    """WAV分析"""
    print("🎵 フォーマット: WAV")
    
    if len(data) >= 44:
        # WAVヘッダー解析
        chunk_size = struct.unpack('<I', data[4:8])[0]
        audio_format = struct.unpack('<H', data[20:22])[0]
        channels = struct.unpack('<H', data[22:24])[0]
        sample_rate = struct.unpack('<I', data[24:28])[0]
        bits_per_sample = struct.unpack('<H', data[34:36])[0]
        
        print(f"📝 フォーマット: {audio_format} (1=PCM)")
        print(f"📝 チャンネル数: {channels}")
        print(f"📝 サンプルレート: {sample_rate} Hz")
        print(f"📝 ビット深度: {bits_per_sample} bit")
        
        # 実際の音声データサイズ
        audio_data_size = len(data) - 44
        print(f"📝 音声データ: {audio_data_size:,} bytes")
        
        print("✅ 利点: WAVは非圧縮のため圧縮効果が期待できる")
        print("💡 最適化: 時間軸相関、周波数軸相関の活用")

def analyze_generic(data):
    """一般的な分析"""
    print("📄 フォーマット: 一般/テキスト")
    
    # 文字種分析
    if len(data) > 0:
        ascii_count = sum(1 for b in data[:4096] if 32 <= b <= 126)
        whitespace_count = sum(1 for b in data[:4096] if b in [9, 10, 13, 32])
        
        text_ratio = (ascii_count + whitespace_count) / min(4096, len(data))
        print(f"📝 テキスト比率: {text_ratio:.1%}")
        
        if text_ratio > 0.8:
            print("✅ 利点: テキストファイルは高い冗長性を持つ")
            print("💡 最適化: 辞書圧縮、パターン認識、統計的圧縮")

def calculate_entropy(data):
    """データのエントロピー計算"""
    if not data:
        return 0
    
    import math
    
    # バイト頻度計算
    freq = [0] * 256
    for byte in data:
        freq[byte] += 1
    
    # エントロピー計算
    entropy = 0
    data_len = len(data)
    
    for f in freq:
        if f > 0:
            p = f / data_len
            entropy -= p * math.log2(p)
    
    return entropy

def test_compression_methods(data):
    """各種圧縮手法のテスト"""
    print("\n🧪 圧縮手法比較:")
    
    import lzma
    import zlib
    import bz2
    
    original_size = len(data)
    
    # LZMA
    try:
        lzma_compressed = lzma.compress(data, preset=6)
        lzma_ratio = (1 - len(lzma_compressed) / original_size) * 100
        print(f"   LZMA: {lzma_ratio:.1f}%")
    except:
        print("   LZMA: エラー")
    
    # ZLIB
    try:
        zlib_compressed = zlib.compress(data, level=9)
        zlib_ratio = (1 - len(zlib_compressed) / original_size) * 100
        print(f"   ZLIB: {zlib_ratio:.1f}%")
    except:
        print("   ZLIB: エラー")
    
    # BZ2
    try:
        bz2_compressed = bz2.compress(data, compresslevel=9)
        bz2_ratio = (1 - len(bz2_compressed) / original_size) * 100
        print(f"   BZ2:  {bz2_ratio:.1f}%")
    except:
        print("   BZ2: エラー")
    
    # NEXUS
    try:
        nexus = NEXUSUnified()
        nexus_compressed = nexus.compress(data)
        nexus_ratio = (1 - len(nexus_compressed) / original_size) * 100
        print(f"   NEXUS: {nexus_ratio:.1f}%")
    except Exception as e:
        print(f"   NEXUS: エラー - {e}")

def main():
    """メイン分析処理"""
    print("🔬 NXZip フォーマット詳細分析")
    print("=" * 70)
    
    # 分析対象ファイル
    sample_dir = Path("NXZip-Python/sample")
    test_files = [
        sample_dir / "出庫実績明細_202412.txt",
        sample_dir / "Python基礎講座3_4月26日-3.mp4", 
        sample_dir / "COT-001.jpg",
        sample_dir / "陰謀論.mp3"
    ]
    
    for file_path in test_files:
        if file_path.exists():
            analyze_file_structure(file_path)
        else:
            print(f"⚠️  ファイルが見つかりません: {file_path}")
    
    # 総合分析
    print(f"\n🎯 総合分析と改善提案")
    print("=" * 70)
    print("1. 📄 テキスト (91.3% → 95%目標):")
    print("   - 現状：非常に良好、目標まで3.7%")
    print("   - 改善：より高度な辞書圧縮、コンテキスト圧縮")
    
    print("\n2. 🎬 動画 (18.3% → 80%目標):")
    print("   - 現状：大幅に不足、61.7%の改善が必要")
    print("   - 原因：H.264は既に高度に圧縮済み")
    print("   - 改善：動きベクトル最適化、残差信号再圧縮")
    
    print("\n3. 🖼️  画像 (3.1% → 80%目標):")
    print("   - 現状：大幅に不足、76.9%の改善が必要")
    print("   - 原因：JPEG DCT圧縮が既に効率的")
    print("   - 改善：DCT係数再配置、量子化テーブル最適化")
    
    print("\n4. 🎵 音声 (1.2% → 80%目標):")
    print("   - 現状：大幅に不足、78.8%の改善が必要")
    print("   - 原因：MP3心理音響圧縮が既に効率的")
    print("   - 改善：フレーム構造最適化、スペクトラム再配置")

if __name__ == "__main__":
    main()
