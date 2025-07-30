#!/usr/bin/env python3
"""
NEXUS v4.0 問題診断ツール - 可逆性と圧縮率の詳細分析
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'nxzip', 'engine'))

import time
import hashlib
from pathlib import Path
from nexus_optimized_v4 import NEXUSOptimizedEngine, OptimizedConfig, simulate_optimized_decompression


def diagnose_compression_issues():
    """圧縮問題診断"""
    print("🔍 NEXUS v4.0 圧縮問題診断")
    print("=" * 80)
    
    # テスト設定
    config = OptimizedConfig(
        max_threads=4,
        chunk_size_mb=1.0,
        fast_mode=True,
        skip_deep_analysis=False,
        compression_level=6
    )
    
    engine = NEXUSOptimizedEngine(config)
    
    # 問題のあったファイルを特定
    sample_dir = Path("sample")
    problem_files = []
    
    # 画像ファイル（可逆性問題があった）
    for file_path in sample_dir.glob("*.jpg"):
        if file_path.stat().st_size < 10 * 1024 * 1024:  # 10MB未満
            problem_files.append(('JPEG画像', file_path))
    
    for file_path in sample_dir.glob("*.png"):
        if file_path.stat().st_size < 20 * 1024 * 1024:  # 20MB未満
            problem_files.append(('PNG画像', file_path))
    
    # 圧縮率が悪いファイル
    for file_path in sample_dir.glob("*.7z"):
        if file_path.stat().st_size < 10 * 1024 * 1024:  # 10MB未満
            problem_files.append(('7Zアーカイブ', file_path))
    
    print(f"🔬 診断対象ファイル: {len(problem_files)}")
    
    for file_type, file_path in problem_files:
        print(f"\n{'='*70}")
        print(f"🧪 詳細診断: {file_path.name}")
        print(f"   📁 タイプ: {file_type}")
        print(f"   📊 サイズ: {file_path.stat().st_size:,} bytes")
        
        try:
            # ファイル読み込み
            with open(file_path, 'rb') as f:
                original_data = f.read()
            
            print(f"   🔑 元データハッシュ: {hashlib.sha256(original_data).hexdigest()[:16]}...")
            
            # 詳細ヘッダー解析
            print(f"   🔍 ヘッダー解析:")
            analyze_file_header(original_data, file_type)
            
            # 圧縮テスト
            print(f"   ⚡ 圧縮テスト実行...")
            compressed_data = engine.optimized_compress(original_data, get_file_type(file_path), 'fast')
            
            compression_ratio = (1 - len(compressed_data) / len(original_data)) * 100
            print(f"      📈 圧縮率: {compression_ratio:.2f}%")
            
            # 詳細解凍検証
            print(f"   🔍 詳細解凍検証:")
            decompressed_data = simulate_optimized_decompression(compressed_data)
            
            # バイト単位比較
            size_match = len(original_data) == len(decompressed_data)
            print(f"      📏 サイズ一致: {'✅' if size_match else '❌'} ({len(original_data)} vs {len(decompressed_data)})")
            
            if size_match:
                # ハッシュ比較
                original_hash = hashlib.sha256(original_data).hexdigest()
                decompressed_hash = hashlib.sha256(decompressed_data).hexdigest()
                hash_match = original_hash == decompressed_hash
                print(f"      🔑 ハッシュ一致: {'✅' if hash_match else '❌'}")
                
                if not hash_match:
                    # 差分解析
                    print(f"      🔍 差分解析:")
                    find_data_differences(original_data, decompressed_data)
            else:
                print(f"      ⚠️ サイズ不一致による可逆性失敗")
            
            # 圧縮データ構造解析
            print(f"   🔍 圧縮データ構造解析:")
            analyze_compressed_structure(compressed_data)
            
        except Exception as e:
            print(f"   ❌ 診断エラー: {e}")
            import traceback
            traceback.print_exc()


def analyze_file_header(data: bytes, file_type: str) -> None:
    """ファイルヘッダー詳細解析"""
    if len(data) < 32:
        print(f"      ⚠️ ファイルサイズが小さすぎます")
        return
    
    header = data[:32]
    print(f"      📝 先頭32バイト: {header.hex()[:64]}...")
    
    if file_type == 'JPEG画像':
        if data[:2] == b'\xff\xd8':
            print(f"      ✅ JPEG SOI確認")
            # EXIF検索
            if b'\xff\xe1' in data[:1024]:
                exif_pos = data.find(b'\xff\xe1')
                print(f"      📋 EXIF位置: {exif_pos}")
        else:
            print(f"      ❌ JPEG形式不正")
    
    elif file_type == 'PNG画像':
        if data[:8] == b'\x89PNG\r\n\x1a\n':
            print(f"      ✅ PNGシグネチャ確認")
            # チャンク解析
            pos = 8
            chunk_count = 0
            while pos < len(data) - 8 and chunk_count < 10:
                try:
                    import struct
                    length = struct.unpack('>I', data[pos:pos+4])[0]
                    chunk_type = data[pos+4:pos+8]
                    print(f"      📦 チャンク{chunk_count}: {chunk_type} ({length} bytes)")
                    pos += 8 + length + 4
                    chunk_count += 1
                except:
                    break
        else:
            print(f"      ❌ PNG形式不正")
    
    elif file_type == '7Zアーカイブ':
        if data[:6] == b'7z\xbc\xaf\x27\x1c':
            print(f"      ✅ 7Zシグネチャ確認")
        else:
            print(f"      ❌ 7Z形式不正")


def find_data_differences(original: bytes, decompressed: bytes) -> None:
    """データ差分検出"""
    min_len = min(len(original), len(decompressed))
    diff_count = 0
    first_diff = None
    
    for i in range(min_len):
        if original[i] != decompressed[i]:
            if first_diff is None:
                first_diff = i
            diff_count += 1
            if diff_count >= 10:  # 最初の10個の差分まで
                break
    
    if first_diff is not None:
        print(f"         🔴 最初の差分位置: {first_diff}")
        print(f"         📊 差分数: {diff_count}+ (最初の{min_len}バイト内)")
        
        # 周辺データ表示
        start = max(0, first_diff - 8)
        end = min(len(original), first_diff + 8)
        print(f"         📝 元データ周辺: {original[start:end].hex()}")
        if end <= len(decompressed):
            print(f"         📝 復元データ周辺: {decompressed[start:end].hex()}")
    else:
        print(f"         ✅ 先頭{min_len}バイトは一致")


def analyze_compressed_structure(compressed: bytes) -> None:
    """圧縮データ構造解析"""
    if len(compressed) < 128:
        print(f"      ⚠️ 圧縮データサイズが小さすぎます")
        return
    
    header = compressed[:128]
    
    if header[:8] == b'NXOPT400':
        print(f"      ✅ NEXUS v4.0ヘッダー確認")
        
        import struct
        try:
            original_size = struct.unpack('<Q', header[8:16])[0]
            chunk_count = struct.unpack('<I', header[16:20])[0]
            timestamp = struct.unpack('<I', header[20:24])[0]
            
            print(f"         📊 元サイズ: {original_size:,} bytes")
            print(f"         🔷 チャンク数: {chunk_count}")
            print(f"         ⏰ タイムスタンプ: {timestamp}")
            
            # ファイルタイプ
            file_type_bytes = header[24:40]
            file_type = file_type_bytes.rstrip(b'\x00').decode('utf-8', errors='ignore')
            print(f"         📁 記録ファイルタイプ: '{file_type}'")
            
            # チェックサム
            header_checksum = struct.unpack('<I', header[40:44])[0]
            print(f"         🔐 ヘッダーチェックサム: {header_checksum:08x}")
            
            # チャンク解析
            pos = 128
            for i in range(min(chunk_count, 5)):  # 最初の5チャンクまで
                if pos + 16 <= len(compressed):
                    chunk_header = compressed[pos:pos+16]
                    chunk_id, chunk_size, chunk_crc = struct.unpack('<III', chunk_header[:12])
                    print(f"         📦 チャンク{i}: ID={chunk_id}, Size={chunk_size}, CRC={chunk_crc:08x}")
                    
                    # チャンクデータの先頭確認
                    if pos + 16 + min(4, chunk_size) <= len(compressed):
                        chunk_prefix = compressed[pos+16:pos+16+min(4, chunk_size)]
                        print(f"            📝 先頭: {chunk_prefix}")
                    
                    pos += 16 + chunk_size
                else:
                    print(f"         ⚠️ チャンク{i}: データ不足")
                    break
                    
        except Exception as e:
            print(f"      ❌ ヘッダー解析エラー: {e}")
    else:
        print(f"      ❌ 不正なヘッダー: {header[:8]}")


def get_file_type(file_path: Path) -> str:
    """ファイルタイプ取得"""
    suffix = file_path.suffix.lower()
    mapping = {
        '.jpg': '画像', '.jpeg': '画像', '.png': '画像',
        '.7z': '圧縮アーカイブ', '.zip': '圧縮アーカイブ',
        '.mp3': '音楽', '.wav': '音楽',
        '.mp4': '動画', '.txt': 'テキスト'
    }
    return mapping.get(suffix, 'その他')


if __name__ == "__main__":
    diagnose_compression_issues()
