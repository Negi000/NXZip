#!/usr/bin/env python3
"""
7-Zip統合エラー修正版
Windows一時ファイル競合問題の対策
"""

import os
import time
import tempfile
import shutil
from pathlib import Path
from typing import Tuple, Dict, Any

try:
    import py7zr
    PY7ZR_AVAILABLE = True
except ImportError:
    PY7ZR_AVAILABLE = False


class SevenZipFixed:
    """7-Zip統合 Windows対応修正版"""
    
    def __init__(self):
        self.temp_counter = 0
    
    def compress_7zip_robust(self, data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """堅牢な7-Zip圧縮 (Windows一時ファイル対策)"""
        if not PY7ZR_AVAILABLE:
            raise ImportError("py7zr not available")
        
        start_time = time.time()
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # 重複回避のための一意な名前生成
                self.temp_counter += 1
                temp_suffix = f"_7z_{os.getpid()}_{self.temp_counter}_{attempt}"
                
                # 明示的な一時ディレクトリ作成
                with tempfile.TemporaryDirectory(prefix="nxz_7zip_") as temp_dir:
                    temp_path = Path(temp_dir)
                    
                    # 入力ファイル
                    input_file = temp_path / f"input{temp_suffix}.bin"
                    with open(input_file, 'wb') as f:
                        f.write(data)
                    
                    # 出力ファイル  
                    output_file = temp_path / f"output{temp_suffix}.7z"
                    
                    # 7z圧縮実行
                    with py7zr.SevenZipFile(output_file, 'w') as archive:
                        archive.write(input_file, 'data.bin')
                    
                    # 結果読み込み
                    if output_file.exists():
                        with open(output_file, 'rb') as f:
                            compressed_data = f.read()
                        
                        total_time = time.time() - start_time
                        
                        return compressed_data, {
                            'method': '7-Zip (修正版)',
                            'original_size': len(data),
                            'compressed_size': len(compressed_data),
                            'compression_time': total_time,
                            'attempt': attempt + 1
                        }
                    else:
                        raise FileNotFoundError("7zファイルが作成されませんでした")
                        
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"7-Zip圧縮失敗 (試行{max_retries}回): {e}")
                else:
                    print(f"🔄 7-Zip試行{attempt + 1}失敗: {e}, 再試行中...")
                    time.sleep(0.1 * (attempt + 1))  # 指数バックオフ
    
    def decompress_7zip_robust(self, compressed_data: bytes) -> bytes:
        """堅牢な7-Zip解凍"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # 重複回避のための一意な名前生成
                self.temp_counter += 1
                temp_suffix = f"_7z_dec_{os.getpid()}_{self.temp_counter}_{attempt}"
                
                with tempfile.TemporaryDirectory(prefix="nxz_7zip_dec_") as temp_dir:
                    temp_path = Path(temp_dir)
                    
                    # 圧縮ファイル書き込み
                    archive_file = temp_path / f"archive{temp_suffix}.7z"
                    with open(archive_file, 'wb') as f:
                        f.write(compressed_data)
                    
                    # 展開ディレクトリ
                    extract_dir = temp_path / f"extract{temp_suffix}"
                    extract_dir.mkdir()
                    
                    # 7z展開
                    with py7zr.SevenZipFile(archive_file, 'r') as archive:
                        archive.extractall(extract_dir)
                    
                    # 結果取得
                    extracted_file = extract_dir / 'data.bin'
                    if extracted_file.exists():
                        with open(extracted_file, 'rb') as f:
                            return f.read()
                    else:
                        # ファイル名が異なる場合を検索
                        extracted_files = list(extract_dir.glob('*'))
                        if extracted_files:
                            with open(extracted_files[0], 'rb') as f:
                                return f.read()
                        else:
                            raise FileNotFoundError("展開されたファイルが見つかりません")
                            
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"7-Zip解凍失敗 (試行{max_retries}回): {e}")
                else:
                    print(f"🔄 7-Zip解凍試行{attempt + 1}失敗: {e}, 再試行中...")
                    time.sleep(0.1 * (attempt + 1))


def test_7zip_fix():
    """7-Zip修正版テスト"""
    if not PY7ZR_AVAILABLE:
        print("❌ py7zr not available")
        return False
    
    print("🔧 7-Zip修正版テスト開始...")
    
    # テストデータ
    test_data = b"Hello, World! This is a test for 7-Zip fix. " * 1000
    print(f"📊 テストデータサイズ: {len(test_data):,} bytes")
    
    try:
        # 7-Zip修正版
        sevenzip = SevenZipFixed()
        
        # 圧縮テスト
        print("🗜️ 圧縮テスト...")
        compressed, info = sevenzip.compress_7zip_robust(test_data)
        ratio = (1 - len(compressed) / len(test_data)) * 100
        
        print(f"✅ 圧縮成功: {len(test_data)} → {len(compressed)} bytes ({ratio:.1f}%)")
        print(f"⏱️ 圧縮時間: {info['compression_time']:.3f}秒 (試行{info['attempt']}回)")
        
        # 解凍テスト
        print("📂 解凍テスト...")
        start_decomp = time.time()
        decompressed = sevenzip.decompress_7zip_robust(compressed)
        decomp_time = time.time() - start_decomp
        
        # 可逆性チェック
        if decompressed == test_data:
            print(f"✅ 解凍成功: {len(decompressed)} bytes")
            print(f"⏱️ 解凍時間: {decomp_time:.3f}秒")
            print("🎉 7-Zip修正版テスト成功！")
            return True
        else:
            print("❌ 可逆性エラー")
            return False
            
    except Exception as e:
        print(f"❌ 7-Zip修正版テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_7zip_fix()
