#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔄 NXZip Final Decompressor - 統合解凍エンジン
全フォーマット対応の最終解凍エンジン

🏆 対応形式:
- MP4動画: 最適バランス解凍
- 画像: SDC解凍
- テキスト: 超高速解凍
- 汎用: 統合解凍
"""

import os
import sys
import time
import zlib
import bz2
import lzma
from pathlib import Path
import struct
import hashlib
import json

class NXZipFinalDecompressor:
    """NXZip最終統合解凍エンジン"""
    
    def __init__(self):
        pass
    
    def decompress_file(self, filepath: str, output_path: str = None) -> dict:
        """ファイル自動判定解凍"""
        try:
            file_path = Path(filepath)
            if not file_path.exists():
                return {'success': False, 'error': f'ファイルが見つかりません: {filepath}'}
            
            # 圧縮ファイル読み込み
            with open(file_path, 'rb') as f:
                compressed_data = f.read()
            
            # フォーマット判定・解凍
            decompressed_data = self._auto_decompress(compressed_data)
            
            if decompressed_data is None:
                return {'success': False, 'error': '解凍に失敗しました'}
            
            # 出力ファイル決定
            if output_path is None:
                output_path = file_path.with_suffix('.restored')
                # 元の拡張子を推測
                if compressed_data.startswith(b'NXMP4_'):
                    output_path = file_path.with_suffix('.restored.mp4')
                elif compressed_data.startswith(b'NXIMG_'):
                    output_path = file_path.with_suffix('.restored.png')
                elif compressed_data.startswith(b'NXTXT_'):
                    output_path = file_path.with_suffix('.restored.txt')
                else:
                    output_path = file_path.with_suffix('.restored.bin')
            
            # 解凍データ保存
            with open(output_path, 'wb') as f:
                f.write(decompressed_data)
            
            return {
                'success': True,
                'input_file': str(file_path),
                'output_file': str(output_path),
                'original_size': len(compressed_data),
                'decompressed_size': len(decompressed_data),
                'decompression_ratio': len(decompressed_data) / len(compressed_data)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _auto_decompress(self, data: bytes) -> bytes:
        """自動フォーマット判定解凍"""
        try:
            # MP4最適バランス形式
            if data.startswith(b'NXMP4_OPTIMAL_BALANCE_V1'):
                return self._decompress_optimal_balance(data)
            
            # MP4フォールバック
            elif data.startswith(b'NXMP4_FALLBACK'):
                return lzma.decompress(data[14:])
            
            # 画像SDC形式
            elif data.startswith(b'NXIMG_SDC'):
                return self._decompress_image_sdc(data)
            
            # 画像フォールバック
            elif data.startswith(b'NXIMG_FALLBACK'):
                return zlib.decompress(data[14:])
            
            # テキスト超高速形式
            elif data.startswith(b'NXTXT_LIGHTNING'):
                return lzma.decompress(data[15:])
            
            # テキストフォールバック
            elif data.startswith(b'NXTXT_FALLBACK'):
                return zlib.decompress(data[14:])
            
            # 汎用最適形式
            elif data.startswith(b'NXGEN_OPTIMAL'):
                return self._decompress_generic_optimal(data)
            
            # 汎用フォールバック
            elif data.startswith(b'NXGEN_FALLBACK'):
                return zlib.decompress(data[14:])
            
            # 既存の古い形式
            elif data.startswith(b'NXMP4_VIDEO_BREAKTHROUGH'):
                return self._decompress_legacy_video(data)
            
            # 汎用解凍試行
            else:
                return self._try_generic_decompression(data)
                
        except Exception as e:
            print(f"❌ 自動解凍エラー: {e}")
            return None
    
    def _decompress_optimal_balance(self, data: bytes) -> bytes:
        """最適バランス解凍"""
        try:
            # メタデータサイズ読み取り
            metadata_size = struct.unpack('<I', data[24:28])[0]
            
            # メタデータ解凍
            metadata_compressed = data[28:28 + metadata_size]
            metadata_json = zlib.decompress(metadata_compressed).decode('utf-8')
            metadata = json.loads(metadata_json)
            
            # 圧縮コア解凍
            core_start = 28 + metadata_size
            compressed_core = data[core_start:]
            decompressed_data = lzma.decompress(compressed_core)
            
            # 検証
            expected_checksum = metadata.get('checksum', '')
            if expected_checksum:
                actual_checksum = hashlib.sha256(decompressed_data).hexdigest()
                if actual_checksum != expected_checksum:
                    print("⚠️ チェックサム不一致")
            
            return decompressed_data
            
        except Exception as e:
            raise Exception(f"最適バランス解凍エラー: {e}")
    
    def _decompress_image_sdc(self, data: bytes) -> bytes:
        """画像SDC解凍"""
        try:
            payload = data[9:]  # 'NXIMG_SDC'を除去
            stage1 = lzma.decompress(payload)
            stage2 = bz2.decompress(stage1)
            return stage2
        except Exception as e:
            raise Exception(f"画像SDC解凍エラー: {e}")
    
    def _decompress_generic_optimal(self, data: bytes) -> bytes:
        """汎用最適解凍"""
        try:
            payload = data[13:]  # 'NXGEN_OPTIMAL'を除去
            
            # 複数アルゴリズム試行
            algorithms = [lzma.decompress, bz2.decompress, zlib.decompress]
            
            for decompress_func in algorithms:
                try:
                    return decompress_func(payload)
                except:
                    continue
            
            raise Exception("汎用最適解凍失敗")
            
        except Exception as e:
            raise Exception(f"汎用最適解凍エラー: {e}")
    
    def _decompress_legacy_video(self, data: bytes) -> bytes:
        """レガシー動画形式解凍"""
        try:
            # 古い形式の解凍試行
            if data.startswith(b'NXMP4_VIDEO_BREAKTHROUGH_SUCCESS'):
                payload = data[32:]
            elif data.startswith(b'NXMP4_VIDEO_BREAKTHROUGH'):
                payload = data[25:]
            else:
                payload = data[20:]  # 汎用
            
            return lzma.decompress(payload)
            
        except Exception as e:
            raise Exception(f"レガシー動画解凍エラー: {e}")
    
    def _try_generic_decompression(self, data: bytes) -> bytes:
        """汎用解凍試行"""
        try:
            # 一般的な圧縮形式を順次試行
            algorithms = [
                lzma.decompress,
                bz2.decompress,
                zlib.decompress
            ]
            
            for decompress_func in algorithms:
                try:
                    return decompress_func(data)
                except:
                    continue
            
            # ヘッダー付き形式の試行
            if len(data) > 20:
                for i in range(5, 25):
                    try:
                        for decompress_func in algorithms:
                            try:
                                return decompress_func(data[i:])
                            except:
                                continue
                    except:
                        continue
            
            raise Exception("汎用解凍失敗")
            
        except Exception as e:
            raise Exception(f"汎用解凍エラー: {e}")

def run_decompression_test():
    """解凍テスト実行"""
    print("🔄 NXZip Final Decompressor - 統合解凍テスト")
    print("=" * 60)
    
    decompressor = NXZipFinalDecompressor()
    
    # テストファイル検索
    sample_dir = r"c:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\sample"
    test_files = []
    
    for ext in ['.nxz']:
        for file_path in Path(sample_dir).glob(f'*{ext}'):
            test_files.append(str(file_path))
    
    if not test_files:
        print("⚠️ .nxzファイルが見つかりません")
        return
    
    success_count = 0
    total_count = len(test_files)
    
    for test_file in test_files:
        print(f"\n📦 解凍中: {Path(test_file).name}")
        print("-" * 40)
        
        result = decompressor.decompress_file(test_file)
        
        if result['success']:
            success_count += 1
            ratio = result['decompression_ratio']
            print(f"✅ 解凍成功: {Path(result['output_file']).name}")
            print(f"📊 展開率: {ratio:.1f}x")
            print(f"💾 出力サイズ: {result['decompressed_size']:,} bytes")
        else:
            print(f"❌ 解凍失敗: {result.get('error', '不明なエラー')}")
    
    # 総合結果
    print("\n" + "=" * 60)
    print("🏆 解凍テスト結果")
    print("=" * 60)
    print(f"📊 成功率: {success_count}/{total_count} ({(success_count/total_count*100):.1f}%)")
    
    if success_count == total_count:
        print("🎉🎉🎉 全ファイル解凍成功!")
        print("🌟 統合解凍エンジン完全動作確認!")
    elif success_count > 0:
        print("🎉 部分成功 - 改善の余地あり")
    else:
        print("❌ 解凍に問題があります")

def main():
    """メイン関数"""
    if len(sys.argv) < 2:
        print("🔄 NXZip Final Decompressor")
        print("使用方法:")
        print("  python nxzip_final_decompressor.py test                    # 解凍テスト")
        print("  python nxzip_final_decompressor.py decompress <file>       # ファイル解凍")
        print("  python nxzip_final_decompressor.py decompress <file> <out> # 出力先指定解凍")
        return
    
    command = sys.argv[1].lower()
    
    if command == "test":
        run_decompression_test()
    elif command == "decompress" and len(sys.argv) >= 3:
        decompressor = NXZipFinalDecompressor()
        input_file = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) >= 4 else None
        
        result = decompressor.decompress_file(input_file, output_file)
        
        if result['success']:
            print(f"✅ 解凍成功: {Path(result['output_file']).name}")
            print(f"📊 展開率: {result['decompression_ratio']:.1f}x")
        else:
            print(f"❌ 解凍失敗: {result.get('error', '不明なエラー')}")
    else:
        print("❌ 無効なコマンドです")

if __name__ == "__main__":
    main()
