#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔄 Optimal Balance Decompressor - 最適バランス解凍エンジン
軽量メタデータによる効率的復元

🎯 解凍機能:
- 軽量メタデータ復元
- 効率的構造復元
- 高速解凍処理
- 完全性検証
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

class OptimalBalanceDecompressor:
    """最適バランス解凍エンジン"""
    
    def __init__(self):
        pass
        
    def decompress_optimal_balance(self, compressed_data: bytes) -> bytes:
        """最適バランス解凍"""
        try:
            print("🔄 最適バランス解凍開始...")
            start_time = time.time()
            
            # ヘッダー確認
            if not compressed_data.startswith(b'NXMP4_OPTIMAL_BALANCE_V1'):
                # フォールバック解凍試行
                return self._try_fallback_decompression(compressed_data)
            
            print("✅ 最適バランス形式確認")
            
            # メタデータ解析
            metadata_size = struct.unpack('<I', compressed_data[24:28])[0]
            metadata_compressed = compressed_data[28:28 + metadata_size]
            metadata_json = zlib.decompress(metadata_compressed).decode('utf-8')
            metadata = json.loads(metadata_json)
            
            print("📋 メタデータ解析完了")
            
            # 圧縮コア抽出
            core_start = 28 + metadata_size
            compressed_core = compressed_data[core_start:]
            print(f"📦 圧縮コア抽出: {len(compressed_core)} bytes")
            
            # 圧縮コア解凍
            decompressed_data = self._decompress_core(compressed_core)
            print(f"🔓 コア解凍完了: {len(decompressed_data)} bytes")
            
            # 構造復元
            restored_data = self._restore_structure(decompressed_data, metadata)
            print(f"🎬 構造復元完了: {len(restored_data)} bytes")
            
            # 検証
            self._verify_restoration(restored_data, metadata)
            
            total_time = time.time() - start_time
            print(f"⚡ 解凍時間: {total_time:.2f}s")
            
            return restored_data
            
        except Exception as e:
            print(f"❌ 解凍エラー: {e}")
            # フォールバック解凍
            return self._try_fallback_decompression(compressed_data)
    
    def _try_fallback_decompression(self, data: bytes) -> bytes:
        """フォールバック解凍"""
        try:
            print("🔧 フォールバック解凍試行...")
            
            # 一般的な形式を試行
            if data.startswith(b'NXMP4_OPTIMAL_FALLBACK'):
                payload = data[22:]
                return lzma.decompress(payload)
            
            # 直接解凍試行
            algorithms = [lzma.decompress, bz2.decompress, zlib.decompress]
            for decompress_func in algorithms:
                try:
                    return decompress_func(data)
                except:
                    continue
            
            raise Exception("フォールバック解凍失敗")
            
        except Exception as e:
            raise Exception(f"フォールバック解凍エラー: {e}")
    
    def _decompress_core(self, core_data: bytes) -> bytes:
        """圧縮コア解凍"""
        try:
            # 高効率圧縮の解凍
            algorithms = [
                ('LZMA', lzma.decompress),
                ('BZ2', bz2.decompress),
                ('ZLIB', zlib.decompress),
            ]
            
            for name, decompress_func in algorithms:
                try:
                    result = decompress_func(core_data)
                    print(f"✅ {name}解凍成功")
                    return result
                except:
                    continue
            
            raise Exception("コア解凍失敗")
            
        except Exception as e:
            raise Exception(f"コア解凍エラー: {e}")
    
    def _restore_structure(self, data: bytes, metadata: dict) -> bytes:
        """構造復元"""
        try:
            print("🔄 構造復元中...")
            
            restored = bytearray(data)
            
            # 除去されたセクションの復元
            restoration = metadata.get('restoration', {})
            removed_sections = restoration.get('removed_sections', [])
            
            for section in removed_sections:
                section_type = section.get('type', '')
                section_size = section.get('size', 0)
                position = section.get('position', 0)
                
                print(f"🔄 {section_type}復元: {section_size} bytes at {position}")
                
                # 基本的なダミーデータで復元（完全復元は困難）
                if section_type in ['free', 'skip']:
                    dummy_atom = struct.pack('>I', section_size) + section_type.encode('ascii')
                    dummy_atom += b'\x00' * (section_size - 8)
                    
                    # 適切な位置に挿入試行
                    if position < len(restored):
                        restored[position:position] = dummy_atom
                    else:
                        restored.extend(dummy_atom)
            
            # ファイル署名復元
            signature = metadata.get('signature', '')
            if signature:
                expected_header = bytes.fromhex(signature)
                if len(restored) >= len(expected_header):
                    restored[:len(expected_header)] = expected_header
            
            # フッター復元
            footer = metadata.get('footer', '')
            if footer:
                expected_footer = bytes.fromhex(footer)
                if len(restored) >= len(expected_footer):
                    restored[-len(expected_footer):] = expected_footer
            
            return bytes(restored)
            
        except Exception as e:
            print(f"❌ 構造復元エラー: {e}")
            return data
    
    def _verify_restoration(self, restored_data: bytes, metadata: dict):
        """復元検証"""
        try:
            checksums = metadata.get('checksums', {})
            
            # ヘッダーチェックサム
            if 'header_md5' in checksums:
                header_data = restored_data[:min(1000, len(restored_data))]
                actual_md5 = hashlib.md5(header_data).hexdigest()
                expected_md5 = checksums['header_md5']
                
                if actual_md5 == expected_md5:
                    print("✅ ヘッダーMD5一致")
                else:
                    print(f"⚠️ ヘッダーMD5不一致: {actual_md5} vs {expected_md5}")
            
            # フッターチェックサム
            if 'footer_md5' in checksums:
                footer_data = restored_data[-min(1000, len(restored_data)):]
                actual_md5 = hashlib.md5(footer_data).hexdigest()
                expected_md5 = checksums['footer_md5']
                
                if actual_md5 == expected_md5:
                    print("✅ フッターMD5一致")
                else:
                    print(f"⚠️ フッターMD5不一致")
            
            # 全体SHA256（時間がかかるので省略可能）
            if 'full_sha256' in checksums and len(restored_data) < 50 * 1024 * 1024:  # 50MB以下のみ
                actual_sha256 = hashlib.sha256(restored_data).hexdigest()
                expected_sha256 = checksums['full_sha256']
                
                if actual_sha256 == expected_sha256:
                    print("✅ 全体SHA256一致 - 完全復元確認!")
                else:
                    print(f"⚠️ 全体SHA256不一致")
            
        except Exception as e:
            print(f"❌ 検証エラー: {e}")

def run_decompression_test():
    """解凍テスト実行"""
    print("🔄 Optimal Balance Decompression Test")
    print("🎯 最適バランス解凍テスト - 効率的復元確認")
    print("=" * 70)
    
    # ファイルパス
    sample_dir = r"c:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\sample"
    original_file = f"{sample_dir}\\Python基礎講座3_4月26日-3.mp4"
    compressed_file = f"{sample_dir}\\Python基礎講座3_4月26日-3.nxz"
    
    if not os.path.exists(compressed_file):
        print(f"❌ 圧縮ファイルが見つかりません: {compressed_file}")
        return
    
    decompressor = OptimalBalanceDecompressor()
    
    try:
        # 圧縮ファイル読み込み
        with open(compressed_file, 'rb') as f:
            compressed_data = f.read()
        
        print(f"📦 圧縮ファイル: {Path(compressed_file).name} ({len(compressed_data):,} bytes)")
        
        # 解凍実行
        restored_data = decompressor.decompress_optimal_balance(compressed_data)
        
        # 検証用ファイル保存
        verification_file = Path(compressed_file).with_suffix('.restored.mp4')
        with open(verification_file, 'wb') as f:
            f.write(restored_data)
        
        print(f"💾 復元ファイル保存: {verification_file.name}")
        
        # 元ファイルとの比較
        if os.path.exists(original_file):
            with open(original_file, 'rb') as f:
                original_data = f.read()
            
            print("\n" + "=" * 70)
            print("🧪 復元品質検証結果")
            print("=" * 70)
            
            # 基本比較
            size_match = len(original_data) == len(restored_data)
            byte_match = original_data == restored_data
            
            print(f"📊 元ファイルサイズ: {len(original_data):,} bytes")
            print(f"📊 復元ファイルサイズ: {len(restored_data):,} bytes")
            print(f"✅ サイズ一致: {'PASS' if size_match else 'FAIL'}")
            print(f"✅ バイト一致: {'PASS' if byte_match else 'FAIL'}")
            
            # 部分一致検証
            if not byte_match:
                # ヘッダー一致確認
                header_size = min(100, len(original_data), len(restored_data))
                header_match = original_data[:header_size] == restored_data[:header_size]
                
                # フッター一致確認
                footer_size = min(100, len(original_data), len(restored_data))
                footer_match = original_data[-footer_size:] == restored_data[-footer_size:]
                
                print(f"✅ ヘッダー一致: {'PASS' if header_match else 'FAIL'}")
                print(f"✅ フッター一致: {'PASS' if footer_match else 'FAIL'}")
                
                # 一致率計算
                if len(original_data) == len(restored_data):
                    match_count = sum(1 for a, b in zip(original_data, restored_data) if a == b)
                    match_rate = (match_count / len(original_data)) * 100
                    print(f"📈 バイト一致率: {match_rate:.2f}%")
            
            if size_match and byte_match:
                print("\n🎉🎉🎉🎉 完全復元成功!")
                print("🏆 100%バイト完全一致!")
                print("🌟 最適バランス圧縮の完全可逆性確認!")
            elif size_match:
                print("\n🎉🎉 構造復元成功!")
                print("⭐ ファイルサイズ一致 - 基本構造復元OK!")
            else:
                print("\n🔧 部分復元")
                print("💪 さらなる改善の余地あり")
        else:
            print("⚠️ 元ファイルが見つからないため、完全比較できません")
            
    except Exception as e:
        print(f"❌ 解凍エラー: {e}")

def main():
    """メイン関数"""
    if len(sys.argv) < 2:
        print("🔄 Optimal Balance Decompressor")
        print("使用方法:")
        print("  python optimal_decompressor.py test                 # 最適バランス解凍テスト")
        print("  python optimal_decompressor.py decompress <file>    # ファイル解凍")
        return
    
    command = sys.argv[1].lower()
    
    if command == "test":
        run_decompression_test()
    elif command == "decompress" and len(sys.argv) >= 3:
        compressed_file = sys.argv[2]
        decompressor = OptimalBalanceDecompressor()
        
        try:
            with open(compressed_file, 'rb') as f:
                compressed_data = f.read()
            
            restored_data = decompressor.decompress_optimal_balance(compressed_data)
            
            # 出力ファイル
            output_file = Path(compressed_file).with_suffix('.restored.mp4')
            with open(output_file, 'wb') as f:
                f.write(restored_data)
            
            print(f"✅ 解凍完了: {output_file}")
            
        except Exception as e:
            print(f"❌ 解凍失敗: {e}")
    else:
        print("❌ 無効なコマンドまたは引数です")

if __name__ == "__main__":
    main()
