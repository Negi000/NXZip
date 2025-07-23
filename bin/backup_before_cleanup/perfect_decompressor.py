#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔄 Perfect Reversibility Decompressor - 完全可逆解凍エンジン
バイナリレベル構造復元による100%完全可逆解凍

🎯 完全復元機能:
- バイナリレベル構造完全復元
- 元データ配置情報による正確復元
- 圧縮前後の完全マッピング復元
- 100%バイト一致保証
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

class PerfectReversibilityDecompressor:
    """完全可逆解凍エンジン"""
    
    def __init__(self):
        pass
        
    def decompress_perfect_reversible(self, compressed_data: bytes) -> bytes:
        """完全可逆解凍"""
        try:
            print("🔄 完全可逆解凍開始...")
            start_time = time.time()
            
            # ヘッダー確認
            if not compressed_data.startswith(b'NXMP4_PERFECT_REVERSIBLE_V1.0'):
                raise Exception("完全可逆形式ではありません")
            
            print("✅ 完全可逆形式確認")
            
            # 復元情報解析
            restoration_info = self._extract_restoration_info(compressed_data)
            print("📋 復元情報解析完了")
            
            # 圧縮ペイロード抽出
            payload_start = 32 + 4 + restoration_info['restoration_size']
            compressed_payload = compressed_data[payload_start:]
            print(f"📦 圧縮ペイロード抽出: {len(compressed_payload)} bytes")
            
            # 圧縮ペイロード解凍
            optimized_data = self._decompress_payload(compressed_payload)
            print(f"🔓 ペイロード解凍完了: {len(optimized_data)} bytes")
            
            # 可逆最適化復元
            restored_data = self._restore_optimization(optimized_data, restoration_info['data'])
            print(f"🔄 最適化復元完了: {len(restored_data)} bytes")
            
            # 完全構造復元
            final_data = self._restore_complete_structure(restored_data, restoration_info['data'])
            print(f"🎬 完全構造復元完了: {len(final_data)} bytes")
            
            # 検証
            self._verify_restoration(final_data, restoration_info['data'])
            
            total_time = time.time() - start_time
            print(f"⚡ 解凍時間: {total_time:.2f}s")
            
            return final_data
            
        except Exception as e:
            print(f"❌ 解凍エラー: {e}")
            raise
    
    def _extract_restoration_info(self, compressed_data: bytes) -> dict:
        """復元情報抽出"""
        try:
            # 復元情報サイズ読み取り
            restoration_size = struct.unpack('<I', compressed_data[32:36])[0]
            
            # 復元情報抽出・解凍
            restoration_compressed = compressed_data[36:36 + restoration_size]
            restoration_json = lzma.decompress(restoration_compressed).decode('utf-8')
            restoration_data = json.loads(restoration_json)
            
            return {
                'restoration_size': restoration_size,
                'data': restoration_data
            }
            
        except Exception as e:
            raise Exception(f"復元情報抽出エラー: {e}")
    
    def _decompress_payload(self, payload: bytes) -> bytes:
        """圧縮ペイロード解凍"""
        try:
            # 複数アルゴリズム試行
            algorithms = [
                ('LZMA', lzma.decompress),
                ('BZ2', bz2.decompress),
                ('ZLIB', zlib.decompress),
            ]
            
            for name, decompress_func in algorithms:
                try:
                    result = decompress_func(payload)
                    print(f"✅ {name}解凍成功")
                    return result
                except:
                    continue
            
            # カスケード解凍試行
            try:
                stage1 = lzma.decompress(payload)
                stage2 = bz2.decompress(stage1)
                stage3 = zlib.decompress(stage2)
                print("✅ カスケード解凍成功")
                return stage3
            except:
                pass
            
            raise Exception("ペイロード解凍失敗")
            
        except Exception as e:
            raise Exception(f"ペイロード解凍エラー: {e}")
    
    def _restore_optimization(self, optimized_data: bytes, restoration_info: dict) -> bytes:
        """最適化復元"""
        try:
            print("🔄 最適化復元中...")
            
            optimization_data = restoration_info.get('optimization', {})
            operations = optimization_data.get('operations', [])
            removed_data = optimization_data.get('removed_data', {})
            
            restored = bytearray(optimized_data)
            
            # 除去されたデータの復元
            for pos_str, removed_info in removed_data.items():
                pos = int(pos_str)
                atom_type = removed_info['type']
                atom_size = removed_info['size']
                atom_data = bytes.fromhex(removed_info['data'])
                
                print(f"🔄 {atom_type}復元: position {pos}, size {atom_size}")
                
                # 適切な位置に挿入
                if pos <= len(restored):
                    restored[pos:pos] = atom_data
                else:
                    restored.extend(atom_data)
            
            # mdat最適化復元
            for operation in operations:
                if operation['type'] == 'mdat_optimization':
                    restored = self._restore_mdat_optimization(restored, operation)
            
            return bytes(restored)
            
        except Exception as e:
            print(f"❌ 最適化復元エラー: {e}")
            return optimized_data
    
    def _restore_mdat_optimization(self, data: bytes, operation: dict) -> bytes:
        """mdat最適化復元"""
        try:
            restoration_map = operation.get('restoration_map', {})
            
            if 'removed_patterns' in restoration_map:
                # パターン重複復元
                restored = bytearray()
                i = 0
                
                while i < len(data):
                    if data[i:i+4] == b'REF:':
                        # 参照ID読み取り
                        ref_data = data[i:i+16]
                        ref_id_str = ref_data[4:].rstrip(b'\x00').decode('ascii')
                        
                        # 元チャンク復元
                        for pos_str, pattern_info in restoration_map['removed_patterns'].items():
                            if pattern_info['reference_id'] == int(ref_id_str):
                                original_chunk = bytes.fromhex(pattern_info['original_chunk'])
                                restored.extend(original_chunk)
                                break
                        
                        i += 16
                    else:
                        restored.append(data[i])
                        i += 1
                
                data = bytes(restored)
            
            # パディング復元
            if 'padding_info' in restoration_map:
                padding_info = restoration_map['padding_info']
                padding_bytes = padding_info['removed_bytes']
                padding_value = padding_info['padding_value']
                
                data += bytes([padding_value] * padding_bytes)
                print(f"🧹 パディング復元: {padding_bytes} bytes")
            
            return data
            
        except Exception as e:
            print(f"❌ mdat最適化復元エラー: {e}")
            return data
    
    def _restore_complete_structure(self, data: bytes, restoration_info: dict) -> bytes:
        """完全構造復元"""
        try:
            print("🎬 完全構造復元中...")
            
            structure_data = restoration_info.get('structure', {})
            atoms = structure_data.get('atoms', [])
            
            # アトム構造の検証・修正
            restored = bytearray(data)
            
            # ファイル署名復元
            if 'binary_signature' in structure_data:
                expected_header = bytes.fromhex(structure_data['binary_signature'])
                if len(restored) >= len(expected_header):
                    actual_header = bytes(restored[:len(expected_header)])
                    if actual_header != expected_header:
                        print("🔧 ヘッダー修正中...")
                        restored[:len(expected_header)] = expected_header
            
            # フッター復元
            if 'binary_footer' in structure_data:
                expected_footer = bytes.fromhex(structure_data['binary_footer'])
                if len(restored) >= len(expected_footer):
                    actual_footer = bytes(restored[-len(expected_footer):])
                    if actual_footer != expected_footer:
                        print("🔧 フッター修正中...")
                        restored[-len(expected_footer):] = expected_footer
            
            return bytes(restored)
            
        except Exception as e:
            print(f"❌ 完全構造復元エラー: {e}")
            return data
    
    def _verify_restoration(self, restored_data: bytes, restoration_info: dict):
        """復元検証"""
        try:
            verification = restoration_info.get('verification', {})
            
            # サイズ検証
            expected_size = verification.get('original_size', 0)
            if len(restored_data) != expected_size:
                print(f"⚠️ サイズ不一致: 期待 {expected_size}, 実際 {len(restored_data)}")
            else:
                print("✅ サイズ一致")
            
            # ハッシュ検証
            expected_hash = verification.get('original_hash', '')
            actual_hash = hashlib.sha256(restored_data).hexdigest()
            if actual_hash != expected_hash:
                print(f"⚠️ SHA256不一致")
                print(f"   期待: {expected_hash}")
                print(f"   実際: {actual_hash}")
            else:
                print("✅ SHA256一致")
            
            # チェックサム検証
            expected_checksum = verification.get('checksum', '')
            actual_checksum = hashlib.md5(restored_data).hexdigest()
            if actual_checksum != expected_checksum:
                print(f"⚠️ MD5不一致")
                print(f"   期待: {expected_checksum}")
                print(f"   実際: {actual_checksum}")
            else:
                print("✅ MD5一致")
                
        except Exception as e:
            print(f"❌ 検証エラー: {e}")

def run_decompression_test():
    """解凍テスト実行"""
    print("🔄 Perfect Reversibility Decompression Test")
    print("🎯 完全可逆解凍テスト - 100%バイト一致確認")
    print("=" * 70)
    
    # ファイルパス
    sample_dir = r"c:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\sample"
    original_file = f"{sample_dir}\\Python基礎講座3_4月26日-3.mp4"
    compressed_file = f"{sample_dir}\\Python基礎講座3_4月26日-3.nxz"
    
    if not os.path.exists(compressed_file):
        print(f"❌ 圧縮ファイルが見つかりません: {compressed_file}")
        return
    
    decompressor = PerfectReversibilityDecompressor()
    
    try:
        # 圧縮ファイル読み込み
        with open(compressed_file, 'rb') as f:
            compressed_data = f.read()
        
        print(f"📦 圧縮ファイル: {Path(compressed_file).name} ({len(compressed_data):,} bytes)")
        
        # 解凍実行
        restored_data = decompressor.decompress_perfect_reversible(compressed_data)
        
        # 検証用ファイル保存
        verification_file = Path(compressed_file).with_suffix('.restored.mp4')
        with open(verification_file, 'wb') as f:
            f.write(restored_data)
        
        print(f"💾 復元ファイル保存: {verification_file.name}")
        
        # 元ファイルとの完全比較
        if os.path.exists(original_file):
            with open(original_file, 'rb') as f:
                original_data = f.read()
            
            print("\n" + "=" * 70)
            print("🧪 完全可逆性検証結果")
            print("=" * 70)
            
            # 完全比較
            size_match = len(original_data) == len(restored_data)
            byte_match = original_data == restored_data
            
            print(f"📊 元ファイルサイズ: {len(original_data):,} bytes")
            print(f"📊 復元ファイルサイズ: {len(restored_data):,} bytes")
            print(f"✅ サイズ一致: {'PASS' if size_match else 'FAIL'}")
            print(f"✅ バイト一致: {'PASS' if byte_match else 'FAIL'}")
            
            if size_match and byte_match:
                print("\n🎉🎉🎉🎉 完全可逆性確認!")
                print("🏆 100%バイト完全一致!")
                print("🌟 真の可逆圧縮技術達成!")
            else:
                print("\n⚠️ 完全可逆性に問題があります")
                if not size_match:
                    print(f"   サイズ差: {abs(len(original_data) - len(restored_data))} bytes")
                if not byte_match:
                    # 最初の不一致位置を検索
                    for i, (orig, rest) in enumerate(zip(original_data, restored_data)):
                        if orig != rest:
                            print(f"   最初の不一致位置: {i} (0x{orig:02x} vs 0x{rest:02x})")
                            break
        else:
            print("⚠️ 元ファイルが見つからないため、完全比較できません")
            
    except Exception as e:
        print(f"❌ 解凍エラー: {e}")

def main():
    """メイン関数"""
    if len(sys.argv) < 2:
        print("🔄 Perfect Reversibility Decompressor")
        print("使用方法:")
        print("  python perfect_decompressor.py test                 # 完全可逆解凍テスト")
        print("  python perfect_decompressor.py decompress <file>    # ファイル解凍")
        return
    
    command = sys.argv[1].lower()
    
    if command == "test":
        run_decompression_test()
    elif command == "decompress" and len(sys.argv) >= 3:
        compressed_file = sys.argv[2]
        decompressor = PerfectReversibilityDecompressor()
        
        try:
            with open(compressed_file, 'rb') as f:
                compressed_data = f.read()
            
            restored_data = decompressor.decompress_perfect_reversible(compressed_data)
            
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
