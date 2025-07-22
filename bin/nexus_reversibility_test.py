#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔄 NEXUS Video Reversibility Test - 完全可逆性検証
91.5%圧縮の完全可逆性を徹底検証

🎯 検証項目:
- バイト完全一致確認
- ハッシュ値比較
- ファイルサイズ確認
- MD5/SHA256検証
- 動画再生可能性確認
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

class VideoReversibilityValidator:
    """動画完全可逆性検証エンジン"""
    
    def __init__(self):
        self.validation_results = []
        
    def validate_compressed_file(self, original_file: str, compressed_file: str) -> dict:
        """圧縮ファイルの完全可逆性検証"""
        print("🔄 完全可逆性検証開始...")
        print("=" * 70)
        
        start_time = time.time()
        validation_result = {
            'success': False,
            'original_file': original_file,
            'compressed_file': compressed_file,
            'tests': {},
            'errors': []
        }
        
        try:
            # 1. ファイル存在確認
            if not os.path.exists(original_file):
                validation_result['errors'].append(f"元ファイルが見つかりません: {original_file}")
                return validation_result
                
            if not os.path.exists(compressed_file):
                validation_result['errors'].append(f"圧縮ファイルが見つかりません: {compressed_file}")
                return validation_result
            
            print(f"📄 元ファイル: {Path(original_file).name}")
            print(f"📦 圧縮ファイル: {Path(compressed_file).name}")
            
            # 2. 元ファイル読み込み
            print("🔍 元ファイル解析中...")
            with open(original_file, 'rb') as f:
                original_data = f.read()
            
            original_size = len(original_data)
            original_md5 = hashlib.md5(original_data).hexdigest()
            original_sha256 = hashlib.sha256(original_data).hexdigest()
            
            print(f"📊 元ファイルサイズ: {original_size:,} bytes")
            print(f"🔐 元MD5: {original_md5}")
            print(f"🔐 元SHA256: {original_sha256[:16]}...")
            
            # 3. 圧縮ファイル読み込み
            print("\n📦 圧縮ファイル解析中...")
            with open(compressed_file, 'rb') as f:
                compressed_data = f.read()
            
            compressed_size = len(compressed_data)
            print(f"📊 圧縮ファイルサイズ: {compressed_size:,} bytes")
            print(f"📈 圧縮率: {((1 - compressed_size/original_size) * 100):.1f}%")
            
            # 4. 解凍処理
            print("\n🔄 解凍処理開始...")
            decompressed_data = self.decompress_video_data(compressed_data)
            
            if decompressed_data is None:
                validation_result['errors'].append("解凍処理に失敗しました")
                return validation_result
            
            decompressed_size = len(decompressed_data)
            print(f"📊 解凍後サイズ: {decompressed_size:,} bytes")
            
            # 5. 完全可逆性検証テスト群
            print("\n🧪 完全可逆性検証テスト実行...")
            print("-" * 50)
            
            # テスト1: サイズ完全一致
            size_match = (original_size == decompressed_size)
            validation_result['tests']['size_match'] = size_match
            print(f"✅ サイズ一致テスト: {'PASS' if size_match else 'FAIL'}")
            if not size_match:
                print(f"   ❌ 元: {original_size:,} vs 解凍後: {decompressed_size:,}")
            
            # テスト2: バイト完全一致
            byte_match = (original_data == decompressed_data)
            validation_result['tests']['byte_match'] = byte_match
            print(f"✅ バイト一致テスト: {'PASS' if byte_match else 'FAIL'}")
            
            # テスト3: MD5ハッシュ一致
            decompressed_md5 = hashlib.md5(decompressed_data).hexdigest()
            md5_match = (original_md5 == decompressed_md5)
            validation_result['tests']['md5_match'] = md5_match
            print(f"✅ MD5一致テスト: {'PASS' if md5_match else 'FAIL'}")
            if not md5_match:
                print(f"   ❌ 元: {original_md5}")
                print(f"   ❌ 解凍: {decompressed_md5}")
            
            # テスト4: SHA256ハッシュ一致
            decompressed_sha256 = hashlib.sha256(decompressed_data).hexdigest()
            sha256_match = (original_sha256 == decompressed_sha256)
            validation_result['tests']['sha256_match'] = sha256_match
            print(f"✅ SHA256一致テスト: {'PASS' if sha256_match else 'FAIL'}")
            
            # テスト5: MP4構造検証
            mp4_structure_valid = self.validate_mp4_structure(decompressed_data)
            validation_result['tests']['mp4_structure'] = mp4_structure_valid
            print(f"✅ MP4構造テスト: {'PASS' if mp4_structure_valid else 'FAIL'}")
            
            # テスト6: 先頭・末尾バイト確認
            header_match = (original_data[:100] == decompressed_data[:100]) if len(original_data) >= 100 and len(decompressed_data) >= 100 else False
            footer_match = (original_data[-100:] == decompressed_data[-100:]) if len(original_data) >= 100 and len(decompressed_data) >= 100 else False
            validation_result['tests']['header_match'] = header_match
            validation_result['tests']['footer_match'] = footer_match
            print(f"✅ ヘッダー一致テスト: {'PASS' if header_match else 'FAIL'}")
            print(f"✅ フッター一致テスト: {'PASS' if footer_match else 'FAIL'}")
            
            # 総合判定
            all_tests = [size_match, byte_match, md5_match, sha256_match, mp4_structure_valid, header_match, footer_match]
            validation_result['success'] = all(all_tests)
            
            # 解凍ファイル保存（検証用）
            verification_file = Path(compressed_file).with_suffix('.verified.mp4')
            with open(verification_file, 'wb') as f:
                f.write(decompressed_data)
            validation_result['verification_file'] = str(verification_file)
            
            # 結果表示
            print("\n" + "=" * 70)
            print("🏆 完全可逆性検証結果")
            print("=" * 70)
            
            passed_tests = sum(all_tests)
            total_tests = len(all_tests)
            
            if validation_result['success']:
                print("🎉🎉🎉🎉 完全可逆性検証 - 完全成功!")
                print("✨ すべてのテストをパス")
                print("🔄 100%完全可逆圧縮確認済み")
                print("🏆 真の圧縮技術革命達成!")
            else:
                print(f"⚠️ 可逆性検証 - 部分成功 ({passed_tests}/{total_tests})")
                print("🔧 改善が必要な項目があります")
            
            print(f"📊 検証テスト結果: {passed_tests}/{total_tests} パス")
            print(f"⚡ 検証時間: {time.time() - start_time:.2f}s")
            print(f"💾 検証ファイル: {verification_file.name}")
            
            return validation_result
            
        except Exception as e:
            validation_result['errors'].append(f"検証エラー: {str(e)}")
            print(f"❌ 検証エラー: {e}")
            return validation_result
    
    def decompress_video_data(self, compressed_data: bytes) -> bytes:
        """動画データ解凍"""
        try:
            print("🔄 解凍アルゴリズム判定中...")
            
            # ヘッダーからアルゴリズム判定
            if compressed_data.startswith(b'NXMP4_VIDEO_BREAKTHROUGH_SUCCESS'):
                print("✨ VIDEO_BREAKTHROUGH_SUCCESS形式検出")
                payload = compressed_data[32:]  # ヘッダー除去
                return self.decompress_video_breakthrough(payload)
            elif compressed_data.startswith(b'NXMP4_VIDEO_BREAKTHROUGH_NEAR'):
                print("⭐ VIDEO_BREAKTHROUGH_NEAR形式検出")
                payload = compressed_data[29:]
                return self.decompress_video_breakthrough(payload)
            elif compressed_data.startswith(b'NXMP4_VIDEO_BREAKTHROUGH_HIGH'):
                print("🎯 VIDEO_BREAKTHROUGH_HIGH形式検出")
                payload = compressed_data[29:]
                return self.decompress_video_breakthrough(payload)
            elif compressed_data.startswith(b'NXMP4_VIDEO_BREAKTHROUGH_BASIC'):
                print("🎪 VIDEO_BREAKTHROUGH_BASIC形式検出")
                payload = compressed_data[30:]
                return self.decompress_video_breakthrough(payload)
            elif compressed_data.startswith(b'NXMP4_VIDEO_FALLBACK'):
                print("🔧 VIDEO_FALLBACK形式検出")
                payload = compressed_data[20:]
                return lzma.decompress(payload)
            else:
                print("❓ 不明な形式 - 汎用解凍試行")
                return self.try_generic_decompress(compressed_data)
                
        except Exception as e:
            print(f"❌ 解凍エラー: {e}")
            return None
    
    def decompress_video_breakthrough(self, compressed_payload: bytes) -> bytes:
        """Video Breakthrough解凍"""
        try:
            print("🎬 Video Breakthrough解凍処理...")
            
            # 逆順解凍（圧縮時の逆順）
            # ステップ1: 並列圧縮の解凍
            stage1_data = self.decompress_parallel_video(compressed_payload)
            print(f"🔄 並列解凍完了: {len(stage1_data):,} bytes")
            
            # ステップ2: 動画最適化の復元
            stage2_data = self.restore_video_optimization(stage1_data)
            print(f"🎥 動画構造復元完了: {len(stage2_data):,} bytes")
            
            # ステップ3: 元のMP4構造復元
            final_data = self.restore_original_mp4_structure(stage2_data)
            print(f"📱 MP4復元完了: {len(final_data):,} bytes")
            
            return final_data
            
        except Exception as e:
            print(f"❌ Video Breakthrough解凍エラー: {e}")
            # フォールバック: 直接LZMA解凍試行
            try:
                return lzma.decompress(compressed_payload)
            except:
                return None
    
    def decompress_parallel_video(self, data: bytes) -> bytes:
        """並列圧縮解凍"""
        try:
            # 主要アルゴリズムで順次解凍試行
            algorithms = [
                ('LZMA', lzma.decompress),
                ('BZ2', bz2.decompress),
                ('ZLIB', zlib.decompress),
            ]
            
            for name, decompress_func in algorithms:
                try:
                    result = decompress_func(data)
                    print(f"✅ {name}解凍成功")
                    return result
                except:
                    continue
            
            # カスケード解凍試行
            try:
                # 3段階逆解凍
                stage1 = lzma.decompress(data)
                stage2 = bz2.decompress(stage1)
                stage3 = zlib.decompress(stage2)
                print("✅ カスケード解凍成功")
                return stage3
            except:
                pass
            
            raise Exception("すべての解凍アルゴリズムで失敗")
            
        except Exception as e:
            print(f"❌ 並列解凍エラー: {e}")
            raise
    
    def restore_video_optimization(self, data: bytes) -> bytes:
        """動画最適化復元"""
        try:
            # 圧縮時の最適化を逆転
            print("🎥 動画最適化復元中...")
            
            # MP4アトム構造の復元
            restored = bytearray()
            pos = 0
            
            while pos < len(data) - 8:
                if pos + 8 > len(data):
                    restored.extend(data[pos:])
                    break
                
                size = struct.unpack('>I', data[pos:pos + 4])[0]
                atom_type = data[pos + 4:pos + 8]
                
                if size == 0:
                    # 残りすべて
                    restored.extend(data[pos:])
                    break
                
                if atom_type == b'mdat':
                    # mdatデータの復元
                    mdat_content = data[pos + 8:pos + size]
                    restored_mdat = self.restore_mdat_content(mdat_content)
                    
                    # 復元されたmdatヘッダー作成
                    new_size = len(restored_mdat) + 8
                    restored.extend(struct.pack('>I', new_size))
                    restored.extend(b'mdat')
                    restored.extend(restored_mdat)
                else:
                    # その他のアトムはそのまま
                    restored.extend(data[pos:pos + size])
                
                pos += size
            
            return bytes(restored)
            
        except Exception as e:
            print(f"❌ 動画最適化復元エラー: {e}")
            # 復元失敗時は元データ返却
            return data
    
    def restore_mdat_content(self, mdat_data: bytes) -> bytes:
        """mdatコンテンツ復元"""
        try:
            # 圧縮時の最適化を可能な限り復元
            # 注意: 一部の最適化は完全復元不可能
            
            # パディング復元試行
            restored = self.restore_video_padding(mdat_data)
            
            # フレーム構造復元試行（限定的）
            # 完全復元は不可能だが、再生可能な形に復元
            
            return restored
            
        except:
            return mdat_data
    
    def restore_video_padding(self, data: bytes) -> bytes:
        """動画パディング復元試行"""
        try:
            # 基本的には不可逆な最適化のため限定的復元
            # ファイル末尾に最小限のパディング追加
            return data + b'\x00' * 16
        except:
            return data
    
    def restore_original_mp4_structure(self, data: bytes) -> bytes:
        """元のMP4構造復元"""
        try:
            # MP4として有効な構造になっているかチェック・修正
            if len(data) < 8:
                return data
            
            # ftypヘッダーが正しく配置されているかチェック
            if data[4:8] == b'ftyp':
                return data  # 既に正しい構造
            
            # ftyp検索・移動
            ftyp_pos = data.find(b'ftyp')
            if ftyp_pos >= 4:
                # ftypを先頭に移動
                ftyp_size = struct.unpack('>I', data[ftyp_pos-4:ftyp_pos])[0]
                ftyp_block = data[ftyp_pos-4:ftyp_pos-4+ftyp_size]
                remaining = data[:ftyp_pos-4] + data[ftyp_pos-4+ftyp_size:]
                return ftyp_block + remaining
            
            return data
            
        except:
            return data
    
    def validate_mp4_structure(self, data: bytes) -> bool:
        """MP4構造検証"""
        try:
            if len(data) < 8:
                return False
            
            # ftypヘッダー確認
            if data[4:8] != b'ftyp':
                return False
            
            # 基本的なアトム構造確認
            pos = 0
            valid_atoms = 0
            
            while pos < len(data) - 8 and pos < 10000:  # 最初の10KBのみチェック
                size = struct.unpack('>I', data[pos:pos + 4])[0]
                atom_type = data[pos + 4:pos + 8]
                
                # 有効なアトムタイプかチェック
                if atom_type in [b'ftyp', b'mdat', b'moov', b'free', b'skip']:
                    valid_atoms += 1
                
                if size == 0:
                    break
                
                pos += size
                
                if pos > len(data):
                    break
            
            return valid_atoms >= 2  # 最低2つの有効なアトムが必要
            
        except:
            return False
    
    def try_generic_decompress(self, data: bytes) -> bytes:
        """汎用解凍試行"""
        try:
            # 一般的な圧縮形式を順次試行
            algorithms = [
                lzma.decompress,
                bz2.decompress,
                zlib.decompress,
            ]
            
            for decompress_func in algorithms:
                try:
                    return decompress_func(data)
                except:
                    continue
            
            return None
            
        except:
            return None

def run_reversibility_test():
    """完全可逆性テスト実行"""
    print("🔄 NEXUS Video Reversibility Test - 完全可逆性検証")
    print("🎯 91.5%圧縮の完全可逆性を徹底検証")
    print("⚡ バイト完全一致・ハッシュ値・構造検証")
    print("=" * 70)
    
    validator = VideoReversibilityValidator()
    
    # テストファイル
    sample_dir = r"c:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\sample"
    original_file = f"{sample_dir}\\Python基礎講座3_4月26日-3.mp4"
    compressed_file = f"{sample_dir}\\Python基礎講座3_4月26日-3.nxz"
    
    if os.path.exists(original_file) and os.path.exists(compressed_file):
        result = validator.validate_compressed_file(original_file, compressed_file)
        
        # 最終判定
        if result['success']:
            print("\n🎉🎉🎉🎉 完全可逆性検証 - 完全成功!")
            print("✅ 91.5%圧縮が完全可逆であることを確認")
            print("🏆 真の圧縮技術革命達成!")
            print("🌟 理論値突破 + 完全可逆性を両立!")
        else:
            print("\n⚠️ 可逆性に問題があります")
            print("❌ 完全可逆性が確認できませんでした")
            if result['errors']:
                print("🔧 エラー詳細:")
                for error in result['errors']:
                    print(f"   - {error}")
    else:
        print("⚠️ テストファイルが見つかりません")
        print(f"元ファイル: {original_file}")
        print(f"圧縮ファイル: {compressed_file}")

def main():
    """メイン関数"""
    if len(sys.argv) < 2:
        print("🔄 NEXUS Video Reversibility Test")
        print("使用方法:")
        print("  python nexus_reversibility_test.py test                    # 完全可逆性テスト")
        print("  python nexus_reversibility_test.py verify <orig> <comp>    # 指定ファイル検証")
        return
    
    command = sys.argv[1].lower()
    
    if command == "test":
        run_reversibility_test()
    elif command == "verify" and len(sys.argv) >= 4:
        validator = VideoReversibilityValidator()
        original_file = sys.argv[2]
        compressed_file = sys.argv[3]
        result = validator.validate_compressed_file(original_file, compressed_file)
        if not result['success']:
            print("❌ 可逆性検証失敗")
    else:
        print("❌ 無効なコマンドまたは引数です")

if __name__ == "__main__":
    main()
