#!/usr/bin/env python3
"""
NXZip 実用化ワークフローテスト改良版v2
SPE暗号化 → NXZ圧縮 → 展開 → SPE復号化の完全ワークフロー

改良点:
1. 既存の成功したworkflow_test.pyベースの維持
2. 圧縮アルゴリズムの部分的改良のみ（互換性維持）
3. より多様なテストファイル対応
4. 動画ファイル(MP4)の明確なテスト対象化
"""

import os
import sys
import hashlib
import time
from pathlib import Path
import tempfile
import shutil
import lzma
import zlib
import bz2

class EnhancedNXZipWorkflowTester:
    """改良版NXZipワークフローテストクラス - 既存ベース維持"""
    
    def __init__(self):
        self.test_results = []
        self.temp_dir = None
        
    def setup(self):
        """テスト環境セットアップ"""
        self.temp_dir = tempfile.mkdtemp(prefix="nxzip_enhanced_")
        print(f"🔧 改良版ワークフローテスト環境: {self.temp_dir}")
    
    def calculate_hash(self, filepath):
        """SHA256ハッシュ計算"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def detect_file_type(self, data):
        """ファイル種別判定"""
        if data.startswith(b'\xFF\xD8\xFF'):
            return 'jpeg'
        elif data.startswith(b'\x89PNG'):
            return 'png'
        elif data.startswith(b'RIFF') and b'WAVE' in data[:20]:
            return 'wav'
        elif data.startswith((b'\x00\x00\x00\x14ftypmp4', b'\x00\x00\x00\x18ftypmp4', b'\x00\x00\x00\x1Cftypmp4')):
            return 'mp4'
        elif data.startswith(b'ID3') or data.startswith(b'\xFF\xFB'):
            return 'mp3'
        else:
            # テキストファイル判定
            try:
                text_data = data.decode('utf-8', errors='ignore')
                if len(text_data.strip()) > 0:
                    # 日本語テキストかASCIIテキストかを判定
                    ascii_ratio = sum(1 for c in text_data if ord(c) < 128) / len(text_data)
                    if ascii_ratio > 0.7:  # 70%以上がASCII
                        return 'text'
                    else:
                        return 'japanese_text'
            except:
                pass
            return 'binary'
    
    def enhanced_compress(self, data, file_type):
        """改良圧縮 - 既存zlib基準を維持しつつ最適化"""
        best_compressed = zlib.compress(data, level=6)  # 既存デフォルト
        best_method = "zlib_6"
        
        # ファイル種別に応じた改良試行
        if file_type in ['text', 'japanese_text']:
            # テキスト改良: より高い圧縮レベル試行
            methods = [
                ('zlib_9', lambda d: zlib.compress(d, level=9)),
                ('bz2_9', lambda d: bz2.compress(d, compresslevel=9)),
            ]
            
        elif file_type == 'wav':
            # WAV改良: より効率的な圧縮
            methods = [
                ('zlib_9', lambda d: zlib.compress(d, level=9)),
                ('bz2_6', lambda d: bz2.compress(d, compresslevel=6)),
            ]
            
        elif file_type == 'mp4':
            # MP4改良: 動画に特化した試行
            methods = [
                ('zlib_9', lambda d: zlib.compress(d, level=9)),
                ('bz2_3', lambda d: bz2.compress(d, compresslevel=3)),
            ]
            
        elif file_type == 'mp3':
            # MP3改良: 音声に特化
            methods = [
                ('bz2_9', lambda d: bz2.compress(d, compresslevel=9)),
                ('zlib_9', lambda d: zlib.compress(d, level=9)),
            ]
            
        else:
            # その他ファイル: 軽い改良
            methods = [
                ('zlib_9', lambda d: zlib.compress(d, level=9)),
            ]
        
        # 最適アルゴリズム選択（失敗時は既存デフォルトを使用）
        for method_name, compress_func in methods:
            try:
                compressed = compress_func(data)
                if len(compressed) < len(best_compressed):
                    best_compressed = compressed
                    best_method = method_name
            except Exception:
                continue
        
        return best_compressed, best_method
    
    def enhanced_decompress(self, compressed_data, method):
        """改良復号化"""
        if method.startswith('bz2'):
            return bz2.decompress(compressed_data)
        else:  # zlib系 (デフォルト)
            return zlib.decompress(compressed_data)
    
    def spe_encrypt(self, input_file, output_file, password="NXZip2025"):
        """SPE暗号化 - 改良版（互換性維持）"""
        with open(input_file, 'rb') as f:
            data = f.read()
        
        # ファイル種別判定
        file_type = self.detect_file_type(data)
        
        # パスワードからキー生成
        key = hashlib.sha256(password.encode()).digest()
        
        # ファイルメタデータ
        file_ext = Path(input_file).suffix.encode()
        file_size = len(data).to_bytes(8, 'little')
        
        # 改良された構造保持圧縮
        compressed_data, method = self.enhanced_compress(data, file_type)
        
        # SPEヘッダー + メタデータ + 暗号化データ (互換性維持)
        spe_header = b"SPE2.0\x00\x00"  # 既存フォーマット維持
        method_bytes = method.encode()[:16].ljust(16, b'\x00')
        metadata = file_size + len(file_ext).to_bytes(2, 'little') + file_ext + method_bytes
        
        # XOR暗号化
        encrypted_data = bytes(a ^ key[i % len(key)] for i, a in enumerate(compressed_data))
        
        with open(output_file, 'wb') as f:
            f.write(spe_header + metadata + encrypted_data)
        
        return len(compressed_data), method, file_type
    
    def spe_decrypt(self, input_file, output_file, password="NXZip2025"):
        """SPE復号化 - 改良版（互換性維持）"""
        with open(input_file, 'rb') as f:
            data = f.read()
        
        if not data.startswith(b"SPE2.0\x00\x00"):
            raise ValueError("無効なSPEファイル")
        
        pos = 8
        file_size = int.from_bytes(data[pos:pos+8], 'little')
        pos += 8
        ext_len = int.from_bytes(data[pos:pos+2], 'little')
        pos += 2
        file_ext = data[pos:pos+ext_len]
        pos += ext_len
        
        # 改良版: メソッド情報の読み取り
        if pos + 16 <= len(data):
            method = data[pos:pos+16].rstrip(b'\x00').decode()
            pos += 16
        else:
            method = "zlib_6"  # 既存デフォルト
        
        encrypted_data = data[pos:]
        
        key = hashlib.sha256(password.encode()).digest()
        compressed_data = bytes(a ^ key[i % len(key)] for i, a in enumerate(encrypted_data))
        
        # 改良版復号化（フォールバック付き）
        try:
            if method.startswith('bz2'):
                decrypted_data = bz2.decompress(compressed_data)
            else:
                decrypted_data = zlib.decompress(compressed_data)
        except Exception:
            # フォールバック: 既存方式
            decrypted_data = zlib.decompress(compressed_data)
        
        if len(decrypted_data) != file_size:
            raise ValueError(f"サイズ不一致: {file_size} vs {len(decrypted_data)}")
        
        with open(output_file, 'wb') as f:
            f.write(decrypted_data)
    
    def compress_to_nxz(self, input_file, output_file):
        """NXZ圧縮 - 既存互換性維持"""
        with open(input_file, 'rb') as f:
            data = f.read()
        
        # NXZヘッダー + メタデータ + LZMA圧縮 (既存維持)
        nxz_header = b"NXZ2.0\x00\x00"
        original_size = len(data).to_bytes(8, 'little')
        timestamp = int(time.time()).to_bytes(8, 'little')
        checksum = hashlib.sha256(data).digest()[:16]
        compressed_data = lzma.compress(data, format=lzma.FORMAT_XZ, preset=6)
        
        with open(output_file, 'wb') as f:
            f.write(nxz_header + original_size + timestamp + checksum + compressed_data)
    
    def extract_from_nxz(self, input_file, output_file):
        """NXZ展開 - 既存互換性維持"""
        with open(input_file, 'rb') as f:
            data = f.read()
        
        if not data.startswith(b"NXZ2.0\x00\x00"):
            raise ValueError("無効なNXZファイル")
        
        pos = 8
        original_size = int.from_bytes(data[pos:pos+8], 'little')
        pos += 8
        timestamp = int.from_bytes(data[pos:pos+8], 'little')
        pos += 8
        expected_checksum = data[pos:pos+16]
        pos += 16
        compressed_data = data[pos:]
        
        decompressed_data = lzma.decompress(compressed_data, format=lzma.FORMAT_XZ)
        
        if len(decompressed_data) != original_size:
            raise ValueError(f"サイズ不一致: {original_size} vs {len(decompressed_data)}")
        
        actual_checksum = hashlib.sha256(decompressed_data).digest()[:16]
        if actual_checksum != expected_checksum:
            raise ValueError("チェックサム不一致")
        
        with open(output_file, 'wb') as f:
            f.write(decompressed_data)
    
    def test_workflow(self, test_file):
        """改良版完全ワークフローテスト"""
        print(f"\n{'='*70}")
        print(f"🚀 改良版完全ワークフローテスト: {os.path.basename(test_file)}")
        print(f"{'='*70}")
        
        original_size = os.path.getsize(test_file)
        original_hash = self.calculate_hash(test_file)
        
        # ファイル種別判定
        with open(test_file, 'rb') as f:
            data = f.read()
        file_type = self.detect_file_type(data)
        
        print(f"📁 元ファイル: {original_size:,} bytes")
        print(f"🔍 ファイル種別: {file_type}")
        print(f"🔐 元Hash: {original_hash[:32]}...")
        
        try:
            base_name = os.path.basename(test_file)
            
            # Phase 1: 改良版SPE暗号化
            print(f"\n🔐 Phase 1: 改良版SPE暗号化")
            encrypted_file = os.path.join(self.temp_dir, f"{base_name}.spe")
            start_time = time.time()
            spe_compressed_size, spe_method, detected_type = self.spe_encrypt(test_file, encrypted_file)
            spe_time = time.time() - start_time
            spe_size = os.path.getsize(encrypted_file)
            spe_ratio = ((original_size - spe_compressed_size) / original_size) * 100
            print(f"   ✅ 暗号化完了: {spe_size:,} bytes, 内部圧縮率: {spe_ratio:.1f}% ({spe_method}) ({spe_time:.2f}秒)")
            
            # Phase 2: NXZ圧縮
            print(f"\n📦 Phase 2: NXZ圧縮")
            nxz_file = os.path.join(self.temp_dir, f"{base_name}.nxz")
            start_time = time.time()
            self.compress_to_nxz(encrypted_file, nxz_file)
            compress_time = time.time() - start_time
            nxz_size = os.path.getsize(nxz_file)
            compression_ratio = ((original_size - nxz_size) / original_size) * 100
            print(f"   ✅ 圧縮完了: {nxz_size:,} bytes, 総合圧縮率: {compression_ratio:.1f}% ({compress_time:.2f}秒)")
            
            # Phase 3: NXZ展開
            print(f"\n📂 Phase 3: NXZ展開")
            extracted_spe = os.path.join(self.temp_dir, f"{base_name}_extracted.spe")
            start_time = time.time()
            self.extract_from_nxz(nxz_file, extracted_spe)
            extract_time = time.time() - start_time
            print(f"   ✅ 展開完了: ({extract_time:.2f}秒)")
            
            # Phase 4: 改良版SPE復号化
            print(f"\n🔓 Phase 4: 改良版SPE復号化")
            restored_file = os.path.join(self.temp_dir, f"{base_name}_restored")
            start_time = time.time()
            self.spe_decrypt(extracted_spe, restored_file)
            decrypt_time = time.time() - start_time
            print(f"   ✅ 復号化完了: ({decrypt_time:.2f}秒)")
            
            # Phase 5: 完全性検証
            print(f"\n✅ Phase 5: 完全性検証")
            restored_hash = self.calculate_hash(restored_file)
            restored_size = os.path.getsize(restored_file)
            
            size_match = (original_size == restored_size)
            hash_match = (original_hash == restored_hash)
            is_reversible = size_match and hash_match
            
            print(f"   サイズ一致: {'✅' if size_match else '❌'} ({original_size:,} vs {restored_size:,})")
            print(f"   Hash一致: {'✅' if hash_match else '❌'}")
            print(f"   復元Hash: {restored_hash[:32]}...")
            
            total_time = spe_time + compress_time + extract_time + decrypt_time
            
            result = {
                "file": os.path.basename(test_file),
                "file_type": file_type,
                "original_size": original_size,
                "nxz_size": nxz_size,
                "compression_ratio": compression_ratio,
                "spe_method": spe_method,
                "total_time": total_time,
                "reversible": is_reversible,
                "size_match": size_match,
                "hash_match": hash_match,
                "status": "✅" if is_reversible else "❌"
            }
            
            if is_reversible:
                print(f"\n🎉 結果: 完全可逆性確認！(圧縮率: {compression_ratio:.1f}%)")
                if compression_ratio > 80:
                    print(f"🔥 優秀な圧縮率達成！")
                elif compression_ratio > 50:
                    print(f"✨ 良好な圧縮率達成！")
            else:
                print(f"\n❌ 結果: 可逆性問題発生")
            
            return result
            
        except Exception as e:
            print(f"\n❌ ワークフローエラー: {e}")
            return {
                "file": os.path.basename(test_file),
                "status": "❌",
                "error": str(e),
                "reversible": False
            }
    
    def run_tests(self):
        """改良版全ファイル形式テスト"""
        print("🚀 NXZip改良版実用化ワークフローテスト開始")
        print("既存成功ベースの部分的改良版 - 互換性維持しつつ性能向上")
        print("SPE改良版暗号化 → NXZ圧縮 → NXZ展開 → SPE復号化")
        print("=" * 80)
        
        # より多様なテストファイル - 動画とテキストを重視
        test_files = [
            # 多様なテキストファイル
            r"c:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\test-data\sample_text.txt",
            r"c:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\test-data\large_test.txt", 
            r"c:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\test-data\repetitive_test.txt",
            r"c:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\sample\出庫実績明細_202412.txt",
            
            # 動画ファイル (明確にテスト対象)
            r"c:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\sample\Python基礎講座3_4月26日-3.mp4",
            
            # 音声ファイル
            r"c:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\sample\陰謀論.mp3",
            r"c:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\sample\generated-music-1752042054079.wav",
            
            # 画像ファイル
            r"c:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\sample\COT-001.jpg",
            r"c:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\sample\COT-012.png"
        ]
        
        existing_files = [f for f in test_files if os.path.exists(f)]
        print(f"📁 テスト対象ファイル数: {len(existing_files)}")
        
        for test_file in existing_files:
            result = self.test_workflow(test_file)
            self.test_results.append(result)
        
        self.print_summary()
    
    def print_summary(self):
        """改良版結果サマリー"""
        print("\n" + "=" * 80)
        print("🏆 NXZip改良版実用化ワークフローテスト結果")
        print("=" * 80)
        
        successful = sum(1 for r in self.test_results if r.get('reversible', False))
        total = len(self.test_results)
        success_rate = (successful / total * 100) if total > 0 else 0
        
        print(f"🎯 完全可逆性成功率: {successful}/{total} ({success_rate:.1f}%)")
        print()
        
        print("📋 SPE改良版+NXZ完全ワークフロー結果:")
        print("┌─────────────────────┬──────────┬──────────┬──────┬────────┬──────────┬──────┐")
        print("│ ファイル            │ 元サイズ │ NXZサイズ│圧縮率│処理時間│ 圧縮手法 │可逆性│")
        print("├─────────────────────┼──────────┼──────────┼──────┼────────┼──────────┼──────┤")
        
        for result in self.test_results:
            if 'error' in result:
                name = result['file'][:19]
                print(f"│ {name:<19} │   ERROR  │   ERROR  │ ERR  │  ERR   │   ERROR  │  ❌  │")
            else:
                name = result['file'][:19]
                orig = f"{result['original_size']:,}"[:8]
                nxz = f"{result['nxz_size']:,}"[:8]
                ratio = f"{result['compression_ratio']:.1f}%"[:5]
                time_s = f"{result['total_time']:.1f}s"[:6]
                method = result.get('spe_method', 'unknown')[:8]
                status = result['status']
                
                print(f"│ {name:<19} │{orig:>8} │{nxz:>8} │{ratio:>5} │{time_s:>6} │{method:>8} │  {status}  │")
        
        print("└─────────────────────┴──────────┴──────────┴──────┴────────┴──────────┴──────┘")
        
        if self.test_results:
            valid_results = [r for r in self.test_results if 'error' not in r]
            if valid_results:
                avg_compression = sum(r['compression_ratio'] for r in valid_results) / len(valid_results)
                avg_time = sum(r['total_time'] for r in valid_results) / len(valid_results)
                
                # ファイル種別別統計
                type_stats = {}
                for result in valid_results:
                    file_type = result.get('file_type', 'unknown')
                    if file_type not in type_stats:
                        type_stats[file_type] = []
                    type_stats[file_type].append(result['compression_ratio'])
                
                print(f"\n📊 改良版統計情報:")
                print(f"   平均圧縮率: {avg_compression:.1f}%")
                print(f"   平均処理時間: {avg_time:.1f}秒")
                
                print(f"\n📈 ファイル種別別圧縮率:")
                for file_type, ratios in type_stats.items():
                    avg_ratio = sum(ratios) / len(ratios)
                    max_ratio = max(ratios)
                    print(f"   {file_type:>12}: 平均 {avg_ratio:.1f}%, 最大 {max_ratio:.1f}%")
        
        # 理論値との比較
        print(f"\n🎯 理論値比較分析:")
        theoretical_targets = {
            'text': 95.0, 'japanese_text': 95.0,
            'mp3': 85.0, 'mp4': 75.0, 'wav': 90.0,
            'jpeg': 85.0, 'png': 80.0, 'binary': 50.0
        }
        
        for result in self.test_results:
            if 'error' not in result:
                file_type = result.get('file_type', 'unknown')
                achieved = result['compression_ratio']
                target = theoretical_targets.get(file_type, 50.0)
                progress = (achieved / target) * 100 if target > 0 else 0
                status_icon = "✅" if progress >= 80 else "⚠️" if progress >= 50 else "❌"
                print(f"   {result['file'][:25]:>25}: {achieved:>5.1f}% / {target:>5.1f}% ({progress:>5.1f}%) {status_icon}")
        
        failed_files = [r for r in self.test_results if not r.get('reversible', False)]
        if failed_files:
            print(f"\n⚠️ 可逆性に問題があるファイル:")
            for result in failed_files:
                if 'error' in result:
                    print(f"   ❌ {result['file']}: {result['error']}")
                else:
                    issues = []
                    if not result.get('size_match', True):
                        issues.append("サイズ不一致")
                    if not result.get('hash_match', True):
                        issues.append("Hash不一致")
                    print(f"   ❌ {result['file']}: {', '.join(issues)}")
        
        print()
        print("🎯 改良版実用化評価:")
        if success_rate >= 95:
            print("✅ 商用レベル: 改良により実用化達成")
        elif success_rate >= 80:
            print("⚠️ 準実用レベル: 改良により大幅向上、微調整で完成")
        else:
            print("❌ 改良継続: 更なるアルゴリズム改良が必要")
        
        # 改良効果まとめ
        if valid_results:
            high_compression = [r for r in valid_results if r['compression_ratio'] > 70]
            if high_compression:
                print(f"\n🔥 改良による高圧縮率達成:")
                for result in high_compression:
                    print(f"   🎖️ {result['file']}: {result['compression_ratio']:.1f}% ({result['file_type']})")
            
            # 動画・テキストの特別評価
            video_results = [r for r in valid_results if r.get('file_type') == 'mp4']
            text_results = [r for r in valid_results if r.get('file_type') in ['text', 'japanese_text']]
            
            if video_results:
                print(f"\n🎬 動画ファイル改良効果:")
                for result in video_results:
                    print(f"   📹 {result['file']}: {result['compression_ratio']:.1f}% (目標75%)")
            
            if text_results:
                print(f"\n📝 テキストファイル改良効果:")
                for result in text_results:
                    print(f"   📄 {result['file']}: {result['compression_ratio']:.1f}% (目標95%)")
    
    def cleanup(self):
        """クリーンアップ"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"\n🧹 クリーンアップ完了: {self.temp_dir}")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        tester = EnhancedNXZipWorkflowTester()
        try:
            tester.setup()
            tester.run_tests()
        finally:
            tester.cleanup()
    else:
        print("使用方法: python workflow_test_enhanced.py test")
        print("改良点:")
        print("  1. 既存成功コードベースの維持（互換性確保）")
        print("  2. 圧縮アルゴリズムの部分的改良（zlib基準 + bz2追加）")
        print("  3. より多様なテキストファイルでのテスト")
        print("  4. MP4動画ファイルの明確なテスト対象化")
        print("  5. ファイル種別別の最適化戦略")

if __name__ == "__main__":
    main()
