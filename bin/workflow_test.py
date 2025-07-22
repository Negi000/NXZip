#!/usr/bin/env python3
"""
NXZip 実用化ワークフローテスト
SPE暗号化 → NXZ圧縮 → 展開 → SPE復号化の完全ワークフロー
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

class NXZipWorkflowTester:
    """NXZip実用化ワークフローテストクラス"""
    
    def __init__(self):
        self.test_results = []
        self.temp_dir = None
        
    def setup(self):
        """テスト環境セットアップ"""
        self.temp_dir = tempfile.mkdtemp(prefix="nxzip_workflow_")
        print(f"🔧 ワークフローテスト環境: {self.temp_dir}")
    
    def calculate_hash(self, filepath):
        """SHA256ハッシュ計算"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def spe_encrypt(self, input_file, output_file, password="NXZip2025"):
        """SPE暗号化"""
        with open(input_file, 'rb') as f:
            data = f.read()
        
        # パスワードからキー生成
        key = hashlib.sha256(password.encode()).digest()
        
        # ファイルメタデータ
        file_ext = Path(input_file).suffix.encode()
        file_size = len(data).to_bytes(8, 'little')
        
        # 構造保持圧縮
        compressed_data = zlib.compress(data, level=6)
        
        # SPEヘッダー + メタデータ + 暗号化データ
        spe_header = b"SPE2.0\x00\x00"
        metadata = file_size + len(file_ext).to_bytes(2, 'little') + file_ext
        
        # XOR暗号化
        encrypted_data = bytes(a ^ key[i % len(key)] for i, a in enumerate(compressed_data))
        
        with open(output_file, 'wb') as f:
            f.write(spe_header + metadata + encrypted_data)
    
    def spe_decrypt(self, input_file, output_file, password="NXZip2025"):
        """SPE復号化"""
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
        encrypted_data = data[pos:]
        
        key = hashlib.sha256(password.encode()).digest()
        compressed_data = bytes(a ^ key[i % len(key)] for i, a in enumerate(encrypted_data))
        decrypted_data = zlib.decompress(compressed_data)
        
        if len(decrypted_data) != file_size:
            raise ValueError(f"サイズ不一致: {file_size} vs {len(decrypted_data)}")
        
        with open(output_file, 'wb') as f:
            f.write(decrypted_data)
    
    def compress_to_nxz(self, input_file, output_file):
        """NXZ圧縮"""
        with open(input_file, 'rb') as f:
            data = f.read()
        
        # NXZヘッダー + メタデータ + LZMA圧縮
        nxz_header = b"NXZ2.0\x00\x00"
        original_size = len(data).to_bytes(8, 'little')
        timestamp = int(time.time()).to_bytes(8, 'little')
        checksum = hashlib.sha256(data).digest()[:16]
        compressed_data = lzma.compress(data, format=lzma.FORMAT_XZ, preset=6)
        
        with open(output_file, 'wb') as f:
            f.write(nxz_header + original_size + timestamp + checksum + compressed_data)
    
    def extract_from_nxz(self, input_file, output_file):
        """NXZ展開"""
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
        """完全ワークフローテスト"""
        print(f"\n{'='*60}")
        print(f"🔬 完全ワークフローテスト: {os.path.basename(test_file)}")
        print(f"{'='*60}")
        
        original_size = os.path.getsize(test_file)
        original_hash = self.calculate_hash(test_file)
        
        print(f"📁 元ファイル: {original_size:,} bytes")
        print(f"🔐 元Hash: {original_hash[:32]}...")
        
        try:
            base_name = os.path.basename(test_file)
            
            # Phase 1: SPE暗号化
            print(f"\n🔐 Phase 1: SPE暗号化")
            encrypted_file = os.path.join(self.temp_dir, f"{base_name}.spe")
            start_time = time.time()
            self.spe_encrypt(test_file, encrypted_file)
            spe_time = time.time() - start_time
            spe_size = os.path.getsize(encrypted_file)
            print(f"   ✅ 暗号化完了: {spe_size:,} bytes ({spe_time:.2f}秒)")
            
            # Phase 2: NXZ圧縮
            print(f"\n📦 Phase 2: NXZ圧縮")
            nxz_file = os.path.join(self.temp_dir, f"{base_name}.nxz")
            start_time = time.time()
            self.compress_to_nxz(encrypted_file, nxz_file)
            compress_time = time.time() - start_time
            nxz_size = os.path.getsize(nxz_file)
            compression_ratio = ((original_size - nxz_size) / original_size) * 100
            print(f"   ✅ 圧縮完了: {nxz_size:,} bytes, 圧縮率: {compression_ratio:.1f}% ({compress_time:.2f}秒)")
            
            # Phase 3: NXZ展開
            print(f"\n📂 Phase 3: NXZ展開")
            extracted_spe = os.path.join(self.temp_dir, f"{base_name}_extracted.spe")
            start_time = time.time()
            self.extract_from_nxz(nxz_file, extracted_spe)
            extract_time = time.time() - start_time
            print(f"   ✅ 展開完了: ({extract_time:.2f}秒)")
            
            # Phase 4: SPE復号化
            print(f"\n🔓 Phase 4: SPE復号化")
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
                "original_size": original_size,
                "nxz_size": nxz_size,
                "compression_ratio": compression_ratio,
                "total_time": total_time,
                "reversible": is_reversible,
                "size_match": size_match,
                "hash_match": hash_match,
                "status": "✅" if is_reversible else "❌"
            }
            
            if is_reversible:
                print(f"\n🎉 結果: 完全可逆性確認！")
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
        """全ファイル形式テスト"""
        print("🚀 NXZip実用化ワークフローテスト開始")
        print("SPE暗号化 → NXZ圧縮 → NXZ展開 → SPE復号化")
        print("=" * 80)
        
        test_files = [
            r"c:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\test-data\sample_text.txt",
            r"c:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\sample\陰謀論.mp3",
            r"c:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\sample\COT-001.jpg",
            r"c:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\sample\COT-012.png",
            r"c:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\sample\generated-music-1752042054079.wav"
        ]
        
        existing_files = [f for f in test_files if os.path.exists(f)]
        
        for test_file in existing_files:
            result = self.test_workflow(test_file)
            self.test_results.append(result)
        
        self.print_summary()
    
    def print_summary(self):
        """結果サマリー"""
        print("\n" + "=" * 80)
        print("🏆 NXZip実用化ワークフローテスト結果")
        print("=" * 80)
        
        successful = sum(1 for r in self.test_results if r.get('reversible', False))
        total = len(self.test_results)
        success_rate = (successful / total * 100) if total > 0 else 0
        
        print(f"🎯 完全可逆性成功率: {successful}/{total} ({success_rate:.1f}%)")
        print()
        
        print("📋 SPE+NXZ完全ワークフロー結果:")
        print("┌─────────────────────┬──────────┬──────────┬──────┬────────┬──────┐")
        print("│ ファイル            │ 元サイズ │ NXZサイズ│圧縮率│処理時間│可逆性│")
        print("├─────────────────────┼──────────┼──────────┼──────┼────────┼──────┤")
        
        for result in self.test_results:
            if 'error' in result:
                name = result['file'][:19]
                print(f"│ {name:<19} │   ERROR  │   ERROR  │ ERR  │  ERR   │  ❌  │")
            else:
                name = result['file'][:19]
                orig = f"{result['original_size']:,}"[:8]
                nxz = f"{result['nxz_size']:,}"[:8]
                ratio = f"{result['compression_ratio']:.1f}%"[:5]
                time_s = f"{result['total_time']:.1f}s"[:6]
                status = result['status']
                
                print(f"│ {name:<19} │{orig:>8} │{nxz:>8} │{ratio:>5} │{time_s:>6} │  {status}  │")
        
        print("└─────────────────────┴──────────┴──────────┴──────┴────────┴──────┘")
        
        if self.test_results:
            valid_results = [r for r in self.test_results if 'error' not in r]
            if valid_results:
                avg_compression = sum(r['compression_ratio'] for r in valid_results) / len(valid_results)
                avg_time = sum(r['total_time'] for r in valid_results) / len(valid_results)
                
                print(f"\n📊 統計:")
                print(f"   平均圧縮率: {avg_compression:.1f}%")
                print(f"   平均処理時間: {avg_time:.1f}秒")
        
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
        print("🎯 実用化評価:")
        if success_rate >= 95:
            print("✅ 実用レベル: 商用展開可能")
        elif success_rate >= 80:
            print("⚠️ 準実用レベル: 一部改善で実用化可能")
        else:
            print("❌ 開発継続: 重要な可逆性問題の解決が必要")
    
    def cleanup(self):
        """クリーンアップ"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"\n🧹 クリーンアップ完了: {self.temp_dir}")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        tester = NXZipWorkflowTester()
        try:
            tester.setup()
            tester.run_tests()
        finally:
            tester.cleanup()
    else:
        print("使用方法: python workflow_test.py test")

if __name__ == "__main__":
    main()
