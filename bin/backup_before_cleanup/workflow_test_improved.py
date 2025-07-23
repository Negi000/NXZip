#!/usr/bin/env python3
"""
NXZip 実用化ワークフローテスト改良版
SPE暗号化 → NXZ圧縮 → 展開 → SPE復号化の完全ワークフロー

改良点:
1. より多様なテキストファイル対応
2. 動画ファイル(MP4)の明確なテスト対象化  
3. 既存ベースエンジンの改良による圧縮率向上
4. 理論値に近づけるアルゴリズム最適化
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

class ImprovedNXZipEngine:
    """改良版NXZipエンジン - 既存ベースから改良"""
    
    def __init__(self):
        self.version = "2.1"
        self.magic_header = b"NXZI21\x00\x00"  # Improved版マジックナンバー
    
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
            # テキストファイル判定の改良
            try:
                text_data = data.decode('utf-8', errors='ignore')
                if len(text_data.strip()) > 0 and text_data.isprintable():
                    return 'text'
            except:
                pass
            return 'binary'
    
    def analyze_text_patterns(self, data):
        """テキストパターン解析 - 既存ベースから改良"""
        try:
            text = data.decode('utf-8', errors='ignore')
            
            # 改良された行パターン解析
            lines = text.split('\n')
            line_patterns = {}
            repetitive_lines = 0
            
            for line in lines:
                stripped_line = line.strip()
                if stripped_line:
                    if stripped_line in line_patterns:
                        line_patterns[stripped_line] += 1
                        repetitive_lines += 1
                    else:
                        line_patterns[stripped_line] = 1
            
            # 文字頻度分析
            char_freq = {}
            for char in text:
                char_freq[char] = char_freq.get(char, 0) + 1
            
            # エントロピー計算
            import math
            entropy = 0
            for freq in char_freq.values():
                p = freq / len(text)
                if p > 0:
                    entropy -= p * math.log2(p)
            
            return {
                'repetitive_ratio': repetitive_lines / len(lines) if lines else 0,
                'unique_lines': len(line_patterns),
                'entropy': entropy,
                'avg_line_length': sum(len(line) for line in lines) / len(lines) if lines else 0,
                'char_diversity': len(char_freq) / 256  # ASCII範囲での文字多様性
            }
            
        except Exception:
            return {'entropy': 8.0, 'repetitive_ratio': 0, 'unique_lines': 0}
    
    def adaptive_compress(self, data, file_type):
        """適応的圧縮 - 既存アルゴリズムから改良"""
        best_compressed = data
        best_method = "none"
        
        if file_type == 'text':
            # テキスト特化改良
            patterns = self.analyze_text_patterns(data)
            
            # 複数アルゴリズムの組み合わせテスト
            methods = [
                ('lzma_extreme', lambda d: lzma.compress(d, format=lzma.FORMAT_XZ, preset=9)),
                ('bz2_high', lambda d: bz2.compress(d, compresslevel=9)),
                ('zlib_max', lambda d: zlib.compress(d, level=9)),
            ]
            
            # 高繰り返し率テキストの特別処理
            if patterns['repetitive_ratio'] > 0.3:
                methods.insert(0, ('lzma_ultra', lambda d: lzma.compress(d, format=lzma.FORMAT_ALONE, preset=9)))
            
        elif file_type == 'wav':
            # WAV音声特化改良
            methods = [
                ('lzma_audio', lambda d: lzma.compress(d, format=lzma.FORMAT_XZ, preset=9)),
                ('custom_rle_lzma', lambda d: self._rle_lzma_compress(d)),
                ('zlib_audio', lambda d: zlib.compress(d, level=9)),
            ]
            
        elif file_type == 'mp4':
            # MP4動画特化改良 - 既存を改良
            methods = [
                ('lzma_video', lambda d: lzma.compress(d, format=lzma.FORMAT_XZ, preset=6)),
                ('bz2_video', lambda d: bz2.compress(d, compresslevel=6)),
                ('multi_stage', lambda d: self._multi_stage_compress(d)),
            ]
            
        elif file_type == 'mp3':
            # MP3音声特化改良
            methods = [
                ('bz2_audio', lambda d: bz2.compress(d, compresslevel=9)),
                ('lzma_audio', lambda d: lzma.compress(d, format=lzma.FORMAT_XZ, preset=8)),
            ]
            
        else:
            # その他ファイル
            methods = [
                ('lzma_general', lambda d: lzma.compress(d, format=lzma.FORMAT_XZ, preset=6)),
                ('zlib_general', lambda d: zlib.compress(d, level=6)),
            ]
        
        # 最適アルゴリズム選択
        for method_name, compress_func in methods:
            try:
                compressed = compress_func(data)
                if len(compressed) < len(best_compressed):
                    best_compressed = compressed
                    best_method = method_name
            except Exception:
                continue
        
        return best_compressed, best_method
    
    def _rle_lzma_compress(self, data):
        """RLE+LZMA改良圧縮"""
        # 単純RLE前処理
        rle_data = bytearray()
        i = 0
        while i < len(data):
            current_byte = data[i]
            count = 1
            while i + count < len(data) and data[i + count] == current_byte and count < 255:
                count += 1
            
            if count >= 4:  # 4回以上の繰り返しでRLE適用
                rle_data.extend([0xFF, current_byte, count])
                i += count
            else:
                rle_data.append(current_byte)
                i += 1
        
        # LZMA圧縮
        return lzma.compress(bytes(rle_data), format=lzma.FORMAT_XZ, preset=9)
    
    def _multi_stage_compress(self, data):
        """多段階圧縮 - 既存から改良"""
        # 段階1: 軽い前処理
        stage1 = zlib.compress(data, level=3)
        
        # 段階2: 高圧縮
        if len(stage1) < len(data) * 0.8:
            stage2 = lzma.compress(stage1, format=lzma.FORMAT_XZ, preset=6)
            return stage2 if len(stage2) < len(stage1) else stage1
        else:
            return lzma.compress(data, format=lzma.FORMAT_XZ, preset=6)

class ImprovedWorkflowTester:
    """改良版ワークフローテスター"""
    
    def __init__(self):
        self.test_results = []
        self.temp_dir = None
        self.engine = ImprovedNXZipEngine()
        
    def setup(self):
        """テスト環境セットアップ"""
        self.temp_dir = tempfile.mkdtemp(prefix="nxzip_improved_")
        print(f"🔧 改良版ワークフローテスト環境: {self.temp_dir}")
    
    def calculate_hash(self, filepath):
        """SHA256ハッシュ計算"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def spe_encrypt(self, input_file, output_file, password="NXZip2025"):
        """SPE暗号化 - 改良版"""
        with open(input_file, 'rb') as f:
            data = f.read()
        
        # パスワードからキー生成
        key = hashlib.sha256(password.encode()).digest()
        
        # ファイルメタデータ
        file_ext = Path(input_file).suffix.encode()
        file_size = len(data).to_bytes(8, 'little')
        
        # 改良された構造保持圧縮
        file_type = self.engine.detect_file_type(data)
        compressed_data, method = self.engine.adaptive_compress(data, file_type)
        
        # SPEヘッダー + メタデータ + 暗号化データ
        spe_header = b"SPE2.1\x00\x00"  # 改良版バージョン
        method_bytes = method.encode()[:16].ljust(16, b'\x00')
        metadata = file_size + len(file_ext).to_bytes(2, 'little') + file_ext + method_bytes
        
        # XOR暗号化
        encrypted_data = bytes(a ^ key[i % len(key)] for i, a in enumerate(compressed_data))
        
        with open(output_file, 'wb') as f:
            f.write(spe_header + metadata + encrypted_data)
        
        return len(compressed_data), method
    
    def spe_decrypt(self, input_file, output_file, password="NXZip2025"):
        """SPE復号化 - 改良版"""
        with open(input_file, 'rb') as f:
            data = f.read()
        
        if not data.startswith(b"SPE2.1\x00\x00"):
            raise ValueError("無効なSPE改良版ファイル")
        
        pos = 8
        file_size = int.from_bytes(data[pos:pos+8], 'little')
        pos += 8
        ext_len = int.from_bytes(data[pos:pos+2], 'little')
        pos += 2
        file_ext = data[pos:pos+ext_len]
        pos += ext_len
        method = data[pos:pos+16].rstrip(b'\x00').decode()
        pos += 16
        encrypted_data = data[pos:]
        
        key = hashlib.sha256(password.encode()).digest()
        compressed_data = bytes(a ^ key[i % len(key)] for i, a in enumerate(encrypted_data))
        
        # 改良版復号化
        decrypted_data = self._decompress_by_method(compressed_data, method)
        
        if len(decrypted_data) != file_size:
            raise ValueError(f"サイズ不一致: {file_size} vs {len(decrypted_data)}")
        
        with open(output_file, 'wb') as f:
            f.write(decrypted_data)
    
    def _decompress_by_method(self, compressed_data, method):
        """方式別復号化"""
        try:
            if 'lzma' in method:
                if 'ultra' in method:
                    return lzma.decompress(compressed_data, format=lzma.FORMAT_ALONE)
                else:
                    return lzma.decompress(compressed_data, format=lzma.FORMAT_XZ)
            elif 'bz2' in method:
                return bz2.decompress(compressed_data)
            elif 'zlib' in method:
                return zlib.decompress(compressed_data)
            elif 'rle_lzma' in method:
                return self._rle_lzma_decompress(compressed_data)
            elif 'multi_stage' in method:
                return self._multi_stage_decompress(compressed_data)
            else:
                return lzma.decompress(compressed_data, format=lzma.FORMAT_XZ)
        except Exception as e:
            # フォールバック
            for decomp_func in [
                lambda d: lzma.decompress(d, format=lzma.FORMAT_XZ),
                lambda d: bz2.decompress(d),
                lambda d: zlib.decompress(d)
            ]:
                try:
                    return decomp_func(compressed_data)
                except:
                    continue
            raise e
    
    def _rle_lzma_decompress(self, compressed_data):
        """RLE+LZMA復号化"""
        lzma_data = lzma.decompress(compressed_data, format=lzma.FORMAT_XZ)
        
        # RLE復号化
        result = bytearray()
        i = 0
        while i < len(lzma_data):
            if i + 2 < len(lzma_data) and lzma_data[i] == 0xFF:
                byte_val = lzma_data[i + 1]
                count = lzma_data[i + 2]
                result.extend([byte_val] * count)
                i += 3
            else:
                result.append(lzma_data[i])
                i += 1
        
        return bytes(result)
    
    def _multi_stage_decompress(self, compressed_data):
        """多段階復号化"""
        try:
            # LZMA復号化試行
            stage1 = lzma.decompress(compressed_data, format=lzma.FORMAT_XZ)
            # zlib復号化試行
            return zlib.decompress(stage1)
        except:
            # 直接LZMA復号化
            return lzma.decompress(compressed_data, format=lzma.FORMAT_XZ)
    
    def compress_to_nxz(self, input_file, output_file):
        """NXZ圧縮 - 改良版"""
        with open(input_file, 'rb') as f:
            data = f.read()
        
        # 改良版NXZヘッダー + メタデータ + LZMA圧縮
        nxz_header = self.engine.magic_header
        original_size = len(data).to_bytes(8, 'little')
        timestamp = int(time.time()).to_bytes(8, 'little')
        checksum = hashlib.sha256(data).digest()[:16]
        
        # 改良された圧縮
        file_type = self.engine.detect_file_type(data)
        compressed_data, method = self.engine.adaptive_compress(data, file_type)
        method_bytes = method.encode()[:16].ljust(16, b'\x00')
        
        with open(output_file, 'wb') as f:
            f.write(nxz_header + original_size + timestamp + checksum + method_bytes + compressed_data)
    
    def extract_from_nxz(self, input_file, output_file):
        """NXZ展開 - 改良版"""
        with open(input_file, 'rb') as f:
            data = f.read()
        
        if not data.startswith(self.engine.magic_header):
            raise ValueError("無効なNXZ改良版ファイル")
        
        pos = 8
        original_size = int.from_bytes(data[pos:pos+8], 'little')
        pos += 8
        timestamp = int.from_bytes(data[pos:pos+8], 'little')
        pos += 8
        expected_checksum = data[pos:pos+16]
        pos += 16
        method = data[pos:pos+16].rstrip(b'\x00').decode()
        pos += 16
        compressed_data = data[pos:]
        
        decompressed_data = self._decompress_by_method(compressed_data, method)
        
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
        
        with open(test_file, 'rb') as f:
            data = f.read()
        file_type = self.engine.detect_file_type(data)
        
        print(f"📁 元ファイル: {original_size:,} bytes")
        print(f"🔍 ファイル種別: {file_type}")
        print(f"🔐 元Hash: {original_hash[:32]}...")
        
        try:
            base_name = os.path.basename(test_file)
            
            # Phase 1: 改良版SPE暗号化
            print(f"\n🔐 Phase 1: 改良版SPE暗号化")
            encrypted_file = os.path.join(self.temp_dir, f"{base_name}.spe")
            start_time = time.time()
            spe_compressed_size, spe_method = self.spe_encrypt(test_file, encrypted_file)
            spe_time = time.time() - start_time
            spe_size = os.path.getsize(encrypted_file)
            spe_ratio = ((original_size - spe_compressed_size) / original_size) * 100
            print(f"   ✅ 暗号化完了: {spe_size:,} bytes, 内部圧縮率: {spe_ratio:.1f}% ({spe_method}) ({spe_time:.2f}秒)")
            
            # Phase 2: 改良版NXZ圧縮
            print(f"\n📦 Phase 2: 改良版NXZ圧縮")
            nxz_file = os.path.join(self.temp_dir, f"{base_name}.nxz")
            start_time = time.time()
            self.compress_to_nxz(encrypted_file, nxz_file)
            compress_time = time.time() - start_time
            nxz_size = os.path.getsize(nxz_file)
            compression_ratio = ((original_size - nxz_size) / original_size) * 100
            print(f"   ✅ 圧縮完了: {nxz_size:,} bytes, 総合圧縮率: {compression_ratio:.1f}% ({compress_time:.2f}秒)")
            
            # Phase 3: 改良版NXZ展開
            print(f"\n📂 Phase 3: 改良版NXZ展開")
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
        print("既存ベースエンジンからの改良版 - 理論値への接近を目指す")
        print("SPE改良版暗号化 → NXZ改良版圧縮 → NXZ展開 → SPE復号化")
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
        
        print("📋 SPE改良版+NXZ改良版完全ワークフロー結果:")
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
                    print(f"   {file_type:>8}: 平均 {avg_ratio:.1f}%, 最大 {max_ratio:.1f}%")
        
        # 理論値との比較
        print(f"\n🎯 理論値比較:")
        theoretical_targets = {
            'text': 95.0,
            'mp3': 85.0,
            'mp4': 75.0,
            'wav': 90.0,
            'jpeg': 85.0,
            'png': 80.0
        }
        
        for result in self.test_results:
            if 'error' not in result:
                file_type = result.get('file_type', 'unknown')
                achieved = result['compression_ratio']
                target = theoretical_targets.get(file_type, 50.0)
                progress = (achieved / target) * 100 if target > 0 else 0
                status_icon = "✅" if progress >= 80 else "⚠️" if progress >= 50 else "❌"
                print(f"   {result['file'][:20]:>20}: {achieved:>5.1f}% / {target:>5.1f}% ({progress:>5.1f}%) {status_icon}")
        
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
    
    def cleanup(self):
        """クリーンアップ"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"\n🧹 クリーンアップ完了: {self.temp_dir}")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        tester = ImprovedWorkflowTester()
        try:
            tester.setup()
            tester.run_tests()
        finally:
            tester.cleanup()
    else:
        print("使用方法: python workflow_test_improved.py test")
        print("改良点:")
        print("  1. より多様なテキストファイルでのテスト")
        print("  2. MP4動画ファイルの明確なテスト対象化")
        print("  3. 既存エンジンベースの適応的圧縮アルゴリズム改良")
        print("  4. 理論値接近のためのファイル種別特化最適化")

if __name__ == "__main__":
    main()
