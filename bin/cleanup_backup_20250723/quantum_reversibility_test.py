#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 nexus_quantum.py 専用可逆性テスト
量子圧縮エンジンの完全可逆性を検証

🎯 テスト対象:
- NXQNT_PNG_V1 形式の解凍
- NXQNT_JPEG_V1 形式の解凍
- SHA256ハッシュ照合による完全一致確認
"""

import os
import sys
import json
import time
import hashlib
import subprocess
import struct
import lzma
from pathlib import Path
from datetime import datetime

class QuantumReversibilityTester:
    """量子圧縮専用可逆性テスター"""
    
    def __init__(self):
        self.test_results = {}
        
    def get_file_hash(self, file_path):
        """ファイルのSHA256ハッシュを計算"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
        
    def compress_with_quantum_engine(self, input_file):
        """nexus_quantum.pyで圧縮実行"""
        try:
            cmd = ['python', 'bin/nexus_quantum.py', str(input_file)]
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  encoding='utf-8', errors='replace')
            
            if result.returncode == 0:
                # 成功時、.nxzファイルパスを推定
                base_name = Path(input_file).stem
                nxz_file = Path(input_file).parent / f"{base_name}.nxz"
                if nxz_file.exists():
                    return True, str(nxz_file)
                else:
                    return False, "NXZファイルが生成されませんでした"
            else:
                return False, result.stderr.strip()
                
        except Exception as e:
            return False, str(e)
            
    def decompress_quantum_nxz(self, nxz_file):
        """量子圧縮NXZファイルの解凍"""
        try:
            with open(nxz_file, 'rb') as f:
                compressed_data = f.read()
            
            # 量子圧縮ヘッダー確認
            if compressed_data.startswith(b'NXQNT_PNG_V1'):
                print(f"   🔍 PNG量子圧縮形式検出")
                header_size = len(b'NXQNT_PNG_V1')
            elif compressed_data.startswith(b'NXQNT_JPEG_V1'):
                print(f"   🔍 JPEG量子圧縮形式検出")
                header_size = len(b'NXQNT_JPEG_V1')
            else:
                return False, f"未対応の量子圧縮形式: {compressed_data[:20]}"
            
            # 量子情報ヘッダーをスキップ
            # float (4 bytes) + unsigned short (2 bytes) = 6 bytes
            quantum_header_size = 6
            payload_start = header_size + quantum_header_size
            
            # エンタングルメントペア数を読み取る
            if len(compressed_data) > header_size + 4:
                entanglement_count = struct.unpack('>H', compressed_data[header_size + 4:header_size + 6])[0]
                print(f"   ⚛️ エンタングルメントペア数: {entanglement_count}")
            
            # LZMA解凍を実行
            payload = compressed_data[payload_start:]
            try:
                decompressed = lzma.decompress(payload)
                print(f"   ✅ LZMA解凍成功: {len(payload)} → {len(decompressed)} bytes")
            except Exception as e:
                return False, f"LZMA解凍失敗: {str(e)}"
                
            # 復元ファイル作成
            base_name = Path(nxz_file).stem
            restored_file = Path(nxz_file).parent / f"{base_name}.quantum_restored"
            
            with open(restored_file, 'wb') as f:
                f.write(decompressed)
                
            print(f"   💾 量子復元完了: {len(decompressed)} bytes")
            return True, str(restored_file)
            
        except Exception as e:
            return False, str(e)
            
    def test_quantum_file(self, file_path):
        """個別ファイルの量子圧縮可逆性テスト"""
        print(f"\\n📄 量子圧縮テスト: {Path(file_path).name}")
        
        # オリジナルファイルのハッシュ
        try:
            original_hash = self.get_file_hash(file_path)
            original_size = os.path.getsize(file_path)
            print(f"   📊 元サイズ: {original_size:,} bytes")
            print(f"   🔒 元ハッシュ: {original_hash[:16]}...")
        except Exception as e:
            print(f"   ❌ ハッシュ計算失敗: {e}")
            return False
            
        test_result = {
            'file': str(file_path),
            'original_size': original_size,
            'original_hash': original_hash,
            'compression_success': False,
            'decompression_success': False,
            'hash_match': False,
            'compression_ratio': 0.0,
            'error': None
        }
        
        # 量子圧縮テスト
        compression_success, compression_result = self.compress_with_quantum_engine(file_path)
        
        if not compression_success:
            test_result['error'] = f"量子圧縮失敗: {compression_result}"
            print(f"   ❌ 量子圧縮失敗: {compression_result}")
            return test_result
            
        test_result['compression_success'] = True
        nxz_file = compression_result
        
        # 圧縮率計算
        compressed_size = os.path.getsize(nxz_file)
        compression_ratio = (1 - compressed_size / original_size) * 100
        test_result['compressed_size'] = compressed_size
        test_result['compression_ratio'] = compression_ratio
        print(f"   📦 量子圧縮後: {compressed_size:,} bytes ({compression_ratio:.1f}%)")
        
        # 量子解凍テスト
        decompression_success, decompression_result = self.decompress_quantum_nxz(nxz_file)
        
        if not decompression_success:
            test_result['error'] = f"量子解凍失敗: {decompression_result}"
            print(f"   ❌ 量子解凍失敗: {decompression_result}")
            return test_result
            
        test_result['decompression_success'] = True
        restored_file = decompression_result
        
        # ハッシュ照合
        try:
            restored_hash = self.get_file_hash(restored_file)
            test_result['restored_hash'] = restored_hash
            
            if original_hash == restored_hash:
                test_result['hash_match'] = True
                print(f"   ✅ 完全可逆性確認: ハッシュ一致")
                print(f"   🎯 量子圧縮性能: {compression_ratio:.1f}%")
            else:
                print(f"   ❌ ハッシュ不一致")
                print(f"      元: {original_hash[:16]}...")
                print(f"      復: {restored_hash[:16]}...")
                
        except Exception as e:
            test_result['error'] = f"復元ハッシュ計算失敗: {e}"
            print(f"   ❌ 復元ハッシュ失敗: {e}")
            
        return test_result
        
    def run_quantum_reversibility_test(self):
        """量子圧縮可逆性テスト実行"""
        print("🔬 nexus_quantum.py 量子圧縮可逆性テスト")
        print("⚛️ 量子もつれアルゴリズムの完全可逆性を検証")
        print("=" * 60)
        
        # テストファイル選定
        test_files = [
            "NXZip-Python/sample/COT-001.png",     # PNG大型
            "NXZip-Python/sample/COT-012.png",     # PNG超大型
            "NXZip-Python/sample/COT-001.jpg",     # JPEG
        ]
        
        results = {
            'test_timestamp': datetime.now().isoformat(),
            'engine': 'nexus_quantum.py',
            'tests': [],
            'summary': {
                'total_tests': 0,
                'passed': 0,
                'failed': 0,
                'success_rate': 0.0
            }
        }
        
        # 各ファイルをテスト
        for file_path in test_files:
            if not os.path.exists(file_path):
                print(f"⚠️ ファイルが存在しません: {file_path}")
                continue
                
            test_result = self.test_quantum_file(file_path)
            results['tests'].append(test_result)
            
        # サマリー計算
        total_tests = len(results['tests'])
        passed_tests = sum(1 for test in results['tests'] if test['hash_match'])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        results['summary']['total_tests'] = total_tests
        results['summary']['passed'] = passed_tests
        results['summary']['failed'] = failed_tests
        results['summary']['success_rate'] = success_rate
        
        print(f"\\n📊 量子圧縮可逆性テスト結果:")
        print(f"   合計: {total_tests}, 成功: {passed_tests}, 失敗: {failed_tests}")
        print(f"   可逆性率: {success_rate:.1f}%")
        
        if success_rate == 100.0:
            print(f"\\n🎉 量子圧縮エンジン完全可逆性達成!")
            print(f"⚛️ 量子もつれアルゴリズムによる完全データ復元確認")
        elif success_rate >= 66.7:
            print(f"\\n✅ 量子圧縮エンジン良好な可逆性")
            print(f"🔧 一部改善の余地あり")
        else:
            print(f"\\n⚠️ 量子圧縮エンジン可逆性要改善")
            
        # 結果をJSONファイルに保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"bin/quantum_reversibility_test_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        print(f"\\n💾 テスト結果保存: {output_file}")
        
        return results

def main():
    """メイン関数"""
    print("🔬 nexus_quantum.py 量子圧縮可逆性テスト")
    tester = QuantumReversibilityTester()
    tester.run_quantum_reversibility_test()

if __name__ == "__main__":
    main()
