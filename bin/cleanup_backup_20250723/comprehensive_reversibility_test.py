#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 NXZip 包括的可逆性テスト
全エンジンの完全可逆性を検証

🎯 テスト対象エンジン:
- nexus_quantum.py      : PNG/JPEG 量子圧縮
- nexus_phase8_turbo.py : 全フォーマット AI強化  
- nexus_optimal_balance.py : テキスト最適化
- nexus_lightning_fast.py  : MP3/WAV音声特化
"""

import os
import sys
import json
import time
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime

class ComprehensiveReversibilityTester:
    """包括的可逆性テスター"""
    
    def __init__(self):
        self.test_results = {}
        self.engines = {
            'nexus_quantum.py': {
                'name': '量子圧縮エンジン',
                'formats': ['.jpg', '.jpeg', '.png'],
                'description': 'PNG/JPEG量子圧縮・量子もつれアルゴリズム'
            },
            'nexus_phase8_turbo.py': {
                'name': 'Phase8 Turbo AI強化エンジン',
                'formats': ['.mp4', '.avi', '.mov', '.txt', '.jpg', '.png', '.mp3', '.wav'],
                'description': '全フォーマット対応・AI強化・並列処理最適化'
            },
            'nexus_optimal_balance.py': {
                'name': '最適バランスエンジン',
                'formats': ['.txt', '.md', '.log', '.csv'],
                'description': 'テキスト特化・構造破壊型・高効率圧縮'
            },
            'nexus_lightning_fast.py': {
                'name': '超高速音声エンジン',
                'formats': ['.mp3', '.wav', '.aac', '.flac'],
                'description': 'MP3/WAV特化・超高速処理・音声最適化'
            }
        }
        
    def get_file_hash(self, file_path):
        """ファイルのSHA256ハッシュを計算"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
        
    def run_engine_compression(self, engine_script, input_file):
        """エンジンによる圧縮実行"""
        try:
            cmd = ['python', f'bin/{engine_script}', str(input_file)]
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  encoding='utf-8', errors='replace')
            
            if result.returncode == 0:
                # 成功時の出力から.nxzファイルパスを抽出
                for line in result.stdout.split('\\n'):
                    if 'SUCCESS:' in line and '.nxz' in line:
                        nxz_path = line.split('SUCCESS: 圧縮完了 - ')[-1].strip()
                        return True, nxz_path
                return True, None
            else:
                return False, result.stderr.strip()
                
        except Exception as e:
            return False, str(e)
            
    def decompress_nxz_file(self, nxz_file):
        """NXZファイルの解凍（複数エンジン対応）"""
        try:
            # NXZファイルを読み込み
            with open(nxz_file, 'rb') as f:
                compressed_data = f.read()
            
            # マジックナンバーチェック（複数形式対応）
            if compressed_data.startswith(b'NXZ\x01'):
                # 標準NXZ形式
                payload = compressed_data[4:]
                print(f"   🔍 標準NXZ形式検出")
            elif compressed_data.startswith(b'NXZ8T'):
                # Phase8 Turbo形式
                payload = compressed_data[5:]
                print(f"   🔍 Phase8 Turbo形式検出")
            else:
                # ヘッダーなし（量子エンジンなど）
                payload = compressed_data
                print(f"   🔍 ヘッダーなし形式検出")
                
            # 解凍実行（複数アルゴリズム試行）
            decompressed = None
            last_error = None
            
            # LZMA解凍を試行
            try:
                import lzma
                decompressed = lzma.decompress(payload)
                print(f"   ✅ LZMA解凍成功")
            except Exception as e:
                last_error = f"LZMA失敗: {str(e)}"
                
            # zlib解凍を試行
            if decompressed is None:
                try:
                    import zlib
                    decompressed = zlib.decompress(payload)
                    print(f"   ✅ zlib解凍成功")
                except Exception as e:
                    last_error = f"zlib失敗: {str(e)}"
                    
            # bz2解凍を試行
            if decompressed is None:
                try:
                    import bz2
                    decompressed = bz2.decompress(payload)
                    print(f"   ✅ bz2解凍成功")
                except Exception as e:
                    last_error = f"bz2失敗: {str(e)}"
                    
            if decompressed is None:
                return False, f"解凍失敗 - 全アルゴリズム試行済み: {last_error}"
                    
            # 復元ファイル作成
            base_name = Path(nxz_file).stem
            restored_file = Path(nxz_file).parent / f"{base_name}.restored"
            
            with open(restored_file, 'wb') as f:
                f.write(decompressed)
                
            print(f"   💾 復元完了: {len(decompressed)} bytes")
            return True, str(restored_file)
            
        except Exception as e:
            return False, str(e)
            
    def test_engine_reversibility(self, engine_script, test_files):
        """エンジンの可逆性テスト"""
        engine_name = self.engines[engine_script]['name']
        engine_formats = self.engines[engine_script]['formats']
        
        print(f"\\n🔬 {engine_name} 可逆性テスト開始")
        print(f"📋 対象フォーマット: {', '.join(engine_formats)}")
        print("=" * 60)
        
        engine_results = {
            'engine': engine_name,
            'script': engine_script,
            'target_formats': engine_formats,
            'tests': [],
            'summary': {
                'total_tests': 0,
                'passed': 0,
                'failed': 0,
                'success_rate': 0.0
            }
        }
        
        # フォーマット別にファイルをフィルター
        compatible_files = []
        for file_path in test_files:
            file_ext = Path(file_path).suffix.lower()
            if any(file_ext == fmt for fmt in engine_formats):
                compatible_files.append(file_path)
                
        if not compatible_files:
            print(f"⚠️ 対応ファイルが見つかりません")
            return engine_results
            
        print(f"📁 テスト対象: {len(compatible_files)}ファイル")
        
        for file_path in compatible_files:
            print(f"\\n📄 テスト: {Path(file_path).name}")
            
            # オリジナルファイルのハッシュ計算
            try:
                original_hash = self.get_file_hash(file_path)
                original_size = os.path.getsize(file_path)
                print(f"   📊 元サイズ: {original_size:,} bytes")
                print(f"   🔒 元ハッシュ: {original_hash[:16]}...")
            except Exception as e:
                print(f"   ❌ ハッシュ計算失敗: {e}")
                continue
                
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
            
            # 圧縮テスト
            compression_success, compression_result = self.run_engine_compression(
                engine_script, file_path)
                
            if not compression_success:
                test_result['error'] = f"圧縮失敗: {compression_result}"
                print(f"   ❌ 圧縮失敗: {compression_result}")
                engine_results['tests'].append(test_result)
                continue
                
            test_result['compression_success'] = True
            
            # NXZファイルパスの特定
            nxz_file = None
            if compression_result:
                nxz_file = compression_result
            else:
                # 推測でファイルパスを決定
                base_name = Path(file_path).stem
                nxz_file = f"NXZip-Python/sample/{base_name}.nxz"
                
            if not os.path.exists(nxz_file):
                # 代替パスを検索
                base_name = Path(file_path).stem
                possible_paths = [
                    f"{base_name}.nxz",
                    f"sample/{base_name}.nxz",
                    f"NXZip-Python/sample/{base_name}.nxz"
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        nxz_file = path
                        break
                        
            if not os.path.exists(nxz_file):
                test_result['error'] = f"NXZファイルが見つかりません: {nxz_file}"
                print(f"   ❌ NXZファイル未発見: {nxz_file}")
                engine_results['tests'].append(test_result)
                continue
                
            # 圧縮率計算
            compressed_size = os.path.getsize(nxz_file)
            compression_ratio = (1 - compressed_size / original_size) * 100
            test_result['compressed_size'] = compressed_size
            test_result['compression_ratio'] = compression_ratio
            print(f"   📦 圧縮後: {compressed_size:,} bytes ({compression_ratio:.1f}%)")
            
            # 解凍テスト
            decompression_success, decompression_result = self.decompress_nxz_file(nxz_file)
            
            if not decompression_success:
                test_result['error'] = f"解凍失敗: {decompression_result}"
                print(f"   ❌ 解凍失敗: {decompression_result}")
                engine_results['tests'].append(test_result)
                continue
                
            test_result['decompression_success'] = True
            restored_file = decompression_result
            
            # ハッシュ照合
            try:
                restored_hash = self.get_file_hash(restored_file)
                test_result['restored_hash'] = restored_hash
                
                if original_hash == restored_hash:
                    test_result['hash_match'] = True
                    print(f"   ✅ 完全可逆性確認: ハッシュ一致")
                    print(f"   🎯 圧縮性能: {compression_ratio:.1f}%")
                else:
                    print(f"   ❌ ハッシュ不一致")
                    print(f"      元: {original_hash[:16]}...")
                    print(f"      復: {restored_hash[:16]}...")
                    
            except Exception as e:
                test_result['error'] = f"復元ハッシュ計算失敗: {e}"
                print(f"   ❌ 復元ハッシュ失敗: {e}")
                
            engine_results['tests'].append(test_result)
            
        # サマリー計算
        total_tests = len(engine_results['tests'])
        passed_tests = sum(1 for test in engine_results['tests'] 
                          if test['hash_match'])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        engine_results['summary']['total_tests'] = total_tests
        engine_results['summary']['passed'] = passed_tests
        engine_results['summary']['failed'] = failed_tests
        engine_results['summary']['success_rate'] = success_rate
        
        print(f"\\n📊 {engine_name} テスト結果:")
        print(f"   合計: {total_tests}, 成功: {passed_tests}, 失敗: {failed_tests}")
        print(f"   可逆性率: {success_rate:.1f}%")
        
        return engine_results
        
    def run_comprehensive_test(self):
        """包括的可逆性テスト実行"""
        print("🔬 NXZip 包括的可逆性テスト")
        print("🎯 全エンジンの完全可逆性を検証")
        print("=" * 60)
        
        # テストファイル収集
        sample_dir = Path("NXZip-Python/sample")
        if not sample_dir.exists():
            print(f"❌ サンプルディレクトリが見つかりません: {sample_dir}")
            return
            
        test_files = []
        for file_path in sample_dir.iterdir():
            if file_path.is_file() and not file_path.name.endswith(('.nxz', '.7z', '.restored')):
                test_files.append(file_path)
                
        print(f"📁 テストファイル: {len(test_files)}個")
        for file_path in test_files:
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"   - {file_path.name} ({size_mb:.1f}MB)")
            
        # 各エンジンをテスト
        all_results = {
            'test_timestamp': datetime.now().isoformat(),
            'test_files_count': len(test_files),
            'engines_tested': len(self.engines),
            'engines': {}
        }
        
        for engine_script in self.engines.keys():
            engine_results = self.test_engine_reversibility(engine_script, test_files)
            all_results['engines'][engine_script] = engine_results
            
        # 総合結果
        print(f"\\n🎊 包括的可逆性テスト完了")
        print("=" * 60)
        
        total_tests = sum(result['summary']['total_tests'] 
                         for result in all_results['engines'].values())
        total_passed = sum(result['summary']['passed'] 
                          for result in all_results['engines'].values())
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        all_results['overall_summary'] = {
            'total_tests': total_tests,
            'total_passed': total_passed,
            'overall_success_rate': overall_success_rate
        }
        
        print(f"📊 総合結果:")
        print(f"   全テスト: {total_tests}")
        print(f"   成功: {total_passed}")
        print(f"   総合可逆性率: {overall_success_rate:.1f}%")
        
        # エンジン別結果
        print(f"\\n🔧 エンジン別結果:")
        for engine_script, results in all_results['engines'].items():
            engine_name = results['engine']
            success_rate = results['summary']['success_rate']
            tests = results['summary']['total_tests']
            print(f"   {engine_name}: {success_rate:.1f}% ({tests}テスト)")
            
        # 結果をJSONファイルに保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"bin/comprehensive_reversibility_test_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
            
        print(f"\\n💾 テスト結果保存: {output_file}")
        
        return all_results

def main():
    """メイン関数"""
    print("🔬 NXZip 包括的可逆性テスト")
    tester = ComprehensiveReversibilityTester()
    tester.run_comprehensive_test()

if __name__ == "__main__":
    main()
