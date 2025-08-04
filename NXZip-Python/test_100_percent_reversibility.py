#!/usr/bin/env python3
"""
NXZip 100%可逆性保証テスト
完全可逆性の実現と検証
"""

import sys
import os
import hashlib
import time
from typing import Dict, Any, List, Tuple

# パスの追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nxzip'))

from nxzip.formats.enhanced_nxz import SuperNXZipFile
from nxzip.engine.nexus_tmc_v91_modular import NEXUSTMCEngineV91


class ReversibilityTester:
    """100%可逆性テスター"""
    
    def __init__(self):
        self.nxz = SuperNXZipFile()
        self.tmc_engine = NEXUSTMCEngineV91(lightweight_mode=True)
        self.test_results = []
    
    def run_comprehensive_test(self) -> bool:
        """包括的可逆性テスト実行"""
        print("🎯 NXZip 100%可逆性保証テスト開始")
        print("=" * 60)
        
        test_cases = self._prepare_test_cases()
        total_tests = len(test_cases)
        passed_tests = 0
        
        for i, (name, data) in enumerate(test_cases):
            print(f"\n📋 テスト {i+1}/{total_tests}: {name}")
            print("-" * 40)
            
            success = self._test_single_case(name, data)
            if success:
                passed_tests += 1
                print(f"✅ {name}: 可逆性OK")
            else:
                print(f"❌ {name}: 可逆性NG")
        
        # 結果サマリー
        success_rate = (passed_tests / total_tests) * 100
        print(f"\n🏆 可逆性テスト結果")
        print("=" * 60)
        print(f"📊 成功率: {success_rate:.1f}% ({passed_tests}/{total_tests})")
        
        if success_rate == 100.0:
            print("🎉 100%可逆性達成！")
            return True
        else:
            print(f"⚠️ 可逆性未達成: {100.0 - success_rate:.1f}%の改善が必要")
            return False
    
    def _prepare_test_cases(self) -> List[Tuple[str, bytes]]:
        """テストケースの準備"""
        test_cases = []
        
        # 1. 基本テキストデータ
        test_cases.append(("小さなテキスト", b"Hello, World! This is a test."))
        test_cases.append(("日本語テキスト", "こんにちは、世界！これはテストです。".encode('utf-8')))
        test_cases.append(("繰り返しパターン", b"ABCD" * 250))  # 1KB
        
        # 2. バイナリデータ
        test_cases.append(("ランダムバイナリ", os.urandom(1024)))
        test_cases.append(("ゼロ埋めデータ", b'\x00' * 1024))
        test_cases.append(("0xFF埋めデータ", b'\xFF' * 1024))
        
        # 3. 構造化データ
        json_data = '{"name": "test", "value": 123, "array": [1,2,3,4,5]}'.encode('utf-8')
        test_cases.append(("JSONデータ", json_data))
        
        # 4. 数値系列（TMC特化テスト）
        numeric_data = b''.join([i.to_bytes(4, 'little') for i in range(256)])
        test_cases.append(("数値系列", numeric_data))
        
        # 5. 混合データ
        mixed_data = b"Text part: " + b'\x00\x01\x02\x03' + "More text".encode('utf-8') + b'\xFF\xFE\xFD\xFC'
        test_cases.append(("混合データ", mixed_data))
        
        # 6. エッジケース
        test_cases.append(("空データ", b""))
        test_cases.append(("1バイト", b"A"))
        test_cases.append(("大きなデータ", b"BigData!" * 1000))  # 9KB
        
        return test_cases
    
    def _test_single_case(self, name: str, original_data: bytes) -> bool:
        """単一ケースのテスト"""
        try:
            print(f"📊 元データ: {len(original_data)} bytes")
            
            # ハッシュ計算
            original_hash = hashlib.sha256(original_data).hexdigest()
            print(f"🔐 元データハッシュ: {original_hash[:16]}...")
            
            # Phase 1: NXZ圧縮（暗号化なし）
            print("🗜️ NXZ圧縮実行中...")
            start_time = time.time()
            nxz_archive = self.nxz.create_archive(original_data, password=None, show_progress=False)
            compress_time = time.time() - start_time
            
            compression_ratio = (1 - len(nxz_archive) / len(original_data)) * 100 if len(original_data) > 0 else 0
            print(f"📈 圧縮完了: {len(nxz_archive)} bytes ({compression_ratio:.1f}% 削減)")
            print(f"⚡ 圧縮時間: {compress_time:.3f}秒")
            
            # Phase 2: NXZ展開
            print("🔓 NXZ展開実行中...")
            start_time = time.time()
            restored_data = self.nxz.extract_archive(nxz_archive, password=None, show_progress=False)
            decompress_time = time.time() - start_time
            
            print(f"📤 展開完了: {len(restored_data)} bytes")
            print(f"⚡ 展開時間: {decompress_time:.3f}秒")
            
            # Phase 3: 完全性検証
            restored_hash = hashlib.sha256(restored_data).hexdigest()
            print(f"🔐 復元ハッシュ: {restored_hash[:16]}...")
            
            # バイト単位比較
            is_identical = original_data == restored_data
            hash_match = original_hash == restored_hash
            size_match = len(original_data) == len(restored_data)
            
            print(f"📏 サイズ一致: {size_match}")
            print(f"🔐 ハッシュ一致: {hash_match}")
            print(f"📋 バイト一致: {is_identical}")
            
            # 詳細分析（不一致の場合）
            if not is_identical:
                self._analyze_differences(original_data, restored_data)
            
            # 結果記録
            result = {
                'name': name,
                'original_size': len(original_data),
                'compressed_size': len(nxz_archive),
                'restored_size': len(restored_data),
                'compression_ratio': compression_ratio,
                'compress_time': compress_time,
                'decompress_time': decompress_time,
                'size_match': size_match,
                'hash_match': hash_match,
                'byte_match': is_identical,
                'success': is_identical and hash_match and size_match
            }
            self.test_results.append(result)
            
            return result['success']
            
        except Exception as e:
            print(f"❌ テストエラー: {e}")
            import traceback
            traceback.print_exc()
            
            self.test_results.append({
                'name': name,
                'error': str(e),
                'success': False
            })
            return False
    
    def _analyze_differences(self, original: bytes, restored: bytes):
        """差異の詳細分析"""
        print("\n🔍 差異詳細分析:")
        
        min_len = min(len(original), len(restored))
        differences = 0
        
        for i in range(min_len):
            if original[i] != restored[i]:
                differences += 1
                if differences <= 5:  # 最初の5個の差異を表示
                    print(f"  位置 {i}: {original[i]:02X} != {restored[i]:02X}")
        
        if len(original) != len(restored):
            print(f"  サイズ差異: {len(original)} vs {len(restored)}")
        
        print(f"  総差異数: {differences}/{min_len}")
        
        # サンプル表示
        if len(original) <= 100:
            print(f"  元データ: {original}")
            print(f"  復元データ: {restored}")
    
    def get_test_summary(self) -> Dict[str, Any]:
        """テスト結果サマリー取得"""
        if not self.test_results:
            return {}
        
        successful = [r for r in self.test_results if r.get('success', False)]
        failed = [r for r in self.test_results if not r.get('success', False)]
        
        return {
            'total_tests': len(self.test_results),
            'successful_tests': len(successful),
            'failed_tests': len(failed),
            'success_rate': len(successful) / len(self.test_results) * 100,
            'avg_compression_ratio': sum(r.get('compression_ratio', 0) for r in successful) / len(successful) if successful else 0,
            'avg_compress_time': sum(r.get('compress_time', 0) for r in successful) / len(successful) if successful else 0,
            'avg_decompress_time': sum(r.get('decompress_time', 0) for r in successful) / len(successful) if successful else 0,
            'failed_cases': [r['name'] for r in failed]
        }


def main():
    """メイン実行"""
    tester = ReversibilityTester()
    
    # 100%可逆性テスト実行
    success = tester.run_comprehensive_test()
    
    # 詳細サマリー表示
    summary = tester.get_test_summary()
    print(f"\n📋 詳細統計")
    print("=" * 60)
    print(f"🎯 成功率: {summary.get('success_rate', 0):.1f}%")
    print(f"📊 平均圧縮率: {summary.get('avg_compression_ratio', 0):.1f}%")
    print(f"⚡ 平均圧縮時間: {summary.get('avg_compress_time', 0):.3f}秒")
    print(f"⚡ 平均展開時間: {summary.get('avg_decompress_time', 0):.3f}秒")
    
    if summary.get('failed_cases'):
        print(f"❌ 失敗ケース: {', '.join(summary['failed_cases'])}")
    
    if success:
        print("\n🏆 100%可逆性達成 - NXZipは完全可逆圧縮システムです！")
        return 0
    else:
        print("\n⚠️ 可逆性未達成 - さらなる改善が必要です")
        return 1


if __name__ == "__main__":
    exit(main())
