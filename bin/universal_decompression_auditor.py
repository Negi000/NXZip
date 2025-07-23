#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 Universal Reversibility Auditor - 包括的可逆性監査システム
全フォーマット・全エンジンの可逆性を徹底検証

🎯 監査対象:
- 全形式エンジン (MP4, MP3, TEXT, etc.)
- 最新最適化エンジン
- 古いバージョンエンジン
- SDCエンジン
- 可逆性完全検証
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
import importlib.util
from typing import Dict, List, Tuple, Any

class UniversalReversibilityAuditor:
    """包括的可逆性監査システム"""
    
    def __init__(self):
        self.audit_results = []
        self.test_data_sets = {}
        self.engine_modules = {}
        
    def discover_engines(self) -> Dict[str, str]:
        """エンジン発見"""
        try:
            print("🔍 エンジン発見中...")
            
            bin_dir = Path(__file__).parent
            engines = {}
            
            # nexus_*.py ファイルを検索
            for py_file in bin_dir.glob("nexus_*.py"):
                if py_file.name != "nexus_reversibility_test.py":  # 自分自身を除外
                    engine_name = py_file.stem
                    engines[engine_name] = str(py_file)
                    print(f"📦 発見: {engine_name}")
            
            # NXZip-Python内のエンジンも検索
            nxzip_dir = bin_dir.parent / "NXZip-Python" / "nxzip" / "engine"
            if nxzip_dir.exists():
                for py_file in nxzip_dir.glob("nexus_*.py"):
                    engine_name = f"nxzip_{py_file.stem}"
                    engines[engine_name] = str(py_file)
                    print(f"📦 発見: {engine_name}")
            
            print(f"🎯 総発見数: {len(engines)} エンジン")
            return engines
            
        except Exception as e:
            print(f"❌ エンジン発見エラー: {e}")
            return {}
    
    def prepare_test_datasets(self) -> Dict[str, bytes]:
        """テストデータセット準備"""
        try:
            print("📋 テストデータセット準備中...")
            
            datasets = {}
            sample_dir = Path(__file__).parent.parent / "NXZip-Python" / "sample"
            
            # MP4テストデータ
            mp4_file = sample_dir / "Python基礎講座3_4月26日-3.mp4"
            if mp4_file.exists():
                with open(mp4_file, 'rb') as f:
                    datasets['MP4'] = f.read()
                print(f"📹 MP4データ: {len(datasets['MP4']):,} bytes")
            
            # MP3テストデータ
            mp3_file = sample_dir / "test_audio.mp3"
            if mp3_file.exists():
                with open(mp3_file, 'rb') as f:
                    datasets['MP3'] = f.read()
                print(f"🎵 MP3データ: {len(datasets['MP3']):,} bytes")
            
            # TEXTテストデータ
            text_file = sample_dir / "test_text.txt"
            if text_file.exists():
                with open(text_file, 'rb') as f:
                    datasets['TEXT'] = f.read()
                print(f"📝 TEXTデータ: {len(datasets['TEXT']):,} bytes")
            
            # 合成テストデータ (小さなファイル)
            if not datasets:
                print("⚠️ サンプルファイルなし - 合成データ作成")
                datasets['SYNTHETIC_MP4'] = self._create_synthetic_mp4()
                datasets['SYNTHETIC_MP3'] = self._create_synthetic_mp3()
                datasets['SYNTHETIC_TEXT'] = self._create_synthetic_text()
            
            print(f"✅ テストデータセット準備完了: {len(datasets)} 種類")
            return datasets
            
        except Exception as e:
            print(f"❌ テストデータ準備エラー: {e}")
            return {}
    
    def _create_synthetic_mp4(self) -> bytes:
        """合成MP4データ作成"""
        # 最小限のMP4構造
        ftyp = b'\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2avc1mp41'
        mdat = b'\x00\x00\x10\x00mdat' + b'\x00' * (4096 - 8)
        return ftyp + mdat
    
    def _create_synthetic_mp3(self) -> bytes:
        """合成MP3データ作成"""
        # MP3ヘッダー + データ
        mp3_header = b'\xFF\xFB\x90\x00'  # MP3フレームヘッダー
        return mp3_header + b'\x00' * 2048
    
    def _create_synthetic_text(self) -> bytes:
        """合成TEXTデータ作成"""
        text = "Hello, World! " * 200  # 繰り返しテキスト
        return text.encode('utf-8')
    
    def load_engine_module(self, engine_name: str, engine_path: str) -> Any:
        """エンジンモジュール動的読み込み"""
        try:
            spec = importlib.util.spec_from_file_location(engine_name, engine_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
            return None
        except Exception as e:
            print(f"❌ {engine_name} 読み込みエラー: {e}")
            return None
    
    def detect_engine_capabilities(self, module: Any) -> Dict[str, Any]:
        """エンジン機能検出"""
        capabilities = {
            'has_compress': False,
            'has_decompress': False,
            'compress_methods': [],
            'decompress_methods': [],
            'engine_classes': []
        }
        
        try:
            # クラスと関数の検出
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                
                if callable(attr):
                    if 'compress' in attr_name.lower() and 'decompress' not in attr_name.lower():
                        capabilities['compress_methods'].append(attr_name)
                        capabilities['has_compress'] = True
                    elif 'decompress' in attr_name.lower():
                        capabilities['decompress_methods'].append(attr_name)
                        capabilities['has_decompress'] = True
                
                # エンジンクラスの検出
                if hasattr(attr, '__name__') and 'engine' in attr.__name__.lower():
                    capabilities['engine_classes'].append(attr_name)
            
            return capabilities
            
        except Exception as e:
            print(f"❌ 機能検出エラー: {e}")
            return capabilities
    
    def test_engine_reversibility(self, engine_name: str, engine_path: str, 
                                test_data: Dict[str, bytes]) -> Dict[str, Any]:
        """エンジン可逆性テスト"""
        print(f"\n🧪 {engine_name} 可逆性テスト開始")
        print("-" * 60)
        
        result = {
            'engine_name': engine_name,
            'engine_path': engine_path,
            'status': 'unknown',
            'capabilities': {},
            'test_results': {},
            'errors': []
        }
        
        try:
            # モジュール読み込み
            module = self.load_engine_module(engine_name, engine_path)
            if not module:
                result['status'] = 'load_failed'
                result['errors'].append('モジュール読み込み失敗')
                return result
            
            print(f"✅ モジュール読み込み成功")
            
            # 機能検出
            capabilities = self.detect_engine_capabilities(module)
            result['capabilities'] = capabilities
            
            print(f"🔍 圧縮メソッド: {capabilities['compress_methods']}")
            print(f"🔍 解凍メソッド: {capabilities['decompress_methods']}")
            print(f"🔍 エンジンクラス: {capabilities['engine_classes']}")
            
            # 可逆性テスト実行
            if capabilities['has_compress'] and capabilities['has_decompress']:
                result['test_results'] = self._run_reversibility_tests(
                    module, capabilities, test_data
                )
                result['status'] = 'tested'
            elif capabilities['has_compress']:
                result['status'] = 'compress_only'
                result['errors'].append('解凍機能なし - 可逆性テスト不可')
                print("⚠️ 解凍機能なし - 可逆性テスト不可")
            else:
                result['status'] = 'no_compress'
                result['errors'].append('圧縮機能なし')
                print("⚠️ 圧縮機能なし")
            
            return result
            
        except Exception as e:
            result['status'] = 'error'
            result['errors'].append(str(e))
            print(f"❌ テストエラー: {e}")
            return result
    
    def _run_reversibility_tests(self, module: Any, capabilities: Dict[str, Any], 
                               test_data: Dict[str, bytes]) -> Dict[str, Any]:
        """可逆性テスト実行"""
        test_results = {}
        
        try:
            # エンジンインスタンス作成試行
            engine_instance = None
            
            for class_name in capabilities['engine_classes']:
                try:
                    engine_class = getattr(module, class_name)
                    engine_instance = engine_class()
                    print(f"✅ エンジンインスタンス作成: {class_name}")
                    break
                except Exception as e:
                    print(f"⚠️ {class_name} インスタンス作成失敗: {e}")
                    continue
            
            if not engine_instance:
                print("⚠️ エンジンインスタンス作成失敗 - 関数レベルテスト試行")
                return self._test_function_level_reversibility(module, capabilities, test_data)
            
            # 各データタイプでテスト
            for data_type, data in test_data.items():
                print(f"\n📋 {data_type} テスト ({len(data):,} bytes)")
                
                test_result = {
                    'data_type': data_type,
                    'original_size': len(data),
                    'compressed_size': 0,
                    'decompressed_size': 0,
                    'compression_ratio': 0.0,
                    'byte_match': False,
                    'hash_match': False,
                    'errors': []
                }
                
                try:
                    # 圧縮テスト
                    if hasattr(engine_instance, 'compress'):
                        compressed_data = engine_instance.compress(data)
                        test_result['compressed_size'] = len(compressed_data)
                        test_result['compression_ratio'] = (1 - len(compressed_data)/len(data)) * 100
                        print(f"✅ 圧縮成功: {len(data)} -> {len(compressed_data)} ({test_result['compression_ratio']:.1f}%)")
                        
                        # 解凍テスト
                        if hasattr(engine_instance, 'decompress'):
                            decompressed_data = engine_instance.decompress(compressed_data)
                            test_result['decompressed_size'] = len(decompressed_data)
                            
                            # 可逆性検証
                            test_result['byte_match'] = (data == decompressed_data)
                            test_result['hash_match'] = (
                                hashlib.sha256(data).hexdigest() == 
                                hashlib.sha256(decompressed_data).hexdigest()
                            )
                            
                            print(f"✅ 解凍成功: {len(compressed_data)} -> {len(decompressed_data)}")
                            print(f"🔍 バイト一致: {'PASS' if test_result['byte_match'] else 'FAIL'}")
                            print(f"🔍 ハッシュ一致: {'PASS' if test_result['hash_match'] else 'FAIL'}")
                            
                            if test_result['byte_match'] and test_result['hash_match']:
                                print("🎉 完全可逆性確認!")
                            else:
                                print("❌ 可逆性問題あり!")
                                
                        else:
                            test_result['errors'].append('解凍メソッドなし')
                    else:
                        test_result['errors'].append('圧縮メソッドなし')
                
                except Exception as e:
                    test_result['errors'].append(str(e))
                    print(f"❌ テストエラー: {e}")
                
                test_results[data_type] = test_result
            
            return test_results
            
        except Exception as e:
            print(f"❌ 可逆性テスト実行エラー: {e}")
            return {'error': str(e)}
    
    def _test_function_level_reversibility(self, module: Any, capabilities: Dict[str, Any], 
                                         test_data: Dict[str, bytes]) -> Dict[str, Any]:
        """関数レベル可逆性テスト"""
        print("🔧 関数レベルテスト実行")
        # 実装省略 - 必要に応じて追加
        return {'status': 'function_level_not_implemented'}
    
    def run_comprehensive_audit(self) -> Dict[str, Any]:
        """包括的監査実行"""
        print("🔍 Universal Reversibility Audit - 包括的可逆性監査")
        print("🎯 全エンジンの可逆性を徹底検証")
        print("=" * 70)
        
        # エンジン発見
        engines = self.discover_engines()
        if not engines:
            print("❌ エンジンが見つかりません")
            return {'status': 'no_engines'}
        
        # テストデータ準備
        test_data = self.prepare_test_datasets()
        if not test_data:
            print("❌ テストデータが準備できません")
            return {'status': 'no_test_data'}
        
        # 各エンジンをテスト
        audit_results = []
        total_engines = len(engines)
        
        print(f"\n🧪 {total_engines} エンジンの可逆性監査開始")
        print("=" * 70)
        
        for i, (engine_name, engine_path) in enumerate(engines.items(), 1):
            print(f"\n[{i}/{total_engines}] {engine_name}")
            
            result = self.test_engine_reversibility(engine_name, engine_path, test_data)
            audit_results.append(result)
            
            # 進捗表示
            progress = (i / total_engines) * 100
            print(f"📈 進捗: {progress:.1f}% ({i}/{total_engines})")
        
        # 総合結果分析
        summary = self._analyze_audit_results(audit_results)
        
        # 結果表示
        self._display_audit_summary(summary, audit_results)
        
        return {
            'status': 'completed',
            'summary': summary,
            'detailed_results': audit_results
        }
    
    def _analyze_audit_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """監査結果分析"""
        summary = {
            'total_engines': len(results),
            'fully_reversible': 0,
            'partially_reversible': 0,
            'not_reversible': 0,
            'compress_only': 0,
            'load_failed': 0,
            'critical_issues': []
        }
        
        for result in results:
            status = result['status']
            
            if status == 'tested':
                test_results = result.get('test_results', {})
                all_reversible = True
                any_reversible = False
                
                for data_type, test_data in test_results.items():
                    if isinstance(test_data, dict):
                        if test_data.get('byte_match', False) and test_data.get('hash_match', False):
                            any_reversible = True
                        else:
                            all_reversible = False
                
                if all_reversible and any_reversible:
                    summary['fully_reversible'] += 1
                elif any_reversible:
                    summary['partially_reversible'] += 1
                else:
                    summary['not_reversible'] += 1
                    summary['critical_issues'].append(f"{result['engine_name']}: 可逆性なし")
            
            elif status == 'compress_only':
                summary['compress_only'] += 1
                summary['critical_issues'].append(f"{result['engine_name']}: 解凍機能なし")
            
            elif status == 'load_failed':
                summary['load_failed'] += 1
        
        return summary
    
    def _display_audit_summary(self, summary: Dict[str, Any], results: List[Dict[str, Any]]):
        """監査結果表示"""
        print("\n" + "=" * 70)
        print("🏆 包括的可逆性監査結果")
        print("=" * 70)
        
        total = summary['total_engines']
        print(f"📊 総エンジン数: {total}")
        print(f"✅ 完全可逆: {summary['fully_reversible']} ({summary['fully_reversible']/total*100:.1f}%)")
        print(f"⚠️ 部分可逆: {summary['partially_reversible']} ({summary['partially_reversible']/total*100:.1f}%)")
        print(f"❌ 非可逆: {summary['not_reversible']} ({summary['not_reversible']/total*100:.1f}%)")
        print(f"🔧 圧縮のみ: {summary['compress_only']} ({summary['compress_only']/total*100:.1f}%)")
        print(f"💥 読み込み失敗: {summary['load_failed']} ({summary['load_failed']/total*100:.1f}%)")
        
        if summary['critical_issues']:
            print(f"\n🚨 重要な問題:")
            for issue in summary['critical_issues'][:10]:  # 最初の10件
                print(f"   - {issue}")
            if len(summary['critical_issues']) > 10:
                print(f"   ... 他 {len(summary['critical_issues']) - 10} 件")
        
        # 完全可逆エンジンリスト
        fully_reversible_engines = []
        for result in results:
            if result['status'] == 'tested':
                test_results = result.get('test_results', {})
                all_reversible = all(
                    test_data.get('byte_match', False) and test_data.get('hash_match', False)
                    for test_data in test_results.values()
                    if isinstance(test_data, dict)
                )
                if all_reversible:
                    fully_reversible_engines.append(result['engine_name'])
        
        if fully_reversible_engines:
            print(f"\n🌟 完全可逆エンジン:")
            for engine in fully_reversible_engines:
                print(f"   ✅ {engine}")
        else:
            print(f"\n⚠️ 完全可逆エンジンなし - 緊急対応が必要")

def main():
    """メイン関数"""
    if len(sys.argv) < 2:
        print("🔍 Universal Reversibility Auditor")
        print("使用方法:")
        print("  python universal_decompression_auditor.py audit    # 包括的可逆性監査")
        return
    
    command = sys.argv[1].lower()
    auditor = UniversalReversibilityAuditor()
    
    if command == "audit":
        result = auditor.run_comprehensive_audit()
        if result['status'] == 'completed':
            print("\n✅ 包括的可逆性監査完了")
        else:
            print(f"\n❌ 監査失敗: {result['status']}")
    else:
        print("❌ 無効なコマンドです")

if __name__ == "__main__":
    main()
