#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 COMPREHENSIVE REVERSIBILITY AUDIT - 全エンジン可逆性監査
すべての最適化エンジンの可逆性を徹底検証

🎯 監査対象:
- 全形式対応エンジン (TEXT, MP3, PNG, MP4)
- 高圧縮率主張エンジンの検証
- 可逆性偽装の検出
- 真の性能評価
"""

import os
import sys
import time
import subprocess
from pathlib import Path
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

class ComprehensiveReversibilityAuditor:
    """包括的可逆性監査システム"""
    
    def __init__(self):
        self.audit_results = []
        self.suspicious_engines = []
        self.verified_engines = []
        
        # テストファイル定義
        self.test_files = {
            'TEXT': r"c:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\test-data\large_test.txt",
            'MP3': r"c:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\sample\sample.mp3",
            'PNG': r"c:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\sample\sample.png", 
            'MP4': r"c:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\sample\Python基礎講座3_4月26日-3.mp4"
        }
        
        # 監査対象エンジン
        self.engines_to_audit = [
            'nexus_lightning_ultra.py',
            'nexus_ultimate_final.py',
            'nexus_ultimate_lightning.py',
            'nexus_optimization_phase6.py',
            'nexus_final_integrated.py',
            'nexus_absolute_final.py',
            'nexus_ai_driven.py',
            'nexus_quantum.py',
            'nexus_optimal_balance.py'  # 参照用（可逆性確認済み）
        ]
    
    def run_comprehensive_audit(self) -> dict:
        """包括的可逆性監査実行"""
        print("🧪 COMPREHENSIVE REVERSIBILITY AUDIT")
        print("🎯 全エンジン可逆性検証 - 偽装圧縮検出")
        print("🔍 真の性能評価システム")
        print("=" * 70)
        
        audit_start = time.time()
        
        # ステップ1: エンジン存在確認
        available_engines = self._check_engine_availability()
        print(f"📊 監査対象エンジン: {len(available_engines)} / {len(self.engines_to_audit)}")
        
        # ステップ2: テストファイル存在確認
        available_files = self._check_test_files()
        print(f"📄 テストファイル: {len(available_files)} / {len(self.test_files)}")
        
        if not available_engines:
            print("❌ 監査対象エンジンが見つかりません")
            return {'success': False, 'error': 'No engines found'}
        
        if not available_files:
            print("❌ テストファイルが見つかりません")
            return {'success': False, 'error': 'No test files found'}
        
        # ステップ3: 並列監査実行
        print("\n🧪 並列可逆性監査開始...")
        print("-" * 70)
        
        audit_results = []
        
        # 各エンジンで各ファイル形式をテスト
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            
            for engine in available_engines:
                for file_format, file_path in available_files.items():
                    if os.path.exists(file_path):
                        future = executor.submit(
                            self._audit_engine_format, engine, file_format, file_path
                        )
                        futures[future] = (engine, file_format)
            
            # 結果収集
            for future in as_completed(futures, timeout=300):  # 5分タイムアウト
                try:
                    result = future.result(timeout=30)  # 個別30秒タイムアウト
                    audit_results.append(result)
                    
                    engine, file_format = futures[future]
                    reversibility = result.get('reversibility_status', 'UNKNOWN')
                    compression = result.get('compression_ratio', 0)
                    
                    if reversibility == 'PERFECT':
                        print(f"✅ {engine} ({file_format}): {compression:.1f}% - 完全可逆")
                    elif reversibility == 'PARTIAL':
                        print(f"⚠️ {engine} ({file_format}): {compression:.1f}% - 部分可逆")
                    elif reversibility == 'FAILED':
                        print(f"❌ {engine} ({file_format}): {compression:.1f}% - 可逆性失敗")
                    else:
                        print(f"🔧 {engine} ({file_format}): エンジンエラー")
                        
                except Exception as e:
                    engine, file_format = futures[future]
                    print(f"⚠️ {engine} ({file_format}): 監査エラー - {str(e)[:50]}")
                    audit_results.append({
                        'engine': engine,
                        'file_format': file_format,
                        'reversibility_status': 'ERROR',
                        'error': str(e)
                    })
        
        # ステップ4: 結果分析
        print("\n" + "=" * 70)
        print("🏆 包括的可逆性監査結果")
        print("=" * 70)
        
        analysis = self._analyze_audit_results(audit_results)
        
        # ステップ5: 疑わしいエンジンの特定
        self._identify_suspicious_engines(audit_results)
        
        # ステップ6: 検証済みエンジンの認定
        self._certify_verified_engines(audit_results)
        
        # 最終レポート
        total_time = time.time() - audit_start
        
        print(f"\n📊 監査統計:")
        print(f"   📋 総テスト数: {len(audit_results)}")
        print(f"   ✅ 完全可逆: {analysis['perfect_count']}")
        print(f"   ⚠️ 部分可逆: {analysis['partial_count']}")
        print(f"   ❌ 可逆失敗: {analysis['failed_count']}")
        print(f"   🔧 エラー: {analysis['error_count']}")
        print(f"   ⚡ 監査時間: {total_time:.1f}s")
        
        print(f"\n🚨 疑わしいエンジン: {len(self.suspicious_engines)}")
        for engine in self.suspicious_engines:
            print(f"   ⚠️ {engine}")
        
        print(f"\n🏆 検証済みエンジン: {len(self.verified_engines)}")
        for engine in self.verified_engines:
            print(f"   ✅ {engine}")
        
        return {
            'success': True,
            'audit_results': audit_results,
            'analysis': analysis,
            'suspicious_engines': self.suspicious_engines,
            'verified_engines': self.verified_engines,
            'audit_time': total_time
        }
    
    def _check_engine_availability(self) -> list:
        """エンジン存在確認"""
        available = []
        bin_dir = Path(".")
        
        for engine in self.engines_to_audit:
            engine_path = bin_dir / engine
            if engine_path.exists():
                available.append(engine)
        
        return available
    
    def _check_test_files(self) -> dict:
        """テストファイル存在確認"""
        available = {}
        
        for file_format, file_path in self.test_files.items():
            if os.path.exists(file_path):
                available[file_format] = file_path
            else:
                print(f"⚠️ テストファイル不在: {file_format} - {file_path}")
        
        return available
    
    def _audit_engine_format(self, engine: str, file_format: str, file_path: str) -> dict:
        """エンジン・形式別監査"""
        try:
            print(f"🧪 監査中: {engine} ({file_format})")
            
            # 元ファイル情報取得
            with open(file_path, 'rb') as f:
                original_data = f.read()
            
            original_size = len(original_data)
            original_hash = hashlib.sha256(original_data).hexdigest()
            
            # 圧縮実行
            compress_start = time.time()
            compress_result = self._run_compression(engine, file_path)
            compress_time = time.time() - compress_start
            
            if not compress_result['success']:
                return {
                    'engine': engine,
                    'file_format': file_format,
                    'reversibility_status': 'ERROR',
                    'error': compress_result.get('error', 'Compression failed'),
                    'compression_time': compress_time
                }
            
            compressed_file = compress_result['output_file']
            compressed_size = os.path.getsize(compressed_file)
            compression_ratio = (1 - compressed_size / original_size) * 100
            
            # 解凍試行
            decompress_start = time.time()
            decompress_result = self._attempt_decompression(compressed_file, engine)
            decompress_time = time.time() - decompress_start
            
            if not decompress_result['success']:
                return {
                    'engine': engine,
                    'file_format': file_format,
                    'reversibility_status': 'FAILED',
                    'compression_ratio': compression_ratio,
                    'compression_time': compress_time,
                    'decompression_error': decompress_result.get('error', 'Unknown'),
                    'decompression_time': decompress_time
                }
            
            # 可逆性検証
            restored_file = decompress_result['restored_file']
            with open(restored_file, 'rb') as f:
                restored_data = f.read()
            
            restored_hash = hashlib.sha256(restored_data).hexdigest()
            size_match = len(original_data) == len(restored_data)
            hash_match = original_hash == restored_hash
            byte_match = original_data == restored_data
            
            # 可逆性判定
            if size_match and hash_match and byte_match:
                reversibility = 'PERFECT'
            elif size_match:
                reversibility = 'PARTIAL'
            else:
                reversibility = 'FAILED'
            
            # クリーンアップ
            try:
                os.remove(compressed_file)
                os.remove(restored_file)
            except:
                pass
            
            return {
                'engine': engine,
                'file_format': file_format,
                'reversibility_status': reversibility,
                'compression_ratio': compression_ratio,
                'original_size': original_size,
                'compressed_size': compressed_size,
                'restored_size': len(restored_data),
                'size_match': size_match,
                'hash_match': hash_match,
                'byte_match': byte_match,
                'compression_time': compress_time,
                'decompression_time': decompress_time
            }
            
        except Exception as e:
            return {
                'engine': engine,
                'file_format': file_format,
                'reversibility_status': 'ERROR',
                'error': str(e)
            }
    
    def _run_compression(self, engine: str, file_path: str) -> dict:
        """圧縮実行"""
        try:
            # エンジンに応じたコマンド実行
            if 'test' in engine or 'compress' in engine:
                # 多くのエンジンはtestコマンドを持つ
                cmd = ['python', engine, 'test']
            else:
                # または直接実行
                cmd = ['python', engine]
            
            # タイムアウト付き実行
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=60,  # 1分タイムアウト
                cwd='.'
            )
            
            if result.returncode != 0:
                return {
                    'success': False,
                    'error': f"Exit code {result.returncode}: {result.stderr[:200]}"
                }
            
            # 出力ファイル検索
            base_name = Path(file_path).stem
            possible_outputs = [
                f"{base_name}.nxz",
                f"{base_name}.compressed",
                f"{Path(file_path).parent}/{base_name}.nxz"
            ]
            
            for output_path in possible_outputs:
                if os.path.exists(output_path):
                    return {
                        'success': True,
                        'output_file': output_path
                    }
            
            # サンプルディレクトリも確認
            sample_dir = r"c:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\sample"
            for output_path in possible_outputs:
                full_path = os.path.join(sample_dir, Path(output_path).name)
                if os.path.exists(full_path):
                    return {
                        'success': True,
                        'output_file': full_path
                    }
            
            return {
                'success': False,
                'error': 'Compressed file not found'
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Compression timeout'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _attempt_decompression(self, compressed_file: str, engine: str) -> dict:
        """解凍試行"""
        try:
            # 複数の解凍方法を試行
            decompression_methods = [
                # 専用デcompressor
                ['python', 'optimal_decompressor.py', 'decompress', compressed_file],
                ['python', 'perfect_decompressor.py', 'decompress', compressed_file],
                # エンジンの解凍機能
                ['python', engine, 'decompress', compressed_file],
                ['python', engine, 'extract', compressed_file],
            ]
            
            for method in decompression_methods:
                try:
                    result = subprocess.run(
                        method,
                        capture_output=True,
                        text=True,
                        timeout=30,
                        cwd='.'
                    )
                    
                    if result.returncode == 0:
                        # 復元ファイル検索
                        possible_restored = [
                            compressed_file.replace('.nxz', '.restored.mp4'),
                            compressed_file.replace('.nxz', '.restored.txt'),
                            compressed_file.replace('.nxz', '.restored.mp3'),
                            compressed_file.replace('.nxz', '.restored.png'),
                            compressed_file.replace('.nxz', '.decompressed'),
                        ]
                        
                        for restored_path in possible_restored:
                            if os.path.exists(restored_path):
                                return {
                                    'success': True,
                                    'restored_file': restored_path
                                }
                
                except:
                    continue
            
            return {
                'success': False,
                'error': 'All decompression methods failed'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _analyze_audit_results(self, results: list) -> dict:
        """監査結果分析"""
        analysis = {
            'perfect_count': 0,
            'partial_count': 0,
            'failed_count': 0,
            'error_count': 0,
            'by_engine': {},
            'by_format': {},
            'suspicious_patterns': []
        }
        
        for result in results:
            status = result.get('reversibility_status', 'UNKNOWN')
            engine = result.get('engine', 'Unknown')
            file_format = result.get('file_format', 'Unknown')
            
            # 全体統計
            if status == 'PERFECT':
                analysis['perfect_count'] += 1
            elif status == 'PARTIAL':
                analysis['partial_count'] += 1
            elif status == 'FAILED':
                analysis['failed_count'] += 1
            else:
                analysis['error_count'] += 1
            
            # エンジン別統計
            if engine not in analysis['by_engine']:
                analysis['by_engine'][engine] = {
                    'perfect': 0, 'partial': 0, 'failed': 0, 'error': 0
                }
            
            if status == 'PERFECT':
                analysis['by_engine'][engine]['perfect'] += 1
            elif status == 'PARTIAL':
                analysis['by_engine'][engine]['partial'] += 1
            elif status == 'FAILED':
                analysis['by_engine'][engine]['failed'] += 1
            else:
                analysis['by_engine'][engine]['error'] += 1
            
            # 形式別統計
            if file_format not in analysis['by_format']:
                analysis['by_format'][file_format] = {
                    'perfect': 0, 'partial': 0, 'failed': 0, 'error': 0
                }
            
            if status == 'PERFECT':
                analysis['by_format'][file_format]['perfect'] += 1
            elif status == 'PARTIAL':
                analysis['by_format'][file_format]['partial'] += 1
            elif status == 'FAILED':
                analysis['by_format'][file_format]['failed'] += 1
            else:
                analysis['by_format'][file_format]['error'] += 1
        
        return analysis
    
    def _identify_suspicious_engines(self, results: list):
        """疑わしいエンジン特定"""
        self.suspicious_engines = []
        
        engine_stats = {}
        for result in results:
            engine = result.get('engine', 'Unknown')
            status = result.get('reversibility_status', 'UNKNOWN')
            compression = result.get('compression_ratio', 0)
            
            if engine not in engine_stats:
                engine_stats[engine] = {
                    'total_tests': 0,
                    'failed_tests': 0,
                    'high_compression_failures': 0,
                    'max_compression': 0
                }
            
            engine_stats[engine]['total_tests'] += 1
            engine_stats[engine]['max_compression'] = max(
                engine_stats[engine]['max_compression'], compression
            )
            
            if status in ['FAILED', 'PARTIAL']:
                engine_stats[engine]['failed_tests'] += 1
                
                # 高圧縮率なのに可逆性失敗は疑わしい
                if compression > 50:
                    engine_stats[engine]['high_compression_failures'] += 1
        
        # 疑わしいパターン検出
        for engine, stats in engine_stats.items():
            failure_rate = stats['failed_tests'] / stats['total_tests'] if stats['total_tests'] > 0 else 0
            
            # 疑わしい条件
            if failure_rate > 0.5:  # 失敗率50%以上
                self.suspicious_engines.append(f"{engine} (失敗率: {failure_rate*100:.1f}%)")
            elif stats['high_compression_failures'] > 0:  # 高圧縮で可逆性失敗
                self.suspicious_engines.append(f"{engine} (高圧縮非可逆)")
    
    def _certify_verified_engines(self, results: list):
        """検証済みエンジン認定"""
        self.verified_engines = []
        
        engine_stats = {}
        for result in results:
            engine = result.get('engine', 'Unknown')
            status = result.get('reversibility_status', 'UNKNOWN')
            
            if engine not in engine_stats:
                engine_stats[engine] = {'total': 0, 'perfect': 0}
            
            engine_stats[engine]['total'] += 1
            if status == 'PERFECT':
                engine_stats[engine]['perfect'] += 1
        
        # 完全可逆エンジン認定
        for engine, stats in engine_stats.items():
            if stats['total'] > 0 and stats['perfect'] == stats['total']:
                self.verified_engines.append(f"{engine} (100%可逆)")

def main():
    """メイン関数"""
    if len(sys.argv) < 2:
        print("🧪 Comprehensive Reversibility Auditor")
        print("使用方法:")
        print("  python comprehensive_reversibility_audit.py audit    # 包括的可逆性監査")
        return
    
    command = sys.argv[1].lower()
    
    if command == "audit":
        auditor = ComprehensiveReversibilityAuditor()
        result = auditor.run_comprehensive_audit()
        
        if result['success']:
            print("\n🎉 包括的可逆性監査完了!")
            
            # 結果ファイル保存
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            report_file = f"reversibility_audit_report_{timestamp}.json"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"📊 詳細レポート保存: {report_file}")
        else:
            print("❌ 監査失敗")
    else:
        print("❌ 無効なコマンドです")

if __name__ == "__main__":
    main()
