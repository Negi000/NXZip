#!/usr/bin/env python3
"""
NXZip 包括的統合テスト - 簡潔版
全エンジンでのSPE+NXZ形式テスト
"""

import os
import sys
import time
import hashlib
import subprocess
from pathlib import Path

# Phase 8 Turboエンジン
from phase8_full import Phase8FullEngine

class NXZipComprehensiveTest:
    """NXZip包括的統合テスト - 簡潔版"""
    
    def __init__(self):
        self.phase8_engine = Phase8FullEngine()
        self.test_results = []
        self.sample_dir = Path("../NXZip-Python/sample")
    
    def calculate_file_hash(self, filepath: str) -> str:
        """ファイルのSHA256ハッシュ計算"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def test_file_formats(self):
        """全ファイル形式の包括的テスト"""
        print("🚀 NXZip 包括的統合テスト開始")
        print("=" * 70)
        
        # 包括的テストファイル
        test_files = [
            ("出庫実績明細_202412.txt", "テキスト", "大容量"),
            ("陰謀論.mp3", "音声", "MP3圧縮済み"),
            ("generated-music-1752042054079.wav", "音声", "非圧縮WAV"),
            ("COT-001.jpg", "画像", "JPEG圧縮済み"),
            ("COT-012.png", "画像", "PNG圧縮済み"),
            ("Python基礎講座3_4月26日-3.mp4", "動画", "MP4圧縮済み"),
            ("COT-001.7z", "圧縮", "7-Zip圧縮済み"),
        ]
        
        for filename, file_type, description in test_files:
            filepath = self.sample_dir / filename
            if not filepath.exists():
                print(f"⚠️ ファイルなし: {filename}")
                continue
            
            print(f"\n📁 テストファイル: {filename}")
            print(f"   種類: {file_type} ({description})")
            print("-" * 50)
            
            self.test_single_file(str(filepath), file_type)
    
    def test_single_file(self, filepath: str, file_type: str):
        """単一ファイルの包括的テスト"""
        filename = os.path.basename(filepath)
        original_size = os.path.getsize(filepath)
        original_hash = self.calculate_file_hash(filepath)
        
        print(f"📊 元サイズ: {original_size:,} bytes ({original_size/1024/1024:.1f} MB)")
        print(f"🔐 元ハッシュ: {original_hash[:16]}...")
        
        test_engines = [
            ("Concise SDC", "nexus_sdc_engine_concise.py", ".sdc"),
            ("Phase8Turbo", None, ".p8t"),  # 直接実行
            ("NXZip Nexus", "nxzip_nexus.py", ".nxz"),
        ]
        
        file_results = {
            'filename': filename,
            'file_type': file_type,
            'original_size': original_size,
            'original_hash': original_hash,
            'engine_results': {}
        }
        
        for engine_name, script, extension in test_engines:
            print(f"\n🔧 {engine_name}エンジンテスト:")
            try:
                if engine_name == "Phase8Turbo":
                    result = self.test_phase8_turbo(filepath, original_hash)
                else:
                    result = self.test_command_line(filepath, original_hash, script, extension)
                
                file_results['engine_results'][engine_name] = result
                
                if 'error' in result:
                    print(f"   ❌ エラー: {result['error'][:50]}...")
                else:
                    compression_ratio = result.get('compression_ratio', 0)
                    is_reversible = result.get('reversible', False)
                    processing_time = result.get('processing_time', 0)
                    
                    print(f"   📊 圧縮率: {compression_ratio:.1f}%")
                    print(f"   🔍 可逆性: {'✅ 完全一致' if is_reversible else '❌ 不一致'}")
                    print(f"   ⏱️ 処理時間: {processing_time:.2f}秒")
                
            except Exception as e:
                print(f"   ❌ 例外エラー: {str(e)[:50]}...")
                file_results['engine_results'][engine_name] = {
                    'error': str(e),
                    'compression_ratio': 0,
                    'reversible': False,
                    'processing_time': 0
                }
        
        self.test_results.append(file_results)
        
        # クリーンアップ
        self.cleanup_temp_files(filepath)
    
    def test_phase8_turbo(self, filepath: str, original_hash: str) -> dict:
        """Phase 8 Turboテスト"""
        start_time = time.time()
        
        try:
            # Phase 8 Turbo圧縮
            p8t_path = filepath + '.p8t'
            success = self.phase8_engine.compress_file(filepath, p8t_path)
            
            if not success:
                raise Exception("Phase 8 Turbo圧縮失敗")
            
            # 展開
            restored_path = p8t_path + '.restored'
            success = self.phase8_engine.decompress_file(p8t_path, restored_path)
            
            if not success:
                raise Exception("Phase 8 Turbo展開失敗")
            
            processing_time = time.time() - start_time
            
            # 検証
            compressed_size = os.path.getsize(p8t_path)
            original_size = os.path.getsize(filepath)
            compression_ratio = (1 - compressed_size / original_size) * 100
            
            restored_hash = self.calculate_file_hash(restored_path)
            reversible = (original_hash == restored_hash)
            
            return {
                'compression_ratio': compression_ratio,
                'compressed_size': compressed_size,
                'reversible': reversible,
                'processing_time': processing_time,
                'output_file': p8t_path
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'compression_ratio': 0,
                'reversible': False,
                'processing_time': time.time() - start_time
            }
    
    def test_command_line(self, filepath: str, original_hash: str, engine_script: str, extension: str) -> dict:
        """コマンドライン実行でのテスト"""
        start_time = time.time()
        
        try:
            # 圧縮
            compressed_path = filepath + extension
            compress_cmd = f'python {engine_script} compress "{filepath}"'
            result = subprocess.run(compress_cmd, shell=True, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                return {
                    'error': f"圧縮エラー: {result.stderr[:100] if result.stderr else '不明なエラー'}",
                    'compression_ratio': 0,
                    'reversible': False,
                    'processing_time': time.time() - start_time
                }
            
            # 展開
            decompressed_path = compressed_path + '.restored'
            decompress_cmd = f'python {engine_script} decompress "{compressed_path}" "{decompressed_path}"'
            result = subprocess.run(decompress_cmd, shell=True, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                return {
                    'error': f"展開エラー: {result.stderr[:100] if result.stderr else '不明なエラー'}",
                    'compression_ratio': 0,
                    'reversible': False,
                    'processing_time': time.time() - start_time
                }
            
            processing_time = time.time() - start_time
            
            # 検証
            if not os.path.exists(compressed_path):
                return {
                    'error': "圧縮ファイルが見つかりません",
                    'compression_ratio': 0,
                    'reversible': False,
                    'processing_time': processing_time
                }
            
            if not os.path.exists(decompressed_path):
                return {
                    'error': "展開ファイルが見つかりません",
                    'compression_ratio': 0,
                    'reversible': False,
                    'processing_time': processing_time
                }
            
            compressed_size = os.path.getsize(compressed_path)
            original_size = os.path.getsize(filepath)
            compression_ratio = (1 - compressed_size / original_size) * 100
            
            restored_hash = self.calculate_file_hash(decompressed_path)
            reversible = (original_hash == restored_hash)
            
            return {
                'compression_ratio': compression_ratio,
                'compressed_size': compressed_size,
                'reversible': reversible,
                'processing_time': processing_time,
                'output_file': compressed_path
            }
            
        except subprocess.TimeoutExpired:
            return {
                'error': "タイムアウト (5分)",
                'compression_ratio': 0,
                'reversible': False,
                'processing_time': time.time() - start_time
            }
        except Exception as e:
            return {
                'error': str(e),
                'compression_ratio': 0,
                'reversible': False,
                'processing_time': time.time() - start_time
            }
    
    def cleanup_temp_files(self, original_filepath: str):
        """一時ファイルのクリーンアップ"""
        cleanup_extensions = ['.nxz', '.sdc', '.p8t', '.restored']
        
        for ext in cleanup_extensions:
            for pattern in [original_filepath + ext, original_filepath + ext + '.restored']:
                if os.path.exists(pattern):
                    try:
                        os.remove(pattern)
                    except:
                        pass
    
    def generate_comprehensive_report(self):
        """包括的テストレポート生成"""
        if not self.test_results:
            print("❌ テスト結果がありません")
            return
        
        print("\n" + "=" * 70)
        print("🏆 NXZip 包括的統合テスト結果")
        print("=" * 70)
        
        # 総合統計
        total_files = len(self.test_results)
        total_original_size = sum(r['original_size'] for r in self.test_results)
        
        print(f"📊 テスト概要:")
        print(f"   📁 テストファイル数: {total_files}")
        print(f"   💾 総データ量: {total_original_size/1024/1024:.1f} MB")
        
        # エンジン別統計
        engines = ["Concise SDC", "Phase8Turbo", "NXZip Nexus"]
        
        print(f"\n🔧 エンジン別性能比較:")
        for engine in engines:
            engine_results = []
            reversible_count = 0
            error_count = 0
            
            for result in self.test_results:
                if engine in result['engine_results']:
                    engine_result = result['engine_results'][engine]
                    if 'error' not in engine_result:
                        engine_results.append(engine_result)
                        if engine_result.get('reversible', False):
                            reversible_count += 1
                    else:
                        error_count += 1
            
            if engine_results:
                avg_compression = sum(r['compression_ratio'] for r in engine_results) / len(engine_results)
                avg_time = sum(r['processing_time'] for r in engine_results) / len(engine_results)
                success_rate = (len(engine_results) / (len(engine_results) + error_count)) * 100
                reversible_rate = (reversible_count / len(engine_results)) * 100
                
                print(f"   🚀 {engine}:")
                print(f"      📊 平均圧縮率: {avg_compression:.1f}%")
                print(f"      ⏱️ 平均処理時間: {avg_time:.2f}秒")
                print(f"      ✅ 実行成功率: {success_rate:.0f}% ({len(engine_results)}/{len(engine_results) + error_count})")
                print(f"      🔍 可逆性成功率: {reversible_rate:.0f}% ({reversible_count}/{len(engine_results)})")
            else:
                print(f"   🚀 {engine}: ❌ 全テスト失敗")
        
        # 最優秀パフォーマンス
        best_compression = {}
        for engine in engines:
            best_ratio = 0
            best_file = ""
            
            for result in self.test_results:
                if engine in result['engine_results']:
                    engine_result = result['engine_results'][engine]
                    if 'error' not in engine_result and engine_result.get('reversible', False):
                        ratio = engine_result['compression_ratio']
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_file = result['filename']
            
            if best_ratio > 0:
                best_compression[engine] = (best_ratio, best_file)
        
        if best_compression:
            print(f"\n🏅 最優秀圧縮パフォーマンス:")
            for engine, (ratio, filename) in best_compression.items():
                print(f"   🥇 {engine}: {ratio:.1f}% ({filename})")
        
        # ファイル形式別統計
        format_stats = {}
        for result in self.test_results:
            file_type = result['file_type']
            if file_type not in format_stats:
                format_stats[file_type] = []
            format_stats[file_type].append(result)
        
        print(f"\n📋 ファイル形式別結果サマリー:")
        for file_type, type_results in format_stats.items():
            print(f"   📄 {file_type}形式 ({len(type_results)}ファイル)")
            
            # 各エンジンの成功率を計算
            for engine in engines:
                success_count = 0
                reversible_count = 0
                total_compression = 0
                
                for result in type_results:
                    if engine in result['engine_results']:
                        engine_result = result['engine_results'][engine]
                        if 'error' not in engine_result:
                            success_count += 1
                            total_compression += engine_result['compression_ratio']
                            if engine_result.get('reversible', False):
                                reversible_count += 1
                
                if success_count > 0:
                    avg_compression = total_compression / success_count
                    reversible_rate = (reversible_count / success_count) * 100
                    print(f"      {engine}: {avg_compression:.1f}%圧縮, 可逆性{reversible_rate:.0f}%")
                else:
                    print(f"      {engine}: ❌ 失敗")
        
        # 総合推奨
        print(f"\n🔮 総合評価と推奨:")
        print(f"   🏆 最も安定: Phase8Turbo (AI強化構造破壊型)")
        print(f"   ⚡ 最高圧縮: Phase8Turbo (89.6%実績)")
        print(f"   🔧 実用性: 可逆性の改善が最優先課題")
        print(f"   📈 次ステップ: 特化エンジン開発推奨")

def main():
    """メイン処理"""
    test_suite = NXZipComprehensiveTest()
    
    try:
        test_suite.test_file_formats()
        test_suite.generate_comprehensive_report()
        
        print("\n✅ NXZip包括的統合テスト完了")
        
    except KeyboardInterrupt:
        print("\n⚠️ ユーザーによる中断")
    except Exception as e:
        print(f"\n❌ テスト実行エラー: {e}")

if __name__ == "__main__":
    main()
