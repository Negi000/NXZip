#!/usr/bin/env python3
"""
NXZip 包括的統合テスト
SPE + NXZ形式 + Phase 8 Turbo 完全テスト
"""

import os
import sys
import time
import hashlib
from pathlib import Path

# NXZip統合テスト - 既存エンジン使用
sys.path.append('.')
# from nexus_sdc_engine import NexusSDCEngine  # エラーがあるためコメントアウト
from nexus_sdc_engine_concise import ConciseSDCEngine

# Phase 8 Turboエンジン
from phase8_full import P        # 技術的推奨事項
        print(f"\n🔮 技術的推奨事項:")
        print(f"   🔧 Concise SDC: 構造破壊型圧縮の安定実装")
        print(f"   ⚡ Phase8Turbo: AI強化による高圧縮率実現")
        print(f"   📦 NXZip Nexus: 従来エンジンとの互換性")
        print(f"   📈 特化最適化: ファイル形式別エンジン推奨")llEngine

class NXZipComprehensiveTest:
    """NXZip包括的統合テスト"""
    
    def __init__(self):
        self.concise_engine = ConciseSDCEngine()
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
            # テキストファイル
            ("出庫実績明細_202412.txt", "テキスト", "大容量"),
            
            # 音声ファイル
            ("陰謀論.mp3", "音声", "MP3圧縮済み"),
            ("generated-music-1752042054079.wav", "音声", "非圧縮WAV"),
            
            # 画像ファイル
            ("COT-001.jpg", "画像", "JPEG圧縮済み"),
            ("COT-012.png", "画像", "PNG圧縮済み"),
            
            # 動画ファイル
            ("Python基礎講座3_4月26日-3.mp4", "動画", "MP4圧縮済み"),
            
            # 圧縮済みファイル
            ("COT-001.7z", "圧縮", "7-Zip圧縮済み"),
            ("Python基礎講座3_4月26日-3.7z", "圧縮", "7-Zip圧縮済み"),
        ]
        
        for filename, file_type, description in test_files:
            filepath = self.sample_dir / filename
            if not filepath.exists():
                print(f"⚠️ ファイルなし: {filename}")
                continue
            
            print(f"\n📁 テストファイル: {filename}")
            print(f"   種類: {file_type} ({description})")
            print("-" * 50)
            
            # 複数エンジンでテスト
            self.test_single_file(str(filepath), file_type)
    
    def test_single_file(self, filepath: str, file_type: str):
        """単一ファイルの包括的テスト"""
        filename = os.path.basename(filepath)
        original_size = os.path.getsize(filepath)
        original_hash = self.calculate_file_hash(filepath)
        
        print(f"📊 元サイズ: {original_size:,} bytes ({original_size/1024/1024:.1f} MB)")
        print(f"🔐 元ハッシュ: {original_hash[:16]}...")
        
        test_engines = [
            ("Concise SDC", self.test_concise_sdc), 
            ("Phase8Turbo", self.test_phase8_turbo),
            ("NXZip Nexus", self.test_nxzip_nexus),
        ]
        
        file_results = {
            'filename': filename,
            'file_type': file_type,
            'original_size': original_size,
            'original_hash': original_hash,
            'engine_results': {}
        }
        
        for engine_name, test_func in test_engines:
            print(f"\n🔧 {engine_name}エンジンテスト:")
            try:
                result = test_func(filepath, original_hash)
                file_results['engine_results'][engine_name] = result
                
                compression_ratio = result.get('compression_ratio', 0)
                is_reversible = result.get('reversible', False)
                processing_time = result.get('processing_time', 0)
                
                print(f"   📊 圧縮率: {compression_ratio:.1f}%")
                print(f"   🔍 可逆性: {'✅ 完全一致' if is_reversible else '❌ 不一致'}")
                print(f"   ⏱️ 処理時間: {processing_time:.2f}秒")
                
            except Exception as e:
                print(f"   ❌ エラー: {str(e)[:50]}...")
                file_results['engine_results'][engine_name] = {
                    'error': str(e),
                    'compression_ratio': 0,
                    'reversible': False,
                    'processing_time': 0
                }
        
        self.test_results.append(file_results)
        
        # クリーンアップ
        self.cleanup_temp_files(filepath)
    
    def test_concise_sdc(self, filepath: str, original_hash: str) -> dict:
        """Concise SDCエンジンテスト"""
        start_time = time.time()
        
        try:
            # Concise SDC圧縮 (.csdc形式)
            compressed_path = filepath + '.csdc'
            self.concise_engine.compress_file(filepath, compressed_path)
            
            # 展開
            decompressed_path = compressed_path + '.restored'
            self.concise_engine.decompress_file(compressed_path, decompressed_path)
            
            processing_time = time.time() - start_time
            
            # 検証
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
        except Exception as e:
            # コマンドライン実行でフォールバック
            return self.test_command_line(filepath, original_hash, "nexus_sdc_engine_concise.py", ".csdc")
    
    def test_nxzip_nexus(self, filepath: str, original_hash: str) -> dict:
        """NXZip Nexusエンジンテスト"""
        return self.test_command_line(filepath, original_hash, "nxzip_nexus.py", ".nxz")
    
    def test_command_line(self, filepath: str, original_hash: str, engine_script: str, extension: str) -> dict:
        """コマンドライン実行でのテスト"""
        import subprocess
        
        start_time = time.time()
        
        try:
            # 圧縮
            compressed_path = filepath + extension
            compress_cmd = f"python {engine_script} compress \"{filepath}\""
            result = subprocess.run(compress_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"圧縮エラー: {result.stderr}")
            
            # 展開
            decompressed_path = compressed_path + '.restored'
            decompress_cmd = f"python {engine_script} decompress \"{compressed_path}\" \"{decompressed_path}\""
            result = subprocess.run(decompress_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"展開エラー: {result.stderr}")
            
            processing_time = time.time() - start_time
            
            # 検証
            if not os.path.exists(compressed_path):
                raise Exception("圧縮ファイルが見つかりません")
            
            if not os.path.exists(decompressed_path):
                raise Exception("展開ファイルが見つかりません")
            
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
            
        except Exception as e:
            return {
                'error': str(e),
                'compression_ratio': 0,
                'reversible': False,
                'processing_time': time.time() - start_time
            }
    
    def test_phase8_turbo(self, filepath: str, original_hash: str) -> dict:
        """Phase 8 Turboテスト"""
        start_time = time.time()
        
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
    
    def cleanup_temp_files(self, original_filepath: str):
        """一時ファイルのクリーンアップ"""
        cleanup_extensions = ['.nxz', '.sdc', '.csdc', '.p8t', '.restored']
        
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
            
            for result in self.test_results:
                if engine in result['engine_results']:
                    engine_result = result['engine_results'][engine]
                    if 'error' not in engine_result:
                        engine_results.append(engine_result)
                        if engine_result.get('reversible', False):
                            reversible_count += 1
            
            if engine_results:
                avg_compression = sum(r['compression_ratio'] for r in engine_results) / len(engine_results)
                avg_time = sum(r['processing_time'] for r in engine_results) / len(engine_results)
                reversible_rate = (reversible_count / len(engine_results)) * 100
                
                print(f"   🚀 {engine}:")
                print(f"      📊 平均圧縮率: {avg_compression:.1f}%")
                print(f"      ⏱️ 平均処理時間: {avg_time:.2f}秒")
                print(f"      🔍 可逆性成功率: {reversible_rate:.0f}% ({reversible_count}/{len(engine_results)})")
        
        # ファイル形式別分析
        format_analysis = {}
        for result in self.test_results:
            file_type = result['file_type']
            if file_type not in format_analysis:
                format_analysis[file_type] = []
            format_analysis[file_type].append(result)
        
        print(f"\n📋 ファイル形式別詳細分析:")
        for file_type, type_results in format_analysis.items():
            print(f"   📄 {file_type}形式 ({len(type_results)}ファイル):")
            
            for result in type_results:
                filename = result['filename'][:25] + ('...' if len(result['filename']) > 25 else '')
                size_mb = result['original_size'] / 1024 / 1024
                
                print(f"      • {filename} ({size_mb:.1f}MB):")
                
                for engine in engines:
                    if engine in result['engine_results']:
                        engine_result = result['engine_results'][engine]
                        if 'error' not in engine_result:
                            ratio = engine_result['compression_ratio']
                            reversible_icon = '✅' if engine_result['reversible'] else '❌'
                            print(f"        {engine}: {ratio:.1f}% {reversible_icon}")
                        else:
                            print(f"        {engine}: ❌ エラー")
        
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
        
        # 推奨改善アクション
        low_performance_files = []
        for result in self.test_results:
            max_compression = 0
            for engine in engines:
                if engine in result['engine_results']:
                    engine_result = result['engine_results'][engine]
                    if 'error' not in engine_result:
                        max_compression = max(max_compression, engine_result['compression_ratio'])
            
            if max_compression < 10:  # 10%未満の圧縮率
                low_performance_files.append((result['filename'], max_compression))
        
        if low_performance_files:
            print(f"\n⚠️ 改善要検討ファイル ({len(low_performance_files)}個):")
            for filename, best_ratio in low_performance_files:
                file_ext = filename.split('.')[-1].upper()
                print(f"   • {filename}: 最高{best_ratio:.1f}% (要{file_ext}特化最適化)")
        
        # 技術的推奨事項
        print(f"\n🔮 技術的推奨事項:")
        print(f"   🎯 Nexus SDC: 構造破壊型圧縮の基盤実装")
        print(f"   ⚡ Phase8Turbo: AI強化による高圧縮率実現")
        print(f"   🔧 Concise SDC: 簡潔版で安定性重視")
        print(f"   � NXZip Nexus: 従来エンジンとの互換性")
        print(f"   📈 特化最適化: ファイル形式別エンジン推奨")

def main():
    """メイン処理"""
    test_suite = NXZipComprehensiveTest()
    
    try:
        test_suite.test_file_formats()
        test_suite.generate_comprehensive_report()
        
        print("\n✅ NXZip包括的統合テスト完了")
        
    except Exception as e:
        print(f"\n❌ テスト実行エラー: {e}")

if __name__ == "__main__":
    main()
