#!/usr/bin/env python3
"""
NEXUS vs 競合ベンチマーク - NXZ統合版
TMC + SPE + NXZ vs 7Z + Zstandard 完全比較テスト
"""

import os
import sys
import time
import subprocess
import tempfile
import shutil
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import statistics

# NEXUS SPE Integrated Engine インポート
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from nexus_spe_integrated_engine import NEXUSSPEIntegratedEngine, NXZFormat
    NEXUS_AVAILABLE = True
except ImportError:
    print("⚠️ NEXUS SPE Integrated Engineが利用できません")
    NEXUS_AVAILABLE = False

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    print("⚠️ Zstandard ライブラリが利用できません")
    ZSTD_AVAILABLE = False

try:
    import lzma
    LZMA_AVAILABLE = True
except ImportError:
    print("⚠️ LZMA ライブラリが利用できません")
    LZMA_AVAILABLE = False


class NEXUSCompetitor:
    """NEXUS SPE統合版競合テスト"""
    
    def __init__(self):
        self.name = "NEXUS-SPE-NXZ"
        if NEXUS_AVAILABLE:
            self.engine = NEXUSSPEIntegratedEngine(max_workers=4, encryption_enabled=True)
        else:
            self.engine = None
    
    def compress(self, data: bytes, level: int = 6, password: str = None) -> Tuple[bytes, Dict[str, Any]]:
        """NXZ形式で圧縮"""
        if not self.engine:
            return data, {'error': 'engine_not_available'}
        
        start_time = time.perf_counter()
        
        try:
            # メタデータ設定
            metadata = {
                'compression_level': level,
                'nexus_version': 'SPE_Integrated_v2',
                'format': 'NXZ'
            }
            
            # NXZ圧縮実行
            compressed_data, compression_info = self.engine.compress_to_nxz(
                data, password=password, metadata=metadata
            )
            
            processing_time = time.perf_counter() - start_time
            
            result_info = {
                'original_size': len(data),
                'compressed_size': len(compressed_data),
                'compression_ratio': compression_info.get('total_compression_ratio', 0),
                'processing_time': processing_time,
                'throughput_mb_s': (len(data) / 1024 / 1024) / processing_time if processing_time > 0 else 0,
                'encrypted': compression_info.get('encrypted', False),
                'format': 'nxz',
                'nexus_info': compression_info
            }
            
            return compressed_data, result_info
            
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            return data, {
                'error': str(e),
                'processing_time': processing_time,
                'original_size': len(data)
            }
    
    def decompress(self, data: bytes, password: str = None) -> Tuple[bytes, Dict[str, Any]]:
        """NXZから展開（部分実装）"""
        if not self.engine:
            return data, {'error': 'engine_not_available'}
        
        start_time = time.perf_counter()
        
        try:
            decompressed_data, decomp_info = self.engine.decompress_from_nxz(data, password)
            processing_time = time.perf_counter() - start_time
            
            decomp_info['processing_time'] = processing_time
            return decompressed_data, decomp_info
            
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            return data, {
                'error': str(e),
                'processing_time': processing_time
            }


class ZstdCompetitor:
    """Zstandard競合クラス"""
    
    def __init__(self):
        self.name = "Zstandard"
        self.available = ZSTD_AVAILABLE
    
    def compress(self, data: bytes, level: int = 6, password: str = None) -> Tuple[bytes, Dict[str, Any]]:
        """Zstd圧縮"""
        if not self.available:
            return data, {'error': 'zstd_not_available'}
        
        start_time = time.perf_counter()
        
        try:
            cctx = zstd.ZstdCompressor(level=level)
            compressed_data = cctx.compress(data)
            
            processing_time = time.perf_counter() - start_time
            
            return compressed_data, {
                'original_size': len(data),
                'compressed_size': len(compressed_data),
                'compression_ratio': (1 - len(compressed_data) / len(data)) * 100 if len(data) > 0 else 0,
                'processing_time': processing_time,
                'throughput_mb_s': (len(data) / 1024 / 1024) / processing_time if processing_time > 0 else 0,
                'level': level,
                'encrypted': bool(password),  # 注意: Zstdは暗号化未対応
                'format': 'zstd'
            }
            
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            return data, {
                'error': str(e),
                'processing_time': processing_time,
                'original_size': len(data)
            }
    
    def decompress(self, data: bytes, password: str = None) -> Tuple[bytes, Dict[str, Any]]:
        """Zstd展開"""
        if not self.available:
            return data, {'error': 'zstd_not_available'}
        
        start_time = time.perf_counter()
        
        try:
            dctx = zstd.ZstdDecompressor()
            decompressed_data = dctx.decompress(data)
            
            processing_time = time.perf_counter() - start_time
            
            return decompressed_data, {
                'decompressed_size': len(decompressed_data),
                'processing_time': processing_time,
                'throughput_mb_s': (len(decompressed_data) / 1024 / 1024) / processing_time if processing_time > 0 else 0,
                'format': 'zstd'
            }
            
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            return data, {
                'error': str(e),
                'processing_time': processing_time
            }


class SevenZipCompetitor:
    """7-Zip競合クラス（外部プロセス）"""
    
    def __init__(self):
        self.name = "7-Zip"
        self.available = self._check_7zip_availability()
    
    def _check_7zip_availability(self) -> bool:
        """7-Zip利用可能性チェック"""
        try:
            # 一般的な7-Zipパス
            possible_paths = [
                "7z",
                "7za", 
                r"C:\Program Files\7-Zip\7z.exe",
                r"C:\Program Files (x86)\7-Zip\7z.exe"
            ]
            
            for path in possible_paths:
                try:
                    result = subprocess.run([path], 
                                          capture_output=True, timeout=5)
                    if result.returncode == 0 or "Usage:" in result.stdout.decode():
                        self.executable = path
                        return True
                except:
                    continue
            
            return False
            
        except Exception:
            return False
    
    def compress(self, data: bytes, level: int = 5, password: str = None) -> Tuple[bytes, Dict[str, Any]]:
        """7-Zip圧縮"""
        if not self.available:
            return data, {'error': '7zip_not_available'}
        
        start_time = time.perf_counter()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                input_file = Path(temp_dir) / "input.bin"
                output_file = Path(temp_dir) / "output.7z"
                
                # 入力ファイル作成
                input_file.write_bytes(data)
                
                # 7-Zipコマンド構築
                cmd = [self.executable, "a", "-t7z", f"-mx={level}", str(output_file), str(input_file)]
                
                if password:
                    cmd.extend([f"-p{password}"])
                
                # 圧縮実行
                result = subprocess.run(cmd, capture_output=True, timeout=60)
                
                if result.returncode == 0 and output_file.exists():
                    compressed_data = output_file.read_bytes()
                    processing_time = time.perf_counter() - start_time
                    
                    return compressed_data, {
                        'original_size': len(data),
                        'compressed_size': len(compressed_data),
                        'compression_ratio': (1 - len(compressed_data) / len(data)) * 100 if len(data) > 0 else 0,
                        'processing_time': processing_time,
                        'throughput_mb_s': (len(data) / 1024 / 1024) / processing_time if processing_time > 0 else 0,
                        'level': level,
                        'encrypted': bool(password),
                        'format': '7z'
                    }
                else:
                    processing_time = time.perf_counter() - start_time
                    return data, {
                        'error': f'7zip_failed: {result.stderr.decode()}',
                        'processing_time': processing_time,
                        'original_size': len(data)
                    }
                    
            except Exception as e:
                processing_time = time.perf_counter() - start_time
                return data, {
                    'error': str(e),
                    'processing_time': processing_time,
                    'original_size': len(data)
                }


class LZMACompetitor:
    """LZMA競合クラス（フォールバック）"""
    
    def __init__(self):
        self.name = "LZMA"
        self.available = LZMA_AVAILABLE
    
    def compress(self, data: bytes, level: int = 6, password: str = None) -> Tuple[bytes, Dict[str, Any]]:
        """LZMA圧縮"""
        if not self.available:
            return data, {'error': 'lzma_not_available'}
        
        start_time = time.perf_counter()
        
        try:
            compressed_data = lzma.compress(data, preset=level)
            processing_time = time.perf_counter() - start_time
            
            return compressed_data, {
                'original_size': len(data),
                'compressed_size': len(compressed_data),
                'compression_ratio': (1 - len(compressed_data) / len(data)) * 100 if len(data) > 0 else 0,
                'processing_time': processing_time,
                'throughput_mb_s': (len(data) / 1024 / 1024) / processing_time if processing_time > 0 else 0,
                'level': level,
                'encrypted': False,  # LZMA自体は暗号化未対応
                'format': 'lzma'
            }
            
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            return data, {
                'error': str(e),
                'processing_time': processing_time,
                'original_size': len(data)
            }


class ComprehensiveBenchmark:
    """包括的ベンチマークシステム"""
    
    def __init__(self):
        self.competitors = {
            'NEXUS-SPE-NXZ': NEXUSCompetitor(),
            'Zstandard': ZstdCompetitor(),
            '7-Zip': SevenZipCompetitor(),
            'LZMA': LZMACompetitor()
        }
        
        self.test_configurations = [
            {'level': 1, 'name': 'Fast'},
            {'level': 6, 'name': 'Balanced'},
            {'level': 9, 'name': 'Maximum'}
        ]
        
        self.encryption_tests = [False, True]
    
    def generate_test_datasets(self) -> Dict[str, bytes]:
        """テストデータセット生成"""
        datasets = {}
        
        # テキストデータ
        text_data = (
            "This is a comprehensive benchmark test for the NEXUS SPE Integrated Engine. "
            "We are comparing compression algorithms including TMC, Zstandard, 7-Zip, and LZMA. "
            "The goal is to achieve superior compression ratios while maintaining fast processing speeds. "
            "Structure-Preserving Encryption (SPE) adds an additional layer of security while preserving data patterns. "
        ) * 500
        datasets['text'] = text_data.encode('utf-8')
        
        # 反復データ
        repetitive_data = b"ABCD1234" * 1000
        datasets['repetitive'] = repetitive_data
        
        # バイナリデータ（疑似ランダム）
        import random
        random.seed(42)
        binary_data = bytes([random.randint(0, 255) for _ in range(8000)])
        datasets['binary'] = binary_data
        
        # 構造化データ（JSON風）
        structured_data = '{"id": %d, "name": "item_%d", "value": %f, "active": %s}' % (1, 1, 3.14159, "true")
        structured_data = (structured_data * 200).encode('utf-8')
        datasets['structured'] = structured_data
        
        return datasets
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """包括的テスト実行"""
        print("🏁 NEXUS vs 競合 包括的ベンチマーク開始")
        print("=" * 80)
        
        datasets = self.generate_test_datasets()
        results = {}
        
        for data_type, test_data in datasets.items():
            print(f"\n📊 データタイプ: {data_type} ({len(test_data)} bytes)")
            print("-" * 50)
            
            type_results = {}
            
            for config in self.test_configurations:
                level = config['level']
                config_name = config['name']
                
                print(f"\n🔧 設定: {config_name} (Level {level})")
                
                config_results = {}
                
                for use_encryption in self.encryption_tests:
                    password = "test_password_2024" if use_encryption else None
                    encryption_label = "暗号化あり" if use_encryption else "暗号化なし"
                    
                    print(f"  🔐 {encryption_label}")
                    
                    encryption_results = {}
                    
                    # 各競合でテスト
                    for name, competitor in self.competitors.items():
                        if not hasattr(competitor, 'available') or competitor.available:
                            try:
                                compressed_data, info = competitor.compress(test_data, level, password)
                                
                                if 'error' not in info:
                                    print(f"    {name:15}: {info['compression_ratio']:6.2f}% | "
                                          f"{info['throughput_mb_s']:5.1f}MB/s | "
                                          f"{info['compressed_size']:6d}B")
                                    
                                    encryption_results[name] = info
                                else:
                                    print(f"    {name:15}: ❌ {info['error']}")
                                    encryption_results[name] = info
                                    
                            except Exception as e:
                                print(f"    {name:15}: ❌ Exception: {str(e)}")
                                encryption_results[name] = {'error': str(e)}
                        else:
                            print(f"    {name:15}: ❌ Not Available")
                            encryption_results[name] = {'error': 'not_available'}
                    
                    config_results[encryption_label] = encryption_results
                
                type_results[config_name] = config_results
            
            results[data_type] = type_results
        
        # 総合評価
        summary = self._generate_summary(results)
        
        return {
            'detailed_results': results,
            'summary': summary,
            'test_timestamp': time.time(),
            'nexus_version': 'SPE_Integrated_NXZ_v2'
        }
    
    def _generate_summary(self, results: Dict) -> Dict[str, Any]:
        """総合評価生成"""
        try:
            summary = {
                'best_compression': {},
                'best_speed': {},
                'best_overall': {},
                'nexus_performance': {}
            }
            
            all_compression_ratios = {'NEXUS-SPE-NXZ': [], 'Zstandard': [], '7-Zip': [], 'LZMA': []}
            all_speeds = {'NEXUS-SPE-NXZ': [], 'Zstandard': [], '7-Zip': [], 'LZMA': []}
            
            # データ収集
            for data_type, type_results in results.items():
                for config_name, config_results in type_results.items():
                    for encryption_label, encryption_results in config_results.items():
                        for competitor, info in encryption_results.items():
                            if 'error' not in info and 'compression_ratio' in info:
                                if competitor in all_compression_ratios:
                                    all_compression_ratios[competitor].append(info['compression_ratio'])
                                    all_speeds[competitor].append(info['throughput_mb_s'])
            
            # 平均値計算
            for competitor in all_compression_ratios:
                if all_compression_ratios[competitor]:
                    avg_compression = statistics.mean(all_compression_ratios[competitor])
                    avg_speed = statistics.mean(all_speeds[competitor])
                    
                    summary['nexus_performance'][competitor] = {
                        'average_compression_ratio': avg_compression,
                        'average_speed_mb_s': avg_speed,
                        'test_count': len(all_compression_ratios[competitor])
                    }
            
            # 最優秀判定
            if summary['nexus_performance']:
                best_comp = max(summary['nexus_performance'].items(), 
                               key=lambda x: x[1]['average_compression_ratio'])
                best_speed = max(summary['nexus_performance'].items(), 
                                key=lambda x: x[1]['average_speed_mb_s'])
                
                summary['best_compression']['winner'] = best_comp[0]
                summary['best_compression']['ratio'] = best_comp[1]['average_compression_ratio']
                
                summary['best_speed']['winner'] = best_speed[0]
                summary['best_speed']['speed'] = best_speed[1]['average_speed_mb_s']
                
                # 総合スコア（圧縮率 + 速度の重み付け）
                overall_scores = {}
                for competitor, perf in summary['nexus_performance'].items():
                    score = perf['average_compression_ratio'] * 0.7 + perf['average_speed_mb_s'] * 0.3
                    overall_scores[competitor] = score
                
                if overall_scores:
                    best_overall = max(overall_scores.items(), key=lambda x: x[1])
                    summary['best_overall']['winner'] = best_overall[0]
                    summary['best_overall']['score'] = best_overall[1]
            
            return summary
            
        except Exception as e:
            return {'error': str(e)}
    
    def print_final_report(self, results: Dict[str, Any]):
        """最終レポート出力"""
        print("\n" + "=" * 80)
        print("🏆 NEXUS vs 競合 最終ベンチマーク結果")
        print("=" * 80)
        
        summary = results.get('summary', {})
        
        if 'nexus_performance' in summary:
            print("\n📈 平均パフォーマンス:")
            for competitor, perf in summary['nexus_performance'].items():
                print(f"  {competitor:15}: {perf['average_compression_ratio']:6.2f}% | "
                      f"{perf['average_speed_mb_s']:5.1f}MB/s | "
                      f"テスト数: {perf['test_count']}")
        
        if 'best_compression' in summary and 'winner' in summary['best_compression']:
            print(f"\n🥇 最高圧縮率: {summary['best_compression']['winner']} "
                  f"({summary['best_compression']['ratio']:.2f}%)")
        
        if 'best_speed' in summary and 'winner' in summary['best_speed']:
            print(f"⚡ 最高速度: {summary['best_speed']['winner']} "
                  f"({summary['best_speed']['speed']:.1f}MB/s)")
        
        if 'best_overall' in summary and 'winner' in summary['best_overall']:
            print(f"🏆 総合優勝: {summary['best_overall']['winner']} "
                  f"(スコア: {summary['best_overall']['score']:.2f})")
        
        # NEXUS特有の特徴
        print(f"\n🎯 NEXUS SPE統合版の特徴:")
        print(f"   ✓ TMC革命的データ構造理解")
        print(f"   ✓ SPE構造保持暗号化")
        print(f"   ✓ NXZv2フォーマット")
        print(f"   ✓ メタデータ保持")
        print(f"   ✓ 統合セキュリティ")
        
        nexus_perf = summary.get('nexus_performance', {}).get('NEXUS-SPE-NXZ', {})
        if nexus_perf:
            print(f"\n📊 NEXUS実績:")
            print(f"   平均圧縮率: {nexus_perf.get('average_compression_ratio', 0):.2f}%")
            print(f"   平均速度: {nexus_perf.get('average_speed_mb_s', 0):.1f}MB/s")


# メイン実行
if __name__ == "__main__":
    benchmark = ComprehensiveBenchmark()
    
    try:
        results = benchmark.run_comprehensive_test()
        benchmark.print_final_report(results)
        
        # 結果をJSONで保存
        output_file = Path(current_dir) / "nexus_spe_nxz_benchmark_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 詳細結果を保存: {output_file}")
        
    except Exception as e:
        print(f"\n❌ ベンチマークエラー: {e}")
        import traceback
        traceback.print_exc()
