#!/usr/bin/env python3
"""
NXZip Performance vs Concept Test
=================================

NXZip Core v2.0のコンセプト目標と実際の性能を比較テスト

Concept Targets:
- FAST Mode: Zstdレベル速度 + Zstdを超える圧縮率
- BALANCED Mode: 7Zレベル圧縮率 + 7Z×2以上の速度  
- MAXIMUM Mode: 高圧縮率重視
- ULTRA Mode: 最高圧縮率（時間無視）

Benchmark vs:
- 7-Zip (7z format)
- Zstd
- Standard gzip/zlib
"""

import os
import sys
import time
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class BenchmarkRunner:
    """ベンチマーク実行クラス"""
    
    def __init__(self):
        self.results = {}
        self.test_files = []
        self.nxzip_available = False
        
    def setup_test_files(self):
        """テストファイルのセットアップ"""
        # 利用可能なテストファイルを検索
        possible_files = [
            "COT-001.jpg",
            "COT-001.png", 
            "出庫実績明細_202412.txt",
            "../NXZip-Python/sample/COT-001.jpg",
            "../NXZip-Python/sample/COT-001.png",
            "../NXZip-Python/sample/出庫実績明細_202412.txt"
        ]
        
        for file_path in possible_files:
            if os.path.exists(file_path):
                self.test_files.append(file_path)
                print(f"✅ Test file found: {file_path}")
        
        if not self.test_files:
            print("❌ No test files found")
            return False
        
        return True
    
    def check_nxzip_availability(self):
        """NXZip Core v2.0の利用可能性チェック"""
        nxzip_files = [
            "nxzip_core.py",
            "nxzip_core_optimized.py",
            "../NXZip-Python/nxzip/cli_unified.py"
        ]
        
        for nxzip_file in nxzip_files:
            if os.path.exists(nxzip_file):
                print(f"✅ NXZip found: {nxzip_file}")
                self.nxzip_available = True
                return nxzip_file
        
        print("❌ NXZip not found")
        self.nxzip_available = False
        return None
    
    def test_nxzip_unified(self, test_file: str) -> Dict:
        """NXZip Unified版のテスト"""
        print(f"\n🚀 Testing NXZip Unified: {test_file}")
        
        results = {}
        file_size = os.path.getsize(test_file) / 1024 / 1024  # MB
        
        # Change to NXZip-Python directory
        original_dir = os.getcwd()
        nxzip_dir = "../NXZip-Python"
        
        if os.path.exists(nxzip_dir):
            os.chdir(nxzip_dir)
            
            try:
                # Test compression
                start_time = time.time()
                cmd = [
                    sys.executable, "-m", "nxzip.cli_unified", 
                    "compress", f"sample/{os.path.basename(test_file)}", 
                    "test_unified.nxz"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                elapsed_time = time.time() - start_time
                
                if result.returncode == 0 and os.path.exists("test_unified.nxz"):
                    compressed_size = os.path.getsize("test_unified.nxz")
                    original_size = os.path.getsize(f"sample/{os.path.basename(test_file)}")
                    compression_ratio = (1 - compressed_size / original_size) * 100
                    speed = file_size / elapsed_time if elapsed_time > 0 else 0
                    
                    results = {
                        'success': True,
                        'time': elapsed_time,
                        'speed': speed,
                        'compression_ratio': compression_ratio,
                        'original_size': original_size,
                        'compressed_size': compressed_size
                    }
                    
                    print(f"  ✅ Success: {elapsed_time:.3f}s, {speed:.1f} MB/s, {compression_ratio:.1f}%")
                    
                    # Cleanup
                    try:
                        os.remove("test_unified.nxz")
                    except:
                        pass
                else:
                    print(f"  ❌ Failed: {result.stderr[:200]}")
                    results = {'success': False, 'error': result.stderr[:200]}
                    
            except Exception as e:
                print(f"  ❌ Exception: {e}")
                results = {'success': False, 'error': str(e)}
            finally:
                os.chdir(original_dir)
        else:
            results = {'success': False, 'error': 'NXZip-Python directory not found'}
        
        return results
    
    def test_reference_compressors(self, test_file: str) -> Dict:
        """参照圧縮ツールのテスト"""
        print(f"\n📊 Testing reference compressors: {test_file}")
        
        results = {}
        file_size = os.path.getsize(test_file) / 1024 / 1024  # MB
        
        # Test Python's built-in compression
        try:
            # Test zlib (similar to gzip)
            with open(test_file, 'rb') as f:
                data = f.read()
            
            import zlib
            start_time = time.time()
            compressed_zlib = zlib.compress(data, level=6)
            zlib_time = time.time() - start_time
            zlib_ratio = (1 - len(compressed_zlib) / len(data)) * 100
            zlib_speed = file_size / zlib_time if zlib_time > 0 else 0
            
            results['zlib'] = {
                'time': zlib_time,
                'speed': zlib_speed,
                'compression_ratio': zlib_ratio,
                'method': 'zlib level 6'
            }
            print(f"  📦 zlib: {zlib_time:.3f}s, {zlib_speed:.1f} MB/s, {zlib_ratio:.1f}%")
            
            # Test lzma (similar to 7z)
            import lzma
            start_time = time.time()
            compressed_lzma = lzma.compress(data, preset=6)
            lzma_time = time.time() - start_time
            lzma_ratio = (1 - len(compressed_lzma) / len(data)) * 100
            lzma_speed = file_size / lzma_time if lzma_time > 0 else 0
            
            results['lzma'] = {
                'time': lzma_time,
                'speed': lzma_speed,
                'compression_ratio': lzma_ratio,
                'method': 'lzma preset 6'
            }
            print(f"  📦 lzma: {lzma_time:.3f}s, {lzma_speed:.1f} MB/s, {lzma_ratio:.1f}%")
            
            # Fast compression (zlib level 1)
            start_time = time.time()
            compressed_fast = zlib.compress(data, level=1)
            fast_time = time.time() - start_time
            fast_ratio = (1 - len(compressed_fast) / len(data)) * 100
            fast_speed = file_size / fast_time if fast_time > 0 else 0
            
            results['fast'] = {
                'time': fast_time,
                'speed': fast_speed,
                'compression_ratio': fast_ratio,
                'method': 'zlib level 1 (fast)'
            }
            print(f"  ⚡ fast: {fast_time:.3f}s, {fast_speed:.1f} MB/s, {fast_ratio:.1f}%")
            
        except Exception as e:
            print(f"  ❌ Reference test failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def analyze_concept_achievement(self, nxzip_results: Dict, reference_results: Dict, file_type: str) -> Dict:
        """コンセプト達成度の分析"""
        analysis = {
            'file_type': file_type,
            'concept_targets': {
                'fast_mode': 'Zstdレベル速度 + Zstdを超える圧縮率',
                'balanced_mode': '7Zレベル圧縮率 + 7Z×2以上の速度',
                'maximum_mode': '高圧縮率重視'
            },
            'achievements': {},
            'overall_score': 0
        }
        
        if not nxzip_results.get('success', False):
            analysis['achievements']['status'] = 'failed'
            analysis['overall_score'] = 0
            return analysis
        
        nxzip_speed = nxzip_results.get('speed', 0)
        nxzip_ratio = nxzip_results.get('compression_ratio', 0)
        
        # Concept achievement analysis
        achievements = {}
        
        # Fast mode concept: Zstd level speed + better compression
        if 'fast' in reference_results:
            ref_fast = reference_results['fast']
            speed_vs_fast = (nxzip_speed / ref_fast['speed']) if ref_fast['speed'] > 0 else 0
            ratio_vs_fast = nxzip_ratio - ref_fast['compression_ratio']
            
            achievements['vs_fast'] = {
                'speed_ratio': speed_vs_fast,
                'compression_advantage': ratio_vs_fast,
                'speed_achieved': speed_vs_fast >= 0.8,  # At least 80% of fast compression speed
                'ratio_achieved': ratio_vs_fast >= 0,    # Better compression ratio
                'concept_met': speed_vs_fast >= 0.8 and ratio_vs_fast >= 0
            }
        
        # Balanced mode concept: 7Z level compression + 2x speed
        if 'lzma' in reference_results:
            ref_lzma = reference_results['lzma']
            speed_vs_lzma = (nxzip_speed / ref_lzma['speed']) if ref_lzma['speed'] > 0 else 0
            ratio_vs_lzma = nxzip_ratio - ref_lzma['compression_ratio']
            
            achievements['vs_lzma'] = {
                'speed_ratio': speed_vs_lzma,
                'compression_difference': ratio_vs_lzma,
                'speed_achieved': speed_vs_lzma >= 2.0,  # 2x faster than lzma
                'ratio_achieved': abs(ratio_vs_lzma) <= 5,  # Within 5% of lzma compression
                'concept_met': speed_vs_lzma >= 2.0 and abs(ratio_vs_lzma) <= 5
            }
        
        # Overall performance vs standard compression
        if 'zlib' in reference_results:
            ref_zlib = reference_results['zlib']
            speed_vs_zlib = (nxzip_speed / ref_zlib['speed']) if ref_zlib['speed'] > 0 else 0
            ratio_vs_zlib = nxzip_ratio - ref_zlib['compression_ratio']
            
            achievements['vs_zlib'] = {
                'speed_ratio': speed_vs_zlib,
                'compression_advantage': ratio_vs_zlib,
                'better_speed': speed_vs_zlib > 1.0,
                'better_compression': ratio_vs_zlib > 0
            }
        
        analysis['achievements'] = achievements
        
        # Calculate overall score
        score = 0
        if achievements.get('vs_fast', {}).get('concept_met', False):
            score += 30
        if achievements.get('vs_lzma', {}).get('concept_met', False):
            score += 40
        if achievements.get('vs_zlib', {}).get('better_speed', False):
            score += 15
        if achievements.get('vs_zlib', {}).get('better_compression', False):
            score += 15
        
        analysis['overall_score'] = score
        
        return analysis
    
    def print_detailed_analysis(self, analysis: Dict):
        """詳細分析結果の表示"""
        print(f"\n🎯 Concept Achievement Analysis - {analysis['file_type']}")
        print("=" * 60)
        
        if analysis['overall_score'] == 0:
            print("❌ Test failed - no performance data available")
            return
        
        print(f"📊 Overall Concept Achievement Score: {analysis['overall_score']}/100")
        
        achievements = analysis.get('achievements', {})
        
        # Fast mode analysis
        if 'vs_fast' in achievements:
            fast = achievements['vs_fast']
            print(f"\n⚡ Fast Mode Concept Analysis:")
            print(f"   Target: Zstdレベル速度 + Zstdを超える圧縮率")
            print(f"   Speed vs Fast: {fast['speed_ratio']:.2f}x ({fast['speed_achieved'] and '✅' or '❌'})")
            print(f"   Compression advantage: {fast['compression_advantage']:+.1f}% ({fast['ratio_achieved'] and '✅' or '❌'})")
            print(f"   Concept achieved: {fast['concept_met'] and '🎉 YES' or '❌ NO'}")
        
        # Balanced mode analysis
        if 'vs_lzma' in achievements:
            lzma = achievements['vs_lzma']
            print(f"\n⚖️ Balanced Mode Concept Analysis:")
            print(f"   Target: 7Zレベル圧縮率 + 7Z×2以上の速度")
            print(f"   Speed vs 7Z: {lzma['speed_ratio']:.2f}x ({lzma['speed_achieved'] and '✅' or '❌'})")
            print(f"   Compression difference: {lzma['compression_difference']:+.1f}% ({lzma['ratio_achieved'] and '✅' or '❌'})")
            print(f"   Concept achieved: {lzma['concept_met'] and '🎉 YES' or '❌ NO'}")
        
        # Overall performance
        if 'vs_zlib' in achievements:
            zlib = achievements['vs_zlib']
            print(f"\n📈 Overall Performance vs Standard:")
            print(f"   Speed vs zlib: {zlib['speed_ratio']:.2f}x ({zlib['better_speed'] and '✅' or '❌'})")
            print(f"   Compression advantage: {zlib['compression_advantage']:+.1f}% ({zlib['better_compression'] and '✅' or '❌'})")
    
    def run_comprehensive_test(self):
        """包括的テストの実行"""
        print("🚀 NXZip Core v2.0 - Concept vs Reality Test")
        print("=" * 60)
        
        # Setup
        if not self.setup_test_files():
            return
        
        nxzip_file = self.check_nxzip_availability()
        if not nxzip_file:
            print("❌ Cannot run test without NXZip")
            return
        
        all_results = {}
        
        # Test each file
        for test_file in self.test_files:
            file_name = os.path.basename(test_file)
            file_type = "image" if file_name.endswith(('.jpg', '.png')) else "text"
            
            print(f"\n{'='*60}")
            print(f"📁 Testing File: {file_name} ({file_type})")
            print("=" * 60)
            
            # Test NXZip
            nxzip_results = self.test_nxzip_unified(test_file)
            
            # Test reference compressors
            reference_results = self.test_reference_compressors(test_file)
            
            # Analyze concept achievement
            analysis = self.analyze_concept_achievement(nxzip_results, reference_results, file_type)
            
            # Store results
            all_results[file_name] = {
                'nxzip': nxzip_results,
                'reference': reference_results,
                'analysis': analysis
            }
            
            # Print detailed analysis
            self.print_detailed_analysis(analysis)
        
        # Final summary
        self.print_final_summary(all_results)
        
        return all_results
    
    def print_final_summary(self, all_results: Dict):
        """最終サマリーの表示"""
        print(f"\n{'='*80}")
        print("🏆 FINAL CONCEPT ACHIEVEMENT SUMMARY")
        print("=" * 80)
        
        total_score = 0
        test_count = 0
        concept_achievements = []
        
        for file_name, results in all_results.items():
            analysis = results.get('analysis', {})
            score = analysis.get('overall_score', 0)
            total_score += score
            test_count += 1
            
            achievements = analysis.get('achievements', {})
            fast_met = achievements.get('vs_fast', {}).get('concept_met', False)
            balanced_met = achievements.get('vs_lzma', {}).get('concept_met', False)
            
            concept_achievements.append({
                'file': file_name,
                'score': score,
                'fast_concept': fast_met,
                'balanced_concept': balanced_met
            })
        
        if test_count > 0:
            average_score = total_score / test_count
            print(f"📊 Average Concept Achievement Score: {average_score:.1f}/100")
            
            # Concept achievement summary
            fast_success = sum(1 for a in concept_achievements if a['fast_concept'])
            balanced_success = sum(1 for a in concept_achievements if a['balanced_concept'])
            
            print(f"\n🎯 Concept Achievement Summary:")
            print(f"   ⚡ Fast Mode Concept: {fast_success}/{test_count} files achieved")
            print(f"   ⚖️ Balanced Mode Concept: {balanced_success}/{test_count} files achieved")
            
            # Overall verdict
            if average_score >= 80:
                print(f"\n🎉 VERDICT: NXZip Core v2.0 EXCEEDS CONCEPT TARGETS!")
            elif average_score >= 60:
                print(f"\n✅ VERDICT: NXZip Core v2.0 meets most concept targets")
            elif average_score >= 40:
                print(f"\n⚠️ VERDICT: NXZip Core v2.0 partially meets concept targets")
            else:
                print(f"\n❌ VERDICT: NXZip Core v2.0 needs improvement to meet concept targets")
        
        print(f"\n{'='*80}")

def main():
    """メイン実行関数"""
    benchmark = BenchmarkRunner()
    return benchmark.run_comprehensive_test()

if __name__ == "__main__":
    main()
