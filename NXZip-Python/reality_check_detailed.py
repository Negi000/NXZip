#!/usr/bin/env python3
"""
NXZip コンセプト vs 現実 - 詳細分析
実際の性能を測定してコンセプト達成度を評価
"""

import os
import sys
import time
import shutil
import subprocess
import warnings
from pathlib import Path

# 警告を抑制
warnings.filterwarnings("ignore", category=RuntimeWarning)

class RealityCheck:
    def __init__(self):
        self.sample_dir = Path("sample")
        self.results = []
        
        # NXZip コンセプト目標値
        self.concepts = {
            "fast_mode": {
                "speed_target": "Zstd級の速度 (100-200 MB/s)",
                "compression_target": "7zipより良好",
                "description": "高速 + 高圧縮率"
            },
            "balanced_mode": {
                "speed_target": "7zipの2倍速 (10-20 MB/s)", 
                "compression_target": "7zip級の圧縮率",
                "description": "バランス型"
            },
            "maximum_mode": {
                "speed_target": "品質重視 (1-10 MB/s)",
                "compression_target": "最高圧縮率",
                "description": "最大圧縮"
            }
        }
        
    def test_nxzip_performance(self, filepath, mode="balanced"):
        """NXZipの実際の性能を測定"""
        try:
            original_size = os.path.getsize(filepath)
            
            # 一時ファイル名
            temp_nxz = f"temp_{mode}.nxz"
            
            # NXZip圧縮実行
            start_time = time.time()
            result = subprocess.run([
                sys.executable, "-m", "nxzip.cli_unified", 
                "compress", str(filepath), temp_nxz
            ], capture_output=True, text=True, timeout=60)
            
            end_time = time.time()
            
            # 結果解析
            if os.path.exists(temp_nxz):
                compressed_size = os.path.getsize(temp_nxz)
                compression_time = end_time - start_time
                speed_mbps = (original_size / (1024 * 1024)) / compression_time
                compression_ratio = (compressed_size / original_size) * 100
                
                # 一時ファイル削除
                os.remove(temp_nxz)
                
                return {
                    "success": True,
                    "original_size": original_size,
                    "compressed_size": compressed_size,
                    "compression_ratio": compression_ratio,
                    "speed_mbps": speed_mbps,
                    "compression_time": compression_time,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            else:
                return {"success": False, "error": "圧縮ファイル未生成"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_reference_compressors(self, filepath):
        """参照圧縮ツールでの性能測定"""
        results = {}
        original_size = os.path.getsize(filepath)
        
        # 7zip測定 (利用可能な場合)
        try:
            temp_7z = "temp_ref.7z"
            start_time = time.time()
            subprocess.run(["7z", "a", "-mx9", temp_7z, str(filepath)], 
                         capture_output=True, timeout=30)
            end_time = time.time()
            
            if os.path.exists(temp_7z):
                compressed_size = os.path.getsize(temp_7z)
                speed = (original_size / (1024 * 1024)) / (end_time - start_time)
                ratio = (compressed_size / original_size) * 100
                results["7zip"] = {"speed": speed, "ratio": ratio}
                os.remove(temp_7z)
        except:
            results["7zip"] = {"speed": "N/A", "ratio": "N/A"}
        
        return results
    
    def analyze_concept_achievement(self, nxzip_result, ref_results, file_type):
        """コンセプト達成度分析"""
        if not nxzip_result["success"]:
            return {
                "achievement": "FAILED",
                "analysis": f"実行失敗: {nxzip_result.get('error', 'Unknown error')}",
                "recommendations": ["パッケージ整合性確認", "依存関係修復"]
            }
        
        speed = nxzip_result["speed_mbps"]
        ratio = nxzip_result["compression_ratio"]
        
        # 速度評価
        speed_grade = "Unknown"
        if speed >= 100:
            speed_grade = "Excellent (Zstd級)"
        elif speed >= 20:
            speed_grade = "Good (7zip 2倍級)"
        elif speed >= 10:
            speed_grade = "Fair (7zip級)"
        elif speed >= 1:
            speed_grade = "Poor (低速)"
        else:
            speed_grade = "Very Poor"
        
        # 圧縮率評価
        compression_grade = "Unknown"
        if ratio <= 5:
            compression_grade = "Excellent"
        elif ratio <= 10:
            compression_grade = "Very Good"
        elif ratio <= 20:
            compression_grade = "Good"
        elif ratio <= 50:
            compression_grade = "Fair"
        else:
            compression_grade = "Poor"
        
        # 総合評価
        overall_grade = "NEEDS_IMPROVEMENT"
        if speed >= 50 and ratio <= 10:
            overall_grade = "EXCELLENT"
        elif speed >= 20 and ratio <= 15:
            overall_grade = "GOOD"
        elif speed >= 10 and ratio <= 25:
            overall_grade = "FAIR"
        
        return {
            "achievement": overall_grade,
            "speed_analysis": f"{speed:.1f} MB/s ({speed_grade})",
            "compression_analysis": f"{ratio:.1f}% ({compression_grade})",
            "file_type": file_type,
            "detailed_results": nxzip_result
        }
    
    def run_comprehensive_test(self):
        """包括的テスト実行"""
        print("🔥 NXZip コンセプト vs 現実 - 詳細分析開始")
        print("=" * 60)
        
        # テストファイル選定
        test_files = []
        for ext in [".jpg", ".png", ".txt", ".mp4", ".wav", ".7z"]:
            files = list(self.sample_dir.glob(f"*{ext}"))
            if files:
                test_files.append((files[0], ext[1:]))
        
        if not test_files:
            print("❌ テストファイルが見つかりません")
            return
        
        print(f"📋 テストファイル数: {len(test_files)}")
        print()
        
        overall_results = []
        
        for filepath, file_type in test_files:
            print(f"📄 Testing: {filepath.name} ({file_type})")
            print(f"📏 サイズ: {os.path.getsize(filepath) / (1024*1024):.2f} MB")
            
            # NXZip測定
            nxzip_result = self.test_nxzip_performance(filepath)
            
            # 参照ツール測定
            ref_results = self.test_reference_compressors(filepath)
            
            # 分析
            analysis = self.analyze_concept_achievement(nxzip_result, ref_results, file_type)
            overall_results.append(analysis)
            
            # 結果表示
            if nxzip_result["success"]:
                print(f"✅ NXZip結果:")
                print(f"   速度: {nxzip_result['speed_mbps']:.1f} MB/s")
                print(f"   圧縮率: {nxzip_result['compression_ratio']:.1f}%")
                print(f"   評価: {analysis['achievement']}")
                print(f"   詳細: {analysis['speed_analysis']}")
                print(f"         {analysis['compression_analysis']}")
            else:
                print(f"❌ NXZip失敗: {nxzip_result.get('error', 'Unknown')}")
            
            print()
        
        # 最終評価
        print("=" * 60)
        print("🎯 最終コンセプト達成度評価")
        print("=" * 60)
        
        excellent_count = sum(1 for r in overall_results if r["achievement"] == "EXCELLENT")
        good_count = sum(1 for r in overall_results if r["achievement"] == "GOOD")
        fair_count = sum(1 for r in overall_results if r["achievement"] == "FAIR")
        needs_improvement = len(overall_results) - excellent_count - good_count - fair_count
        
        print(f"📊 結果分布:")
        print(f"   🏆 EXCELLENT: {excellent_count}/{len(overall_results)}")
        print(f"   🥈 GOOD: {good_count}/{len(overall_results)}")
        print(f"   🥉 FAIR: {fair_count}/{len(overall_results)}")
        print(f"   ⚠️  NEEDS_IMPROVEMENT: {needs_improvement}/{len(overall_results)}")
        
        # 総合判定
        success_rate = (excellent_count + good_count) / len(overall_results) * 100
        print(f"\n🎖️  総合達成率: {success_rate:.1f}%")
        
        if success_rate >= 80:
            final_grade = "🏆 CONCEPT ACHIEVED"
        elif success_rate >= 60:
            final_grade = "🥈 MOSTLY ACHIEVED"
        elif success_rate >= 40:
            final_grade = "🥉 PARTIALLY ACHIEVED"
        else:
            final_grade = "⚠️  NEEDS MAJOR IMPROVEMENT"
        
        print(f"🏅 最終評価: {final_grade}")
        
        # 推奨事項
        print(f"\n💡 推奨改善点:")
        if needs_improvement > len(overall_results) * 0.5:
            print("   1. 基本性能の大幅改善が必要")
            print("   2. アルゴリズム最適化")
            print("   3. 実装の見直し")
        elif fair_count > 0:
            print("   1. 特定ファイル形式の最適化")
            print("   2. 速度向上の施策")
            print("   3. 圧縮率改善")
        else:
            print("   1. 現状維持で良好")
            print("   2. 微調整による更なる向上")
        
        return overall_results

if __name__ == "__main__":
    checker = RealityCheck()
    results = checker.run_comprehensive_test()
