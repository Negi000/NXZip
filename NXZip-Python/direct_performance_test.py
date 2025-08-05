#!/usr/bin/env python3
"""
NXZip 直接テスト - リアルタイム性能評価
"""

import os
import sys
import time
import warnings
from pathlib import Path

# 警告を抑制
warnings.filterwarnings("ignore")

# NXZipを直接インポート
sys.path.insert(0, '.')
from nxzip.cli_unified import compress_file

class DirectPerformanceTest:
    def __init__(self):
        self.sample_dir = Path("sample")
        
    def measure_nxzip_direct(self, input_file, output_file):
        """NXZipを直接実行して性能測定"""
        try:
            original_size = os.path.getsize(input_file)
            
            # 圧縮実行
            start_time = time.time()
            success = compress_file(str(input_file), str(output_file))
            end_time = time.time()
            
            if success and os.path.exists(output_file):
                compressed_size = os.path.getsize(output_file)
                compression_time = end_time - start_time
                speed_mbps = (original_size / (1024 * 1024)) / compression_time
                compression_ratio = (compressed_size / original_size) * 100
                
                return {
                    "success": True,
                    "original_size": original_size,
                    "compressed_size": compressed_size,
                    "compression_ratio": compression_ratio,
                    "speed_mbps": speed_mbps,
                    "compression_time": compression_time
                }
            else:
                return {"success": False, "error": "圧縮失敗"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def run_direct_test(self):
        """直接テスト実行"""
        print("🚀 NXZip 直接性能テスト")
        print("=" * 50)
        
        # テストファイル選定
        test_files = [
            ("COT-001.jpg", "jpg"),
            ("COT-001.png", "png"), 
            ("出庫実績明細_202412.txt", "txt"),
            ("generated-music-1752042054079.wav", "wav")
        ]
        
        results = []
        
        for filename, file_type in test_files:
            filepath = self.sample_dir / filename
            if not filepath.exists():
                print(f"❌ ファイル未発見: {filename}")
                continue
                
            output_file = f"direct_test_{file_type}.nxz"
            
            print(f"\n📄 Testing: {filename}")
            print(f"📏 サイズ: {os.path.getsize(filepath) / (1024*1024):.2f} MB")
            
            # 測定実行
            result = self.measure_nxzip_direct(filepath, output_file)
            
            if result["success"]:
                print(f"✅ 成功!")
                print(f"   🏃 速度: {result['speed_mbps']:.1f} MB/s")
                print(f"   🗜️  圧縮率: {result['compression_ratio']:.1f}%")
                print(f"   ⏱️  時間: {result['compression_time']:.2f}秒")
                print(f"   📊 元サイズ: {result['original_size']/1024/1024:.2f} MB")
                print(f"   📦 圧縮後: {result['compressed_size']/1024/1024:.2f} MB")
                
                # コンセプト評価
                speed = result['speed_mbps']
                ratio = result['compression_ratio']
                
                if speed >= 50 and ratio <= 10:
                    grade = "🏆 EXCELLENT (コンセプト達成!)"
                elif speed >= 20 and ratio <= 20:
                    grade = "🥈 GOOD (コンセプト近い)"
                elif speed >= 10 and ratio <= 30:
                    grade = "🥉 FAIR (改善必要)"
                else:
                    grade = "⚠️  NEEDS_IMPROVEMENT"
                
                print(f"   🎯 評価: {grade}")
                
                # クリーンアップ
                if os.path.exists(output_file):
                    os.remove(output_file)
                    
                results.append({
                    "filename": filename,
                    "file_type": file_type,
                    "result": result,
                    "grade": grade
                })
                
            else:
                print(f"❌ 失敗: {result.get('error', 'Unknown error')}")
        
        # 総合評価
        print("\n" + "=" * 50)
        print("🎯 総合評価")
        print("=" * 50)
        
        if results:
            avg_speed = sum(r["result"]["speed_mbps"] for r in results) / len(results)
            avg_ratio = sum(r["result"]["compression_ratio"] for r in results) / len(results)
            
            print(f"📊 平均性能:")
            print(f"   速度: {avg_speed:.1f} MB/s")
            print(f"   圧縮率: {avg_ratio:.1f}%")
            
            # コンセプト目標との比較
            print(f"\n🎯 コンセプト目標比較:")
            print(f"   Fast Mode目標: 100-200 MB/s + 高圧縮")
            print(f"   現実: {avg_speed:.1f} MB/s")
            
            if avg_speed >= 100:
                concept_achievement = "🏆 Fast Modeコンセプト達成!"
            elif avg_speed >= 50:
                concept_achievement = "🥈 Fast Mode近し (50%達成)"
            elif avg_speed >= 20:
                concept_achievement = "🥉 Balanced Mode相当"
            else:
                concept_achievement = "⚠️  コンセプト未達成"
            
            print(f"   結論: {concept_achievement}")
        else:
            print("❌ 測定データなし")
        
        return results

if __name__ == "__main__":
    tester = DirectPerformanceTest()
    tester.run_direct_test()
