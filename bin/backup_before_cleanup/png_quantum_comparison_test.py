#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 PNG量子圧縮比較テスト
元々のPNGとJPEGから変換されたPNGの量子圧縮性能を比較
"""

import os
import sys
import time
from pathlib import Path

# 親ディレクトリから究極エンジンをインポート
sys.path.append('bin')
from nexus_ultimate_final import UltimateCompressionEngine

def run_png_comparison_test():
    """PNG量子圧縮比較テスト"""
    print("🔬 PNG量子圧縮比較テスト")
    print("=" * 80)
    print("🎯 目標: JPEGから変換されたPNGでも同等の量子圧縮性能を確認")
    print("=" * 80)
    
    engine = UltimateCompressionEngine()
    
    # テスト対象PNG ファイル
    sample_dir = "NXZip-Python/sample"
    png_files = [
        {
            'file': f"{sample_dir}/COT-012.png",
            'description': '元々のPNG（大容量）',
            'source': 'Original PNG'
        },
        {
            'file': f"{sample_dir}/COT-001.png", 
            'description': 'JPEGから変換されたPNG',
            'source': 'Converted from JPEG'
        }
    ]
    
    results = []
    total_start = time.time()
    
    for png_test in png_files:
        test_file = png_test['file']
        if os.path.exists(test_file):
            print(f"\n🔬 PNG量子テスト: {Path(test_file).name}")
            print(f"   📋 説明: {png_test['description']}")
            print(f"   📄 元形式: {png_test['source']}")
            print("-" * 60)
            
            result = engine.ultimate_compress_file(test_file)
            if result['success']:
                result['source'] = png_test['source']
                result['description'] = png_test['description']
                results.append(result)
            else:
                print(f"❌ エラー: {result.get('error', '不明')}")
        else:
            print(f"⚠️ ファイルが見つかりません: {test_file}")
    
    total_time = time.time() - total_start
    
    # PNG比較結果表示
    if results:
        print(f"\n🔬 PNG量子圧縮比較結果")
        print("=" * 80)
        
        print(f"📊 PNG量子圧縮性能比較:")
        for result in results:
            achievement = result['achievement_rate']
            
            if achievement >= 90:
                status = "🏆 量子革命的成功"
            elif achievement >= 70:
                status = "✅ 量子大幅改善"
            elif achievement >= 50:
                status = "⚠️ 量子部分改善"
            else:
                status = "❌ 量子改善不足"
            
            print(f"\n   {status}")
            print(f"   📁 ファイル: {Path(result['output_file']).stem}")
            print(f"   📄 元形式: {result['source']}")
            print(f"   💾 サイズ: {result['original_size']:,} bytes → {result['compressed_size']:,} bytes")
            print(f"   🎯 圧縮率: {result['compression_ratio']:.1f}% (目標: {result['theoretical_target']:.1f}%)")
            print(f"   📈 達成率: {achievement:.1f}%")
            print(f"   ⚡ 処理時間: {result['processing_time']:.2f}s ({result['speed_mbps']:.1f} MB/s)")
        
        # 量子圧縮技術の一貫性評価
        if len(results) >= 2:
            achievements = [r['achievement_rate'] for r in results]
            compression_ratios = [r['compression_ratio'] for r in results]
            
            avg_achievement = sum(achievements) / len(achievements)
            achievement_variance = max(achievements) - min(achievements)
            
            avg_compression = sum(compression_ratios) / len(compression_ratios)
            compression_variance = max(compression_ratios) - min(compression_ratios)
            
            print(f"\n📊 量子技術一貫性評価:")
            print(f"   平均達成率: {avg_achievement:.1f}%")
            print(f"   達成率のばらつき: {achievement_variance:.1f}%")
            print(f"   平均圧縮率: {avg_compression:.1f}%")
            print(f"   圧縮率のばらつき: {compression_variance:.1f}%")
            
            # 一貫性判定
            if achievement_variance <= 5.0:
                consistency = "🏆 高い一貫性"
            elif achievement_variance <= 10.0:
                consistency = "✅ 良好な一貫性"
            elif achievement_variance <= 20.0:
                consistency = "⚠️ 普通の一貫性"
            else:
                consistency = "❌ 低い一貫性"
            
            print(f"   一貫性評価: {consistency}")
            
            if avg_achievement >= 90:
                overall_status = "🎉 PNG量子圧縮技術が実用レベルで確立！"
            elif avg_achievement >= 70:
                overall_status = "🚀 PNG量子圧縮技術の大幅な成功を確認"
            else:
                overall_status = "🔧 PNG量子圧縮技術の更なる改善が必要"
            
            print(f"\n{overall_status}")
        
        print(f"\n📈 総合評価:")
        print(f"   総処理時間: {total_time:.1f}s")
        print(f"   テストファイル数: {len(results)}")

def main():
    """メイン関数"""
    run_png_comparison_test()

if __name__ == "__main__":
    main()
