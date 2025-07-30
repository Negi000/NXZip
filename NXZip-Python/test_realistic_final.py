#!/usr/bin/env python3
"""
NEXUS v6.1 現実的目標設定版 最終テスト
実現可能な目標での性能評価
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'nxzip', 'engine'))

import time
import hashlib
from pathlib import Path
from nxzip.engine.nexus_v6_1_final import NEXUSEngineReversibilityGuaranteed


def test_with_realistic_targets():
    """現実的目標でのテスト"""
    print("🎯 NEXUS v6.1 現実的目標設定版 最終テスト")
    print("=" * 80)
    
    # 現実的な目標設定
    realistic_targets = {
        'jpg': 8.0,    # JPEG（既に圧縮済み）
        'png': 2.0,    # PNG（可逆圧縮済み）
        'mp4': 15.0,   # 動画（既に圧縮済み）
        'wav': 50.0,   # 音声（非圧縮）
        'mp3': 5.0,    # MP3（高圧縮済み）
        'txt': 50.0,   # テキスト（圧縮しやすい）
        '7z': 0.5      # アーカイブ（圧縮困難）
    }
    
    engine = NEXUSEngineReversibilityGuaranteed()
    
    # テストファイル処理
    sample_dir = Path("sample")
    test_files = []
    
    if sample_dir.exists():
        for ext in ['*.jpg', '*.png', '*.mp4', '*.wav', '*.mp3', '*.txt', '*.7z']:
            test_files.extend(sample_dir.glob(ext))
    
    results = []
    perfect_files = 0
    target_achieved = 0
    reversible_files = 0
    
    for file_path in test_files:
        print(f"\n{'='*50}")
        print(f"📁 {file_path.name}")
        
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            file_type = file_path.suffix.lower().lstrip('.')
            realistic_target = realistic_targets.get(file_type, 10.0)
            
            start_time = time.perf_counter()
            compressed, info = engine.compress_with_reversibility_check(data, file_type)
            compression_time = time.perf_counter() - start_time
            
            # 手動目標評価
            achieved = info['compression_ratio'] >= realistic_target
            if achieved:
                target_achieved += 1
            
            if info['reversible']:
                reversible_files += 1
            
            # 完璧な結果判定
            if (info['reversible'] and achieved and 
                len(compressed) < len(data) and 
                info['compression_ratio'] > 0):
                perfect_files += 1
                status = "✅ PERFECT"
            elif info['reversible'] and len(compressed) < len(data):
                status = "✅ GOOD"
            elif info['reversible']:
                status = "⚠️ OK"
            else:
                status = "❌ FAILED"
            
            print(f"   📊 サイズ: {len(data)/1024/1024:.2f}MB")
            print(f"   📈 圧縮率: {info['compression_ratio']:.2f}%")
            print(f"   🎯 目標: {realistic_target:.1f}% → {'✅達成' if achieved else '❌未達成'}")
            print(f"   ⚡ スループット: {info['throughput_mb_s']:.2f}MB/s")
            print(f"   🔄 可逆性: {'✅' if info['reversible'] else '❌'}")
            print(f"   📈 膨張防止: {'✅' if len(compressed) < len(data) else '❌'}")
            print(f"   🏆 総合: {status}")
            
            results.append({
                'file': file_path.name,
                'type': file_type,
                'ratio': info['compression_ratio'],
                'target': realistic_target,
                'achieved': achieved,
                'reversible': info['reversible'],
                'status': status,
                'throughput': info['throughput_mb_s']
            })
            
        except Exception as e:
            print(f"   ❌ エラー: {e}")
    
    # 最終評価
    total_files = len(results)
    print(f"\n{'='*80}")
    print(f"🏆 最終評価結果")
    print(f"{'='*80}")
    print(f"📊 処理結果:")
    print(f"   📁 総ファイル数: {total_files}")
    print(f"   ✅ PERFECT: {perfect_files}/{total_files} ({perfect_files/total_files*100:.1f}%)")
    print(f"   🎯 目標達成: {target_achieved}/{total_files} ({target_achieved/total_files*100:.1f}%)")
    print(f"   🔄 可逆性成功: {reversible_files}/{total_files} ({reversible_files/total_files*100:.1f}%)")
    
    # ファイルタイプ別詳細
    print(f"\n📊 ファイルタイプ別詳細:")
    types_summary = {}
    for result in results:
        file_type = result['type']
        if file_type not in types_summary:
            types_summary[file_type] = []
        types_summary[file_type].append(result)
    
    for file_type, type_results in types_summary.items():
        avg_ratio = sum(r['ratio'] for r in type_results) / len(type_results)
        achieved_count = sum(1 for r in type_results if r['achieved'])
        reversible_count = sum(1 for r in type_results if r['reversible'])
        avg_target = sum(r['target'] for r in type_results) / len(type_results)
        
        print(f"   {file_type.upper()}: {avg_ratio:.1f}% (目標: {avg_target:.1f}%)")
        print(f"      達成: {achieved_count}/{len(type_results)}, 可逆: {reversible_count}/{len(type_results)}")
    
    # 成績評価
    overall_grade = calculate_final_grade(
        perfect_files / total_files,
        target_achieved / total_files,
        reversible_files / total_files
    )
    
    print(f"\n🎖️ 総合成績: {overall_grade}")
    
    # 改善提案
    print(f"\n💡 改善提案:")
    if reversible_files / total_files < 0.8:
        print(f"   🔧 可逆性の改善が最優先（現在: {reversible_files/total_files*100:.1f}%）")
    elif target_achieved / total_files < 0.6:
        print(f"   📈 圧縮率向上が必要（現在: {target_achieved/total_files*100:.1f}%）")
    else:
        print(f"   ✅ 良好な性能を達成しています")
    
    print(f"\n🚀 次の改良方針:")
    print(f"   1. 7zファイルの可逆性問題解決")
    print(f"   2. 画像ファイルの圧縮率向上")
    print(f"   3. 解析エラーの修正")


def calculate_final_grade(perfect_rate: float, target_rate: float, reversible_rate: float) -> str:
    """最終成績計算"""
    score = (perfect_rate * 40 + target_rate * 30 + reversible_rate * 30) * 100
    
    if score >= 85:
        return "A+ (優秀)"
    elif score >= 75:
        return "A (良好)"
    elif score >= 65:
        return "B+ (普通)"
    elif score >= 55:
        return "B (改善必要)"
    elif score >= 45:
        return "C (要改良)"
    else:
        return "D (再設計必要)"


if __name__ == "__main__":
    test_with_realistic_targets()
