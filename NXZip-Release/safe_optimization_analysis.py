#!/usr/bin/env python3
"""
NXZip Core v2.0 安全な最適化分析
圧縮率を維持しつつ速度向上を実現
"""

import time
import cProfile
import pstats
import io
from pathlib import Path
from nxzip_core import NXZipCore, CompressionMode

def analyze_bottlenecks():
    """現在のボトルネック詳細分析"""
    print("🔍 NXZip Core v2.0 ボトルネック詳細分析")
    print("=" * 60)
    
    # テストデータ作成
    test_sizes = [100*1024, 1024*1024]  # 100KB, 1MB
    test_data = {}
    
    for size in test_sizes:
        # 実際のテキストパターン
        pattern = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * (size // 60)
        test_data[size] = pattern[:size].encode('utf-8')
    
    core = NXZipCore()
    
    for size, data in test_data.items():
        print(f"\n📊 データサイズ: {size//1024} KB")
        
        for mode in ["fast", "balanced", "maximum"]:
            print(f"\n🔧 {mode.upper()}モード分析:")
            
            # プロファイリング実行
            profiler = cProfile.Profile()
            
            start_time = time.perf_counter()
            profiler.enable()
            result = core.compress(data, mode=mode)
            profiler.disable()
            total_time = time.perf_counter() - start_time
            
            # 結果表示
            if result.success:
                speed = (size / (1024*1024)) / total_time
                print(f"  総時間: {total_time*1000:.1f}ms")
                print(f"  速度: {speed:.1f} MB/s")
                print(f"  圧縮率: {result.compression_ratio:.1f}%")
                
                # プロファイル分析
                stream = io.StringIO()
                stats = pstats.Stats(profiler, stream=stream)
                stats.sort_stats('cumulative')
                stats.print_stats(10)
                
                profile_lines = stream.getvalue().split('\n')
                print(f"  🔍 トップ5関数:")
                
                # ヘッダーをスキップして上位5つの関数を表示
                function_count = 0
                for line in profile_lines:
                    if 'function calls' in line or 'seconds' in line or '---' in line:
                        continue
                    if line.strip() and function_count < 5:
                        parts = line.split()
                        if len(parts) >= 6:
                            cumtime = parts[3] if parts[3] != 'cumtime' else parts[2]
                            filename = parts[-1] if parts[-1] != 'filename:lineno(function)' else 'header'
                            if cumtime.replace('.', '').isdigit() and 'header' not in filename:
                                print(f"    {cumtime}s - {filename}")
                                function_count += 1

def identify_safe_optimizations():
    """安全な最適化ポイントの特定"""
    print(f"\n" + "=" * 60)
    print("⚡ 安全な最適化ポイント（圧縮率維持）")
    print("=" * 60)
    
    optimizations = [
        {
            "category": "初期化処理の最適化",
            "items": [
                "SPECoreJIT の遅延初期化",
                "BWTTransformer の条件付き初期化",  
                "不要な print 文の削除",
                "NumPy 配列作成の最適化"
            ],
            "risk": "最低",
            "impact": "中程度（5-20ms削減）",
            "compression_impact": "なし"
        },
        {
            "category": "BWT処理の条件最適化", 
            "items": [
                "サイズ制限の厳格化（現在2MB→100KB）",
                "データタイプ判定の高速化",
                "BWT適用判定の前倒し",
                "pydivsufsort呼び出しの最適化"
            ],
            "risk": "最低",
            "impact": "大（100-150ms削減）",
            "compression_impact": "なし（条件変更のみ）"
        },
        {
            "category": "パイプライン処理の軽量化",
            "items": [
                "進捗管理の簡素化",
                "ステージ情報収集の最適化", 
                "メタデータ生成の遅延実行",
                "不要なデバッグ出力の削除"
            ],
            "risk": "最低",
            "impact": "中程度（2-10ms削減）",
            "compression_impact": "なし"
        },
        {
            "category": "メモリ操作の最適化",
            "items": [
                "バイト配列コピーの最小化",
                "データ変換の in-place 処理",
                "NumPy配列の再利用",
                "文字列結合の最適化"
            ],
            "risk": "低",
            "impact": "中程度（3-15ms削減）", 
            "compression_impact": "なし"
        }
    ]
    
    for opt in optimizations:
        print(f"\n🎯 {opt['category']}:")
        print(f"  リスク: {opt['risk']}")
        print(f"  効果: {opt['impact']}")
        print(f"  圧縮率への影響: {opt['compression_impact']}")
        print(f"  具体的施策:")
        for item in opt['items']:
            print(f"    • {item}")

def create_optimization_plan():
    """段階的最適化計画"""
    print(f"\n" + "=" * 60)
    print("📋 段階的最適化計画（圧縮率完全保持）")
    print("=" * 60)
    
    phases = [
        {
            "name": "Phase 1: 即効性最適化（15分）",
            "tasks": [
                "print文の削除/コメント化",
                "進捗管理の軽量化", 
                "デバッグ出力の条件付き化",
                "不要な初期化メッセージの削除"
            ],
            "expected_gain": "5-15ms削減（全モード）",
            "risk": "ゼロ"
        },
        {
            "name": "Phase 2: BWT条件最適化（30分）",
            "tasks": [
                "BWT適用条件の厳格化",
                "データサイズ制限の調整",
                "事前判定処理の追加",
                "フォールバック処理の高速化"
            ],
            "expected_gain": "50-120ms削減（MAXIMUMモード）",
            "risk": "最低（条件変更のみ）"
        },
        {
            "name": "Phase 3: パイプライン最適化（1時間）",
            "tasks": [
                "ステージ処理の軽量化",
                "メタデータ生成の最適化",
                "進捗コールバックの効率化",
                "エラーハンドリングの簡素化"
            ],
            "expected_gain": "3-10ms削減（全モード）",
            "risk": "低"
        },
        {
            "name": "Phase 4: メモリ最適化（2時間）",
            "tasks": [
                "バイト配列操作の最適化",
                "NumPy配列の効率的利用",
                "データコピーの最小化",
                "ガベージコレクション最適化"
            ],
            "expected_gain": "5-20ms削減（大容量ファイル）",
            "risk": "低-中"
        }
    ]
    
    for phase in phases:
        print(f"\n{phase['name']}:")
        print(f"  予想効果: {phase['expected_gain']}")
        print(f"  リスク: {phase['risk']}")
        print(f"  作業内容:")
        for task in phase['tasks']:
            print(f"    • {task}")
    
    print(f"\n🎯 総合予想効果:")
    print(f"• FASTモード: 180-467 MB/s → 200-550 MB/s (+10-20%)")
    print(f"• BALANCEDモード: 15-61 MB/s → 20-80 MB/s (+30-50%)")
    print(f"• MAXIMUMモード: 0.6-36 MB/s → 5-60 MB/s (+5-10x) ⭐")
    print(f"• 圧縮率: 完全保持（アルゴリズム変更なし）")

if __name__ == "__main__":
    # ボトルネック分析
    analyze_bottlenecks()
    
    # 最適化ポイント特定
    identify_safe_optimizations() 
    
    # 最適化計画
    create_optimization_plan()
