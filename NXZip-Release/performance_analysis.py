#!/usr/bin/env python3
"""
NXZip Core v2.0 パフォーマンス分析ツール
Java化の必要性を評価
"""

import time
import cProfile
import pstats
import io
from pathlib import Path
import sys

# NXZip Core インポート
from nxzip_core import NXZipCore

def create_test_data(size_mb: float, data_type: str = "text"):
    """テストデータ生成"""
    size_bytes = int(size_mb * 1024 * 1024)
    
    if data_type == "text":
        # 実際のテキストパターン
        base_text = """
        Lorem ipsum dolor sit amet, consectetur adipiscing elit.
        Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
        Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.
        Nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in
        reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
        """
        text_cycle = base_text * (size_bytes // len(base_text) + 1)
        return text_cycle[:size_bytes].encode('utf-8')
        
    elif data_type == "binary":
        # バイナリパターン（PE実行ファイル風）
        import numpy as np
        data = bytearray(size_bytes)
        
        # MZヘッダー
        data[:2] = b'MZ'
        
        # 繰り返しパターン追加
        for i in range(100, size_bytes - 20, 200):
            if i % 1000 == 0:
                data[i:i+10] = b'\x90' * 10  # NOP命令
            elif i % 500 == 0:
                data[i:i+4] = b'\xff\x15\x00\x00'  # call指令
        
        return bytes(data)
    
    elif data_type == "repetitive":
        # 高冗長性データ
        pattern = b'AAAA' * 100 + b'BBBB' * 100 + b'CCCC' * 100
        repetitions = (size_bytes // len(pattern) + 1)
        full_data = pattern * repetitions
        return full_data[:size_bytes]

def profile_nxzip_compression(data: bytes, mode: str = "fast"):
    """NXZip圧縮のプロファイリング"""
    core = NXZipCore()
    
    # プロファイラー設定
    profiler = cProfile.Profile()
    
    # 圧縮実行
    profiler.enable()
    result = core.compress(data, mode=mode, filename="test_data")
    profiler.disable()
    
    # プロファイル結果解析
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # 上位20関数
    
    return result, stream.getvalue()

def benchmark_modes():
    """各モードのベンチマーク"""
    print("🚀 NXZip Core v2.0 パフォーマンス分析")
    print("=" * 60)
    
    # テストデータセット
    test_cases = [
        (0.1, "text", "小さなテキスト"),
        (1.0, "text", "中規模テキスト"),
        (5.0, "text", "大規模テキスト"),
        (1.0, "binary", "バイナリデータ"),
        (1.0, "repetitive", "高冗長データ")
    ]
    
    modes = ["fast", "balanced", "maximum"]
    
    results = []
    
    for size_mb, data_type, description in test_cases:
        print(f"\n📊 {description} ({size_mb} MB, {data_type})")
        print("-" * 40)
        
        data = create_test_data(size_mb, data_type)
        actual_size = len(data)
        
        for mode in modes:
            print(f"\n🔧 {mode.upper()}モード:")
            
            # 複数回実行して平均取得
            times = []
            ratios = []
            
            for run in range(3):
                start_time = time.perf_counter()
                result = NXZipCore().compress(data, mode=mode)
                end_time = time.perf_counter()
                
                if result.success:
                    execution_time = end_time - start_time
                    times.append(execution_time)
                    ratios.append(result.compression_ratio)
                else:
                    print(f"   Run {run+1}: ❌ 失敗")
                    continue
            
            if times:
                avg_time = sum(times) / len(times)
                avg_ratio = sum(ratios) / len(ratios)
                speed_mbps = (actual_size / (1024*1024)) / avg_time
                
                print(f"   平均時間: {avg_time*1000:.1f}ms")
                print(f"   平均速度: {speed_mbps:.1f} MB/s")
                print(f"   圧縮率: {avg_ratio:.1f}%")
                
                results.append({
                    'size_mb': size_mb,
                    'data_type': data_type,
                    'mode': mode,
                    'time': avg_time,
                    'speed_mbps': speed_mbps,
                    'ratio': avg_ratio
                })
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("📈 パフォーマンスサマリー")
    print("=" * 60)
    
    # モード別平均速度
    mode_speeds = {}
    for mode in modes:
        mode_results = [r for r in results if r['mode'] == mode]
        if mode_results:
            avg_speed = sum(r['speed_mbps'] for r in mode_results) / len(mode_results)
            mode_speeds[mode] = avg_speed
            print(f"{mode.upper()}モード平均: {avg_speed:.1f} MB/s")
        
    # ボトルネック分析
    print(f"\n🔍 速度分析:")
    print(f"- Pythonzlib基準: ~200-300 MB/s")
    print(f"- Pythonlzma基準: ~60-100 MB/s") 
    print(f"- NXZip現状: {list(mode_speeds.values())}")
    
    # 最も遅いケースの詳細プロファイリング
    if results:
        slowest = min(results, key=lambda x: x['speed_mbps'])
        print(f"\n🐌 最低速度ケース: {slowest['data_type']} {slowest['mode']}モード")
        print(f"   速度: {slowest['speed_mbps']:.1f} MB/s")
        
        # 詳細プロファイリング実行
        test_data = create_test_data(slowest['size_mb'], slowest['data_type'])
        result, profile_output = profile_nxzip_compression(test_data, slowest['mode'])
        
        print(f"\n📊 プロファイリング結果（上位関数）:")
        lines = profile_output.split('\n')
        for line in lines[5:15]:  # ヘッダースキップして上位10表示
            if line.strip():
                print(f"   {line}")

def java_migration_analysis():
    """Java移行の効果分析"""
    print(f"\n" + "=" * 60)
    print("☕ Java移行効果分析")
    print("=" * 60)
    
    print("""
🔍 現在のボトルネック予想:
1. パイプライン処理のオーバーヘッド
2. Python関数呼び出しコスト
3. バイト配列操作の非効率性
4. TMC変換の計算コスト
5. SPE統合処理の複雑性

☕ Java移行のメリット:
✅ JVMの最適化（JIT compilation）
✅ マルチスレッド処理の効率化
✅ メモリ管理の最適化
✅ バイト配列操作の高速化
✅ ガベージコレクションの効率性

⚠️ Java移行のデメリット:
❌ 開発コストの増大
❌ プラットフォーム依存の複雑化
❌ Pythonエコシステムとの分離
❌ NumPy/SciPyライブラリの恩恵喪失
❌ デバッグの困難さ

🎯 推奨アプローチ:
1. まずPython内最適化を実施
2. Cython/Numbaによる高速化
3. クリティカルパスのC++拡張
4. 最後にJava移行を検討
    """)

if __name__ == "__main__":
    # 基本ベンチマーク実行
    benchmark_modes()
    
    # Java移行分析
    java_migration_analysis()
    
    print(f"\n✅ パフォーマンス分析完了")
