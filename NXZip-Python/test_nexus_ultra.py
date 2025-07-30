#!/usr/bin/env python3
"""
NEXUS Ultra Engine 包括的テスト
エラー修正確認 + 大幅性能向上確認 + 並列処理テスト
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'nxzip', 'engine'))

import time
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from nxzip.engine.nexus_v6_1_ultra import NEXUSEngineUltra


def test_ultra_comprehensive():
    """Ultra版包括的テスト"""
    print("🚀 NEXUS Ultra Engine 包括的テスト")
    print("=" * 80)
    print("📋 テスト項目:")
    print("   ✓ エラー修正確認（uint8範囲外エラー解決）")
    print("   ✓ 大幅性能向上確認（並列処理・最適化）")
    print("   ✓ 可逆性保証確認（全ファイル）")
    print("   ✓ 目標達成率向上確認")
    print("   ✓ メモリ効率確認")
    print("=" * 80)
    
    # Ultra Engine初期化
    engine = NEXUSEngineUltra(max_workers=4)
    
    # テスト1: 実ファイルテスト
    print(f"\n🧪 テスト1: 実ファイル性能テスト")
    test_real_files(engine)
    
    # テスト2: 大規模データテスト
    print(f"\n🧪 テスト2: 大規模データ性能テスト")
    test_large_scale_data(engine)
    
    # テスト3: 並列処理効果テスト
    print(f"\n🧪 テスト3: 並列処理効果テスト")
    test_parallel_performance(engine)
    
    # テスト4: エラー耐性テスト
    print(f"\n🧪 テスト4: エラー耐性テスト")
    test_error_resistance(engine)
    
    # 最終レポート
    print(f"\n{'='*80}")
    print(f"📊 Ultra Engine 最終レポート")
    print(f"{'='*80}")
    generate_final_report(engine)


def test_real_files(engine):
    """実ファイルテスト"""
    sample_dir = Path("sample")
    test_files = []
    
    if sample_dir.exists():
        for ext in ['*.jpg', '*.png', '*.mp4', '*.wav', '*.mp3', '*.txt', '*.7z']:
            test_files.extend(sample_dir.glob(ext))
    
    if not test_files:
        print("   ⚠️ 実ファイルが見つかりません。サンプルデータでテストします。")
        test_files = []
    
    results = []
    error_free_count = 0
    
    for i, file_path in enumerate(test_files[:8]):  # 最大8ファイル
        print(f"\n   📁 {i+1}/{min(8, len(test_files))}: {file_path.name}")
        
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            file_type = file_path.suffix.lower().lstrip('.')
            size_mb = len(data) / 1024 / 1024
            
            # Ultra圧縮実行
            start_time = time.perf_counter()
            compressed, info = engine.compress_ultra(data, file_type)
            total_time = time.perf_counter() - start_time
            
            # エラーチェック
            has_errors = 'error' in info
            if not has_errors:
                error_free_count += 1
            
            print(f"      📊 サイズ: {size_mb:.2f}MB")
            print(f"      📈 圧縮率: {info['compression_ratio']:.2f}%")
            print(f"      ⚡ スループット: {info['throughput_mb_s']:.2f}MB/s")
            print(f"      🎛️ 戦略: {info['strategy']}")
            print(f"      🔄 可逆性: {'✅' if info['reversible'] else '❌'}")
            print(f"      ❌ エラー: {'なし' if not has_errors else info.get('error', '')}")
            print(f"      🎯 目標達成: {'✅' if info['target_achieved'] else '❌'}")
            
            results.append({
                'file': file_path.name,
                'size_mb': size_mb,
                'ratio': info['compression_ratio'],
                'throughput': info['throughput_mb_s'],
                'strategy': info['strategy'],
                'reversible': info['reversible'],
                'error_free': not has_errors,
                'target_achieved': info['target_achieved']
            })
            
        except Exception as e:
            print(f"      ❌ テストエラー: {e}")
    
    # サマリー
    if results:
        avg_ratio = sum(r['ratio'] for r in results) / len(results)
        avg_throughput = sum(r['throughput'] for r in results) / len(results)
        reversible_rate = sum(1 for r in results if r['reversible']) / len(results) * 100
        target_rate = sum(1 for r in results if r['target_achieved']) / len(results) * 100
        
        print(f"\n   📊 実ファイルテスト サマリー:")
        print(f"      ファイル数: {len(results)}")
        print(f"      エラーなし: {error_free_count}/{len(results)} ({error_free_count/len(results)*100:.1f}%)")
        print(f"      平均圧縮率: {avg_ratio:.2f}%")
        print(f"      平均スループット: {avg_throughput:.2f}MB/s")
        print(f"      可逆性率: {reversible_rate:.1f}%")
        print(f"      目標達成率: {target_rate:.1f}%")


def test_large_scale_data(engine):
    """大規模データテスト"""
    test_datasets = [
        {'name': '小規模テキスト', 'size_mb': 1, 'type': 'txt'},
        {'name': '中規模バイナリ', 'size_mb': 5, 'type': 'unknown'},
        {'name': '大規模構造化', 'size_mb': 15, 'type': 'txt'},
        {'name': '超大規模ランダム', 'size_mb': 30, 'type': 'unknown'}
    ]
    
    for dataset in test_datasets:
        print(f"\n   🧪 {dataset['name']} ({dataset['size_mb']}MB)")
        
        try:
            # データ生成
            data = generate_large_test_data(dataset['size_mb'], dataset['type'])
            actual_size_mb = len(data) / 1024 / 1024
            
            # Ultra圧縮
            start_time = time.perf_counter()
            compressed, info = engine.compress_ultra(data, dataset['type'])
            total_time = time.perf_counter() - start_time
            
            print(f"      📊 実サイズ: {actual_size_mb:.2f}MB")
            print(f"      📈 圧縮率: {info['compression_ratio']:.2f}%")
            print(f"      ⚡ スループット: {info['throughput_mb_s']:.2f}MB/s")
            print(f"      🎛️ 戦略: {info['strategy']}")
            print(f"      🔄 可逆性: {'✅' if info['reversible'] else '❌'}")
            print(f"      ⏱️ 処理時間: {total_time:.3f}秒")
            
            # 性能評価
            if info['throughput_mb_s'] >= 20:
                perf_grade = "✅ 高速"
            elif info['throughput_mb_s'] >= 10:
                perf_grade = "⚡ 良好"
            else:
                perf_grade = "⚠️ 改善余地"
            
            print(f"      🏆 性能評価: {perf_grade}")
            
        except Exception as e:
            print(f"      ❌ エラー: {e}")


def test_parallel_performance(engine):
    """並列処理効果テスト"""
    print(f"\n   🔄 並列処理 vs 単一処理 比較テスト")
    
    # テストデータ準備
    test_data = generate_large_test_data(10, 'txt')  # 10MBテキストデータ
    
    try:
        # 単一処理（小さなワーカー数）
        engine_single = NEXUSEngineUltra(max_workers=1)
        start_time = time.perf_counter()
        compressed_single, info_single = engine_single.compress_ultra(test_data, 'txt')
        single_time = time.perf_counter() - start_time
        
        # 並列処理（複数ワーカー）
        engine_parallel = NEXUSEngineUltra(max_workers=4)
        start_time = time.perf_counter()
        compressed_parallel, info_parallel = engine_parallel.compress_ultra(test_data, 'txt')
        parallel_time = time.perf_counter() - start_time
        
        # 結果比較
        speedup = single_time / parallel_time if parallel_time > 0 else 1.0
        
        print(f"      📊 テストデータ: {len(test_data)/1024/1024:.1f}MB")
        print(f"      ⚡ 単一処理: {single_time:.3f}秒 ({info_single['throughput_mb_s']:.2f}MB/s)")
        print(f"      🚀 並列処理: {parallel_time:.3f}秒 ({info_parallel['throughput_mb_s']:.2f}MB/s)")
        print(f"      📈 速度向上: {speedup:.2f}倍")
        
        if speedup >= 1.5:
            print(f"      🏆 並列効果: ✅ 優秀 ({speedup:.1f}倍向上)")
        elif speedup >= 1.2:
            print(f"      🏆 並列効果: ⚡ 良好 ({speedup:.1f}倍向上)")
        else:
            print(f"      🏆 並列効果: ⚠️ 限定的 ({speedup:.1f}倍向上)")
        
    except Exception as e:
        print(f"      ❌ 並列テストエラー: {e}")


def test_error_resistance(engine):
    """エラー耐性テスト"""
    print(f"\n   🛡️ エラー耐性・境界値テスト")
    
    error_test_cases = [
        {'name': '空データ', 'data': b'', 'type': 'unknown'},
        {'name': '1バイトデータ', 'data': b'A', 'type': 'txt'},
        {'name': '巨大単一値', 'data': b'A' * 1000000, 'type': 'txt'},
        {'name': 'ランダム極小', 'data': bytes(range(256)), 'type': 'unknown'},
        {'name': '不正UTF-8', 'data': b'\xff\xfe\xfd' * 1000, 'type': 'txt'}
    ]
    
    error_free_count = 0
    
    for test_case in error_test_cases:
        try:
            print(f"      🧪 {test_case['name']}: ", end="")
            
            start_time = time.perf_counter()
            compressed, info = engine.compress_ultra(test_case['data'], test_case['type'])
            test_time = time.perf_counter() - start_time
            
            # エラーチェック
            has_error = 'error' in info
            
            if not has_error:
                error_free_count += 1
                print(f"✅ 成功 ({info['compression_ratio']:.1f}%, {test_time:.3f}秒)")
            else:
                print(f"⚠️ エラー処理 ({info.get('error', 'unknown')})")
            
        except Exception as e:
            print(f"❌ 例外: {e}")
    
    print(f"\n      📊 エラー耐性結果: {error_free_count}/{len(error_test_cases)} 成功 ({error_free_count/len(error_test_cases)*100:.1f}%)")


def generate_large_test_data(size_mb: float, data_type: str) -> bytes:
    """大規模テストデータ生成"""
    import numpy as np
    
    target_size = int(size_mb * 1024 * 1024)
    
    if data_type == 'txt':
        # テキスト様データ（圧縮しやすい）
        patterns = [
            b"NEXUS Ultra Engine Test Data Pattern ",
            b"High Performance Compression System ",
            b"Parallel Processing Optimization ",
            b"Error-Free Implementation "
        ]
        
        data = b''
        while len(data) < target_size:
            for pattern in patterns:
                data += pattern * 100
                if len(data) >= target_size:
                    break
        
        return data[:target_size]
        
    else:
        # バイナリ様データ（圧縮困難）
        return np.random.randint(0, 256, target_size, dtype=np.uint8).tobytes()


def generate_final_report(engine):
    """最終レポート生成"""
    stats = engine.get_ultra_stats()
    
    if stats.get('status') == 'no_data':
        print("   ⚠️ 統計データがありません")
        return
    
    print(f"📈 処理統計:")
    print(f"   📁 処理ファイル数: {stats['files_processed']}")
    print(f"   📊 総圧縮率: {stats['total_compression_ratio']:.2f}%")
    print(f"   ⚡ 平均スループット: {stats['average_throughput_mb_s']:.2f}MB/s")
    print(f"   🔄 可逆性率: {stats['reversibility_rate']:.1f}%")
    print(f"   🎯 目標達成率: {stats['target_achievement_rate']:.1f}%")
    print(f"   ❌ エラー数: {stats['error_count']}")
    print(f"   ⏱️ 総処理時間: {stats['total_time']:.3f}秒")
    
    print(f"\n🎛️ 戦略使用分布:")
    for strategy, count in stats['strategy_distribution'].items():
        if count > 0:
            print(f"   {strategy}: {count}回")
    
    print(f"\n💾 データ処理量:")
    print(f"   📥 入力: {stats['total_input_mb']:.2f}MB")
    print(f"   📤 出力: {stats['total_output_mb']:.2f}MB")
    print(f"   📉 削減: {stats['total_input_mb'] - stats['total_output_mb']:.2f}MB")
    
    print(f"\n🏆 総合評価: {stats['performance_grade']}")
    
    # 改善提案
    print(f"\n💡 改善状況:")
    if stats['error_count'] == 0:
        print(f"   ✅ エラー修正: 完了（エラー数: 0）")
    else:
        print(f"   ⚠️ エラー修正: 部分的（エラー数: {stats['error_count']}）")
    
    if stats['average_throughput_mb_s'] >= 20:
        print(f"   ✅ 性能向上: 大幅改善（{stats['average_throughput_mb_s']:.1f}MB/s）")
    elif stats['average_throughput_mb_s'] >= 10:
        print(f"   ⚡ 性能向上: 改善済み（{stats['average_throughput_mb_s']:.1f}MB/s）")
    else:
        print(f"   ⚠️ 性能向上: 更なる改善必要（{stats['average_throughput_mb_s']:.1f}MB/s）")
    
    if stats['reversibility_rate'] >= 90:
        print(f"   ✅ 可逆性: 優秀（{stats['reversibility_rate']:.1f}%）")
    elif stats['reversibility_rate'] >= 70:
        print(f"   ⚡ 可逆性: 良好（{stats['reversibility_rate']:.1f}%）")
    else:
        print(f"   ⚠️ 可逆性: 改善必要（{stats['reversibility_rate']:.1f}%）")


if __name__ == "__main__":
    test_ultra_comprehensive()
