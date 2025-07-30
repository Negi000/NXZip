#!/usr/bin/env python3
"""
NEXUS Ultra Engine 大規模並列テスト - 改良版
Ultra Engineを使用した高性能並列処理テスト
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'nxzip', 'engine'))

import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from nxzip.engine.nexus_v6_1_ultra import NEXUSEngineUltra


def test_large_scale_parallel():
    """大規模並列テスト - Ultra Engine版"""
    print("🚀 NEXUS Ultra Engine 大規模並列テスト")
    print("=" * 80)
    
    # Ultra Engine設定
    max_workers = 8
    engine = NEXUSEngineUltra(max_workers=max_workers)
    
    # 大規模データセット
    large_datasets = [
        {
            'name': '中規模データ（5MB）',
            'size_mb': 5,
            'pattern_type': 'mixed',
            'file_type': 'txt'
        },
        {
            'name': '大規模データ（15MB）',
            'size_mb': 15,
            'pattern_type': 'structured',
            'file_type': 'txt'
        },
        {
            'name': '超大規模データ（30MB）',
            'size_mb': 30,
            'pattern_type': 'random',
            'file_type': 'unknown'
        },
        {
            'name': '巨大データ（50MB）',
            'size_mb': 50,
            'pattern_type': 'mixed',
            'file_type': 'txt'
        }
    ]
    
    for dataset_info in large_datasets:
        print(f"\n{'='*70}")
        print(f"🧪 {dataset_info['name']}")
        print(f"   📊 予定サイズ: {dataset_info['size_mb']}MB")
        print(f"   🎯 パターン: {dataset_info['pattern_type']}")
        
        # データ生成
        print("   📝 テストデータ生成中...")
        data = generate_test_data(dataset_info['size_mb'], dataset_info['pattern_type'])
        actual_size_mb = len(data) / 1024 / 1024
        print(f"   ✅ データ生成完了: {actual_size_mb:.1f}MB")
        
        try:
            # Ultra並列圧縮実行
            print("   🚀 Ultra並列圧縮実行中...")
            start_time = time.perf_counter()
            compressed, info = engine.compress_ultra(data, dataset_info['file_type'])
            total_time = time.perf_counter() - start_time
            
            # 結果
            compression_ratio = info['compression_ratio']
            throughput = info['throughput_mb_s']
            
            print(f"   ✅ 圧縮完了!")
            print(f"      📈 圧縮率: {compression_ratio:.2f}%")
            print(f"      ⚡ スループット: {throughput:.2f}MB/s")
            print(f"      ⏱️ 処理時間: {total_time:.3f}秒")
            print(f"      💾 圧縮前: {len(data):,} bytes ({actual_size_mb:.1f}MB)")
            print(f"      💾 圧縮後: {len(compressed):,} bytes ({len(compressed)/1024/1024:.1f}MB)")
            print(f"      🎛️ 戦略: {info['strategy']}")
            print(f"      🔄 可逆性: {'✅' if info['reversible'] else '❌'}")
            
            # 性能評価
            if throughput >= 50:
                perf_grade = "🏆 超高速"
            elif throughput >= 25:
                perf_grade = "� 高速"
            elif throughput >= 10:
                perf_grade = "⚡ 良好"
            else:
                perf_grade = "⚠️ 改善必要"
            
            print(f"      🏆 性能評価: {perf_grade}")
            
        except Exception as e:
            print(f"   ❌ エラー: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 最終レポート
    print(f"\n{'='*80}")
    print(f"📈 Ultra Engine 大規模テスト完了レポート")
    print(f"{'='*80}")
    
    # Ultra Engine統計
    stats = engine.get_ultra_stats()
    
    if stats.get('status') != 'no_data':
        print(f"🎯 処理統計:")
        print(f"   📊 総ファイル数: {stats['files_processed']}")
        print(f"   💾 総データ処理量: {stats['total_input_mb']:.1f}MB")
        print(f"   📈 総圧縮率: {stats['total_compression_ratio']:.2f}%")
        print(f"   ⚡ 平均スループット: {stats['average_throughput_mb_s']:.2f}MB/s")
        print(f"   ⏱️ 総処理時間: {stats['total_time']:.3f}秒")
        
        print(f"\n🎛️ 戦略使用分布:")
        for strategy, count in stats['strategy_distribution'].items():
            if count > 0:
                print(f"   {strategy}: {count}回")
        
        print(f"\n🏆 Ultra Engine 評価:")
        print(f"   グレード: {stats['performance_grade']}")
        print(f"   可逆性率: {stats['reversibility_rate']:.1f}%")
        print(f"   目標達成率: {stats['target_achievement_rate']:.1f}%")
        print(f"   エラー数: {stats['error_count']}")
    
    print(f"\n🎉 Ultra Engine 大規模並列テスト完了!")


def generate_test_data(size_mb: float, pattern_type: str) -> bytes:
    """テストデータ生成"""
    target_size = int(size_mb * 1024 * 1024)
    
    if pattern_type == 'mixed':
        # 混合パターン（構造化+ランダム）
        structured_part = b"NEXUS-TEST-PATTERN-" * (target_size // 40)
        random_part = np.random.randint(0, 256, target_size // 2, dtype=np.uint8).tobytes()
        repeating_part = b"ABCDEFGHIJKLMNOP" * (target_size // 32)
        
        data = structured_part + random_part + repeating_part
        
    elif pattern_type == 'structured':
        # 構造化パターン（圧縮しやすい）
        base_pattern = b"NEXUS-PARALLEL-ENGINE-TEST-DATA-" * (target_size // 64)
        numeric_pattern = bytes(range(256)) * (target_size // 512)
        repeat_pattern = b"0123456789" * (target_size // 20)
        
        data = base_pattern + numeric_pattern + repeat_pattern
        
    elif pattern_type == 'random':
        # ランダムパターン（圧縮困難）
        data = np.random.randint(0, 256, target_size, dtype=np.uint8).tobytes()
        
    else:
        # デフォルト
        data = b"DEFAULT-TEST-DATA" * (target_size // 17)
    
    # サイズ調整
    if len(data) > target_size:
        data = data[:target_size]
    elif len(data) < target_size:
        padding_needed = target_size - len(data)
        data += b"PADDING" * (padding_needed // 7)
        data += b"P" * (padding_needed % 7)
    
    return data


if __name__ == "__main__":
    test_large_scale_parallel()
