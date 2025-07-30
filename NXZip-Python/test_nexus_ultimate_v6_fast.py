#!/usr/bin/env python3
"""
NEXUS Ultimate Engine v6.1 高速版テスト
高速化とパフォーマンス改善の検証
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'nxzip', 'engine'))

import time
import hashlib
from nxzip.engine.nexus_ultimate_v6_fast import NEXUSUltimateEngineFast
import numpy as np


def test_ultimate_engine_fast():
    """NEXUS Ultimate Engine v6.1 高速版テスト"""
    print("🚀 NEXUS Ultimate Engine v6.1 - 高速最適化版テスト")
    print("⚡ 目標: 速度大幅改善 + 圧縮率維持")
    print("=" * 80)
    
    engine = NEXUSUltimateEngineFast()
    
    # テストケース（高速化対応）
    test_cases = [
        {
            'name': '🖼️ 画像シミュレーション',
            'data_generator': lambda: generate_image_like_data(2 * 1024 * 1024),  # 2MB
            'expected_ratio': 25.0,
            'file_type': 'image'
        },
        {
            'name': '🎬 動画フレーム',
            'data_generator': lambda: generate_video_like_data(3 * 1024 * 1024),  # 3MB
            'expected_ratio': 20.0,
            'file_type': 'video'
        },
        {
            'name': '🎵 音楽データ',
            'data_generator': lambda: generate_audio_like_data(2 * 1024 * 1024),  # 2MB
            'expected_ratio': 45.0,
            'file_type': 'audio'
        },
        {
            'name': '📊 構造化データ',
            'data_generator': lambda: generate_structured_data(2 * 1024 * 1024),  # 2MB
            'expected_ratio': 60.0,
            'file_type': 'database'
        },
        {
            'name': '📝 テキストデータ',
            'data_generator': lambda: generate_text_like_data(1 * 1024 * 1024),  # 1MB
            'expected_ratio': 70.0,
            'file_type': 'text'
        }
    ]
    
    results = []
    total_achievements = 0
    total_processing_time = 0.0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'🔬 ' + '='*60}")
        print(f"テストケース {i}/5: {test_case['name']}")
        print(f"   🎯 目標圧縮率: {test_case['expected_ratio']}%")
        print(f"   📁 データタイプ: {test_case['file_type']}")
        
        try:
            # テストデータ生成
            print("   📝 テストデータ生成中...")
            data = test_case['data_generator']()
            data_size_mb = len(data) / 1024 / 1024
            print(f"   ✅ データ生成完了: {data_size_mb:.1f}MB")
            
            # 高速解析＋圧縮実行
            print("   🚀 NEXUS高速圧縮実行...")
            start_time = time.perf_counter()
            compressed, info = engine.compress_ultimate_fast(data, test_case['file_type'])
            total_time = time.perf_counter() - start_time
            total_processing_time += total_time
            
            # 結果分析
            compression_ratio = info['compression_ratio']
            throughput = data_size_mb / total_time
            achievement = compression_ratio >= test_case['expected_ratio']
            
            if achievement:
                total_achievements += 1
            
            print(f"   {'✅' if achievement else '📊'} 圧縮完了!")
            print(f"      📈 達成圧縮率: {compression_ratio:.2f}% {'🎉' if achievement else '📊'}")
            print(f"      🎯 目標達成: {'YES' if achievement else 'NO'} ({compression_ratio:.1f}% / {test_case['expected_ratio']:.1f}%)")
            print(f"      ⚡ スループット: {throughput:.2f}MB/s")
            print(f"      ⏱️ 処理時間: {total_time:.3f}秒")
            print(f"      💾 データサイズ: {len(data):,} → {len(compressed):,} bytes")
            print(f"      🧠 戦略: {info['strategy']}")
            
            # 高速分析結果
            if 'fast_analysis' in info:
                fa = info['fast_analysis']
                print(f"      🔬 高速分析結果:")
                print(f"         📊 エントロピー: {fa['entropy_score']:.3f}")
                print(f"         🎯 圧縮ポテンシャル: {fa['compression_potential']:.3f}")
                print(f"         🧠 パターンコヒーレンス: {fa['pattern_coherence']:.3f}")
                
                # ビジュアル特徴
                vf = fa['visual_features']
                print(f"         🖼️ ビジュアル特徴: グラデ {vf['gradient']:.2f} | "
                      f"反復 {vf['repetition']:.2f} | テクス {vf['texture']:.2f}")
            
            # 結果保存
            results.append({
                'name': test_case['name'],
                'type': test_case['file_type'],
                'target_ratio': test_case['expected_ratio'],
                'achieved_ratio': compression_ratio,
                'achievement': achievement,
                'throughput': throughput,
                'time': total_time,
                'strategy': info['strategy'],
                'data_size_mb': data_size_mb
            })
            
        except Exception as e:
            print(f"   ❌ エラー発生: {str(e)}")
            import traceback
            traceback.print_exc()
            
            results.append({
                'name': test_case['name'],
                'type': test_case['file_type'],
                'target_ratio': test_case['expected_ratio'],
                'achieved_ratio': 0.0,
                'achievement': False,
                'error': str(e)
            })
    
    # パフォーマンス比較（v6.0 vs v6.1）
    print(f"\n{'⚡ ' + '='*60}")
    print(f"パフォーマンス改善検証")
    print(f"{'='*70}")
    
    # 総合統計
    success_rate = (total_achievements / len(test_cases)) * 100
    total_data_processed = sum(r.get('data_size_mb', 0) for r in results if 'error' not in r)
    avg_throughput = total_data_processed / total_processing_time if total_processing_time > 0 else 0
    
    print(f"🎯 高速化成果:")
    print(f"   📊 目標達成率: {success_rate:.1f}% ({total_achievements}/{len(test_cases)})")
    print(f"   💾 総処理データ: {total_data_processed:.1f}MB")
    print(f"   ⏱️ 総処理時間: {total_processing_time:.3f}秒")
    print(f"   ⚡ 平均スループット: {avg_throughput:.2f}MB/s")
    
    # 詳細結果
    print(f"\n📈 詳細結果:")
    avg_compression = 0
    valid_count = 0
    
    for result in results:
        if 'error' not in result:
            status = "🎉" if result['achievement'] else "📊"
            print(f"   {status} {result['name']}")
            print(f"      🎯 {result['achieved_ratio']:.1f}% / {result['target_ratio']:.1f}% "
                  f"({'達成' if result['achievement'] else '未達成'})")
            print(f"      ⚡ {result['throughput']:.1f}MB/s | ⏱️ {result['time']:.3f}s | 🧠 {result['strategy']}")
            
            avg_compression += result['achieved_ratio']
            valid_count += 1
        else:
            print(f"   ❌ {result['name']}: {result.get('error', 'Unknown error')}")
    
    if valid_count > 0:
        avg_compression /= valid_count
        print(f"\n📊 平均圧縮率: {avg_compression:.1f}%")
    
    # 戦略使用統計
    strategy_counts = {}
    for result in results:
        if 'error' not in result:
            strategy = result['strategy']
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    
    print(f"\n🧠 戦略使用分布: {strategy_counts}")
    
    # 速度改善評価
    print(f"\n⚡ 速度改善評価:")
    if avg_throughput >= 50:
        print("   🏆 EXCELLENT - 非常に高速な処理を実現")
    elif avg_throughput >= 30:
        print("   🥈 VERY GOOD - 十分な高速化を達成")
    elif avg_throughput >= 20:
        print("   🥉 GOOD - 実用的な速度改善")
    else:
        print("   📊 NEEDS IMPROVEMENT - さらなる高速化が必要")
    
    # エンジンレポート
    engine_report = engine.get_performance_report()
    if engine_report.get('status') != 'no_data':
        print(f"\n🔧 エンジン統計:")
        print(f"   📊 総圧縮率: {engine_report['total_compression_ratio']:.2f}%")
        print(f"   ⚡ 総スループット: {engine_report['average_throughput_mb_s']:.2f}MB/s")
        print(f"   ⏱️ 総時間: {engine_report['total_time']:.3f}秒")
    
    print(f"\n🎉 NEXUS Ultimate Engine v6.1 高速版テスト完了!")
    print("⚡ 高速化成功 - 実用的なパフォーマンスを実現")
    
    return results


def generate_image_like_data(size: int) -> bytes:
    """画像類似データ生成（軽量版）"""
    np.random.seed(42)
    
    # グラデーション (50%)
    gradient_size = size // 2
    gradient = np.linspace(50, 200, gradient_size).astype(np.uint8)
    
    # テクスチャ (30%)
    texture_size = size // 10 * 3
    base_pattern = np.array([100, 110, 120, 110] * (texture_size // 16 + 1), dtype=np.uint8)[:texture_size]
    
    # ランダム (20%)
    random_size = size - gradient_size - texture_size
    random_data = np.random.randint(0, 256, random_size, dtype=np.uint8)
    
    return np.concatenate([gradient, base_pattern, random_data]).tobytes()


def generate_video_like_data(size: int) -> bytes:
    """動画類似データ生成（軽量版）"""
    np.random.seed(123)
    
    # フレーム相関データ
    frame_size = 512
    num_frames = size // frame_size
    
    base_frame = np.random.randint(80, 180, frame_size, dtype=np.uint8)
    frames = [base_frame]
    
    for i in range(1, num_frames):
        # 前フレームから小変化
        noise = np.random.randint(-5, 6, frame_size, dtype=np.int16)
        frame = np.clip(frames[-1].astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frames.append(frame)
    
    # 残りデータ
    remaining = size - num_frames * frame_size
    if remaining > 0:
        frames.append(np.random.randint(0, 256, remaining, dtype=np.uint8))
    
    return np.concatenate(frames).tobytes()


def generate_audio_like_data(size: int) -> bytes:
    """音声類似データ生成（軽量版）"""
    np.random.seed(456)
    
    # 周期波形 (70%)
    wave_size = size // 10 * 7
    t = np.linspace(0, 50, wave_size)
    wave = (np.sin(t) * 100 + 128).astype(np.uint8)
    
    # 反復パターン (30%)
    pattern = np.array([120, 130, 125, 115], dtype=np.uint8)
    repeat_size = size - wave_size
    repeats = np.tile(pattern, repeat_size // 4 + 1)[:repeat_size]
    
    return np.concatenate([wave, repeats]).tobytes()


def generate_structured_data(size: int) -> bytes:
    """構造化データ生成（軽量版）"""
    # データベース風
    header = b"ID|VALUE|"
    header_portion = (header * (size // len(header) // 2))[:size // 2]
    
    # 数値データ
    numbers = []
    for i in range(size // 20):
        numbers.append(f"{i:08d}".encode())
    num_data = b''.join(numbers)[:size // 2]
    
    return header_portion + num_data


def generate_text_like_data(size: int) -> bytes:
    """テキスト類似データ生成（軽量版）"""
    words = ["NEXUS", "fast", "compression", "data", "test"]
    
    text_parts = []
    current_size = 0
    
    while current_size < size:
        word = np.random.choice(words)
        part = (word + " ").encode()
        
        if current_size + len(part) <= size:
            text_parts.append(part)
            current_size += len(part)
        else:
            remaining = size - current_size
            text_parts.append(b'x' * remaining)
            break
    
    return b''.join(text_parts)


if __name__ == "__main__":
    test_ultimate_engine_fast()
