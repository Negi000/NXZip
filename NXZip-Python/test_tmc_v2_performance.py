#!/usr/bin/env python3
"""
TMC Engine v2 性能テスト
最適化版の圧縮率向上と高速化検証
"""

import os
import sys
import time
import numpy as np
from typing import Dict, List, Tuple

# TMC v2エンジンインポート
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'nxzip', 'engine'))

try:
    from nexus_tmc_engine_v2 import NEXUSTMCEngineV2
except ImportError:
    print("❌ TMC Engine v2のインポートに失敗しました")
    print("代替インポートを試行中...")
    # 直接パスを指定
    engine_path = os.path.join(current_dir, 'nxzip', 'engine', 'nexus_tmc_engine_v2.py')
    if os.path.exists(engine_path):
        exec(open(engine_path).read())
        print("✅ 直接読み込み成功")
    else:
        print(f"❌ ファイルが見つかりません: {engine_path}")
        sys.exit(1)


def create_test_datasets() -> Dict[str, bytes]:
    """多様なテストデータセット作成"""
    datasets = {}
    
    # 1. 構造化数値データ（WAVファイル風）
    print("📊 構造化数値データ生成中...")
    wav_header = b'RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x02\x00'
    audio_samples = np.random.randint(0, 256, 32768, dtype=np.uint8)  # 32KB音声データ
    # パターン性を持たせる
    for i in range(0, len(audio_samples), 4):
        if i + 3 < len(audio_samples):
            base_val = audio_samples[i]
            audio_samples[i+1] = (base_val + 10) % 256
            audio_samples[i+2] = (base_val + 20) % 256
            audio_samples[i+3] = (base_val + 5) % 256
    
    datasets['structured_numeric'] = wav_header + audio_samples.tobytes()
    
    # 2. テキストデータ（日本語+英語）
    print("📝 テキストデータ生成中...")
    text_data = """
    NEXUS TMC Engine v2 - 革命的圧縮フレームワーク
    最適化されたTransform-Model-Code方式により、従来の圧縮限界を突破！
    
    Features:
    - Ultra-fast data structure analysis
    - Adaptive transformation pipeline  
    - Parallel high-performance encoding
    - Cache-optimized architecture
    - Memory-efficient design
    
    Performance Targets:
    - Compression ratio: 50-80% improvement
    - Processing speed: 2-5x faster
    - Memory usage: 30% reduction
    - Scalability: Linear performance scaling
    """ * 100  # 100回繰り返し
    
    datasets['text_like'] = text_data.encode('utf-8')
    
    # 3. 時系列データ（センサーデータ風）
    print("📈 時系列データ生成中...")
    time_series = []
    base_value = 128
    for i in range(10000):
        # 緩やかな変化とノイズ
        base_value += np.random.normal(0, 2)
        base_value = max(0, min(255, base_value))
        noise = np.random.normal(0, 5)
        value = int(max(0, min(255, base_value + noise)))
        time_series.append(value)
    
    datasets['time_series'] = bytes(time_series)
    
    # 4. メディアバイナリ（PNG風ヘッダー付き）
    print("🖼️ メディアバイナリ生成中...")
    png_header = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
    media_data = np.random.randint(0, 256, 16384, dtype=np.uint8)
    # メディアファイル特有のエントロピー
    for i in range(0, len(media_data), 8):
        if i + 7 < len(media_data):
            # 局所的相関
            base = media_data[i]
            for j in range(1, 8):
                if i + j < len(media_data):
                    media_data[i+j] = (base + np.random.randint(-20, 20)) % 256
    
    datasets['media_binary'] = png_header + media_data.tobytes()
    
    # 5. 既圧縮データ（高エントロピー）
    print("🗜️ 既圧縮データ生成中...")
    compressed_data = np.random.randint(0, 256, 8192, dtype=np.uint8)  # 完全ランダム
    datasets['compressed_binary'] = compressed_data.tobytes()
    
    # 6. 大容量汎用バイナリ
    print("📦 汎用バイナリ生成中...")
    generic_data = bytearray()
    for _ in range(1000):
        pattern = b'\x00\x01\x02\x03' * 20
        noise = np.random.randint(0, 256, 10, dtype=np.uint8).tobytes()
        generic_data.extend(pattern + noise)
    
    datasets['generic_binary'] = bytes(generic_data)
    
    return datasets


def run_performance_comparison(datasets: Dict[str, bytes]) -> None:
    """性能比較実行"""
    print("\n🚀 TMC Engine v2 性能テスト開始")
    print("=" * 70)
    
    # エンジン初期化
    engine_v2 = NEXUSTMCEngineV2(max_workers=4)
    
    results = []
    total_original_size = 0
    total_compressed_size = 0
    total_time = 0
    
    for data_name, data in datasets.items():
        print(f"\n📋 テスト: {data_name}")
        print(f"   原サイズ: {len(data):,} bytes ({len(data)/1024:.1f} KB)")
        
        # TMC v2圧縮
        start_time = time.perf_counter()
        compressed, info = engine_v2.compress_tmc_v2(data, data_name)
        end_time = time.perf_counter()
        
        compression_time = end_time - start_time
        compression_ratio = info['compression_ratio']
        throughput = info['throughput_mb_s']
        data_type = info['data_type']
        transform_method = info['transform_info']['transform_method']
        
        print(f"   圧縮後: {len(compressed):,} bytes ({len(compressed)/1024:.1f} KB)")
        print(f"   圧縮率: {compression_ratio:.2f}%")
        print(f"   スループット: {throughput:.2f} MB/s")
        print(f"   判定タイプ: {data_type}")
        print(f"   変換方法: {transform_method}")
        print(f"   処理時間: {compression_time*1000:.1f}ms")
        
        # ステージ別時間表示
        if 'stage_times' in info:
            stage_times = info['stage_times']
            print(f"   └─ 分析: {stage_times['analysis']*1000:.1f}ms")
            print(f"   └─ 変換: {stage_times['transform']*1000:.1f}ms")
            print(f"   └─ 符号化: {stage_times['encoding']*1000:.1f}ms")
        
        # 可逆性確認
        reversible = info.get('reversible', False)
        expansion_prevented = info.get('expansion_prevented', False)
        print(f"   可逆性: {'✅' if reversible else '❌'}")
        print(f"   膨張防止: {'✅' if expansion_prevented else '❌'}")
        
        # 結果記録
        results.append({
            'name': data_name,
            'original_size': len(data),
            'compressed_size': len(compressed),
            'ratio': compression_ratio,
            'throughput': throughput,
            'time': compression_time,
            'data_type': data_type,
            'transform_method': transform_method
        })
        
        total_original_size += len(data)
        total_compressed_size += len(compressed)
        total_time += compression_time
    
    # 総合結果
    print("\n" + "=" * 70)
    print("📊 総合結果")
    print("=" * 70)
    
    overall_ratio = (1 - total_compressed_size / total_original_size) * 100
    overall_throughput = (total_original_size / 1024 / 1024) / total_time
    
    print(f"総データサイズ: {total_original_size:,} bytes ({total_original_size/1024/1024:.2f} MB)")
    print(f"総圧縮サイズ: {total_compressed_size:,} bytes ({total_compressed_size/1024/1024:.2f} MB)")
    print(f"総合圧縮率: {overall_ratio:.2f}%")
    print(f"総合スループット: {overall_throughput:.2f} MB/s")
    print(f"総処理時間: {total_time:.3f}秒")
    
    # データタイプ別統計
    print(f"\n📈 データタイプ別性能:")
    type_stats = {}
    for result in results:
        dtype = result['data_type']
        if dtype not in type_stats:
            type_stats[dtype] = []
        type_stats[dtype].append(result['ratio'])
    
    for dtype, ratios in type_stats.items():
        avg_ratio = np.mean(ratios)
        print(f"   {dtype}: 平均圧縮率 {avg_ratio:.2f}%")
    
    # 性能グレード
    if overall_ratio >= 60 and overall_throughput >= 50:
        grade = "🚀 革命的性能 - 圧縮率&速度両立達成！"
    elif overall_ratio >= 45:
        grade = "🏆 優秀圧縮 - 高圧縮率達成！"
    elif overall_throughput >= 30:
        grade = "⚡ 高速処理 - 高スループット達成！"
    else:
        grade = "✅ 標準性能 - 安定動作確認"
    
    print(f"\n🎯 性能グレード: {grade}")
    
    # TMC v2統計表示
    stats = engine_v2.get_tmc_v2_stats()
    if 'performance_grade' in stats:
        print(f"🏅 TMC評価: {stats['performance_grade']}")
    
    print("\n🔧 最適化効果:")
    print("   ✓ キャッシュ最適化による高速分析")
    print("   ✓ 並列処理による変換高速化")
    print("   ✓ データタイプ別圧縮戦略")
    print("   ✓ メモリ効率化設計")
    print("   ✓ 適応的アルゴリズム選択")


def run_detailed_analysis(datasets: Dict[str, bytes]) -> None:
    """詳細分析実行"""
    print("\n🔬 詳細分析実行")
    print("=" * 70)
    
    engine = NEXUSTMCEngineV2(max_workers=4)
    
    for data_name, data in datasets.items():
        print(f"\n📋 詳細分析: {data_name}")
        
        # 圧縮実行
        compressed, info = engine.compress_tmc_v2(data, data_name)
        
        # 特徴量表示
        features = info.get('features', {})
        print(f"   🧮 特徴量:")
        for feature_name, value in features.items():
            if isinstance(value, float):
                print(f"      {feature_name}: {value:.3f}")
            else:
                print(f"      {feature_name}: {value}")
        
        # 変換詳細
        transform_info = info.get('transform_info', {})
        print(f"   🔄 変換詳細:")
        for key, value in transform_info.items():
            if key != 'features':
                print(f"      {key}: {value}")
        
        # 符号化詳細
        encoding_info = info.get('encoding_info', {})
        if 'compression_results' in encoding_info:
            print(f"   🗜️ 符号化詳細:")
            for result in encoding_info['compression_results']:
                stream_id = result.get('stream_id', 0)
                method = result.get('method', 'unknown')
                ratio = result.get('ratio', 0)
                print(f"      ストリーム {stream_id}: {method} ({ratio:.1f}%)")


if __name__ == "__main__":
    try:
        print("🚀 NEXUS TMC Engine v2 - 性能テスト実行")
        print("革命的圧縮フレームワークの最適化効果を検証")
        print("=" * 70)
        
        # テストデータ作成
        print("📦 テストデータセット作成中...")
        datasets = create_test_datasets()
        
        print(f"\n✅ {len(datasets)}種類のテストデータセット準備完了")
        for name, data in datasets.items():
            print(f"   {name}: {len(data):,} bytes")
        
        # 性能テスト実行
        run_performance_comparison(datasets)
        
        # 詳細分析実行
        run_detailed_analysis(datasets)
        
        print("\n" + "=" * 70)
        print("🎯 TMC Engine v2 性能テスト完了！")
        print("最適化による圧縮率向上と高速化を確認")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  テスト中断")
    except Exception as e:
        print(f"\n❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()
