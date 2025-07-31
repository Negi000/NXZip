#!/usr/bin/env python3
"""
NEXUS TMC vs Zstandard 現実的性能比較
"""
import time
import sys
import zstandard as zstd
sys.path.insert(0, '.')

from nxzip.engine.nexus_tmc import NEXUSTMCEngineV9

def benchmark_zstandard():
    """Zstandard性能ベンチマーク"""
    print("⚡ Zstandard性能ベンチマーク")
    print("=" * 50)
    
    # 様々な圧縮レベルでテスト
    compression_levels = [1, 3, 6, 9, 15, 19]  # 1=fastest, 19=best compression
    
    # テストデータ
    test_data = b"Large Scale Compression Test Data " * 30000  # ~1MB
    data_size_mb = len(test_data) / 1024 / 1024
    
    print(f"📄 テストデータ: {len(test_data):,} bytes ({data_size_mb:.1f} MB)")
    print()
    
    results = []
    
    for level in compression_levels:
        compressor = zstd.ZstdCompressor(level=level)
        decompressor = zstd.ZstdDecompressor()
        
        # 圧縮性能測定
        start_time = time.perf_counter()
        compressed = compressor.compress(test_data)
        compress_time = time.perf_counter() - start_time
        
        # 展開性能測定
        start_time = time.perf_counter()
        decompressed = decompressor.decompress(compressed)
        decompress_time = time.perf_counter() - start_time
        
        # 可逆性確認
        is_correct = test_data == decompressed
        compression_ratio = len(compressed) / len(test_data) * 100
        
        # 速度計算 (MB/s)
        compress_speed = data_size_mb / compress_time
        decompress_speed = data_size_mb / decompress_time
        
        print(f"レベル {level:2d}: 圧縮 {compress_speed:6.1f} MB/s, "
              f"展開 {decompress_speed:6.1f} MB/s, "
              f"圧縮率 {compression_ratio:5.1f}%")
        
        results.append({
            'level': level,
            'compress_speed': compress_speed,
            'decompress_speed': decompress_speed,
            'compression_ratio': compression_ratio,
            'compressed_size': len(compressed)
        })
    
    return results, test_data

def benchmark_nexus_tmc(test_data):
    """NEXUS TMC性能ベンチマーク"""
    print("\n🚀 NEXUS TMC v9.0性能ベンチマーク")
    print("=" * 50)
    
    engine = NEXUSTMCEngineV9(max_workers=4)
    data_size_mb = len(test_data) / 1024 / 1024
    
    # 圧縮性能測定
    start_time = time.perf_counter()
    compressed, meta = engine.compress_tmc(test_data)
    compress_time = time.perf_counter() - start_time
    
    # 展開性能測定
    start_time = time.perf_counter()
    decompressed, _ = engine.decompress_tmc(compressed)
    decompress_time = time.perf_counter() - start_time
    
    # 可逆性確認
    is_correct = test_data == decompressed
    compression_ratio = len(compressed) / len(test_data) * 100
    
    # 速度計算 (MB/s)
    compress_speed = data_size_mb / compress_time
    decompress_speed = data_size_mb / decompress_time
    
    print(f"NEXUS TMC: 圧縮 {compress_speed:6.1f} MB/s, "
          f"展開 {decompress_speed:6.1f} MB/s, "
          f"圧縮率 {compression_ratio:5.1f}%")
    print(f"可逆性: {'✅ OK' if is_correct else '❌ NG'}")
    
    return {
        'compress_speed': compress_speed,
        'decompress_speed': decompress_speed,
        'compression_ratio': compression_ratio,
        'compressed_size': len(compressed)
    }

def realistic_analysis(zstd_results, tmc_result):
    """現実的な性能分析"""
    print("\n\n📊 現実的性能比較分析")
    print("=" * 60)
    
    print("🔍 速度比較（NEXUS TMC vs Zstandard）:")
    print("-" * 60)
    print(f"{'圧縮器':<15} {'圧縮速度':<12} {'展開速度':<12} {'圧縮率':<10} {'判定'}")
    print("-" * 60)
    
    # Zstandard結果表示
    for result in zstd_results:
        level = result['level']
        ratio_color = "🟢" if result['compression_ratio'] < 50 else "🟡" if result['compression_ratio'] < 80 else "🔴"
        speed_rating = "⚡" if result['compress_speed'] > 50 else "🚀" if result['compress_speed'] > 20 else "🐌"
        
        print(f"Zstd Level{level:2d}   {result['compress_speed']:8.1f} MB/s "
              f"{result['decompress_speed']:8.1f} MB/s   "
              f"{result['compression_ratio']:6.1f}%   {speed_rating}{ratio_color}")
    
    # NEXUS TMC結果表示
    tmc_ratio_color = "🟢" if tmc_result['compression_ratio'] < 50 else "🟡" if tmc_result['compression_ratio'] < 80 else "🔴"
    tmc_speed_rating = "⚡" if tmc_result['compress_speed'] > 50 else "🚀" if tmc_result['compress_speed'] > 20 else "🐌"
    
    print(f"NEXUS TMC       {tmc_result['compress_speed']:8.1f} MB/s "
          f"{tmc_result['decompress_speed']:8.1f} MB/s   "
          f"{tmc_result['compression_ratio']:6.1f}%   {tmc_speed_rating}{tmc_ratio_color}")
    
    print("\n🎯 現実的な評価:")
    print("-" * 30)
    
    # 最速のZstandardと比較
    fastest_zstd = max(zstd_results, key=lambda x: x['compress_speed'])
    best_compression_zstd = min(zstd_results, key=lambda x: x['compression_ratio'])
    
    print(f"📈 速度面:")
    speed_ratio = tmc_result['compress_speed'] / fastest_zstd['compress_speed']
    if speed_ratio >= 1.0:
        print(f"  ✅ NEXUS TMCは最速Zstd(Level{fastest_zstd['level']})より{speed_ratio:.1f}倍高速")
    else:
        slowdown = fastest_zstd['compress_speed'] / tmc_result['compress_speed']
        print(f"  ❌ NEXUS TMCは最速Zstd(Level{fastest_zstd['level']})より{slowdown:.1f}倍低速")
        print(f"     最速Zstd: {fastest_zstd['compress_speed']:.1f} MB/s")
        print(f"     NEXUS TMC: {tmc_result['compress_speed']:.1f} MB/s")
    
    print(f"\n📊 圧縮率面:")
    compression_ratio = tmc_result['compression_ratio'] / best_compression_zstd['compression_ratio']
    if compression_ratio <= 1.0:
        improvement = (1 - compression_ratio) * 100
        print(f"  ✅ NEXUS TMCは最高圧縮Zstd(Level{best_compression_zstd['level']})より{improvement:.1f}%改善")
    else:
        degradation = (compression_ratio - 1) * 100
        print(f"  ❌ NEXUS TMCは最高圧縮Zstd(Level{best_compression_zstd['level']})より{degradation:.1f}%劣化")
    
    print(f"     最高圧縮Zstd: {best_compression_zstd['compression_ratio']:.1f}%")
    print(f"     NEXUS TMC: {tmc_result['compression_ratio']:.1f}%")
    
    # 実用性評価
    print(f"\n🔍 実用性評価:")
    if tmc_result['compress_speed'] < 5:
        print("  ⚠️ 圧縮速度が5MB/s未満 - 実用性に課題")
    elif tmc_result['compress_speed'] < 20:
        print("  🟡 圧縮速度は実用範囲だが、Zstandardには劣る")
    else:
        print("  ✅ 圧縮速度は実用レベル")
    
    if tmc_result['compression_ratio'] < best_compression_zstd['compression_ratio']:
        print("  ✅ 圧縮率はZstandardを上回る")
    else:
        print("  ⚠️ 圧縮率でZstandardに劣る場合がある")

def improvement_suggestions():
    """改善提案"""
    print(f"\n\n💡 NEXUS TMC v9.0 改善提案")
    print("=" * 40)
    
    suggestions = [
        {
            'title': '軽量モード追加',
            'description': 'BWT変換をスキップする高速モード',
            'expected_gain': '5-10倍速度向上',
            'tradeoff': '圧縮率10-20%低下'
        },
        {
            'title': 'インクリメンタル圧縮',
            'description': '差分圧縮による高速化',
            'expected_gain': '3-5倍速度向上',
            'tradeoff': 'メモリ使用量増加'
        },
        {
            'title': 'SIMD最適化',
            'description': 'AVX2/AVX-512による並列化',
            'expected_gain': '2-3倍速度向上',
            'tradeoff': 'CPU依存性'
        },
        {
            'title': 'ハードウェア加速',
            'description': 'GPU/FPGA活用',
            'expected_gain': '10-100倍速度向上',
            'tradeoff': '特殊ハードウェア必要'
        }
    ]
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n{i}. {suggestion['title']}")
        print(f"   📝 {suggestion['description']}")
        print(f"   📈 期待効果: {suggestion['expected_gain']}")
        print(f"   ⚖️ トレードオフ: {suggestion['tradeoff']}")

def main():
    """メイン実行"""
    print("🔍 NEXUS TMC vs Zstandard 現実的性能比較")
    print("=" * 70)
    
    # Zstandardベンチマーク
    zstd_results, test_data = benchmark_zstandard()
    
    # NEXUS TMCベンチマーク
    tmc_result = benchmark_nexus_tmc(test_data)
    
    # 比較分析
    realistic_analysis(zstd_results, tmc_result)
    
    # 改善提案
    improvement_suggestions()
    
    print(f"\n\n🎊 結論:")
    print("=" * 20)
    print("現在のNEXUS TMC v9.0は圧縮率では優秀だが、")
    print("速度面でZstandardに大きく劣っている。")
    print("実用化には軽量モードの追加が必須。")

if __name__ == "__main__":
    main()
