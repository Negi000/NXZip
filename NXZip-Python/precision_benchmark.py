#!/usr/bin/env python3
"""
高精度ベンチマーク: NXZip 目標達成度確認
軽量モード: Zstandardレベル
通常モード: 7-Zipの2倍高速 + 7-Zipレベル圧縮
"""

import time
import zlib
import lzma
from nxzip.engine.nexus_simple_fast import SimpleNEXUSEngine

def create_large_test_data(size_kb: int = 1000) -> bytes:
    """大きなテストデータで精度向上"""
    # より現実的なデータパターン
    text_data = "This is a comprehensive compression benchmark test file. " * 100
    binary_data = bytes(range(256)) * 50
    repeated_data = b"COMPRESSION_TEST_PATTERN_" * 200
    structured_data = b"BEGIN_SECTION\n" + b"data_line\n" * 100 + b"END_SECTION\n"
    
    # 複雑な混合データ
    pattern = (text_data.encode() + binary_data + repeated_data + structured_data)
    repetitions = (size_kb * 1024) // len(pattern) + 1
    mixed_data = pattern * repetitions
    
    return mixed_data[:size_kb * 1024]

def precise_benchmark(func, *args, iterations=5):
    """高精度ベンチマーク（複数回実行の平均）"""
    times = []
    result = None
    
    for i in range(iterations):
        start = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()
        times.append(end - start)
    
    # 最高・最低を除外した平均
    if len(times) >= 3:
        times.sort()
        avg_time = sum(times[1:-1]) / (len(times) - 2)
    else:
        avg_time = sum(times) / len(times)
    
    return result, avg_time

def benchmark_standards_precise(data: bytes):
    """高精度標準圧縮ベンチマーク"""
    print(f"\n📊 高精度標準圧縮ベンチマーク (データサイズ: {len(data)//1024}KB):")
    
    # zlib-3 (Zstandard相当)
    print("   測定中: zlib-3 (Zstd相当)...")
    (zlib_compressed, _), zlib_time = precise_benchmark(
        lambda: (zlib.compress(data, level=3), None)
    )
    zlib_ratio = (1 - len(zlib_compressed) / len(data)) * 100
    zlib_speed = (len(data) / (1024 * 1024)) / zlib_time  # MB/s
    print(f"   📦 zlib-3: {zlib_ratio:.1f}% 圧縮, {zlib_time:.4f}秒, {zlib_speed:.1f}MB/s")
    
    # lzma-5 (7-Zip相当)
    print("   測定中: lzma-5 (7Zip相当)...")
    (lzma_compressed, _), lzma_time = precise_benchmark(
        lambda: (lzma.compress(data, preset=5), None)
    )
    lzma_ratio = (1 - len(lzma_compressed) / len(data)) * 100
    lzma_speed = (len(data) / (1024 * 1024)) / lzma_time  # MB/s
    print(f"   🗜️  lzma-5: {lzma_ratio:.1f}% 圧縮, {lzma_time:.4f}秒, {lzma_speed:.1f}MB/s")
    
    return {
        'zstd_equivalent': {
            'ratio': zlib_ratio, 
            'time': zlib_time, 
            'speed': zlib_speed,
            'size': len(zlib_compressed)
        },
        '7zip_equivalent': {
            'ratio': lzma_ratio, 
            'time': lzma_time, 
            'speed': lzma_speed,
            'size': len(lzma_compressed)
        }
    }

def benchmark_nxzip_precise(data: bytes):
    """高精度NXZipベンチマーク"""
    print(f"\n🚀 高精度NXZipベンチマーク:")
    
    # 軽量モード
    print("   測定中: NXZip軽量モード...")
    def compress_light():
        engine = SimpleNEXUSEngine(lightweight_mode=True)
        return engine.compress(data)
    
    (compressed_light, info_light), light_time = precise_benchmark(compress_light)
    light_ratio = info_light['compression_ratio']
    light_speed = (len(data) / (1024 * 1024)) / light_time  # MB/s
    print(f"   ⚡ NXZip軽量: {light_ratio:.1f}% 圧縮, {light_time:.4f}秒, {light_speed:.1f}MB/s")
    
    # 通常モード
    print("   測定中: NXZip通常モード...")
    def compress_normal():
        engine = SimpleNEXUSEngine(lightweight_mode=False)
        return engine.compress(data)
    
    (compressed_normal, info_normal), normal_time = precise_benchmark(compress_normal)
    normal_ratio = info_normal['compression_ratio']
    normal_speed = (len(data) / (1024 * 1024)) / normal_time  # MB/s
    print(f"   🎯 NXZip通常: {normal_ratio:.1f}% 圧縮, {normal_time:.4f}秒, {normal_speed:.1f}MB/s")
    
    return {
        'light': {
            'ratio': light_ratio, 
            'time': light_time, 
            'speed': light_speed,
            'size': len(compressed_light)
        },
        'normal': {
            'ratio': normal_ratio, 
            'time': normal_time, 
            'speed': normal_speed,
            'size': len(compressed_normal)
        }
    }

def analyze_detailed_results(standards: dict, nxzip: dict, data_size: int):
    """詳細な結果分析"""
    print("\n📊 詳細目標達成度分析:")
    
    # 軽量モード vs Zstandard
    zstd = standards['zstd_equivalent']
    light = nxzip['light']
    
    print(f"\n⚡ 軽量モード vs Zstandard詳細比較:")
    print(f"   📦 Zstd  : {zstd['ratio']:.2f}% 圧縮, {zstd['time']:.4f}秒, {zstd['speed']:.1f}MB/s")
    print(f"   ⚡ 軽量  : {light['ratio']:.2f}% 圧縮, {light['time']:.4f}秒, {light['speed']:.1f}MB/s")
    
    if light['time'] > 0 and zstd['time'] > 0:
        speed_factor = zstd['time'] / light['time']
        ratio_diff = light['ratio'] - zstd['ratio']
        speed_ratio = light['speed'] / zstd['speed']
        
        print(f"   🏃 速度比較: {speed_factor:.2f}x {'高速' if speed_factor > 1 else '低速'} ({speed_ratio:.2f}x throughput)")
        print(f"   📈 圧縮率差: {ratio_diff:+.2f}% {'向上' if ratio_diff > 0 else '低下'}")
        
        # 目標判定: 速度±30%、圧縮率±3%以内
        speed_ok = 0.7 <= speed_factor <= 1.3
        ratio_ok = abs(ratio_diff) <= 3.0
        zstd_goal = speed_ok and ratio_ok
        
        print(f"   🎯 Zstandardレベル目標: {'✅ 達成' if zstd_goal else '❌ 未達成'}")
        if not speed_ok:
            print(f"      🔧 速度要改善: 目標0.7-1.3x、現在{speed_factor:.2f}x")
        if not ratio_ok:
            print(f"      🔧 圧縮率要改善: 目標±3%以内、現在{ratio_diff:+.2f}%")
    else:
        zstd_goal = False
        print("   ❌ 計測エラー")
    
    # 通常モード vs 7-Zip
    zip7 = standards['7zip_equivalent']
    normal = nxzip['normal']
    
    print(f"\n🎯 通常モード vs 7-Zip詳細比較:")
    print(f"   🗜️  7Zip : {zip7['ratio']:.2f}% 圧縮, {zip7['time']:.4f}秒, {zip7['speed']:.1f}MB/s")
    print(f"   🎯 通常  : {normal['ratio']:.2f}% 圧縮, {normal['time']:.4f}秒, {normal['speed']:.1f}MB/s")
    
    if normal['time'] > 0 and zip7['time'] > 0:
        speed_factor = zip7['time'] / normal['time']
        ratio_diff = normal['ratio'] - zip7['ratio']
        speed_ratio = normal['speed'] / zip7['speed']
        
        print(f"   🏃 速度比較: {speed_factor:.2f}x {'高速' if speed_factor > 1 else '低速'} ({speed_ratio:.2f}x throughput)")
        print(f"   📈 圧縮率差: {ratio_diff:+.2f}% {'向上' if ratio_diff > 0 else '低下'}")
        
        # 目標判定: 2倍以上高速、圧縮率-3%以内
        speed_ok = speed_factor >= 2.0
        ratio_ok = ratio_diff >= -3.0
        zip_goal = speed_ok and ratio_ok
        
        print(f"   🎯 7-Zip 2倍高速目標: {'✅ 達成' if zip_goal else '❌ 未達成'}")
        if not speed_ok:
            print(f"      🔧 速度要改善: 目標2.0x以上、現在{speed_factor:.2f}x")
        if not ratio_ok:
            print(f"      🔧 圧縮率要改善: 目標-3%以内、現在{ratio_diff:+.2f}%")
    else:
        zip_goal = False
        print("   ❌ 計測エラー")
    
    # 総合評価と改善提案
    print(f"\n📈 総合評価:")
    overall_success = zstd_goal and zip_goal
    print(f"   🎯 両目標達成: {'✅ 成功' if overall_success else '❌ 改善必要'}")
    
    if not overall_success:
        print(f"\n🔧 改善提案:")
        if not zstd_goal:
            print(f"   ⚡ 軽量モード改善: より効率的なzlib設定を検討")
        if not zip_goal:
            print(f"   🎯 通常モード改善: LZMA設定の最適化が必要")
    
    return overall_success

def main():
    print("=== NXZip 高精度目標確認ベンチマーク ===")
    print("🎯 軽量モード目標: Zstandardレベル (速度±30%, 圧縮率±3%)")
    print("🎯 通常モード目標: 7-Zipの2倍高速 + 圧縮率-3%以内")
    
    # 大きなテストデータで精度向上 (1MB)
    test_data = create_large_test_data(1000)
    print(f"\n📊 テストデータ: {len(test_data)} bytes ({len(test_data)//1024}KB)")
    
    # 高精度ベンチマーク実行
    print("🔄 高精度測定開始（複数回実行の平均値）...")
    standard_results = benchmark_standards_precise(test_data)
    nxzip_results = benchmark_nxzip_precise(test_data)
    
    # 詳細分析
    success = analyze_detailed_results(standard_results, nxzip_results, len(test_data))
    
    if success:
        print(f"\n🎉 素晴らしい！両目標を達成しました！")
        print(f"   ⚡ 軽量モード: Zstandardレベル達成")
        print(f"   🎯 通常モード: 7-Zip 2倍高速達成")
    else:
        print(f"\n🔧 目標未達成。最適化が必要です。")

if __name__ == "__main__":
    main()
