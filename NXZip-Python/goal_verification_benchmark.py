#!/usr/bin/env python3
"""
目標確認ベンチマーク: NXZip vs 標準圧縮
軽量モード: Zstandardレベル
通常モード: 7-Zipの2倍高速 + 7-Zipレベル圧縮
"""

import time
import zlib
import lzma
from nxzip.engine.nexus_simple_fast import SimpleNEXUSEngine

def create_test_data(size_kb: int = 100) -> bytes:
    """リアルなテストデータ作成"""
    text_data = "This is a test file for compression benchmarking. " * 50
    binary_data = bytes(range(256)) * 10
    repeated_data = b"ABCD" * 1000
    
    mixed_data = (text_data.encode() + binary_data + repeated_data) * (size_kb // 10)
    return mixed_data[:size_kb * 1024]

def benchmark_standards(data: bytes):
    """標準圧縮のベンチマーク"""
    print("\n📊 標準圧縮ベンチマーク:")
    
    # zlib (Zstandard相当)
    start = time.time()
    zlib_compressed = zlib.compress(data, level=3)  # Zstd default level相当
    zlib_time = time.time() - start
    zlib_ratio = (1 - len(zlib_compressed) / len(data)) * 100
    print(f"   📦 zlib-3 (Zstd相当): {zlib_ratio:.1f}% 圧縮, {zlib_time:.3f}秒")
    
    # LZMA (7-Zip相当)
    start = time.time()
    lzma_compressed = lzma.compress(data, preset=5)  # 7-Zip level 5相当
    lzma_time = time.time() - start
    lzma_ratio = (1 - len(lzma_compressed) / len(data)) * 100
    print(f"   🗜️  lzma-5 (7Zip相当): {lzma_ratio:.1f}% 圧縮, {lzma_time:.3f}秒")
    
    return {
        'zstd_equivalent': {'ratio': zlib_ratio, 'time': zlib_time},
        '7zip_equivalent': {'ratio': lzma_ratio, 'time': lzma_time}
    }

def benchmark_nxzip(data: bytes):
    """NXZip エンジンベンチマーク"""
    print("\n🚀 NXZip ベンチマーク:")
    
    # 軽量モード
    engine_light = SimpleNEXUSEngine(lightweight_mode=True)
    start = time.time()
    compressed_light, info_light = engine_light.compress(data)
    light_time = time.time() - start
    light_ratio = info_light['compression_ratio']
    print(f"   ⚡ NXZip軽量: {light_ratio:.1f}% 圧縮, {light_time:.3f}秒")
    
    # 通常モード
    engine_normal = SimpleNEXUSEngine(lightweight_mode=False)
    start = time.time()
    compressed_normal, info_normal = engine_normal.compress(data)
    normal_time = time.time() - start
    normal_ratio = info_normal['compression_ratio']
    print(f"   🎯 NXZip通常: {normal_ratio:.1f}% 圧縮, {normal_time:.3f}秒")
    
    return {
        'light': {'ratio': light_ratio, 'time': light_time},
        'normal': {'ratio': normal_ratio, 'time': normal_time}
    }

def analyze_goal_achievement(standards: dict, nxzip: dict):
    """目標達成度の分析"""
    print("\n📊 目標達成度分析:")
    
    # 軽量モード vs Zstandard
    zstd_time = standards['zstd_equivalent']['time']
    zstd_ratio = standards['zstd_equivalent']['ratio']
    light_time = nxzip['light']['time']
    light_ratio = nxzip['light']['ratio']
    
    print(f"\n⚡ 軽量モード vs Zstandard:")
    if light_time > 0 and zstd_time > 0:
        speed_factor = zstd_time / light_time if light_time > 0 else 0
        ratio_diff = light_ratio - zstd_ratio
        
        print(f"   速度比較: {speed_factor:.1f}x {'高速' if speed_factor > 1 else '低速'}")
        print(f"   圧縮率: {ratio_diff:+.1f}% {'向上' if ratio_diff > 0 else '低下'}")
        
        # 目標: Zstandardと同等レベル (±20%以内の速度、±5%以内の圧縮率)
        speed_ok = 0.8 <= speed_factor <= 1.2
        ratio_ok = abs(ratio_diff) <= 5.0
        zstd_goal = speed_ok and ratio_ok
        
        print(f"   🎯 Zstandardレベル目標: {'✅ 達成' if zstd_goal else '❌ 未達成'}")
        if not speed_ok:
            print(f"      速度要改善: 目標±20%以内、現在{speed_factor:.1f}x")
        if not ratio_ok:
            print(f"      圧縮率要改善: 目標±5%以内、現在{ratio_diff:+.1f}%")
    else:
        zstd_goal = False
        print("   ❌ 計測エラー")
    
    # 通常モード vs 7-Zip
    zip_time = standards['7zip_equivalent']['time']
    zip_ratio = standards['7zip_equivalent']['ratio']
    normal_time = nxzip['normal']['time']
    normal_ratio = nxzip['normal']['ratio']
    
    print(f"\n🎯 通常モード vs 7-Zip:")
    if normal_time > 0 and zip_time > 0:
        speed_factor = zip_time / normal_time if normal_time > 0 else 0
        ratio_diff = normal_ratio - zip_ratio
        
        print(f"   速度比較: {speed_factor:.1f}x {'高速' if speed_factor > 1 else '低速'}")
        print(f"   圧縮率: {ratio_diff:+.1f}% {'向上' if ratio_diff > 0 else '低下'}")
        
        # 目標: 7-Zipの2倍高速 + 同等圧縮率
        speed_ok = speed_factor >= 2.0
        ratio_ok = ratio_diff >= -5.0  # 5%以内の低下まで許容
        zip_goal = speed_ok and ratio_ok
        
        print(f"   🎯 7-Zip 2倍高速目標: {'✅ 達成' if zip_goal else '❌ 未達成'}")
        if not speed_ok:
            print(f"      速度要改善: 目標2.0x以上、現在{speed_factor:.1f}x")
        if not ratio_ok:
            print(f"      圧縮率要改善: 目標-5%以内、現在{ratio_diff:+.1f}%")
    else:
        zip_goal = False
        print("   ❌ 計測エラー")
    
    # 総合評価
    print(f"\n📈 総合評価:")
    overall_success = zstd_goal and zip_goal
    print(f"   🎯 両目標達成: {'✅ 成功' if overall_success else '❌ 改善必要'}")
    
    return overall_success

def main():
    print("=== NXZip 目標確認ベンチマーク ===")
    print("🎯 軽量モード目標: Zstandardレベル")
    print("🎯 通常モード目標: 7-Zipの2倍高速 + 同等圧縮")
    
    # テストデータ作成 (100KB)
    test_data = create_test_data(100)
    print(f"\n📊 テストデータ: {len(test_data)} bytes ({len(test_data)//1024}KB)")
    
    # ベンチマーク実行
    standard_results = benchmark_standards(test_data)
    nxzip_results = benchmark_nxzip(test_data)
    
    # 目標達成度分析
    success = analyze_goal_achievement(standard_results, nxzip_results)
    
    if success:
        print(f"\n🎉 おめでとうございます！両目標を達成しました！")
    else:
        print(f"\n🔧 改善が必要です。シンプルエンジンを最適化しましょう。")

if __name__ == "__main__":
    main()
