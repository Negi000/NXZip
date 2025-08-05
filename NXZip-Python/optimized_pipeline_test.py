#!/usr/bin/env python3
"""
最適化されたパイプライン順序テスト
新順序: TMC変換 → 圧縮 → SPE暗号化
"""

import os
import sys
import time

# NXZip-Releaseディレクトリを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'NXZip-Release'))

from nxzip_core import NXZipCore, CompressionMode
import zlib
import lzma

def test_optimized_pipeline():
    """最適化されたパイプライン順序をテスト"""
    print("🚀 最適化されたパイプライン順序テスト")
    print("=" * 60)
    
    # テストデータ準備
    test_data = b"Hello World! " * 1000  # 繰り返しデータで効果を見やすく
    print(f"📊 テストデータ: {len(test_data)} bytes")
    print(f"   内容: 繰り返しテキスト（圧縮しやすいデータ）")
    
    # NXZip Core初期化
    nxzip = NXZipCore()
    
    # 暗号化キー生成
    encryption_key = b"test_key_1234567890123456"[:32]  # 32バイトキー
    
    print(f"\n🔥 最適化されたパイプライン実行:")
    print(f"   順序: TMC変換 → 圧縮 → SPE暗号化")
    
    start_time = time.time()
    result = nxzip.compress(test_data, mode="balanced", encryption_key=encryption_key)
    compression_time = time.time() - start_time
    
    if result.success:
        print(f"✅ 圧縮成功!")
        print(f"   原始サイズ: {result.original_size} bytes")
        print(f"   圧縮サイズ: {result.compressed_size} bytes") 
        print(f"   圧縮率: {result.compression_ratio:.2f}%")
        print(f"   圧縮時間: {compression_time:.3f}秒")
        print(f"   エンジン: {result.engine}")
        
        # パイプライン詳細分析
        print(f"\n📋 パイプライン詳細:")
        stages = result.metadata.get('stages', [])
        for i, (stage_name, stage_info) in enumerate(stages, 1):
            print(f"   Step {i}: {stage_name}")
            
            if stage_name == 'tmc_transform':
                transforms = stage_info.get('transforms_applied', [])
                original_size = stage_info.get('original_size', 0)
                transformed_size = stage_info.get('transformed_size', 0)
                print(f"     変換: {transforms}")
                print(f"     サイズ: {original_size} → {transformed_size} bytes")
                
            elif stage_name == 'primary_compression':
                method = stage_info.get('method', 'unknown')
                input_size = stage_info.get('input_size', 0)
                output_size = stage_info.get('output_size', 0)
                stage_ratio = stage_info.get('stage_ratio', 0)
                print(f"     方法: {method}")
                print(f"     サイズ: {input_size} → {output_size} bytes")
                print(f"     段階圧縮率: {stage_ratio:.2f}%")
                
            elif stage_name == 'spe_encryption':
                spe_applied = stage_info.get('spe_applied', False)
                encrypted = stage_info.get('encrypted', False)
                print(f"     SPE適用: {spe_applied}")
                print(f"     暗号化: {encrypted}")
                if spe_applied:
                    original_size = stage_info.get('original_size', 0)
                    spe_size = stage_info.get('spe_size', 0)
                    print(f"     サイズ: {original_size} → {spe_size} bytes")
        
        # 比較のため標準ライブラリでのみ圧縮
        print(f"\n📊 比較: 標準ライブラリのみ")
        
        # zlib圧縮
        zlib_start = time.time()
        zlib_compressed = zlib.compress(test_data, level=6)
        zlib_time = time.time() - zlib_start
        zlib_ratio = (1 - len(zlib_compressed) / len(test_data)) * 100
        print(f"   zlib(level=6): {len(zlib_compressed)} bytes, {zlib_ratio:.2f}%, {zlib_time:.3f}秒")
        
        # lzma圧縮
        lzma_start = time.time()
        lzma_compressed = lzma.compress(test_data, preset=6)
        lzma_time = time.time() - lzma_start
        lzma_ratio = (1 - len(lzma_compressed) / len(test_data)) * 100
        print(f"   lzma(preset=6): {len(lzma_compressed)} bytes, {lzma_ratio:.2f}%, {lzma_time:.3f}秒")
        
        # 効果判定
        print(f"\n🎯 最適化効果分析:")
        
        if result.compression_ratio > zlib_ratio:
            print(f"   ✅ zlib比較: +{result.compression_ratio - zlib_ratio:.2f}% 改善")
        else:
            print(f"   ❌ zlib比較: {zlib_ratio - result.compression_ratio:.2f}% 劣化")
            
        if result.compression_ratio > lzma_ratio:
            print(f"   ✅ lzma比較: +{result.compression_ratio - lzma_ratio:.2f}% 改善")
        else:
            print(f"   ❌ lzma比較: {lzma_ratio - result.compression_ratio:.2f}% 劣化")
        
        # 理論値との比較
        theoretical_best = max(zlib_ratio, lzma_ratio)
        if result.compression_ratio >= theoretical_best * 0.95:  # 95%以上なら実用的
            print(f"   🎉 理論最適値の95%以上を達成！ ({result.compression_ratio:.2f}% vs {theoretical_best:.2f}%)")
        else:
            print(f"   ⚠️ 理論最適値に届かず ({result.compression_ratio:.2f}% vs {theoretical_best:.2f}%)")
        
    else:
        print(f"❌ 圧縮失敗: {result.error_message}")

def test_different_data_types():
    """異なるデータタイプでテスト"""
    print(f"\n🧪 異なるデータタイプでのテスト")
    print("=" * 60)
    
    nxzip = NXZipCore()
    encryption_key = b"test_key_1234567890123456"[:32]
    
    test_cases = [
        ("繰り返しテキスト", b"Hello World! " * 500),
        ("ランダムデータ", os.urandom(2000)),
        ("ゼロデータ", b'\x00' * 2000),
        ("数値配列風", b'\x01\x02\x03\x04' * 500),
    ]
    
    for name, data in test_cases:
        print(f"\n📝 {name}: {len(data)} bytes")
        
        result = nxzip.compress(data, mode="balanced", encryption_key=encryption_key)
        
        if result.success:
            # 標準圧縮との比較
            zlib_size = len(zlib.compress(data, level=6))
            zlib_ratio = (1 - zlib_size / len(data)) * 100
            
            improvement = result.compression_ratio - zlib_ratio
            
            print(f"   NXZip: {result.compressed_size} bytes ({result.compression_ratio:.1f}%)")
            print(f"   zlib:  {zlib_size} bytes ({zlib_ratio:.1f}%)")
            print(f"   差分:  {improvement:+.1f}% {'✅' if improvement >= 0 else '❌'}")
        else:
            print(f"   ❌ 圧縮失敗: {result.error_message}")

if __name__ == "__main__":
    test_optimized_pipeline()
    test_different_data_types()
