#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TMC v9.0 革新機能統合テスト
並列パイプライン + サブリニアLZ77 + 適応的チャンク分割
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nxzip'))

import asyncio
import time
from nxzip.engine.nexus_tmc_v4_unified import NEXUSTMCEngineV9

async def test_tmc_v9_innovations():
    """TMC v9.0革新機能の総合テスト"""
    print("🚀 TMC v9.0 革新機能統合テスト開始")
    print("="*60)
    
    # エンジン初期化
    engine = NEXUSTMCEngineV9(max_workers=4)
    
    # テストデータ生成
    test_cases = [
        {
            "name": "大型構造化データ",
            "data": b'{"user_id": 12345, "name": "test_user", "data": [1,2,3,4,5]}' * 2000,
            "description": "40KB JSONライクデータ - 適応的チャンク分割テスト"
        },
        {
            "name": "高冗長性データ",
            "data": b"ABCDEFGHIJKLMNOP" * 8192,  # 128KB
            "description": "128KB 高冗長性データ - サブリニアLZ77テスト"
        },
        {
            "name": "混合エントロピーデータ", 
            "data": b"HIGH_ENTROPY_" + os.urandom(10240) + b"LOW_ENTROPY_" + (b"pattern" * 1000),
            "description": "混合エントロピー - 並列パイプライン効率テスト"
        }
    ]
    
    for test_case in test_cases:
        data = test_case["data"]
        print(f"\n📊 テスト: {test_case['name']}")
        print(f"データサイズ: {len(data):,} bytes")
        print(f"説明: {test_case['description']}")
        print("-" * 50)
        
        # v9.0 非同期並列圧縮
        print("  🔥 v9.0 非同期並列圧縮実行中...")
        start_time = time.time()
        
        try:
            compressed_data, compression_info = await engine.compress_tmc_v9_async(data)
            v9_time = time.time() - start_time
            
            print(f"  ✅ v9.0圧縮成功:")
            print(f"     圧縮率: {compression_info['compression_ratio']:.2f}%")
            print(f"     処理時間: {v9_time:.3f}秒")
            print(f"     スループット: {compression_info['throughput_mbps']:.2f} MB/s")
            print(f"     チャンク数: {compression_info['chunk_count']}")
            print(f"     サブリニアLZ77使用: {'✅' if compression_info['sublinear_lz77_used'] else '❌'}")
            
            # パイプライン統計
            pipeline_stats = compression_info.get('pipeline_stats', {})
            print(f"     並列処理済み: {pipeline_stats.get('total_processed', 0)}タスク")
            
            # 革新機能表示
            innovations = compression_info.get('innovations', [])
            print(f"     革新機能: {', '.join(innovations)}")
            
        except Exception as e:
            print(f"  ❌ v9.0圧縮エラー: {e}")
            import traceback
            traceback.print_exc()
        
        # 従来v8.1との比較
        print("  📈 v8.1との比較...")
        try:
            start_time = time.time()
            # v8.1比較（フォールバック）
            v8_compressed, v8_info = engine.compress_tmc(data)
            v8_time = time.time() - start_time
            
            v8_ratio = (1 - len(v8_compressed) / len(data)) * 100
            print(f"     v8.1圧縮率: {v8_ratio:.2f}%")
            print(f"     v8.1処理時間: {v8_time:.3f}秒")
            
            if v9_time > 0:
                speedup = v8_time / v9_time
                ratio_improvement = compression_info['compression_ratio'] - v8_ratio
                print(f"     🚀 v9.0高速化: {speedup:.2f}倍")
                print(f"     📈 圧縮率改善: {ratio_improvement:+.2f}%")
            
        except Exception as e:
            print(f"     v8.1比較エラー: {e}")

async def test_sublinear_lz77_performance():
    """サブリニアLZ77性能専用テスト"""
    print("\n🔍 サブリニアLZ77性能テスト")
    print("="*40)
    
    engine = NEXUSTMCEngineV9()
    
    # LZ77に最適なテストデータ
    test_patterns = [
        {
            "name": "高反復パターン",
            "data": b"The quick brown fox jumps over the lazy dog. " * 2000,
            "expected": "高圧縮率期待"
        },
        {
            "name": "辞書効果データ",
            "data": b"function test() { return 'Hello World'; }\n" * 1000 + 
                   b"var result = test();\nconsole.log(result);\n" * 500,
            "expected": "辞書マッチ効果期待"
        }
    ]
    
    for pattern in test_patterns:
        data = pattern["data"]
        print(f"\n📊 {pattern['name']}: {len(data):,} bytes")
        print(f"期待効果: {pattern['expected']}")
        
        # サブリニアLZ77直接テスト
        start_time = time.time()
        compressed, lz77_info = engine.sublinear_lz77.encode_sublinear(data)
        lz77_time = time.time() - start_time
        
        print(f"  結果:")
        print(f"    圧縮率: {lz77_info.get('compression_ratio', 0):.2f}%")
        print(f"    処理時間: {lz77_time:.3f}秒")
        print(f"    理論計算量: {lz77_info.get('theoretical_complexity', 'N/A')}")
        print(f"    トークン数: {lz77_info.get('token_count', 0):,}")

if __name__ == "__main__":
    # 非同期テスト実行
    asyncio.run(test_tmc_v9_innovations())
    asyncio.run(test_sublinear_lz77_performance())
