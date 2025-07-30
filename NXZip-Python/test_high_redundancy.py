#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高冗長テキストデータでの可逆性テスト
TMC v9.0の残る問題を特定・解決
"""

import sys
import time
import asyncio
sys.path.append('.')

from nxzip.engine.nexus_tmc_v4_unified import NEXUSTMCEngineV9

def test_high_redundancy():
    print("🔍 高冗長テキストデータ可逆性テスト開始")
    print("=" * 60)
    
    # 高冗長テキストを生成（ベンチマークと同様）
    test_text = ('高冗長性テキストデータの圧縮テスト。' * 100) * 16  # 約100KB
    test_data = test_text.encode('utf-8')
    print(f'テストデータサイズ: {len(test_data):,} bytes')
    
    # エンジン初期化
    engine = NEXUSTMCEngineV9()
    
    # TMC v9.0非同期圧縮テスト
    print("\n=== TMC v9.0 非同期圧縮テスト ===")
    try:
        # 非同期関数を正しく呼び出し
        compressed, info = asyncio.run(engine.compress_tmc_v9_async(test_data))
        print(f'圧縮結果: {len(test_data):,} -> {len(compressed):,} bytes')
        print(f'圧縮率: {info.get("compression_ratio", 0):.1f}%')
        print(f'圧縮方式: {info.get("method", "unknown")}')
        
        # 展開テスト
        print("\n=== TMC v9.0 展開テスト ===")
        decompressed, decomp_info = engine.decompress_tmc(compressed)
        print(f'展開結果: {len(compressed):,} -> {len(decompressed):,} bytes')
        
        # 可逆性確認
        is_identical = (test_data == decompressed)
        print(f'可逆性: {"✅" if is_identical else "❌"}')
        
        if not is_identical:
            print("\n🔍 可逆性失敗の詳細分析:")
            print(f'元データサイズ: {len(test_data):,} bytes')
            print(f'復元データサイズ: {len(decompressed):,} bytes')
            
            if len(test_data) != len(decompressed):
                print("❌ サイズ不一致が原因")
                size_diff = len(decompressed) - len(test_data)
                print(f"サイズ差: {size_diff:+,} bytes")
            else:
                print("⚠️ サイズは一致、内容が異なる")
                # 最初の不一致を探す
                for i in range(min(len(test_data), len(decompressed))):
                    if test_data[i] != decompressed[i]:
                        print(f'最初の不一致位置: {i:,}')
                        start = max(0, i-10)
                        end = min(len(test_data), i+10)
                        
                        print(f'元データ[{start}:{end}]: {test_data[start:end]}')
                        print(f'復元データ[{start}:{end}]: {decompressed[start:end]}')
                        break
        
    except Exception as e:
        print(f"❌ TMC v9.0テストエラー: {e}")
        import traceback
        traceback.print_exc()
    
    # TMC v7.0従来方式でのテスト（比較）
    print("\n=== TMC v7.0 従来方式テスト（比較） ===")
    try:
        compressed_v7, info_v7 = engine.compress_tmc(test_data)
        print(f'v7.0圧縮結果: {len(test_data):,} -> {len(compressed_v7):,} bytes')
        print(f'v7.0圧縮率: {info_v7.get("compression_ratio", 0):.1f}%')
        
        decompressed_v7, _ = engine.decompress_tmc(compressed_v7)
        is_identical_v7 = (test_data == decompressed_v7)
        print(f'v7.0可逆性: {"✅" if is_identical_v7 else "❌"}')
        
    except Exception as e:
        print(f"❌ TMC v7.0テストエラー: {e}")

if __name__ == "__main__":
    test_high_redundancy()
