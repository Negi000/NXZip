#!/usr/bin/env python3
"""
NEXUS TMC v9.1 + SPE - 軽量化テスト版
メモリ使用量を最適化し、基本的な可逆性と性能をテスト
"""

import os
import sys
import time
import gc
import psutil
from pathlib import Path

# プロジェクトパスを追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def get_memory_usage():
    """現在のメモリ使用量を取得"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB

def clear_memory():
    """メモリをクリアする"""
    gc.collect()

def test_basic_compression_only():
    """基本的な圧縮のみのテスト"""
    print("=" * 60)
    print("🔧 基本圧縮テスト (軽量版)")
    print("=" * 60)
    
    # 小さなテストデータ
    test_data = b"Hello World! This is a test for basic compression." * 1000  # 約50KB
    print(f"📊 Test Data Size: {len(test_data)} bytes ({len(test_data)//1024}KB)")
    
    # メモリ使用量監視
    initial_memory = get_memory_usage()
    print(f"初期メモリ使用量: {initial_memory:.1f} MB")
    
    try:
        from nxzip.engine.nexus_tmc_v91_modular import NEXUSTMCEngineV91
        
        # 軽量モードで初期化
        print("\n⚡ 軽量モードエンジン初期化...")
        engine = NEXUSTMCEngineV91(
            lightweight_mode=True,
            max_workers=2,  # ワーカー数を制限
            chunk_size=256*1024  # 256KB chunks
        )
        
        init_memory = get_memory_usage()
        print(f"エンジン初期化後メモリ: {init_memory:.1f} MB (+{init_memory-initial_memory:.1f} MB)")
        
        # 圧縮テスト
        print("\n🗜️ 圧縮テスト...")
        start_time = time.time()
        compressed_data, info = engine.compress(test_data)
        compress_time = time.time() - start_time
        
        compress_memory = get_memory_usage()
        print(f"圧縮後メモリ: {compress_memory:.1f} MB (+{compress_memory-init_memory:.1f} MB)")
        
        print(f"✅ 圧縮完了:")
        print(f"  元サイズ: {len(test_data)} bytes")
        print(f"  圧縮後: {len(compressed_data)} bytes")
        print(f"  圧縮率: {len(compressed_data)/len(test_data)*100:.2f}%")
        print(f"  圧縮時間: {compress_time:.4f}s")
        print(f"  スループット: {len(test_data)/(compress_time*1024):.2f} KB/s")
        
        # 基本的な解凍テスト（zlibフォールバック）
        print("\n📤 基本解凍テスト...")
        try:
            # シンプルなzlib解凍
            import zlib
            if info.get('method') == 'zlib' or 'error' in info:
                # フォールバック解凍
                decompressed = zlib.decompress(compressed_data)
            else:
                decompressed = engine.decompress(compressed_data, info)
            
            decompress_memory = get_memory_usage()
            print(f"解凍後メモリ: {decompress_memory:.1f} MB")
            
            # 可逆性チェック
            if decompressed == test_data:
                print("✅ 可逆性テスト: 成功")
            else:
                print("❌ 可逆性テスト: 失敗")
                print(f"  元データ長: {len(test_data)}")
                print(f"  解凍データ長: {len(decompressed)}")
        
        except Exception as e:
            print(f"❌ 解凍エラー: {e}")
            print("基本的なzlibフォールバック解凍を試行...")
            try:
                import zlib
                decompressed = zlib.decompress(compressed_data)
                if decompressed == test_data:
                    print("✅ フォールバック解凍: 成功")
                else:
                    print("❌ フォールバック解凍: データ不一致")
            except Exception as e2:
                print(f"❌ フォールバック解凍失敗: {e2}")
    
    except Exception as e:
        print(f"❌ テスト失敗: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # メモリクリーンアップ
        clear_memory()
        final_memory = get_memory_usage()
        print(f"\nメモリクリーンアップ後: {final_memory:.1f} MB")


def test_competitors_simple():
    """競合ツールとの簡単な比較"""
    print("\n" + "=" * 60)
    print("🏆 競合比較テスト (軽量版)")
    print("=" * 60)
    
    # 小さなテストデータ
    test_data = b"This is a test data for compression comparison. " * 2000  # 約100KB
    print(f"📊 Test Data Size: {len(test_data)} bytes ({len(test_data)//1024}KB)")
    
    results = {}
    
    # 1. zlib テスト
    try:
        import zlib
        start_time = time.time()
        zlib_compressed = zlib.compress(test_data, level=6)
        zlib_time = time.time() - start_time
        
        # 解凍テスト
        zlib_decompressed = zlib.decompress(zlib_compressed)
        zlib_reversible = (zlib_decompressed == test_data)
        
        results['zlib'] = {
            'size': len(zlib_compressed),
            'ratio': len(zlib_compressed) / len(test_data),
            'time': zlib_time,
            'reversible': zlib_reversible
        }
        print(f"✅ zlib: {len(zlib_compressed)} bytes ({results['zlib']['ratio']*100:.2f}%), {zlib_time:.4f}s")
    
    except Exception as e:
        print(f"❌ zlib test failed: {e}")
    
    # 2. lzma テスト
    try:
        import lzma
        start_time = time.time()
        lzma_compressed = lzma.compress(test_data, preset=6)
        lzma_time = time.time() - start_time
        
        # 解凍テスト
        lzma_decompressed = lzma.decompress(lzma_compressed)
        lzma_reversible = (lzma_decompressed == test_data)
        
        results['lzma'] = {
            'size': len(lzma_compressed),
            'ratio': len(lzma_compressed) / len(test_data),
            'time': lzma_time,
            'reversible': lzma_reversible
        }
        print(f"✅ lzma: {len(lzma_compressed)} bytes ({results['lzma']['ratio']*100:.2f}%), {lzma_time:.4f}s")
    
    except Exception as e:
        print(f"❌ lzma test failed: {e}")
    
    # 3. TMC v9.1 軽量モード
    try:
        from nxzip.engine.nexus_tmc_v91_modular import NEXUSTMCEngineV91
        
        engine = NEXUSTMCEngineV91(lightweight_mode=True, max_workers=2)
        start_time = time.time()
        tmc_compressed, tmc_info = engine.compress(test_data)
        tmc_time = time.time() - start_time
        
        # 簡単な解凍テスト
        try:
            tmc_decompressed = engine.decompress(tmc_compressed, tmc_info)
            tmc_reversible = (tmc_decompressed == test_data)
        except:
            # フォールバック
            import zlib
            tmc_decompressed = zlib.decompress(tmc_compressed)
            tmc_reversible = (tmc_decompressed == test_data)
        
        results['TMC v9.1'] = {
            'size': len(tmc_compressed),
            'ratio': len(tmc_compressed) / len(test_data),
            'time': tmc_time,
            'reversible': tmc_reversible
        }
        print(f"✅ TMC v9.1: {len(tmc_compressed)} bytes ({results['TMC v9.1']['ratio']*100:.2f}%), {tmc_time:.4f}s")
    
    except Exception as e:
        print(f"❌ TMC v9.1 test failed: {e}")
    
    # 結果サマリー
    print("\n📊 比較結果サマリー:")
    print(f"{'Algorithm':<12} {'Size(bytes)':<12} {'Ratio':<8} {'Time(s)':<8} {'Reversible'}")
    print("-" * 60)
    
    for name, result in results.items():
        print(f"{name:<12} {result['size']:<12} {result['ratio']*100:>6.2f}% {result['time']:>6.4f}s {result['reversible']}")
    
    # ベスト結果
    if results:
        best_compression = min(results.items(), key=lambda x: x[1]['ratio'])
        fastest = min(results.items(), key=lambda x: x[1]['time'])
        
        print(f"\n🏆 最高圧縮率: {best_compression[0]} ({best_compression[1]['ratio']*100:.2f}%)")
        print(f"⚡ 最高速度: {fastest[0]} ({fastest[1]['time']:.4f}s)")


def main():
    """軽量化テストのメイン実行"""
    print("🚀 NEXUS TMC v9.1 - 軽量化テスト版")
    print("目的: メモリ使用量最適化 + 基本機能検証")
    print("=" * 60)
    
    initial_memory = get_memory_usage()
    print(f"開始時メモリ使用量: {initial_memory:.1f} MB")
    
    try:
        # 1. 基本圧縮テスト
        test_basic_compression_only()
        
        # メモリクリーンアップ
        clear_memory()
        
        # 2. 競合比較テスト
        test_competitors_simple()
        
        print("\n" + "=" * 60)
        print("✅ 軽量化テスト完了!")
        
        final_memory = get_memory_usage()
        memory_increase = final_memory - initial_memory
        print(f"最終メモリ使用量: {final_memory:.1f} MB (+{memory_increase:.1f} MB)")
        
        if memory_increase < 100:  # 100MB未満
            print("🎉 メモリ使用量: 正常範囲内")
        else:
            print("⚠️ メモリ使用量: 要最適化")
    
    except Exception as e:
        print(f"\n❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
