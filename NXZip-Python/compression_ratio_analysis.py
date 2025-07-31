#!/usr/bin/env python3
"""
NEXUS TMC 軽量モード圧縮率分析
Zstandardとの詳細比較とトレードオフ分析
"""

import os
import sys
import time
import hashlib
import zstandard as zstd
from pathlib import Path

# NEXUS TMC エンジンをインポート
sys.path.append(os.path.dirname(__file__))
from lightweight_mode import NEXUSTMCLightweight

def create_test_data():
    """様々な種類のテストデータを生成"""
    test_files = {}
    
    # 1. テキストデータ（高圧縮率期待）
    text_data = """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
    Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
    Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
    Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
    """ * 1000  # 繰り返しでパターンを作る
    test_files['text'] = text_data.encode('utf-8')
    
    # 2. 構造化データ（JSON形式）
    json_data = '{"id": %d, "name": "user_%d", "email": "user_%d@example.com", "active": %s},' % (1, 1, 1, 'true')
    json_data = '[' + (json_data * 1000)[:-1] + ']'
    test_files['json'] = json_data.encode('utf-8')
    
    # 3. バイナリデータ（低圧縮率期待）
    import random
    random.seed(42)
    binary_data = bytes([random.randint(0, 255) for _ in range(50000)])
    test_files['binary'] = binary_data
    
    # 4. 繰り返しパターンデータ
    pattern_data = b'ABCDEFGHIJ' * 5000
    test_files['pattern'] = pattern_data
    
    # 5. 実際のファイルからのデータ（存在する場合）
    sample_dir = Path("sample")
    if sample_dir.exists():
        for file_path in sample_dir.glob("*.txt"):
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
                    if len(content) > 1000:  # 最低サイズ確保
                        test_files[f'real_{file_path.stem}'] = content[:100000]  # 最大100KB
                        break
            except:
                pass
    
    return test_files

def analyze_compression_ratio(data_name, data, engines):
    """圧縮率とパフォーマンスの詳細分析"""
    print(f"\n=== {data_name} データ分析 ===")
    print(f"原始データサイズ: {len(data):,} bytes")
    
    results = {}
    
    for engine_name, engine in engines.items():
        try:
            # 圧縮実行
            start_time = time.time()
            
            if 'NEXUS TMC' in engine_name:
                # NEXUS TMCの場合はメタデータも取得
                compressed, meta = engine['compress_with_meta'](data)
            else:
                compressed = engine['compress'](data)
                meta = None
            
            compress_time = time.time() - start_time
            
            # 展開実行
            start_time = time.time()
            
            if 'NEXUS TMC' in engine_name:
                decompressed = engine['decompress_with_meta'](compressed, meta)
            else:
                decompressed = engine['decompress'](compressed)
                
            decompress_time = time.time() - start_time
            
            # データ整合性チェック
            if decompressed != data:
                print(f"⚠️ {engine_name}: データ整合性エラー!")
                continue
            
            # 圧縮率計算
            compression_ratio = len(compressed) / len(data)
            space_saved = (1 - compression_ratio) * 100
            
            # 速度計算
            compress_speed = len(data) / (1024 * 1024 * compress_time)  # MB/s
            decompress_speed = len(data) / (1024 * 1024 * decompress_time)  # MB/s
            
            results[engine_name] = {
                'compressed_size': len(compressed),
                'compression_ratio': compression_ratio,
                'space_saved': space_saved,
                'compress_time': compress_time,
                'decompress_time': decompress_time,
                'compress_speed': compress_speed,
                'decompress_speed': decompress_speed
            }
            
            print(f"\n{engine_name}:")
            print(f"  圧縮後サイズ: {len(compressed):,} bytes")
            print(f"  圧縮率: {compression_ratio:.3f}")
            print(f"  容量削減: {space_saved:.1f}%")
            print(f"  圧縮速度: {compress_speed:.1f} MB/s")
            print(f"  展開速度: {decompress_speed:.1f} MB/s")
            
        except Exception as e:
            print(f"❌ {engine_name}: エラー - {e}")
            results[engine_name] = None
    
    return results

def compare_engines():
    """各エンジンの比較分析"""
    print("NEXUS TMC 軽量モード vs Zstandard 圧縮率比較分析")
    print("=" * 60)
    
    # エンジン設定
    nexus_lightweight = NEXUSTMCLightweight()
    
    engines = {
        'Zstandard (レベル1)': {
            'compress': lambda data: zstd.compress(data, level=1),
            'decompress': lambda data: zstd.decompress(data)
        },
        'Zstandard (レベル3)': {
            'compress': lambda data: zstd.compress(data, level=3),
            'decompress': lambda data: zstd.decompress(data)
        },
        'Zstandard (レベル6)': {
            'compress': lambda data: zstd.compress(data, level=6),
            'decompress': lambda data: zstd.decompress(data)
        },
        'Zstandard (レベル9)': {
            'compress': lambda data: zstd.compress(data, level=9),
            'decompress': lambda data: zstd.decompress(data)
        },
        'NEXUS TMC 軽量': {
            'compress_with_meta': nexus_lightweight.compress_fast,
            'decompress_with_meta': nexus_lightweight.decompress_fast
        }
    }
    
    # テストデータ生成
    test_data = create_test_data()
    
    # 各データセットで分析
    all_results = {}
    for data_name, data in test_data.items():
        all_results[data_name] = analyze_compression_ratio(data_name, data, engines)
    
    # 総合比較
    print("\n" + "=" * 60)
    print("総合比較サマリー")
    print("=" * 60)
    
    # 平均値計算
    engine_averages = {}
    for engine_name in engines.keys():
        ratios = []
        speeds = []
        for data_name, results in all_results.items():
            if results.get(engine_name):
                ratios.append(results[engine_name]['compression_ratio'])
                speeds.append(results[engine_name]['compress_speed'])
        
        if ratios:
            engine_averages[engine_name] = {
                'avg_ratio': sum(ratios) / len(ratios),
                'avg_speed': sum(speeds) / len(speeds)
            }
    
    print("\n平均圧縮率と速度:")
    for engine_name, avg in engine_averages.items():
        print(f"{engine_name}:")
        print(f"  平均圧縮率: {avg['avg_ratio']:.3f}")
        print(f"  平均圧縮速度: {avg['avg_speed']:.1f} MB/s")
        print(f"  平均容量削減: {(1-avg['avg_ratio'])*100:.1f}%")
    
    # トレードオフ分析
    print("\n" + "=" * 60)
    print("トレードオフ分析")
    print("=" * 60)
    
    if 'NEXUS TMC 軽量' in engine_averages and 'Zstandard (レベル3)' in engine_averages:
        nexus_light = engine_averages['NEXUS TMC 軽量']
        zstd_3 = engine_averages['Zstandard (レベル3)']
        
        ratio_diff = (nexus_light['avg_ratio'] - zstd_3['avg_ratio']) * 100
        speed_diff = (nexus_light['avg_speed'] / zstd_3['avg_speed'] - 1) * 100
        
        print(f"\nNEXUS TMC 軽量 vs Zstandard レベル3:")
        print(f"  圧縮率差: {ratio_diff:+.1f}% (正の値=軽量モードの方が低圧縮)")
        print(f"  速度差: {speed_diff:+.1f}% (正の値=軽量モードの方が高速)")
        
        if ratio_diff > 0:
            print(f"  → 軽量モードは圧縮率で {ratio_diff:.1f}% 劣るが、速度で {speed_diff:.1f}% 優位")
        else:
            print(f"  → 軽量モードは圧縮率で {-ratio_diff:.1f}% 優位、速度でも {speed_diff:.1f}% 優位")

def detailed_ratio_breakdown():
    """詳細な圧縮率分解分析"""
    print("\n" + "=" * 60)
    print("詳細圧縮率分解分析")
    print("=" * 60)
    
    # 段階的圧縮分析用のサンプルデータ
    sample_text = "The quick brown fox jumps over the lazy dog. " * 2000
    data = sample_text.encode('utf-8')
    
    print(f"分析対象データサイズ: {len(data):,} bytes")
    
    # 各段階での圧縮効果
    print("\n圧縮段階別効果:")
    
    # 1. 生データ
    print(f"1. 生データ: {len(data):,} bytes")
    
    # 2. Zstandard各レベル
    for level in [1, 3, 6, 9, 15]:
        try:
            compressed = zstd.compress(data, level=level)
            ratio = len(compressed) / len(data)
            print(f"2. Zstandard レベル{level}: {len(compressed):,} bytes (圧縮率: {ratio:.3f})")
        except:
            pass
    
    # 3. NEXUS TMC軽量モード
    try:
        nexus_light = NEXUSTMCLightweight()
        compressed = nexus_light.compress(data)
        ratio = len(compressed) / len(data)
        print(f"3. NEXUS TMC 軽量: {len(compressed):,} bytes (圧縮率: {ratio:.3f})")
    except Exception as e:
        print(f"3. NEXUS TMC 軽量: エラー - {e}")

if __name__ == "__main__":
    try:
        compare_engines()
        detailed_ratio_breakdown()
        
        print("\n" + "=" * 60)
        print("分析完了!")
        print("軽量モードは速度重視のため、圧縮率では多少の妥協がありますが、")
        print("実用的なバランスを提供します。")
        print("=" * 60)
        
    except Exception as e:
        print(f"分析中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
