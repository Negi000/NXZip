#!/usr/bin/env python3
"""
NXZip TMC v9.1 パフォーマンステスト
SPE + NEXUS TMC統合テスト & 可逆性確認
"""

import os
import sys
import time
import random
import hashlib
from pathlib import Path

# NXZip-Pythonパスを追加
sys.path.insert(0, str(Path(__file__).parent / "NXZip-Python"))

try:
    from nxzip.engine.nexus_tmc_v91_modular import NEXUSTMCEngineV91
    print("✅ NEXUSTMCEngineV91 インポート成功")
except ImportError as e:
    print(f"❌ NEXUSTMCEngineV91 インポートエラー: {e}")
    sys.exit(1)


def generate_test_data():
    """多様なテストデータを生成"""
    test_cases = {}
    
    # 1. テキストデータ（反復パターン）
    text_data = "Hello World! " * 1000 + "NXZip TMC Test Data " * 500
    test_cases["text_repetitive"] = text_data.encode('utf-8')
    
    # 2. 自然言語テキスト
    natural_text = """
    NXZip TMC v9.1は次世代モジュラー圧縮プラットフォームです。
    Transform-Model-Code圧縮フレームワークにより、高度な圧縮率を実現します。
    SPE（Structure-Preserving Encryption）統合により、データの構造を保持しながら
    暗号化を行います。分離されたコンポーネント統合により、各モジュールが
    独立して最適化され、全体として高性能を発揮します。
    """ * 200
    test_cases["text_natural"] = natural_text.encode('utf-8')
    
    # 3. 数値配列（整数）
    int_array = bytes([i % 256 for i in range(0, 10000)])
    test_cases["sequential_int"] = int_array
    
    # 4. 浮動小数点数配列（シミュレート）
    float_data = b''.join([
        int(1000 * (0.5 + 0.3 * (i % 100) / 100)).to_bytes(4, 'little') 
        for i in range(2500)
    ])
    test_cases["float_array"] = float_data
    
    # 5. ランダムバイナリ（圧縮困難）
    random.seed(42)  # 再現可能なランダム
    random_data = bytes([random.randint(0, 255) for _ in range(5000)])
    test_cases["generic_binary"] = random_data
    
    # 6. 混合データ
    mixed_data = (
        "HEADER:" + "="*50 + "\n"
        + text_data[:500] + "\n"
        + "BINARY_SECTION:\n"
    ).encode('utf-8') + random_data[:1000] + int_array[:1000]
    test_cases["mixed_data"] = mixed_data
    
    return test_cases


def calculate_hash(data: bytes) -> str:
    """データのSHA256ハッシュを計算"""
    return hashlib.sha256(data).hexdigest()


def test_engine_mode(engine_name: str, lightweight_mode: bool, test_data: dict):
    """特定のエンジンモードをテスト"""
    print(f"\n{'='*60}")
    print(f"🧪 {engine_name} テスト開始")
    print(f"{'='*60}")
    
    try:
        # エンジン初期化
        engine = NEXUSTMCEngineV91(
            max_workers=2 if lightweight_mode else 4,
            chunk_size=256*1024 if lightweight_mode else 1024*1024,
            lightweight_mode=lightweight_mode
        )
        print(f"✅ エンジン初期化成功: {engine_name}")
        
        results = {}
        total_original_size = 0
        total_compressed_size = 0
        total_compression_time = 0.0
        total_decompression_time = 0.0
        reversibility_tests = 0
        reversibility_passed = 0
        
        for data_type, original_data in test_data.items():
            print(f"\n--- {data_type} テスト ---")
            original_hash = calculate_hash(original_data)
            original_size = len(original_data)
            total_original_size += original_size
            
            print(f"📊 元データ: {original_size:,} bytes, Hash: {original_hash[:16]}...")
            
            # 圧縮テスト
            start_time = time.time()
            try:
                compressed_data, compression_info = engine.compress(original_data)
                compression_time = time.time() - start_time
                total_compression_time += compression_time
                
                compressed_size = len(compressed_data)
                total_compressed_size += compressed_size
                compression_ratio = compression_info.get('compression_ratio', 0)
                
                print(f"🗜️  圧縮完了: {compressed_size:,} bytes ({compression_ratio:.1f}% 圧縮)")
                print(f"⏱️  圧縮時間: {compression_time:.3f}秒")
                print(f"📈 スループット: {compression_info.get('throughput_mbps', 0):.1f} MB/s")
                print(f"🔧 エンジン: {compression_info.get('engine_version', 'Unknown')}")
                print(f"🎯 データタイプ: {compression_info.get('data_type', 'Unknown')}")
                print(f"🔄 変換適用: {compression_info.get('transform_applied', False)}")
                print(f"🔐 SPE有効: {compression_info.get('spe_enabled', False)}")
                
                # 解凍テスト & 可逆性確認
                start_time = time.time()
                try:
                    decompressed_data = engine.decompress(compressed_data, compression_info)
                    decompression_time = time.time() - start_time
                    total_decompression_time += decompression_time
                    
                    decompressed_hash = calculate_hash(decompressed_data)
                    reversibility_tests += 1
                    
                    if decompressed_hash == original_hash:
                        reversibility_passed += 1
                        print(f"✅ 可逆性確認成功: Hash一致")
                        print(f"⏱️  解凍時間: {decompression_time:.3f}秒")
                        
                        results[data_type] = {
                            'original_size': original_size,
                            'compressed_size': compressed_size,
                            'compression_ratio': compression_ratio,
                            'compression_time': compression_time,
                            'decompression_time': decompression_time,
                            'reversible': True,
                            'compression_info': compression_info
                        }
                    else:
                        print(f"❌ 可逆性エラー: Hash不一致")
                        print(f"   元Hash:   {original_hash[:16]}...")
                        print(f"   解凍Hash: {decompressed_hash[:16]}...")
                        
                        results[data_type] = {
                            'original_size': original_size,
                            'compressed_size': compressed_size,
                            'compression_ratio': compression_ratio,
                            'compression_time': compression_time,
                            'decompression_time': decompression_time,
                            'reversible': False,
                            'error': 'Hash不一致'
                        }
                        
                except Exception as e:
                    print(f"❌ 解凍エラー: {e}")
                    results[data_type] = {
                        'original_size': original_size,
                        'compressed_size': compressed_size,
                        'compression_ratio': compression_ratio,
                        'compression_time': compression_time,
                        'reversible': False,
                        'error': f'解凍エラー: {e}'
                    }
                    
            except Exception as e:
                print(f"❌ 圧縮エラー: {e}")
                results[data_type] = {
                    'original_size': original_size,
                    'error': f'圧縮エラー: {e}'
                }
        
        # 総合統計
        print(f"\n{'='*40}")
        print(f"📊 {engine_name} 総合結果")
        print(f"{'='*40}")
        print(f"📁 総入力サイズ: {total_original_size:,} bytes")
        print(f"🗜️  総圧縮サイズ: {total_compressed_size:,} bytes")
        
        if total_original_size > 0:
            overall_ratio = (1 - total_compressed_size / total_original_size) * 100
            print(f"📈 総合圧縮率: {overall_ratio:.1f}%")
        
        print(f"⏱️  総圧縮時間: {total_compression_time:.3f}秒")
        print(f"⏱️  総解凍時間: {total_decompression_time:.3f}秒")
        
        if total_compression_time > 0:
            overall_throughput = (total_original_size / (1024 * 1024)) / total_compression_time
            print(f"📈 平均スループット: {overall_throughput:.1f} MB/s")
        
        print(f"🔄 可逆性テスト: {reversibility_passed}/{reversibility_tests} 成功")
        if reversibility_tests > 0:
            reversibility_rate = (reversibility_passed / reversibility_tests) * 100
            print(f"✅ 可逆性成功率: {reversibility_rate:.1f}%")
        
        # エンジン統計
        try:
            engine_stats = engine.get_nxzip_stats()
            print(f"\n🔧 エンジン統計:")
            print(f"   TMC変換効率: {engine_stats.get('tmc_transform_efficiency', 0):.1f}%")
            print(f"   分離コンポーネント: {engine_stats.get('modular_components_active', 0)}個")
            print(f"   処理チャンク数: {engine_stats.get('chunks_processed', 0)}")
        except Exception as e:
            print(f"⚠️ エンジン統計取得エラー: {e}")
        
        return results
        
    except Exception as e:
        print(f"❌ {engine_name} テスト失敗: {e}")
        return {}


def main():
    """メインテスト実行"""
    print("🚀 NXZip TMC v9.1 パフォーマンステスト開始")
    print("📦 SPE + NEXUS TMC統合テスト & 可逆性確認")
    
    # テストデータ生成
    print("\n📋 テストデータ生成中...")
    test_data = generate_test_data()
    
    total_test_size = sum(len(data) for data in test_data.values())
    print(f"✅ テストデータ生成完了: {len(test_data)}種類, 総サイズ {total_test_size:,} bytes")
    
    # 各モードをテスト
    all_results = {}
    
    # 1. 軽量モード（Zstandardレベル目標）
    lightweight_results = test_engine_mode(
        "軽量モード (Zstandardレベル)", 
        lightweight_mode=True, 
        test_data=test_data
    )
    all_results["lightweight"] = lightweight_results
    
    # 2. 通常モード（7-Zip超越目標）
    normal_results = test_engine_mode(
        "通常モード (7-Zip超越レベル)", 
        lightweight_mode=False, 
        test_data=test_data
    )
    all_results["normal"] = normal_results
    
    # 比較分析
    print(f"\n{'='*60}")
    print(f"📊 モード比較分析")
    print(f"{'='*60}")
    
    for data_type in test_data.keys():
        if (data_type in lightweight_results and data_type in normal_results and
            'compression_ratio' in lightweight_results[data_type] and 
            'compression_ratio' in normal_results[data_type]):
            
            light_ratio = lightweight_results[data_type]['compression_ratio']
            normal_ratio = normal_results[data_type]['compression_ratio']
            light_time = lightweight_results[data_type]['compression_time']
            normal_time = normal_results[data_type]['compression_time']
            
            print(f"\n--- {data_type} ---")
            print(f"圧縮率: 軽量 {light_ratio:.1f}% vs 通常 {normal_ratio:.1f}%")
            print(f"速度:   軽量 {light_time:.3f}s vs 通常 {normal_time:.3f}s")
            
            if normal_time > 0:
                speed_improvement = light_time / normal_time
                print(f"速度向上: {speed_improvement:.2f}x")
    
    # 今後の課題
    print(f"\n{'='*60}")
    print(f"🎯 今後の課題と改善点")
    print(f"{'='*60}")
    
    # 可逆性問題の分析
    reversibility_issues = []
    for mode, results in all_results.items():
        for data_type, result in results.items():
            if not result.get('reversible', False):
                reversibility_issues.append(f"{mode}モード - {data_type}")
    
    if reversibility_issues:
        print(f"🔴 可逆性問題:")
        for issue in reversibility_issues:
            print(f"   - {issue}")
    else:
        print(f"✅ 全テストケースで可逆性確認済み")
    
    # パフォーマンス改善点
    print(f"\n🔧 パフォーマンス改善点:")
    print(f"1. TMC変換効率の向上")
    print(f"2. 分離コンポーネント間の最適化")
    print(f"3. SPE統合の完全実装")
    print(f"4. 並列処理パイプラインの調整")
    print(f"5. メモリ管理の最適化")
    
    print(f"\n✅ テスト完了!")


if __name__ == "__main__":
    main()
