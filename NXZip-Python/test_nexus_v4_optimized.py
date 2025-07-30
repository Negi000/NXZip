#!/usr/bin/env python3
"""
NEXUS Optimized Engine v4.0 実ファイル高速テスト
パフォーマンス問題を解決した高速版のテスト
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'nxzip', 'engine'))

import time
import hashlib
from pathlib import Path
from nexus_optimized_v4 import NEXUSOptimizedEngine, OptimizedConfig, simulate_optimized_decompression


def test_optimized_real_files():
    """最適化エンジン実ファイルテスト"""
    print("⚡ NEXUS最適化エンジン v4.0 実ファイル高速テスト")
    print("=" * 80)
    
    # サンプルディレクトリ
    sample_dir = Path("sample")
    
    if not sample_dir.exists():
        print(f"❌ サンプルディレクトリが見つかりません: {sample_dir}")
        return
    
    # 高速設定
    fast_config = OptimizedConfig(
        max_threads=4,  # 適度なスレッド数
        chunk_size_mb=1.0,  # 1MBチャンク
        memory_limit_gb=6.0,
        fast_mode=True,
        skip_deep_analysis=False,  # 軽量解析は有効
        compression_level=6  # バランス重視
    )
    
    # 超高速設定
    ultra_fast_config = OptimizedConfig(
        max_threads=2,
        chunk_size_mb=2.0,  # 大きなチャンク
        memory_limit_gb=4.0,
        fast_mode=True,
        skip_deep_analysis=True,  # 解析完全スキップ
        compression_level=3  # 高速圧縮
    )
    
    # テスト設定選択
    config_tests = [
        ('高速モード', fast_config),
        ('超高速モード', ultra_fast_config)
    ]
    
    # テスト対象ファイル選択
    test_files = []
    
    for file_path in sample_dir.iterdir():
        if file_path.is_file():
            try:
                file_size = file_path.stat().st_size
                if file_size < 50 * 1024 * 1024:  # 50MB未満で高速テスト
                    test_files.append({
                        'path': file_path,
                        'name': file_path.name,
                        'size': file_size,
                        'type': get_optimized_file_type(file_path)
                    })
            except:
                continue
    
    # ファイルサイズでソート
    test_files.sort(key=lambda x: x['size'])
    test_files = test_files[:8]  # 最初の8ファイルのみ
    
    print(f"🔬 高速テスト対象: {len(test_files)} ファイル")
    for file_info in test_files:
        print(f"   📄 {file_info['name']} ({file_info['size'] / 1024:.1f}KB) [{file_info['type']}]")
    
    # 各設定でテスト
    all_results = []
    
    for config_name, config in config_tests:
        print(f"\n{'='*80}")
        print(f"⚡ {config_name}テスト開始")
        print(f"{'='*80}")
        
        engine = NEXUSOptimizedEngine(config)
        config_results = []
        
        for i, file_info in enumerate(test_files):
            print(f"\n{'='*60}")
            print(f"🧪 {config_name} {i+1}/{len(test_files)}: {file_info['name']}")
            print(f"   📁 ファイルタイプ: {file_info['type']}")
            print(f"   📊 ファイルサイズ: {file_info['size']:,} bytes ({file_info['size']/1024:.1f}KB)")
            
            try:
                # ファイル読み込み
                with open(file_info['path'], 'rb') as f:
                    original_data = f.read()
                
                original_hash = hashlib.sha256(original_data).hexdigest()
                
                # 最適化圧縮
                print("   ⚡ 最適化圧縮実行...")
                compress_start = time.perf_counter()
                
                compressed_data = engine.optimized_compress(
                    original_data, 
                    file_info['type'], 
                    'fast'
                )
                
                compress_time = time.perf_counter() - compress_start
                
                compression_ratio = (1 - len(compressed_data) / len(original_data)) * 100
                throughput = file_info['size'] / 1024 / 1024 / compress_time
                
                print(f"   ✅ 圧縮完了!")
                print(f"      📈 圧縮率: {compression_ratio:.2f}%")
                print(f"      ⚡ スループット: {throughput:.2f}MB/s")
                print(f"      ⏱️ 圧縮時間: {compress_time:.3f}秒")
                
                # 可逆性検証
                print("   🔍 可逆性検証...")
                try:
                    decompressed_data = simulate_optimized_decompression(compressed_data)
                    decompressed_hash = hashlib.sha256(decompressed_data).hexdigest()
                    is_reversible = (original_hash == decompressed_hash and 
                                   len(original_data) == len(decompressed_data))
                    
                    print(f"      🏆 可逆性: {'✅ 完全' if is_reversible else '❌ 不完全'}")
                except Exception as e:
                    print(f"      ⚠️ 可逆性検証エラー: {e}")
                    is_reversible = False
                
                # 他圧縮形式との簡易比較
                print("   📊 基準比較...")
                lzma_comparison = quick_lzma_comparison(original_data)
                
                if lzma_comparison:
                    ratio_diff = compression_ratio - lzma_comparison['ratio']
                    print(f"      LZMA基準: {lzma_comparison['ratio']:.2f}% (NEXUS{ratio_diff:+.2f}%)")
                    performance_vs_lzma = f"{ratio_diff:+.1f}%"
                else:
                    performance_vs_lzma = "N/A"
                
                # 結果保存
                result = {
                    'config': config_name,
                    'name': file_info['name'],
                    'type': file_info['type'],
                    'original_size': len(original_data),
                    'compressed_size': len(compressed_data),
                    'compression_ratio': compression_ratio,
                    'compress_time': compress_time,
                    'throughput': throughput,
                    'is_reversible': is_reversible,
                    'vs_lzma': performance_vs_lzma
                }
                
                config_results.append(result)
                
            except Exception as e:
                print(f"   ❌ テストエラー: {str(e)}")
                
                config_results.append({
                    'config': config_name,
                    'name': file_info['name'],
                    'type': file_info['type'],
                    'error': str(e),
                    'is_reversible': False
                })
        
        all_results.extend(config_results)
        
        # 設定別サマリー
        print(f"\n📊 {config_name} サマリー:")
        successful = [r for r in config_results if not r.get('error') and r.get('is_reversible', False)]
        if successful:
            avg_ratio = sum(r['compression_ratio'] for r in successful) / len(successful)
            avg_throughput = sum(r['throughput'] for r in successful) / len(successful)
            
            print(f"   ✅ 成功: {len(successful)}/{len(config_results)}")
            print(f"   📈 平均圧縮率: {avg_ratio:.2f}%")
            print(f"   ⚡ 平均スループット: {avg_throughput:.2f}MB/s")
        
        # エンジンレポート
        engine_report = engine.get_optimization_report()
        stats = engine_report['performance_stats']
        print(f"   📊 エンジン統計:")
        print(f"      処理ファイル: {stats['total_files_processed']}")
        print(f"      平均スループット: {stats['average_throughput']:.2f}MB/s")
    
    # === 総合レポート ===
    print(f"\n{'='*80}")
    print(f"📈 NEXUS v4.0 最適化テスト総合レポート")
    print(f"{'='*80}")
    
    if all_results:
        # 設定別比較
        config_stats = {}
        for result in all_results:
            if result.get('error'):
                continue
                
            config = result['config']
            if config not in config_stats:
                config_stats[config] = []
            config_stats[config].append(result)
        
        print(f"🔄 設定別パフォーマンス比較:")
        for config_name, results in config_stats.items():
            successful = [r for r in results if r.get('is_reversible', False)]
            if successful:
                avg_ratio = sum(r['compression_ratio'] for r in successful) / len(successful)
                avg_throughput = sum(r['throughput'] for r in successful) / len(successful)
                avg_time = sum(r['compress_time'] for r in successful) / len(successful)
                
                print(f"\n   {config_name}:")
                print(f"      📈 平均圧縮率: {avg_ratio:.2f}%")
                print(f"      ⚡ 平均スループット: {avg_throughput:.2f}MB/s")
                print(f"      ⏱️ 平均処理時間: {avg_time:.3f}秒")
                print(f"      ✅ 成功率: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")
        
        # ファイルタイプ別分析
        print(f"\n📁 ファイルタイプ別最適パフォーマンス:")
        type_best = {}
        for result in all_results:
            if result.get('error') or not result.get('is_reversible', False):
                continue
            
            file_type = result['type']
            if file_type not in type_best or result['throughput'] > type_best[file_type]['throughput']:
                type_best[file_type] = result
        
        for file_type, best_result in type_best.items():
            print(f"   {file_type}:")
            print(f"      🏆 最高スループット: {best_result['throughput']:.2f}MB/s ({best_result['config']})")
            print(f"      📈 圧縮率: {best_result['compression_ratio']:.2f}%")
            print(f"      📄 ファイル: {best_result['name']}")
        
        # 全体統計
        all_successful = [r for r in all_results if not r.get('error') and r.get('is_reversible', False)]
        if all_successful:
            best_throughput = max(all_successful, key=lambda r: r['throughput'])
            best_compression = max(all_successful, key=lambda r: r['compression_ratio'])
            
            print(f"\n🥇 全体最高記録:")
            print(f"   ⚡ 最高スループット: {best_throughput['throughput']:.2f}MB/s")
            print(f"      📄 {best_throughput['name']} ({best_throughput['config']})")
            print(f"   📈 最高圧縮率: {best_compression['compression_ratio']:.2f}%")
            print(f"      📄 {best_compression['name']} ({best_compression['config']})")
            
            # LZMA比較統計
            lzma_improvements = [r for r in all_successful if r.get('vs_lzma', 'N/A') != 'N/A' and '+' in str(r['vs_lzma'])]
            if lzma_improvements:
                print(f"\n🚀 LZMA超越実績:")
                print(f"   🏆 改善ファイル: {len(lzma_improvements)}/{len(all_successful)}")
                print(f"   📊 改善率: {len(lzma_improvements)/len(all_successful)*100:.1f}%")
    
    print(f"\n🎉 NEXUS v4.0 最適化テスト完了!")
    return all_results


def get_optimized_file_type(file_path: Path) -> str:
    """最適化ファイルタイプ判定"""
    suffix = file_path.suffix.lower()
    
    type_mapping = {
        '.txt': 'テキスト',
        '.mp4': '動画',
        '.mp3': '音楽',
        '.wav': '音楽',
        '.jpg': '画像',
        '.jpeg': '画像',
        '.png': '画像',
        '.gif': '画像',
        '.pdf': 'ドキュメント',
        '.py': 'プログラム',
        '.js': 'プログラム',
        '.7z': '圧縮アーカイブ',
        '.zip': '圧縮アーカイブ',
        '.rar': '圧縮アーカイブ'
    }
    
    return type_mapping.get(suffix, 'その他')


def quick_lzma_comparison(data: bytes) -> dict:
    """高速LZMA比較"""
    try:
        import lzma
        start_time = time.perf_counter()
        lzma_compressed = lzma.compress(data, preset=6)  # バランス設定
        lzma_time = time.perf_counter() - start_time
        lzma_ratio = (1 - len(lzma_compressed) / len(data)) * 100
        
        return {
            'size': len(lzma_compressed),
            'ratio': lzma_ratio,
            'time': lzma_time
        }
    except:
        return None


if __name__ == "__main__":
    test_optimized_real_files()
