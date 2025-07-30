#!/usr/bin/env python3
"""
NEXUS Enhanced Engine v5.0 実ファイルテスト
可逆性保証 & 高圧縮率版の検証
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'nxzip', 'engine'))

import time
import hashlib
from pathlib import Path
from nexus_enhanced_v5 import NEXUSEnhancedEngine, EnhancedConfig, simulate_enhanced_decompression


def test_enhanced_real_files():
    """拡張エンジン実ファイルテスト"""
    print("🔒 NEXUS拡張エンジン v5.0 実ファイルテスト - 可逆性保証 & 高圧縮率")
    print("=" * 80)
    
    # サンプルディレクトリ
    sample_dir = Path("sample")
    
    if not sample_dir.exists():
        print(f"❌ サンプルディレクトリが見つかりません: {sample_dir}")
        return
    
    # 可逆性保証設定
    reversible_config = EnhancedConfig(
        max_threads=4,
        chunk_size_mb=1.0,
        memory_limit_gb=6.0,
        ensure_reversibility=True,  # 可逆性強制保証
        strict_mode=True,
        aggressive_compression=True,  # 高圧縮
        multi_pass_compression=True,
        compression_level=9  # 最高圧縮
    )
    
    # 超高圧縮設定
    ultra_config = EnhancedConfig(
        max_threads=6,
        chunk_size_mb=0.5,  # 小さなチャンク
        memory_limit_gb=8.0,
        ensure_reversibility=True,
        strict_mode=True,
        aggressive_compression=True,
        multi_pass_compression=True,
        adaptive_algorithms=True,
        compression_level=9,
        enable_entropy_coding=True
    )
    
    # テスト設定
    test_configs = [
        ('可逆性保証モード', reversible_config),
        ('超高圧縮モード', ultra_config)
    ]
    
    # テスト対象ファイル選択
    test_files = []
    
    for file_path in sample_dir.iterdir():
        if file_path.is_file():
            try:
                file_size = file_path.stat().st_size
                if file_size < 30 * 1024 * 1024:  # 30MB未満で集中テスト
                    test_files.append({
                        'path': file_path,
                        'name': file_path.name,
                        'size': file_size,
                        'type': get_enhanced_file_type(file_path)
                    })
            except:
                continue
    
    # ファイルサイズでソート
    test_files.sort(key=lambda x: x['size'])
    test_files = test_files[:6]  # 最初の6ファイル
    
    print(f"🔬 拡張テスト対象: {len(test_files)} ファイル")
    for file_info in test_files:
        print(f"   📄 {file_info['name']} ({file_info['size'] / 1024:.1f}KB) [{file_info['type']}]")
    
    # 各設定でテスト
    all_results = []
    
    for config_name, config in test_configs:
        print(f"\n{'='*80}")
        print(f"🔒 {config_name}テスト開始")
        print(f"{'='*80}")
        
        engine = NEXUSEnhancedEngine(config)
        config_results = []
        
        for i, file_info in enumerate(test_files):
            print(f"\n{'='*70}")
            print(f"🧪 {config_name} {i+1}/{len(test_files)}: {file_info['name']}")
            print(f"   📁 ファイルタイプ: {file_info['type']}")
            print(f"   📊 ファイルサイズ: {file_info['size']:,} bytes ({file_info['size']/1024:.1f}KB)")
            
            try:
                # ファイル読み込み
                with open(file_info['path'], 'rb') as f:
                    original_data = f.read()
                
                original_hash = hashlib.sha256(original_data).hexdigest()
                print(f"   🔑 元データハッシュ: {original_hash[:16]}...")
                
                # 拡張圧縮
                print("   🔒 拡張圧縮実行...")
                compress_start = time.perf_counter()
                
                compressed_data = engine.enhanced_compress(
                    original_data, 
                    file_info['type'], 
                    'maximum'
                )
                
                compress_time = time.perf_counter() - compress_start
                
                compression_ratio = (1 - len(compressed_data) / len(original_data)) * 100
                throughput = file_info['size'] / 1024 / 1024 / compress_time
                
                print(f"   ✅ 圧縮完了!")
                print(f"      📈 圧縮率: {compression_ratio:.2f}%")
                print(f"      ⚡ スループット: {throughput:.2f}MB/s")
                print(f"      ⏱️ 圧縮時間: {compress_time:.3f}秒")
                
                # 厳格可逆性検証
                print("   🔍 厳格可逆性検証...")
                verify_start = time.perf_counter()
                
                try:
                    decompressed_data = simulate_enhanced_decompression(compressed_data)
                    verify_time = time.perf_counter() - verify_start
                    
                    # 複数検証
                    size_match = len(original_data) == len(decompressed_data)
                    decompressed_hash = hashlib.sha256(decompressed_data).hexdigest()
                    hash_match = original_hash == decompressed_hash
                    bytes_match = original_data == decompressed_data
                    
                    is_reversible = size_match and hash_match and bytes_match
                    
                    print(f"      📏 サイズ一致: {'✅' if size_match else '❌'} ({len(original_data)} vs {len(decompressed_data)})")
                    print(f"      🔑 ハッシュ一致: {'✅' if hash_match else '❌'}")
                    print(f"      🔒 バイト一致: {'✅' if bytes_match else '❌'}")
                    print(f"      🏆 可逆性: {'✅ 完全' if is_reversible else '❌ 不完全'}")
                    print(f"      ⏱️ 検証時間: {verify_time:.3f}秒")
                    
                    if not is_reversible and size_match:
                        # 差分解析
                        print(f"      🔍 差分解析:")
                        find_byte_differences(original_data, decompressed_data)
                        
                except Exception as e:
                    print(f"      ❌ 可逆性検証エラー: {e}")
                    is_reversible = False
                
                # 他圧縮形式との詳細比較
                print("   📊 詳細比較分析...")
                comparison = comprehensive_compression_comparison(original_data)
                
                best_ratio = compression_ratio
                improvements = []
                
                for method, comp_result in comparison.items():
                    if comp_result and 'ratio' in comp_result:
                        ratio_diff = compression_ratio - comp_result['ratio']
                        speed_ratio = throughput / (comp_result.get('throughput', 1))
                        
                        print(f"      {method:>12}: {comp_result['ratio']:6.2f}% | {comp_result.get('throughput', 0):.1f}MB/s | NEXUS{ratio_diff:+.1f}% (速度x{speed_ratio:.1f})")
                        
                        if ratio_diff > 0:
                            improvements.append(f"{method}比+{ratio_diff:.1f}%")
                
                if improvements:
                    print(f"      🏆 圧縮率改善: {', '.join(improvements)}")
                
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
                    'comparison': comparison,
                    'improvements': improvements
                }
                
                config_results.append(result)
                
            except Exception as e:
                print(f"   ❌ テストエラー: {str(e)}")
                import traceback
                traceback.print_exc()
                
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
        processed = [r for r in config_results if not r.get('error')]
        
        if processed:
            avg_ratio = sum(r['compression_ratio'] for r in processed) / len(processed)
            avg_throughput = sum(r['throughput'] for r in processed) / len(processed)
            reversible_rate = len(successful) / len(processed) * 100
            
            print(f"   ✅ 処理成功: {len(processed)}/{len(config_results)}")
            print(f"   🔒 可逆成功: {len(successful)}/{len(processed)} ({reversible_rate:.1f}%)")
            print(f"   📈 平均圧縮率: {avg_ratio:.2f}%")
            print(f"   ⚡ 平均スループット: {avg_throughput:.2f}MB/s")
            
            # 最高記録
            if successful:
                best_compression = max(successful, key=lambda r: r['compression_ratio'])
                best_speed = max(successful, key=lambda r: r['throughput'])
                
                print(f"   🥇 最高圧縮率: {best_compression['compression_ratio']:.2f}% ({best_compression['name']})")
                print(f"   🥇 最高速度: {best_speed['throughput']:.2f}MB/s ({best_speed['name']})")
        
        # エンジンレポート
        engine_report = engine.get_enhanced_report()
        stats = engine_report['performance_stats']
        print(f"   📊 エンジン統計:")
        print(f"      処理ファイル: {stats['total_files_processed']}")
        print(f"      平均スループット: {stats['average_throughput']:.2f}MB/s")
        print(f"      可逆成功率: {stats.get('reversibility_success_rate', 0):.1f}%")
    
    # === 総合分析レポート ===
    print(f"\n{'='*80}")
    print(f"🔒 NEXUS v5.0 拡張エンジン総合分析レポート")
    print(f"{'='*80}")
    
    if all_results:
        # 可逆性分析
        all_successful = [r for r in all_results if not r.get('error') and r.get('is_reversible', False)]
        all_processed = [r for r in all_results if not r.get('error')]
        
        print(f"🔒 可逆性分析:")
        print(f"   📊 総テスト数: {len(all_results)}")
        print(f"   ✅ 処理成功: {len(all_processed)}")
        print(f"   🔒 可逆成功: {len(all_successful)}")
        print(f"   📈 可逆成功率: {len(all_successful)/len(all_processed)*100:.1f}%" if all_processed else "N/A")
        
        # 設定別比較
        config_stats = {}
        for result in all_successful:
            config = result['config']
            if config not in config_stats:
                config_stats[config] = []
            config_stats[config].append(result)
        
        print(f"\n🔄 設定別パフォーマンス:")
        for config_name, results in config_stats.items():
            if results:
                avg_ratio = sum(r['compression_ratio'] for r in results) / len(results)
                avg_throughput = sum(r['throughput'] for r in results) / len(results)
                avg_time = sum(r['compress_time'] for r in results) / len(results)
                
                print(f"   {config_name}:")
                print(f"      📈 平均圧縮率: {avg_ratio:.2f}%")
                print(f"      ⚡ 平均スループット: {avg_throughput:.2f}MB/s")
                print(f"      ⏱️ 平均処理時間: {avg_time:.3f}秒")
                print(f"      ✅ 可逆成功: {len(results)}ファイル")
        
        # ファイルタイプ別最適性能
        print(f"\n📁 ファイルタイプ別最高性能:")
        type_best = {}
        for result in all_successful:
            file_type = result['type']
            if file_type not in type_best or result['compression_ratio'] > type_best[file_type]['compression_ratio']:
                type_best[file_type] = result
        
        for file_type, best_result in type_best.items():
            print(f"   {file_type}:")
            print(f"      🏆 最高圧縮率: {best_result['compression_ratio']:.2f}%")
            print(f"      ⚡ スループット: {best_result['throughput']:.2f}MB/s")
            print(f"      📄 ファイル: {best_result['name']} ({best_result['config']})")
            
            if best_result.get('improvements'):
                print(f"      🚀 改善実績: {', '.join(best_result['improvements'])}")
        
        # 全体最高記録
        if all_successful:
            overall_best_compression = max(all_successful, key=lambda r: r['compression_ratio'])
            overall_best_speed = max(all_successful, key=lambda r: r['throughput'])
            
            print(f"\n🥇 全体最高記録:")
            print(f"   📈 最高圧縮率: {overall_best_compression['compression_ratio']:.2f}%")
            print(f"      📄 {overall_best_compression['name']} ({overall_best_compression['config']})")
            print(f"   ⚡ 最高スループット: {overall_best_speed['throughput']:.2f}MB/s")
            print(f"      📄 {overall_best_speed['name']} ({overall_best_speed['config']})")
        
        # 改善統計
        all_improvements = []
        for result in all_successful:
            if result.get('improvements'):
                all_improvements.extend(result['improvements'])
        
        if all_improvements:
            print(f"\n🚀 圧縮改善統計:")
            print(f"   📊 改善記録: {len(all_improvements)} 件")
            print(f"   📈 改善率: {len(all_improvements)/len(all_successful)*100:.1f}%")
            
            # 改善分析
            improvement_methods = {}
            for improvement in all_improvements:
                method = improvement.split('比')[0]
                if method not in improvement_methods:
                    improvement_methods[method] = 0
                improvement_methods[method] += 1
            
            print(f"   🎯 改善対象:")
            for method, count in sorted(improvement_methods.items(), key=lambda x: x[1], reverse=True):
                print(f"      {method}: {count} 件")
    
    print(f"\n🎉 NEXUS v5.0 拡張エンジンテスト完了!")
    print(f"   🔒 可逆性保証システム検証完了")
    print(f"   📈 高圧縮率システム検証完了")
    return all_results


def get_enhanced_file_type(file_path: Path) -> str:
    """拡張ファイルタイプ判定"""
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


def comprehensive_compression_comparison(data: bytes) -> dict:
    """包括的圧縮比較"""
    results = {}
    
    # LZMA（複数プリセット）
    for preset in [1, 3, 6, 9]:
        try:
            import lzma
            start_time = time.perf_counter()
            compressed = lzma.compress(data, preset=preset)
            comp_time = time.perf_counter() - start_time
            ratio = (1 - len(compressed) / len(data)) * 100
            throughput = len(data) / 1024 / 1024 / comp_time if comp_time > 0 else 0
            
            results[f'LZMA-{preset}'] = {
                'size': len(compressed),
                'ratio': ratio,
                'time': comp_time,
                'throughput': throughput
            }
        except:
            results[f'LZMA-{preset}'] = None
    
    # GZIP
    try:
        import gzip
        start_time = time.perf_counter()
        compressed = gzip.compress(data, compresslevel=9)
        comp_time = time.perf_counter() - start_time
        ratio = (1 - len(compressed) / len(data)) * 100
        throughput = len(data) / 1024 / 1024 / comp_time if comp_time > 0 else 0
        
        results['GZIP'] = {
            'size': len(compressed),
            'ratio': ratio,
            'time': comp_time,
            'throughput': throughput
        }
    except:
        results['GZIP'] = None
    
    # BZIP2
    try:
        import bz2
        start_time = time.perf_counter()
        compressed = bz2.compress(data, compresslevel=9)
        comp_time = time.perf_counter() - start_time
        ratio = (1 - len(compressed) / len(data)) * 100
        throughput = len(data) / 1024 / 1024 / comp_time if comp_time > 0 else 0
        
        results['BZIP2'] = {
            'size': len(compressed),
            'ratio': ratio,
            'time': comp_time,
            'throughput': throughput
        }
    except:
        results['BZIP2'] = None
    
    # ZLIB
    try:
        import zlib
        start_time = time.perf_counter()
        compressed = zlib.compress(data, level=9)
        comp_time = time.perf_counter() - start_time
        ratio = (1 - len(compressed) / len(data)) * 100
        throughput = len(data) / 1024 / 1024 / comp_time if comp_time > 0 else 0
        
        results['ZLIB'] = {
            'size': len(compressed),
            'ratio': ratio,
            'time': comp_time,
            'throughput': throughput
        }
    except:
        results['ZLIB'] = None
    
    return results


def find_byte_differences(original: bytes, decompressed: bytes) -> None:
    """バイト差分詳細解析"""
    min_len = min(len(original), len(decompressed))
    differences = []
    
    for i in range(min_len):
        if original[i] != decompressed[i]:
            differences.append(i)
            if len(differences) >= 10:  # 最初の10個まで
                break
    
    if differences:
        print(f"         🔴 差分位置: {differences}")
        for pos in differences[:3]:  # 最初の3個の詳細
            start = max(0, pos - 4)
            end = min(len(original), pos + 5)
            print(f"         📍 位置{pos}: 元={original[start:end].hex()} vs 復元={decompressed[start:end].hex()}")
    
    if len(original) != len(decompressed):
        print(f"         📏 サイズ差: {len(original) - len(decompressed)} bytes")


if __name__ == "__main__":
    test_enhanced_real_files()
