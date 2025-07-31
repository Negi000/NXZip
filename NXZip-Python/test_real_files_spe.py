#!/usr/bin/env python3
"""
SPE+TMC Real File Testing Suite

実際のサンプルファイルでのSPE統合テスト
"""

import os
import sys
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List

# プロジェクトパスを追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from test_spe_integration import SPE_TMC_IntegratedEngine


def find_sample_files(sample_dir: str = "sample") -> List[Path]:
    """サンプルファイルを検索"""
    sample_path = Path(sample_dir)
    
    if not sample_path.exists():
        print(f"⚠️ Sample directory not found: {sample_path}")
        return []
    
    # 元ファイル（非圧縮）を優先
    extensions = ['.txt', '.mp4', '.wav', '.mp3', '.jpg', '.png']
    sample_files = []
    
    for ext in extensions:
        files = list(sample_path.glob(f"*{ext}"))
        sample_files.extend(files)
    
    return sample_files


def test_real_file(file_path: Path, max_size_mb: int = 10) -> Dict[str, Any]:
    """実ファイルでのテスト（サイズ制限あり）"""
    
    if not file_path.exists():
        return {'error': f'File not found: {file_path}'}
    
    file_size = file_path.stat().st_size
    if file_size > max_size_mb * 1024 * 1024:
        print(f"⚠️ File too large: {file_path.name} ({file_size // (1024*1024)}MB) - skipping")
        return {'error': 'File too large'}
    
    try:
        # ファイル読み込み
        with open(file_path, 'rb') as f:
            data = f.read()
        
        print(f"\n🔍 Testing: {file_path.name} ({len(data) // 1024}KB)")
        
        results = {}
        
        # 各モードでテスト
        for lightweight in [False, True]:
            for encryption in [False, True]:
                mode_name = f"{'Light' if lightweight else 'Standard'}_{'Encrypted' if encryption else 'Plain'}"
                
                engine = SPE_TMC_IntegratedEngine(
                    lightweight_mode=lightweight,
                    enable_encryption=encryption
                )
                
                try:
                    # 圧縮
                    start_time = time.time()
                    compressed_data, info = engine.compress_with_spe(
                        data, 
                        password="test123" if encryption else None
                    )
                    compress_time = time.time() - start_time
                    
                    # 解凍
                    start_time = time.time()
                    decompressed_data = engine.decompress_with_spe(
                        compressed_data, 
                        info, 
                        password="test123" if encryption else None
                    )
                    decompress_time = time.time() - start_time
                    
                    # 検証
                    original_hash = hashlib.sha256(data).hexdigest()
                    decompressed_hash = hashlib.sha256(decompressed_data).hexdigest()
                    is_valid = (original_hash == decompressed_hash)
                    
                    compression_ratio = (1 - len(compressed_data) / len(data)) * 100 if len(data) > 0 else 0
                    
                    results[mode_name] = {
                        'compression_ratio': compression_ratio,
                        'compress_time': compress_time,
                        'decompress_time': decompress_time,
                        'compressed_size': len(compressed_data),
                        'is_valid': is_valid,
                        'throughput': (len(data) / (1024 * 1024)) / (compress_time + decompress_time) if (compress_time + decompress_time) > 0 else 0
                    }
                    
                    print(f"  {mode_name}: {compression_ratio:.1f}% ratio, {results[mode_name]['throughput']:.1f} MB/s, {'✅' if is_valid else '❌'}")
                    
                except Exception as e:
                    results[mode_name] = {'error': str(e)}
                    print(f"  {mode_name}: ❌ Error - {e}")
        
        return {
            'file_name': file_path.name,
            'file_size': len(data),
            'results': results
        }
        
    except Exception as e:
        return {'error': f'Failed to process {file_path.name}: {e}'}


def generate_summary_report(test_results: List[Dict[str, Any]]):
    """テスト結果の要約レポート生成"""
    print("\n" + "=" * 80)
    print("📋 REAL FILE TEST SUMMARY REPORT")
    print("=" * 80)
    
    valid_results = [r for r in test_results if 'error' not in r and 'results' in r]
    
    if not valid_results:
        print("❌ No valid test results")
        return
    
    print(f"\n📊 Tested {len(valid_results)} files successfully")
    
    # モード別統計
    mode_stats = {}
    for result in valid_results:
        for mode_name, mode_result in result['results'].items():
            if 'error' in mode_result:
                continue
                
            if mode_name not in mode_stats:
                mode_stats[mode_name] = {
                    'ratios': [],
                    'speeds': [],
                    'valid_count': 0,
                    'total_count': 0
                }
            
            mode_stats[mode_name]['total_count'] += 1
            mode_stats[mode_name]['ratios'].append(mode_result['compression_ratio'])
            mode_stats[mode_name]['speeds'].append(mode_result['throughput'])
            
            if mode_result['is_valid']:
                mode_stats[mode_name]['valid_count'] += 1
    
    print(f"\n{'Mode':<20} {'Avg.Ratio%':<12} {'Avg.Speed':<12} {'Success':<10}")
    print("-" * 60)
    
    for mode_name, stats in mode_stats.items():
        if stats['ratios']:
            avg_ratio = sum(stats['ratios']) / len(stats['ratios'])
            avg_speed = sum(stats['speeds']) / len(stats['speeds'])
            success_rate = (stats['valid_count'] / stats['total_count']) * 100
            
            print(f"{mode_name:<20} {avg_ratio:<12.1f} {avg_speed:<12.1f} {success_rate:<10.0f}%")
    
    # ベストパフォーマンスファイル
    print(f"\n🏆 Best Performance by Category:")
    print("-" * 40)
    
    best_ratio = max(valid_results, key=lambda r: max(
        [res.get('compression_ratio', 0) for res in r['results'].values() if 'error' not in res]
    ))
    best_ratio_value = max([res.get('compression_ratio', 0) for res in best_ratio['results'].values() if 'error' not in res])
    
    best_speed = max(valid_results, key=lambda r: max(
        [res.get('throughput', 0) for res in r['results'].values() if 'error' not in res]
    ))
    best_speed_value = max([res.get('throughput', 0) for res in best_speed['results'].values() if 'error' not in res])
    
    print(f"Best Compression: {best_ratio['file_name']} ({best_ratio_value:.1f}%)")
    print(f"Best Speed: {best_speed['file_name']} ({best_speed_value:.1f} MB/s)")


def main():
    """メイン実行"""
    print("🎯 NEXUS TMC v9.1 + SPE Real File Testing")
    print("=" * 60)
    
    # サンプルファイル検索
    sample_files = find_sample_files()
    
    if not sample_files:
        print("❌ No sample files found")
        return
    
    print(f"📁 Found {len(sample_files)} sample files")
    
    # テスト実行
    test_results = []
    
    for file_path in sample_files[:10]:  # 最大10ファイルまで
        result = test_real_file(file_path)
        test_results.append(result)
    
    # 要約レポート
    generate_summary_report(test_results)
    
    print("\n✅ Real file testing completed!")


if __name__ == "__main__":
    main()
