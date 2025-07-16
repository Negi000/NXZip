#!/usr/bin/env python3
"""
🚀 NXZip NEXUS Benchmark Suite
NXZip NEXUSの性能評価とベンチマークシステム

Copyright (c) 2025 NXZip Project
"""

import os
import time
import json
from nxzip_nexus import NXZipNEXUS
from typing import Dict, List, Any

class NEXUSBenchmark:
    """NEXUS性能評価システム"""
    
    def __init__(self):
        self.nexus = NXZipNEXUS()
        self.results = []
    
    def run_comprehensive_benchmark(self, output_file: str = "nexus_benchmark_results.json"):
        """包括的ベンチマーク実行"""
        print("🚀 NXZip NEXUS 包括的ベンチマーク開始")
        print("=" * 60)
        
        # テストケース定義
        test_cases = [
            {
                'name': '小サイズテキスト',
                'data': 'Hello World! ' * 100,
                'filename': 'small.txt',
                'category': 'text'
            },
            {
                'name': '中サイズ日本語テキスト',
                'data': ('こんにちはNXZip NEXUS！これは中サイズのテストです。' * 1000),
                'filename': 'medium_japanese.txt',
                'category': 'text'
            },
            {
                'name': '大サイズ繰り返しテキスト',
                'data': ('NEXUS圧縮テストデータ。高圧縮率を実現するための繰り返しパターン。' * 5000),
                'filename': 'large_text.txt',
                'category': 'text'
            },
            {
                'name': 'JSON構造化データ',
                'data': json.dumps({
                    'nexus': 'compression test',
                    'data': list(range(1000)),
                    'metadata': {'version': 1.0, 'success': True}
                }),
                'filename': 'data.json',
                'category': 'structured'
            },
            {
                'name': 'XML文書',
                'data': ('<?xml version="1.0"?><nexus><test id="1">compression</test></nexus>' * 500),
                'filename': 'document.xml',
                'category': 'structured'
            },
            {
                'name': 'バイナリデータ（低エントロピー）',
                'data': bytes([i % 10 for i in range(100000)]),
                'filename': 'low_entropy.bin',
                'category': 'binary'
            },
            {
                'name': 'バイナリデータ（高エントロピー）',
                'data': bytes([i % 256 for i in range(50000)]),
                'filename': 'high_entropy.bin',
                'category': 'binary'
            }
        ]
        
        # ベンチマーク実行
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📊 テスト {i}/{len(test_cases)}: {test_case['name']}")
            
            # データ準備
            if isinstance(test_case['data'], str):
                data = test_case['data'].encode('utf-8')
            else:
                data = test_case['data']
            
            print(f"📁 ファイル: {test_case['filename']}")
            print(f"📊 サイズ: {len(data):,} bytes")
            print(f"🏷️  カテゴリ: {test_case['category']}")
            
            # 圧縮実行
            start_time = time.time()
            try:
                compressed, stats = self.nexus.compress(
                    data, 
                    test_case['filename'], 
                    show_progress=False
                )
                
                # 結果記録
                result = {
                    'test_name': test_case['name'],
                    'filename': test_case['filename'],
                    'category': test_case['category'],
                    'original_size': len(data),
                    'compressed_size': len(compressed),
                    'compression_ratio': stats['compression_ratio'],
                    'detected_format': stats['detected_format'],
                    'processing_time': stats['processing_time'],
                    'speed_mbps': stats['speed_mbps'],
                    'timestamp': time.time(),
                    'nexus_version': stats['nexus_version']
                }
                
                self.results.append(result)
                
                # 結果表示
                print(f"🔍 検出形式: {stats['detected_format']}")
                print(f"📈 圧縮率: {stats['compression_ratio']:.3f}%")
                print(f"⚡ 処理速度: {stats['speed_mbps']:.2f} MB/s")
                print(f"⏱️  時間: {stats['processing_time']:.3f}秒")
                
                # 性能評価
                if stats['compression_ratio'] >= 99.0:
                    print("🏆 優秀: 99%超の圧縮率!")
                elif stats['compression_ratio'] >= 95.0:
                    print("✅ 良好: 95%超の圧縮率")
                elif stats['compression_ratio'] >= 90.0:
                    print("📈 普通: 90%超の圧縮率")
                else:
                    print("⚠️  要改善: 90%未満")
                
            except Exception as e:
                print(f"❌ エラー: {e}")
                self.results.append({
                    'test_name': test_case['name'],
                    'filename': test_case['filename'],
                    'category': test_case['category'],
                    'error': str(e),
                    'timestamp': time.time()
                })
            
            print("-" * 50)
        
        # 統計分析
        self._analyze_results()
        
        # 結果保存
        self._save_results(output_file)
        
        print(f"\n📁 結果保存: {output_file}")
    
    def _analyze_results(self):
        """結果分析"""
        print("\n🏆 NXZip NEXUS ベンチマーク総合結果")
        print("=" * 50)
        
        successful_results = [r for r in self.results if 'error' not in r]
        
        if not successful_results:
            print("❌ 成功したテストがありません")
            return
        
        # 全体統計
        total_tests = len(self.results)
        successful_tests = len(successful_results)
        success_rate = (successful_tests / total_tests) * 100
        
        avg_compression = sum(r['compression_ratio'] for r in successful_results) / len(successful_results)
        avg_speed = sum(r['speed_mbps'] for r in successful_results) / len(successful_results)
        
        print(f"📊 テスト実行: {successful_tests}/{total_tests}")
        print(f"📈 成功率: {success_rate:.1f}%")
        print(f"📊 平均圧縮率: {avg_compression:.3f}%")
        print(f"⚡ 平均処理速度: {avg_speed:.2f} MB/s")
        
        # カテゴリ別分析
        categories = {}
        for result in successful_results:
            category = result['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(result)
        
        print("\n📊 カテゴリ別性能:")
        for category, results in categories.items():
            avg_ratio = sum(r['compression_ratio'] for r in results) / len(results)
            avg_speed_cat = sum(r['speed_mbps'] for r in results) / len(results)
            print(f"  🏷️  {category}: {avg_ratio:.2f}% | {avg_speed_cat:.2f} MB/s")
        
        # 最高性能
        best_compression = max(successful_results, key=lambda x: x['compression_ratio'])
        fastest_processing = max(successful_results, key=lambda x: x['speed_mbps'])
        
        print(f"\n🏆 最高圧縮率: {best_compression['compression_ratio']:.3f}% ({best_compression['test_name']})")
        print(f"⚡ 最高速度: {fastest_processing['speed_mbps']:.2f} MB/s ({fastest_processing['test_name']})")
        
        # 総合評価
        if avg_compression >= 99.0:
            print("\n🎉🏆🎊 NEXUS 完全勝利! 世界最高クラスの圧縮性能!")
        elif avg_compression >= 95.0:
            print("\n🎉 NEXUS 大成功! 優秀な圧縮性能!")
        elif avg_compression >= 90.0:
            print("\n📈 NEXUS 成功! 良好な圧縮性能!")
        else:
            print("\n⚠️  NEXUS 改善余地あり")
    
    def _save_results(self, filename: str):
        """結果をJSONファイルに保存"""
        output_data = {
            'benchmark_info': {
                'nexus_version': 'NEXUS v1.0',
                'timestamp': time.time(),
                'total_tests': len(self.results),
                'successful_tests': len([r for r in self.results if 'error' not in r])
            },
            'results': self.results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

def main():
    """メイン実行関数"""
    benchmark = NEXUSBenchmark()
    
    # 現在時刻でファイル名生成
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"nexus_benchmark_{timestamp}.json"
    
    benchmark.run_comprehensive_benchmark(output_file)

if __name__ == "__main__":
    main()
