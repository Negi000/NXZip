#!/usr/bin/env python3
"""
拡張ベンチマークテスト - 複数ファイル
"""
import sys
import os

sys.path.insert(0, '.')
from final_comprehensive_benchmark import ComprehensiveCompressionBenchmark

def run_extended_test():
    """拡張テストの実行"""
    print("🚀 拡張ベンチマークテスト開始")
    
    benchmark = ComprehensiveCompressionBenchmark(verbose=True)
    
    # より多くのテストファイル
    test_files = [
        "./README.md",
        "./PROJECT_STATUS.md", 
        "./TECHNICAL.md",
        "./sample/出庫実績明細_202412.txt",
        "./sample/COT-001.jpg",
        "./sample/COT-001.png",
        "./sample/COT-012.png",
        "./sample/generated-music-1752042054079.wav",
        "./sample/陰謀論.mp3",
        "./sample/Python基礎講座3_4月26日-3.mp4"
    ]
    
    # 実際に存在するファイルのみをテスト
    existing_files = []
    for f in test_files:
        if os.path.exists(f):
            size_mb = os.path.getsize(f) / (1024 * 1024)
            print(f"✅ {os.path.basename(f)} ({size_mb:.1f} MB)")
            existing_files.append(f)
        else:
            print(f"⚠️ {f} - ファイルが見つかりません")
    
    print(f"\n📊 テスト対象: {len(existing_files)} ファイル")
    
    if existing_files:
        results = benchmark.run_comprehensive_test(existing_files)
        
        # 詳細分析
        print("\n" + "=" * 70)
        print("📈 詳細パフォーマンス分析")
        print("=" * 70)
        
        # ファイル種別ごとの分析
        analyze_by_file_type(results)
        
        # エンジン別ランキング
        engine_ranking(results)
        
    else:
        print("❌ テストできるファイルがありません")

def analyze_by_file_type(results):
    """ファイル種別ごとの分析"""
    file_types = {
        'テキスト': ['.txt', '.md'],
        '画像': ['.jpg', '.png'],
        '音声': ['.mp3', '.wav'], 
        '動画': ['.mp4']
    }
    
    for type_name, extensions in file_types.items():
        type_results = {}
        for file_name, file_results in results.items():
            if any(file_name.lower().endswith(ext) for ext in extensions):
                type_results[file_name] = file_results
        
        if type_results:
            print(f"\n📂 {type_name}ファイル:")
            print("-" * 30)
            
            # 各エンジンの平均性能
            engine_avg = {}
            for file_results in type_results.values():
                for engine_name, result in file_results.items():
                    if result['success']:
                        if engine_name not in engine_avg:
                            engine_avg[engine_name] = {
                                'compression_ratios': [],
                                'compression_speeds': [],
                                'decompression_speeds': []
                            }
                        engine_avg[engine_name]['compression_ratios'].append(result['compression_ratio'])
                        engine_avg[engine_name]['compression_speeds'].append(result['compression_speed'])
                        engine_avg[engine_name]['decompression_speeds'].append(result['decompression_speed'])
            
            # 平均値表示
            for engine_name, stats in engine_avg.items():
                if stats['compression_ratios']:
                    avg_comp = sum(stats['compression_ratios']) / len(stats['compression_ratios'])
                    avg_speed = sum(stats['compression_speeds']) / len(stats['compression_speeds'])
                    print(f"  {engine_name}: 圧縮率 {avg_comp:.1%}, 速度 {avg_speed:.1f} MB/s")

def engine_ranking(results):
    """エンジン別ランキング"""
    print("\n🏆 エンジンランキング")
    print("=" * 50)
    
    # 全エンジンの統計
    engine_stats = {}
    
    for file_results in results.values():
        for engine_name, result in file_results.items():
            if result['success']:
                if engine_name not in engine_stats:
                    engine_stats[engine_name] = {
                        'compression_ratios': [],
                        'compression_speeds': [],
                        'decompression_speeds': []
                    }
                
                stats = engine_stats[engine_name]
                stats['compression_ratios'].append(result['compression_ratio'])
                stats['compression_speeds'].append(result['compression_speed'])
                stats['decompression_speeds'].append(result['decompression_speed'])
    
    # 各カテゴリでランキング
    categories = [
        ('圧縮率', 'compression_ratios'),
        ('圧縮速度', 'compression_speeds'),
        ('展開速度', 'decompression_speeds')
    ]
    
    for category_name, stat_key in categories:
        print(f"\n🥇 {category_name}ランキング:")
        
        ranking = []
        for engine_name, stats in engine_stats.items():
            if stats[stat_key]:
                avg_value = sum(stats[stat_key]) / len(stats[stat_key])
                ranking.append((engine_name, avg_value))
        
        # 降順でソート
        ranking.sort(key=lambda x: x[1], reverse=True)
        
        for i, (engine_name, value) in enumerate(ranking[:3]):
            medal = ['🥇', '🥈', '🥉'][i]
            if stat_key == 'compression_ratios':
                print(f"  {medal} {engine_name}: {value:.1%}")
            else:
                print(f"  {medal} {engine_name}: {value:.1f} MB/s")

if __name__ == "__main__":
    run_extended_test()
