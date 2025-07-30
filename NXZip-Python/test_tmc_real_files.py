#!/usr/bin/env python3
"""
NEXUS TMC Engine 実ファイルテスト
Transform-Model-Code アルゴリズムの実ファイル性能検証
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'NXZip-Python', 'nxzip', 'engine'))

import time
import hashlib
from pathlib import Path

# TMCエンジンをインポート
try:
    from nexus_tmc_engine import NEXUSTMCEngine, DataType
except ImportError:
    sys.path.append('NXZip-Python/nxzip/engine')
    from nexus_tmc_engine import NEXUSTMCEngine, DataType


def test_tmc_real_files():
    """TMC Engine 実ファイルテスト"""
    print("🚀 NEXUS TMC Engine - 実ファイル革命的圧縮テスト")
    print("=" * 80)
    print("📋 TMC (Transform-Model-Code) 特徴:")
    print("   🧠 データ構造自動分析 (Analyze & Dispatch)")
    print("   🔄 適応的変換処理 (Transform)")
    print("   ⚡ 並列符号化システム (Code)")
    print("   🎯 データタイプ別最適化")
    print("=" * 80)
    
    # TMCエンジン初期化
    engine = NEXUSTMCEngine(max_workers=4)
    
    # 実ファイル収集
    sample_dir = Path("sample")
    test_files = []
    
    if sample_dir.exists():
        for file_path in sample_dir.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith('.'):
                test_files.append(file_path)
    
    if not test_files:
        print("❌ sampleフォルダにファイルが見つかりません")
        return
    
    print(f"📁 検出ファイル数: {len(test_files)}")
    
    # テスト実行
    results = []
    data_type_stats = {}
    transform_method_stats = {}
    
    for i, file_path in enumerate(test_files):
        print(f"\n{'='*70}")
        print(f"📁 {i+1}/{len(test_files)}: {file_path.name}")
        
        try:
            # ファイル読み込み
            with open(file_path, 'rb') as f:
                original_data = f.read()
            
            if len(original_data) == 0:
                print(f"   ⚠️ 空ファイル、スキップ")
                continue
            
            file_type = file_path.suffix.lower().lstrip('.')
            size_mb = len(original_data) / 1024 / 1024
            original_hash = hashlib.sha256(original_data).hexdigest()
            
            print(f"   📊 サイズ: {size_mb:.2f}MB")
            print(f"   🎯 ファイルタイプ: {file_type}")
            print(f"   🔐 ハッシュ: {original_hash[:16]}...")
            
            # TMC圧縮実行
            start_time = time.perf_counter()
            compressed, info = engine.compress_tmc(original_data, file_type)
            compression_time = time.perf_counter() - start_time
            
            # TMC分析結果表示
            data_type = info.get('data_type', 'unknown')
            features = info.get('features', {})
            transform_info = info.get('transform_info', {})
            encoding_info = info.get('encoding_info', {})
            
            print(f"\n   🧠 TMC分析結果:")
            print(f"      📊 検出データタイプ: {data_type}")
            print(f"      📈 エントロピー: {features.get('entropy', 0):.2f}")
            print(f"      🔗 自己相関: {features.get('auto_correlation', 0):.3f}")
            print(f"      🏗️ 構造スコア: {features.get('type_structure_score', 0):.3f}")
            print(f"      📝 テキストスコア: {features.get('text_score', 0):.3f}")
            print(f"      📊 時系列スコア: {features.get('time_series_score', 0):.3f}")
            
            print(f"\n   🔄 TMC変換結果:")
            print(f"      🛠️ 変換方法: {transform_info.get('transform_method', 'none')}")
            print(f"      📦 ストリーム数: {transform_info.get('stream_count', 1)}")
            if 'type_size' in transform_info:
                print(f"      🎯 型サイズ: {transform_info['type_size']}バイト")
            if 'decomposition_score' in transform_info:
                print(f"      📈 分解品質: {transform_info['decomposition_score']:.3f}")
            
            print(f"\n   ⚡ TMC符号化結果:")
            print(f"      🚀 並列ストリーム: {encoding_info.get('stream_count', 1)}")
            compression_results = encoding_info.get('compression_results', [])
            if compression_results:
                methods = [r.get('method', 'unknown') for r in compression_results]
                method_counts = {}
                for method in methods:
                    method_counts[method] = method_counts.get(method, 0) + 1
                for method, count in method_counts.items():
                    print(f"      📦 {method}: {count}ストリーム")
            
            # 総合結果
            print(f"\n   🏆 TMC総合結果:")
            print(f"      📈 圧縮率: {info['compression_ratio']:.2f}%")
            print(f"      ⚡ スループット: {info['throughput_mb_s']:.2f}MB/s")
            print(f"      ⏱️ 処理時間: {info['total_time']:.3f}秒")
            print(f"      💾 圧縮前: {len(original_data):,} bytes")
            print(f"      💾 圧縮後: {len(compressed):,} bytes")
            print(f"      🔄 可逆性: {'✅' if info['reversible'] else '❌'}")
            print(f"      📉 膨張防止: {'✅' if info['expansion_prevented'] else '❌'}")
            
            # 性能評価
            if info['compression_ratio'] >= 50:
                perf_grade = "🏆 優秀"
            elif info['compression_ratio'] >= 25:
                perf_grade = "✅ 良好"
            elif info['compression_ratio'] >= 10:
                perf_grade = "⚡ 普通"
            else:
                perf_grade = "⚠️ 改善余地"
            
            print(f"      🎖️ TMC評価: {perf_grade}")
            
            # 統計更新
            data_type_stats[data_type] = data_type_stats.get(data_type, 0) + 1
            transform_method = transform_info.get('transform_method', 'none')
            transform_method_stats[transform_method] = transform_method_stats.get(transform_method, 0) + 1
            
            results.append({
                'file': file_path.name,
                'size_mb': size_mb,
                'file_type': file_type,
                'data_type': data_type,
                'transform_method': transform_method,
                'compression_ratio': info['compression_ratio'],
                'throughput': info['throughput_mb_s'],
                'reversible': info['reversible'],
                'expansion_prevented': info['expansion_prevented'],
                'features': features,
                'transform_info': transform_info,
                'encoding_info': encoding_info
            })
            
        except Exception as e:
            print(f"   ❌ TMCエラー: {str(e)}")
            results.append({
                'file': file_path.name,
                'error': str(e)
            })
    
    # 最終分析レポート
    print(f"\n{'='*80}")
    print(f"📊 NEXUS TMC Engine 革命的圧縮分析レポート")
    print(f"{'='*80}")
    
    successful_results = [r for r in results if 'error' not in r]
    total_files = len(successful_results)
    
    if total_files == 0:
        print("❌ 成功したテストがありません")
        return
    
    # 基本統計
    avg_compression = sum(r['compression_ratio'] for r in successful_results) / total_files
    avg_throughput = sum(r['throughput'] for r in successful_results) / total_files
    perfect_reversible = sum(1 for r in successful_results if r['reversible'])
    expansion_prevented = sum(1 for r in successful_results if r['expansion_prevented'])
    
    print(f"📈 TMC基本性能:")
    print(f"   📁 処理ファイル数: {total_files}")
    print(f"   📊 平均圧縮率: {avg_compression:.2f}%")
    print(f"   ⚡ 平均スループット: {avg_throughput:.2f}MB/s")
    print(f"   🔄 可逆性率: {perfect_reversible}/{total_files} ({perfect_reversible/total_files*100:.1f}%)")
    print(f"   📉 膨張防止率: {expansion_prevented}/{total_files} ({expansion_prevented/total_files*100:.1f}%)")
    
    # データタイプ分析
    print(f"\n🧠 TMCデータタイプ分析:")
    for data_type, count in data_type_stats.items():
        percentage = count / total_files * 100
        print(f"   {data_type}: {count}ファイル ({percentage:.1f}%)")
    
    # 変換方法分析
    print(f"\n🔄 TMC変換方法分析:")
    for method, count in transform_method_stats.items():
        percentage = count / total_files * 100
        print(f"   {method}: {count}ファイル ({percentage:.1f}%)")
    
    # データタイプ別性能
    print(f"\n📊 データタイプ別TMC性能:")
    for data_type in data_type_stats.keys():
        type_results = [r for r in successful_results if r['data_type'] == data_type]
        if type_results:
            type_avg_compression = sum(r['compression_ratio'] for r in type_results) / len(type_results)
            type_avg_throughput = sum(r['throughput'] for r in type_results) / len(type_results)
            type_reversible = sum(1 for r in type_results if r['reversible']) / len(type_results) * 100
            
            print(f"   {data_type.upper()}:")
            print(f"      平均圧縮率: {type_avg_compression:.1f}%")
            print(f"      平均スループット: {type_avg_throughput:.1f}MB/s")
            print(f"      可逆性率: {type_reversible:.1f}%")
    
    # TMCエンジン統計
    tmc_stats = engine.get_tmc_stats()
    if tmc_stats.get('status') != 'no_data':
        print(f"\n📈 TMCエンジン詳細統計:")
        print(f"   総処理量: {tmc_stats['total_input_mb']:.1f}MB")
        print(f"   総圧縮率: {tmc_stats['total_compression_ratio']:.2f}%")
        print(f"   総処理時間: {tmc_stats['total_time']:.3f}秒")
        
        print(f"\n🔄 TMC変換メソッド分布:")
        for method, count in tmc_stats.get('transform_method_distribution', {}).items():
            print(f"   {method}: {count}回")
        
        print(f"\n📦 TMC圧縮メソッド分布:")
        for method, count in tmc_stats.get('compression_method_distribution', {}).items():
            print(f"   {method}: {count}回")
    
    # TMC革命的評価
    print(f"\n🏆 TMC革命的評価:")
    
    if avg_compression >= 40 and perfect_reversible == total_files:
        grade = "🎉 革命的成功 - TMC設計思想完全実現!"
        print(f"   {grade}")
        print(f"   🚀 データ構造理解による圧縮革命達成")
    elif avg_compression >= 30 and perfect_reversible >= total_files * 0.9:
        grade = "✅ 大成功 - TMCアプローチ有効性証明"
        print(f"   {grade}")
        print(f"   🎯 適応的変換の威力を実証")
    elif avg_compression >= 20:
        grade = "⚡ 成功 - TMC基本機能動作確認"
        print(f"   {grade}")
        print(f"   🔧 一部最適化でさらなる向上可能")
    else:
        grade = "⚠️ 改良必要 - TMC実装要調整"
        print(f"   {grade}")
        print(f"   🛠️ TMCアルゴリズム調整が必要")
    
    print(f"\n💡 TMC次期開発提案:")
    print(f"   🧠 機械学習モデルの高度化")
    print(f"   🔄 変換アルゴリズムの最適化")
    print(f"   ⚡ 並列処理効率の向上")
    print(f"   📊 データタイプ判定精度の改善")
    
    print(f"\n🎯 TMC革命完了!")


if __name__ == "__main__":
    test_tmc_real_files()
