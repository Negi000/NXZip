#!/usr/bin/env python3
"""
完全可逆性100%達成テスト - Ultra Engine v2使用
sampleフォルダの実ファイルで可逆性100%を必ず達成
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'nxzip', 'engine'))

import time
import hashlib
from pathlib import Path
from nxzip.engine.nexus_v6_1_ultra_v2 import NEXUSEngineUltraV2


def test_perfect_reversibility_v2():
    """Ultra Engine v2 完全可逆性100%達成テスト"""
    print("🎯 NEXUS Ultra Engine v2 - 完全可逆性100%達成テスト")
    print("=" * 80)
    print("📋 v2エンジン特徴:")
    print("   ✓ 完全可逆性100%必須保証")
    print("   ✓ 膨張防止100%必須保証")
    print("   ✓ エラー0件必須保証")
    print("   ✓ フォールバック原形保持")
    print("   ✓ 複数解凍方法試行")
    print("=" * 80)
    
    # Ultra Engine v2初期化
    engine = NEXUSEngineUltraV2(max_workers=4)
    
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
    perfect_count = 0
    reversible_count = 0
    expansion_prevented_count = 0
    error_free_count = 0
    
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
            print(f"   🎯 タイプ: {file_type}")
            print(f"   🔐 ハッシュ: {original_hash[:16]}...")
            
            # Ultra v2圧縮実行
            start_time = time.perf_counter()
            compressed, info = engine.compress_ultra_v2(original_data, file_type)
            compression_time = time.perf_counter() - start_time
            
            # 厳密可逆性検証
            is_perfectly_reversible = verify_strict_reversibility(
                original_data, compressed, original_hash
            )
            
            # 膨張チェック
            expansion_prevented = len(compressed) <= len(original_data)
            
            # エラーチェック
            has_error = 'error' in info
            
            # 統計更新
            if not has_error:
                error_free_count += 1
            
            if expansion_prevented:
                expansion_prevented_count += 1
            
            if is_perfectly_reversible:
                reversible_count += 1
            
            # 完璧判定（v2では必ず達成すべき）
            is_perfect = is_perfectly_reversible and expansion_prevented and not has_error
            
            if is_perfect:
                perfect_count += 1
                status = "✅ PERFECT v2"
            else:
                status = "❌ v2 FAILURE (設計問題)"
            
            # 結果表示
            print(f"   📈 圧縮率: {info['compression_ratio']:.2f}%")
            print(f"   ⚡ スループット: {info['throughput_mb_s']:.2f}MB/s")
            print(f"   🎛️ 戦略: {info['strategy']}")
            print(f"   🔄 可逆性: {'✅ 100%' if is_perfectly_reversible else '❌ 失敗'}")
            print(f"   📉 膨張防止: {'✅' if expansion_prevented else '❌'}")
            print(f"   ❌ エラー: {'なし' if not has_error else 'あり'}")
            print(f"   🎯 v2完璧: {'✅' if info.get('perfect_result', False) else '❌'}")
            print(f"   🏆 最終判定: {status}")
            
            results.append({
                'file': file_path.name,
                'size_mb': size_mb,
                'file_type': file_type,
                'compression_ratio': info['compression_ratio'],
                'throughput': info['throughput_mb_s'],
                'strategy': info['strategy'],
                'perfectly_reversible': is_perfectly_reversible,
                'expansion_prevented': expansion_prevented,
                'error_free': not has_error,
                'v2_perfect': info.get('perfect_result', False),
                'perfect': is_perfect,
                'status': status
            })
            
        except Exception as e:
            print(f"   ❌ テストエラー: {str(e)}")
            results.append({
                'file': file_path.name,
                'error': str(e),
                'perfectly_reversible': False,
                'expansion_prevented': False,
                'error_free': False,
                'v2_perfect': False,
                'perfect': False,
                'status': "❌ EXCEPTION"
            })
    
    # 最終結果分析
    print(f"\n{'='*80}")
    print(f"📊 Ultra Engine v2 完全可逆性100%テスト 最終結果")
    print(f"{'='*80}")
    
    total_files = len(results)
    
    print(f"📈 v2エンジン成果:")
    print(f"   📁 総ファイル数: {total_files}")
    print(f"   ✅ PERFECT: {perfect_count}/{total_files} ({perfect_count/total_files*100:.1f}%)")
    print(f"   🔄 可逆性100%: {reversible_count}/{total_files} ({reversible_count/total_files*100:.1f}%)")
    print(f"   📉 膨張防止: {expansion_prevented_count}/{total_files} ({expansion_prevented_count/total_files*100:.1f}%)")
    print(f"   ❌ エラーフリー: {error_free_count}/{total_files} ({error_free_count/total_files*100:.1f}%)")
    
    # v2特別指標
    v2_perfect_count = sum(1 for r in results if r.get('v2_perfect', False))
    print(f"   🎯 v2完璧機能: {v2_perfect_count}/{total_files} ({v2_perfect_count/total_files*100:.1f}%)")
    
    # ファイルタイプ別v2成果
    print(f"\n📊 ファイルタイプ別v2成果:")
    file_types = {}
    for result in results:
        if 'error' in result:
            continue
        ftype = result['file_type']
        if ftype not in file_types:
            file_types[ftype] = {
                'total': 0,
                'reversible': 0,
                'perfect': 0,
                'v2_perfect': 0,
                'avg_ratio': 0,
                'avg_throughput': 0
            }
        
        file_types[ftype]['total'] += 1
        if result['perfectly_reversible']:
            file_types[ftype]['reversible'] += 1
        if result['perfect']:
            file_types[ftype]['perfect'] += 1
        if result.get('v2_perfect', False):
            file_types[ftype]['v2_perfect'] += 1
        file_types[ftype]['avg_ratio'] += result['compression_ratio']
        file_types[ftype]['avg_throughput'] += result['throughput']
    
    for ftype, stats in file_types.items():
        if stats['total'] > 0:
            reversible_rate = stats['reversible'] / stats['total'] * 100
            perfect_rate = stats['perfect'] / stats['total'] * 100
            v2_perfect_rate = stats['v2_perfect'] / stats['total'] * 100
            avg_ratio = stats['avg_ratio'] / stats['total']
            avg_throughput = stats['avg_throughput'] / stats['total']
            
            print(f"   {ftype.upper()}:")
            print(f"      可逆性: {stats['reversible']}/{stats['total']} ({reversible_rate:.1f}%)")
            print(f"      従来PERFECT: {stats['perfect']}/{stats['total']} ({perfect_rate:.1f}%)")
            print(f"      v2PERFECT: {stats['v2_perfect']}/{stats['total']} ({v2_perfect_rate:.1f}%)")
            print(f"      平均圧縮率: {avg_ratio:.1f}%")
            print(f"      平均スループット: {avg_throughput:.1f}MB/s")
    
    # 改善度評価
    print(f"\n📈 v2改善度評価:")
    
    if reversible_count == total_files:
        reversibility_improvement = "🎉 可逆性100%完全達成!"
    else:
        failed_files = total_files - reversible_count
        reversibility_improvement = f"⚠️ 可逆性問題残存: {failed_files}ファイル"
    
    if expansion_prevented_count == total_files:
        expansion_improvement = "🎉 膨張防止100%完全達成!"
    else:
        expanded_files = total_files - expansion_prevented_count
        expansion_improvement = f"⚠️ 膨張問題残存: {expanded_files}ファイル"
    
    print(f"   🔄 可逆性: {reversibility_improvement}")
    print(f"   📉 膨張防止: {expansion_improvement}")
    print(f"   ❌ エラー: {error_free_count}/{total_files} エラーフリー")
    
    # v2エンジン最終判定
    print(f"\n🏆 Ultra Engine v2 最終判定:")
    
    if reversible_count == total_files and expansion_prevented_count == total_files and error_free_count == total_files:
        grade = "🎉 v2完全成功 - 100%完璧達成!"
        print(f"   {grade}")
        print(f"   🎯 Ultra Engine v2が設計通り完璧に動作")
        print(f"   ✨ 全ファイル形式で完全可逆性100%達成")
    elif perfect_count / total_files >= 0.95:
        grade = "✅ v2ほぼ完璧 - 95%以上達成"
        print(f"   {grade}")
        print(f"   🎯 Ultra Engine v2がほぼ完璧に動作")
    elif perfect_count / total_files >= 0.80:
        grade = "⚡ v2良好 - 80%以上達成"
        print(f"   {grade}")
        print(f"   🔧 一部調整でv2完璧達成可能")
    else:
        grade = "⚠️ v2要修正 - 80%未満"
        print(f"   {grade}")
        print(f"   🔧 v2エンジン設計の見直しが必要")
    
    # 問題ファイル特定（v2で失敗したもの）
    print(f"\n⚠️ v2問題ファイル:")
    problem_files = [r for r in results if not r.get('perfect', False)]
    
    if problem_files:
        for result in problem_files[:5]:
            print(f"   📁 {result['file']}: {result['status']}")
            if 'error' in result:
                print(f"      エラー: {result['error']}")
    else:
        print(f"   ✅ v2問題ファイルなし - 完全成功!")
    
    # Ultra Engine v2統計
    stats = engine.get_ultra_v2_stats()
    if stats.get('status') != 'no_data':
        print(f"\n📈 Ultra Engine v2 詳細統計:")
        print(f"   平均スループット: {stats['average_throughput_mb_s']:.2f}MB/s")
        print(f"   総圧縮率: {stats['total_compression_ratio']:.2f}%")
        print(f"   可逆性率: {stats['reversibility_rate']:.1f}%")
        print(f"   膨張防止率: {stats['expansion_prevention_rate']:.1f}%")
        print(f"   完璧達成率: {stats['perfect_achievement_rate']:.1f}%")
        print(f"   エラー数: {stats['error_count']}")
        print(f"   性能グレード: {stats['performance_grade']}")
        
        print(f"\n🎛️ v2戦略使用分布:")
        for strategy, count in stats['strategy_distribution'].items():
            if count > 0:
                print(f"   {strategy}: {count}回")


def verify_strict_reversibility(original: bytes, compressed: bytes, original_hash: str) -> bool:
    """厳密可逆性検証（v2基準）"""
    try:
        # 複数解凍方法を試行
        import lzma
        import zlib
        import bz2
        
        decompression_methods = [
            ('lzma', lzma.decompress),
            ('zlib', zlib.decompress),
            ('bz2', bz2.decompress)
        ]
        
        # 原形保持チェック
        if compressed == original:
            return True
        
        for method_name, decompress_func in decompression_methods:
            try:
                decompressed = decompress_func(compressed)
                decompressed_hash = hashlib.sha256(decompressed).hexdigest()
                
                # 厳密な一致チェック
                if (decompressed_hash == original_hash and 
                    len(decompressed) == len(original)):
                    return True
                    
            except Exception:
                continue
        
        return False
        
    except Exception:
        return False


if __name__ == "__main__":
    test_perfect_reversibility_v2()
