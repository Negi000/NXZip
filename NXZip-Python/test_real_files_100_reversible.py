#!/usr/bin/env python3
"""
NEXUS Ultra Engine 実ファイル可逆性100%達成版
sampleフォルダの実ファイルで可逆性100%を目指す
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'nxzip', 'engine'))

import time
import hashlib
from pathlib import Path
from nxzip.engine.nexus_v6_1_ultra import NEXUSEngineUltra


def test_real_files_100_reversible():
    """実ファイル可逆性100%テスト"""
    print("🎯 NEXUS Ultra Engine - 実ファイル可逆性100%達成テスト")
    print("=" * 80)
    print("📋 テスト方針:")
    print("   ✓ sampleフォルダの実ファイルのみテスト")
    print("   ✓ 可逆性100%必須（100%未満は失敗扱い）")
    print("   ✓ 膨張防止100%必須")
    print("   ✓ エラー0件必須")
    print("=" * 80)
    
    # Ultra Engine初期化
    engine = NEXUSEngineUltra(max_workers=4)
    
    # 実ファイル収集
    sample_dir = Path("sample")
    test_files = []
    
    if sample_dir.exists():
        # すべての実ファイルを収集
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
            
            # Ultra圧縮実行
            start_time = time.perf_counter()
            compressed, info = engine.compress_ultra(original_data, file_type)
            compression_time = time.perf_counter() - start_time
            
            # 詳細可逆性テスト
            is_perfectly_reversible = test_perfect_reversibility(
                original_data, compressed, original_hash
            )
            
            # 膨張チェック
            expansion_prevented = len(compressed) < len(original_data)
            
            # エラーチェック
            has_error = 'error' in info
            
            # 統計更新
            if not has_error:
                error_free_count += 1
            
            if expansion_prevented:
                expansion_prevented_count += 1
            
            if is_perfectly_reversible:
                reversible_count += 1
            
            if is_perfectly_reversible and expansion_prevented and not has_error:
                perfect_count += 1
                status = "✅ PERFECT"
            elif is_perfectly_reversible and expansion_prevented:
                status = "⚠️ 可逆・圧縮OK（軽微エラー）"
            elif is_perfectly_reversible:
                status = "⚠️ 可逆OK（膨張問題）"
            elif expansion_prevented:
                status = "❌ 圧縮OK（可逆性問題）"
            else:
                status = "❌ FAILED"
            
            # 結果表示
            print(f"   📈 圧縮率: {info['compression_ratio']:.2f}%")
            print(f"   ⚡ スループット: {info['throughput_mb_s']:.2f}MB/s")
            print(f"   🎛️ 戦略: {info['strategy']}")
            print(f"   🔄 可逆性: {'✅ 100%' if is_perfectly_reversible else '❌ 失敗'}")
            print(f"   📉 膨張防止: {'✅' if expansion_prevented else '❌'}")
            print(f"   ❌ エラー: {'なし' if not has_error else 'あり'}")
            print(f"   🏆 総合: {status}")
            
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
                'perfect': is_perfectly_reversible and expansion_prevented and not has_error,
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
                'perfect': False,
                'status': "❌ EXCEPTION"
            })
    
    # 最終結果分析
    print(f"\n{'='*80}")
    print(f"📊 実ファイル可逆性100%テスト 最終結果")
    print(f"{'='*80}")
    
    total_files = len(results)
    
    print(f"📈 基本統計:")
    print(f"   📁 総ファイル数: {total_files}")
    print(f"   ✅ PERFECT: {perfect_count}/{total_files} ({perfect_count/total_files*100:.1f}%)")
    print(f"   🔄 可逆性100%: {reversible_count}/{total_files} ({reversible_count/total_files*100:.1f}%)")
    print(f"   📉 膨張防止: {expansion_prevented_count}/{total_files} ({expansion_prevented_count/total_files*100:.1f}%)")
    print(f"   ❌ エラーフリー: {error_free_count}/{total_files} ({error_free_count/total_files*100:.1f}%)")
    
    # ファイルタイプ別詳細分析
    print(f"\n📊 ファイルタイプ別可逆性分析:")
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
                'avg_ratio': 0,
                'avg_throughput': 0
            }
        
        file_types[ftype]['total'] += 1
        if result['perfectly_reversible']:
            file_types[ftype]['reversible'] += 1
        if result['perfect']:
            file_types[ftype]['perfect'] += 1
        file_types[ftype]['avg_ratio'] += result['compression_ratio']
        file_types[ftype]['avg_throughput'] += result['throughput']
    
    for ftype, stats in file_types.items():
        if stats['total'] > 0:
            reversible_rate = stats['reversible'] / stats['total'] * 100
            perfect_rate = stats['perfect'] / stats['total'] * 100
            avg_ratio = stats['avg_ratio'] / stats['total']
            avg_throughput = stats['avg_throughput'] / stats['total']
            
            print(f"   {ftype.upper()}:")
            print(f"      可逆性: {stats['reversible']}/{stats['total']} ({reversible_rate:.1f}%)")
            print(f"      PERFECT: {stats['perfect']}/{stats['total']} ({perfect_rate:.1f}%)")
            print(f"      平均圧縮率: {avg_ratio:.1f}%")
            print(f"      平均スループット: {avg_throughput:.1f}MB/s")
    
    # 問題ファイル特定
    print(f"\n⚠️ 問題ファイル:")
    problem_files = [r for r in results if not r['perfect']]
    
    if problem_files:
        for result in problem_files[:5]:  # 最大5件表示
            print(f"   📁 {result['file']}: {result['status']}")
            if 'error' in result:
                print(f"      エラー: {result['error']}")
    else:
        print(f"   ✅ 問題ファイルなし - すべてPERFECT!")
    
    # 最終判定
    print(f"\n🏆 最終判定:")
    
    if reversible_count == total_files and expansion_prevented_count == total_files and error_free_count == total_files:
        grade = "🎉 完全成功 - 可逆性100%達成!"
        print(f"   {grade}")
        print(f"   🎯 すべてのファイルで完璧な結果を達成しました")
    elif reversible_count / total_files >= 0.9:
        grade = "✅ 高成功率 - 可逆性90%以上達成"
        print(f"   {grade}")
        print(f"   🎯 ほぼすべてのファイルで可逆性を達成")
    elif reversible_count / total_files >= 0.7:
        grade = "⚡ 良好 - 可逆性70%以上達成"
        print(f"   {grade}")
        print(f"   🔧 一部改善が必要ですが良好な結果")
    else:
        grade = "⚠️ 要改善 - 可逆性70%未満"
        print(f"   {grade}")
        print(f"   🔧 可逆性の大幅な改善が必要")
    
    # 改善提案
    print(f"\n💡 改善提案:")
    if reversible_count < total_files:
        print(f"   🔧 可逆性問題の解決が最優先")
        print(f"   📋 解凍アルゴリズムの見直しが必要")
    
    if expansion_prevented_count < total_files:
        print(f"   📉 膨張防止機能の強化が必要")
    
    if error_free_count < total_files:
        print(f"   ❌ エラーハンドリングの改善が必要")
    
    # Ultra Engine統計
    stats = engine.get_ultra_stats()
    if stats.get('status') != 'no_data':
        print(f"\n📈 Ultra Engine 統計:")
        print(f"   平均スループット: {stats['average_throughput_mb_s']:.2f}MB/s")
        print(f"   総圧縮率: {stats['total_compression_ratio']:.2f}%")
        print(f"   エラー数: {stats['error_count']}")


def test_perfect_reversibility(original_data: bytes, compressed: bytes, original_hash: str) -> bool:
    """完璧な可逆性テスト"""
    try:
        # 複数の解凍方法を試行
        import lzma
        import zlib
        import bz2
        
        decompression_methods = [
            ('lzma', lzma.decompress),
            ('zlib', zlib.decompress),
            ('bz2', bz2.decompress)
        ]
        
        for method_name, decompress_func in decompression_methods:
            try:
                decompressed = decompress_func(compressed)
                decompressed_hash = hashlib.sha256(decompressed).hexdigest()
                
                # ハッシュとサイズの完全一致チェック
                if (decompressed_hash == original_hash and 
                    len(decompressed) == len(original_data)):
                    return True
                    
            except Exception:
                continue
        
        return False
        
    except Exception:
        return False


if __name__ == "__main__":
    test_real_files_100_reversible()
