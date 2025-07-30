#!/usr/bin/env python3
"""
NEXUS Advanced Engine v3.0 実ファイルテスト
既圧縮ファイル（JPEG/PNG/MP4）に対する超高度圧縮テスト
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'nxzip', 'engine'))

import time
import hashlib
from pathlib import Path
from nexus_advanced_v3 import NEXUSAdvancedEngine, AdvancedCompressionConfig


def test_nexus_v3_real_files():
    """NEXUS v3.0 実ファイルテスト"""
    print("🔥 NEXUS Advanced Engine v3.0 実ファイル超高度圧縮テスト")
    print("=" * 80)
    
    # サンプルディレクトリ
    sample_dir = Path("sample")
    
    if not sample_dir.exists():
        print(f"❌ サンプルディレクトリが見つかりません: {sample_dir}")
        return
    
    # 超高度設定
    config = AdvancedCompressionConfig(
        use_gpu=False,
        use_multiprocessing=True,
        use_threading=True,
        max_threads=8,
        max_processes=4,
        chunk_size_mb=1,  # 小さなチャンクで精密処理
        memory_limit_gb=12.0,
        
        # 高度機能有効化
        deep_analysis_enabled=True,
        entropy_reconstruction=True,
        multilevel_structure_analysis=True,
        hybrid_transformation=True,
        adaptive_chunking=True,
        
        # ファイル特化最適化
        jpeg_optimization=True,
        png_optimization=True,
        mp4_optimization=True,
        audio_optimization=True,
        text_optimization=True,
        
        # 超高品質モード
        ultra_mode=True
    )
    
    engine = NEXUSAdvancedEngine(config)
    
    # テスト対象ファイル選択
    test_files = []
    
    for file_path in sample_dir.iterdir():
        if file_path.is_file():
            # 全ファイル対象（圧縮ファイルも含む）
            try:
                file_size = file_path.stat().st_size
                if file_size < 200 * 1024 * 1024:  # 200MB未満
                    test_files.append({
                        'path': file_path,
                        'name': file_path.name,
                        'size': file_size,
                        'type': get_advanced_file_type(file_path)
                    })
            except:
                continue
    
    # ファイルサイズでソート
    test_files.sort(key=lambda x: x['size'])
    
    print(f"🔬 超高度圧縮テスト対象: {len(test_files)} ファイル")
    for file_info in test_files:
        print(f"   📄 {file_info['name']} ({file_info['size'] / 1024:.1f}KB) [{file_info['type']}]")
    
    # 各ファイルで超高度圧縮テスト
    results = []
    
    for i, file_info in enumerate(test_files):
        print(f"\n{'='*70}")
        print(f"🧪 超高度圧縮テスト {i+1}/{len(test_files)}: {file_info['name']}")
        print(f"   📁 ファイルタイプ: {file_info['type']}")
        print(f"   📊 ファイルサイズ: {file_info['size']:,} bytes ({file_info['size']/1024:.1f}KB)")
        
        try:
            # ファイル読み込み
            print("   📖 ファイル読み込み...")
            with open(file_info['path'], 'rb') as f:
                original_data = f.read()
            
            # 元データハッシュ
            original_hash = hashlib.sha256(original_data).hexdigest()
            print(f"   🔑 データハッシュ: {original_hash[:16]}...")
            
            # === NEXUS v3.0 超高度圧縮 ===
            print("   🔥 NEXUS v3.0 超高度圧縮実行...")
            compress_start = time.perf_counter()
            
            compressed_data = engine.advanced_compress(
                original_data, 
                file_info['type'], 
                'balanced'  # ultra品質も試せる
            )
            
            compress_time = time.perf_counter() - compress_start
            
            compression_ratio = (1 - len(compressed_data) / len(original_data)) * 100
            throughput = file_info['size'] / 1024 / 1024 / compress_time
            
            print(f"   ✅ 超高度圧縮完了!")
            print(f"      📈 圧縮率: {compression_ratio:.2f}%")
            print(f"      ⚡ スループット: {throughput:.2f}MB/s")
            print(f"      ⏱️ 圧縮時間: {compress_time:.3f}秒")
            print(f"      💾 圧縮前: {len(original_data):,} bytes")
            print(f"      💾 圧縮後: {len(compressed_data):,} bytes")
            
            # === 可逆性検証 ===
            print("   🔍 可逆性検証...")
            try:
                decompressed_data = simulate_nexus_v3_decompression(compressed_data)
                
                size_match = len(original_data) == len(decompressed_data)
                decompressed_hash = hashlib.sha256(decompressed_data).hexdigest()
                hash_match = original_hash == decompressed_hash
                bytes_match = original_data == decompressed_data
                is_reversible = size_match and hash_match and bytes_match
                
                print(f"      🏆 可逆性: {'✅ 完全' if is_reversible else '❌ 不完全'}")
            except Exception as e:
                print(f"      ⚠️ 可逆性検証エラー: {e}")
                is_reversible = False
            
            # === 他圧縮形式との比較 ===
            print("   📊 他形式比較...")
            comparison = compare_with_standard_compression(original_data, file_info['name'])
            
            # 結果保存
            result = {
                'name': file_info['name'],
                'type': file_info['type'],
                'original_size': len(original_data),
                'compressed_size': len(compressed_data),
                'compression_ratio': compression_ratio,
                'compress_time': compress_time,
                'throughput': throughput,
                'is_reversible': is_reversible,
                'comparison': comparison
            }
            
            results.append(result)
            
            # 比較結果表示
            print(f"   📈 圧縮性能比較:")
            improvements = []
            for method, comp_result in comparison.items():
                if comp_result and 'ratio' in comp_result:
                    ratio_diff = compression_ratio - comp_result['ratio']
                    print(f"      {method:>8}: {comp_result['ratio']:6.2f}% (NEXUS{ratio_diff:+.2f}%)")
                    if ratio_diff > 0:
                        improvements.append(f"{method}比+{ratio_diff:.1f}%")
            
            if improvements:
                print(f"      🏆 改善: {', '.join(improvements)}")
                
        except Exception as e:
            print(f"   ❌ テストエラー: {str(e)}")
            import traceback
            traceback.print_exc()
            
            results.append({
                'name': file_info['name'],
                'type': file_info['type'],
                'error': str(e),
                'is_reversible': False
            })
    
    # === 超高度圧縮総合レポート ===
    print(f"\n{'='*80}")
    print(f"📊 NEXUS v3.0 超高度圧縮総合レポート")
    print(f"{'='*80}")
    
    if results:
        successful_results = [r for r in results if not r.get('error') and r.get('is_reversible', False)]
        all_results = [r for r in results if not r.get('error')]
        
        print(f"🎯 テスト結果サマリー:")
        print(f"   📊 総テスト数: {len(results)}")
        print(f"   ✅ 成功（可逆）: {len(successful_results)}")
        print(f"   🔄 処理完了: {len(all_results)}")
        print(f"   📈 可逆成功率: {len(successful_results)/len(results)*100:.1f}%")
        
        if all_results:
            print(f"\n🏆 圧縮性能統計:")
            
            # 全体統計
            avg_ratio = sum(r['compression_ratio'] for r in all_results) / len(all_results)
            avg_throughput = sum(r['throughput'] for r in all_results) / len(all_results)
            
            print(f"   📈 平均圧縮率: {avg_ratio:.2f}%")
            print(f"   ⚡ 平均スループット: {avg_throughput:.2f}MB/s")
            
            # ファイルタイプ別分析
            type_stats = {}
            for result in all_results:
                file_type = result['type']
                if file_type not in type_stats:
                    type_stats[file_type] = []
                type_stats[file_type].append(result)
            
            print(f"\n📁 ファイルタイプ別性能:")
            for file_type, type_results in type_stats.items():
                avg_type_ratio = sum(r['compression_ratio'] for r in type_results) / len(type_results)
                avg_type_throughput = sum(r['throughput'] for r in type_results) / len(type_results)
                
                print(f"   {file_type} ({len(type_results)} ファイル):")
                print(f"      📈 平均圧縮率: {avg_type_ratio:.2f}%")
                print(f"      ⚡ 平均スループット: {avg_type_throughput:.2f}MB/s")
                
                # 個別ファイル
                for result in type_results:
                    status = "✅" if result.get('is_reversible', False) else "⚠️"
                    print(f"         {status} {result['name'][:35]:35} | {result['compression_ratio']:6.2f}%")
            
            # 最高性能
            best_compression = max(all_results, key=lambda r: r['compression_ratio'])
            best_speed = max(all_results, key=lambda r: r['throughput'])
            
            print(f"\n🥇 最高パフォーマンス:")
            print(f"   📈 最高圧縮率: {best_compression['name']} ({best_compression['compression_ratio']:.2f}%)")
            print(f"   ⚡ 最高速度: {best_speed['name']} ({best_speed['throughput']:.2f}MB/s)")
            
            # LZMA比較統計
            lzma_improvements = []
            for result in all_results:
                if 'comparison' in result and 'LZMA' in result['comparison']:
                    lzma_result = result['comparison']['LZMA']
                    if lzma_result and 'ratio' in lzma_result:
                        improvement = result['compression_ratio'] - lzma_result['ratio']
                        if improvement > 0:
                            lzma_improvements.append(improvement)
            
            if lzma_improvements:
                avg_lzma_improvement = sum(lzma_improvements) / len(lzma_improvements)
                print(f"\n🚀 LZMA超越実績:")
                print(f"   📈 平均改善: +{avg_lzma_improvement:.2f}%")
                print(f"   🏆 改善ファイル数: {len(lzma_improvements)}/{len(all_results)}")
                print(f"   📊 改善率: {len(lzma_improvements)/len(all_results)*100:.1f}%")
        
        # エラーファイル
        error_results = [r for r in results if r.get('error')]
        if error_results:
            print(f"\n❌ 処理失敗ファイル:")
            for result in error_results:
                print(f"   • {result['name']}: {result.get('error', '不明')}")
    
    # エンジンレポート
    engine_report = engine.get_advanced_report()
    print(f"\n🔧 エンジン統計:")
    stats = engine_report['processing_stats']
    print(f"   📊 処理ファイル数: {stats['total_files_processed']}")
    print(f"   ⚡ 平均スループット: {stats['average_throughput']:.2f}MB/s")
    
    print(f"\n🎉 NEXUS v3.0 超高度圧縮テスト完了!")
    
    return results


def get_advanced_file_type(file_path: Path) -> str:
    """高度ファイルタイプ判定"""
    suffix = file_path.suffix.lower()
    
    # 詳細タイプマッピング
    type_mapping = {
        '.txt': 'テキスト',
        '.mp4': '動画',
        '.mp3': '音楽',
        '.wav': '音楽',
        '.jpg': '画像',
        '.jpeg': '画像',
        '.png': '画像',
        '.gif': '画像',
        '.bmp': '画像',
        '.pdf': 'ドキュメント',
        '.docx': 'ドキュメント',
        '.xlsx': 'スプレッドシート',
        '.py': 'プログラム',
        '.js': 'プログラム',
        '.html': 'ウェブ',
        '.css': 'ウェブ',
        '.7z': '圧縮アーカイブ',
        '.zip': '圧縮アーカイブ',
        '.rar': '圧縮アーカイブ'
    }
    
    return type_mapping.get(suffix, 'その他')


def compare_with_standard_compression(data: bytes, filename: str) -> dict:
    """標準圧縮形式との比較"""
    results = {}
    
    # LZMA (7-Zip相当) - 複数プリセット
    for preset in [3, 6, 9]:
        try:
            import lzma
            start_time = time.perf_counter()
            lzma_compressed = lzma.compress(data, preset=preset)
            lzma_time = time.perf_counter() - start_time
            lzma_ratio = (1 - len(lzma_compressed) / len(data)) * 100
            
            results[f'LZMA-{preset}'] = {
                'size': len(lzma_compressed),
                'ratio': lzma_ratio,
                'time': lzma_time
            }
        except:
            results[f'LZMA-{preset}'] = None
    
    # 最良LZMAを基準に
    lzma_results = [r for r in results.values() if r is not None]
    if lzma_results:
        best_lzma = max(lzma_results, key=lambda x: x['ratio'])
        results['LZMA'] = best_lzma
    
    # GZIP
    try:
        import gzip
        start_time = time.perf_counter()
        gzip_compressed = gzip.compress(data, compresslevel=9)
        gzip_time = time.perf_counter() - start_time
        gzip_ratio = (1 - len(gzip_compressed) / len(data)) * 100
        
        results['GZIP'] = {
            'size': len(gzip_compressed),
            'ratio': gzip_ratio,
            'time': gzip_time
        }
    except:
        results['GZIP'] = None
    
    # BZIP2
    try:
        import bz2
        start_time = time.perf_counter()
        bz2_compressed = bz2.compress(data, compresslevel=9)
        bz2_time = time.perf_counter() - start_time
        bz2_ratio = (1 - len(bz2_compressed) / len(data)) * 100
        
        results['BZIP2'] = {
            'size': len(bz2_compressed),
            'ratio': bz2_ratio,
            'time': bz2_time
        }
    except:
        results['BZIP2'] = None
    
    return results


def simulate_nexus_v3_decompression(compressed_data: bytes) -> bytes:
    """NEXUS v3.0 解凍シミュレーション"""
    try:
        # v3.0 ヘッダー解析
        if len(compressed_data) < 256:
            return compressed_data
        
        header = compressed_data[:256]
        
        # マジックナンバー確認
        if header[:8] != b'NXADV300':
            return compressed_data
        
        import struct
        
        # 基本情報抽出
        original_size = struct.unpack('<Q', header[8:16])[0]
        chunk_count = struct.unpack('<I', header[16:20])[0]
        
        # チャンクデータ解凍
        decompressed_chunks = []
        current_pos = 256  # v3.0 ヘッダーサイズ
        
        for chunk_idx in range(chunk_count):
            if current_pos + 32 > len(compressed_data):
                break
            
            # v3.0 チャンクヘッダー
            chunk_header = compressed_data[current_pos:current_pos + 32]
            chunk_id = struct.unpack('<I', chunk_header[0:4])[0]
            chunk_size = struct.unpack('<I', chunk_header[4:8])[0]
            
            current_pos += 32
            
            # チャンクデータ
            if current_pos + chunk_size > len(compressed_data):
                chunk_size = len(compressed_data) - current_pos
            
            chunk_data = compressed_data[current_pos:current_pos + chunk_size]
            current_pos += chunk_size
            
            # チャンク解凍（複数手法に対応）
            decompressed_chunk = decompress_v3_chunk(chunk_data)
            decompressed_chunks.append((chunk_id, decompressed_chunk))
        
        # 結合
        decompressed_chunks.sort(key=lambda x: x[0])
        result = b''.join(chunk[1] for chunk in decompressed_chunks)
        
        return result
        
    except Exception as e:
        print(f"      解凍エラー: {e}")
        return compressed_data


def decompress_v3_chunk(chunk_data: bytes) -> bytes:
    """v3.0 チャンク解凍"""
    # ハイブリッド圧縮の判定と解凍
    if len(chunk_data) >= 8:
        method_header = chunk_data[:8]
        
        # メソッド判定
        if method_header.startswith(b'lzma'):
            try:
                import lzma
                return lzma.decompress(chunk_data[8:])
            except:
                pass
        elif method_header.startswith(b'gzip'):
            try:
                import gzip
                return gzip.decompress(chunk_data[8:])
            except:
                pass
        elif method_header.startswith(b'bz2'):
            try:
                import bz2
                return bz2.decompress(chunk_data[8:])
            except:
                pass
    
    # 標準LZMA解凍試行
    try:
        import lzma
        return lzma.decompress(chunk_data)
    except:
        return chunk_data


if __name__ == "__main__":
    test_nexus_v3_real_files()
