#!/usr/bin/env python3
"""
NEXUS実ファイルテスト - sample/ディレクトリの実際のファイルでテスト
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'nxzip', 'engine'))

import time
import hashlib
from pathlib import Path
from nexus_parallel_engine_clean import NEXUSParallelEngine, ParallelConfig


def test_nexus_with_real_files():
    """実ファイルでのNEXUSテスト"""
    print("🗂️ NEXUS実ファイル圧縮テスト")
    print("=" * 80)
    
    # サンプルディレクトリ
    sample_dir = Path("sample")
    
    if not sample_dir.exists():
        print(f"❌ サンプルディレクトリが見つかりません: {sample_dir}")
        return
    
    # NEXUS設定
    config = ParallelConfig(
        use_gpu=False,  # 安定性重視
        use_multiprocessing=True,
        use_threading=True,
        max_threads=6,
        max_processes=3,
        chunk_size_mb=2,
        memory_limit_gb=8.0
    )
    
    engine = NEXUSParallelEngine(config)
    
    # テスト対象ファイル選択（圧縮ファイル以外）
    test_files = []
    
    for file_path in sample_dir.iterdir():
        if file_path.is_file():
            # 既に圧縮されたファイル（.7z）は除外
            if not file_path.suffix.lower() in ['.7z']:
                # ファイルサイズチェック（100MB以下）
                try:
                    file_size = file_path.stat().st_size
                    if file_size < 100 * 1024 * 1024:  # 100MB未満
                        test_files.append({
                            'path': file_path,
                            'name': file_path.name,
                            'size': file_size,
                            'type': get_file_type(file_path)
                        })
                except:
                    continue
    
    # ファイルサイズでソート
    test_files.sort(key=lambda x: x['size'])
    
    print(f"🔬 テスト対象ファイル: {len(test_files)} 個")
    for file_info in test_files:
        print(f"   📄 {file_info['name']} ({file_info['size'] / 1024:.1f}KB) [{file_info['type']}]")
    
    # 各ファイルでテスト
    results = []
    
    for i, file_info in enumerate(test_files):
        print(f"\n{'='*70}")
        print(f"🧪 テスト {i+1}/{len(test_files)}: {file_info['name']}")
        print(f"   📁 ファイルタイプ: {file_info['type']}")
        print(f"   📊 ファイルサイズ: {file_info['size']:,} bytes ({file_info['size']/1024:.1f}KB)")
        
        try:
            # ファイル読み込み
            print("   📖 ファイル読み込み中...")
            with open(file_info['path'], 'rb') as f:
                original_data = f.read()
            
            # 元データのハッシュ
            original_hash = hashlib.sha256(original_data).hexdigest()
            print(f"   🔑 元データハッシュ: {original_hash[:16]}...")
            
            # 品質設定（ファイルタイプ別）
            quality = determine_quality_for_file_type(file_info['type'])
            print(f"   🎯 圧縮品質: {quality}")
            
            # === NEXUS圧縮テスト ===
            print("   🔄 NEXUS圧縮実行中...")
            compress_start = time.perf_counter()
            compressed_data = engine.parallel_compress(original_data, quality)
            compress_time = time.perf_counter() - compress_start
            
            compression_ratio = (1 - len(compressed_data) / len(original_data)) * 100
            throughput = file_info['size'] / 1024 / 1024 / compress_time  # MB/s
            
            print(f"   ✅ 圧縮完了!")
            print(f"      📈 圧縮率: {compression_ratio:.2f}%")
            print(f"      ⚡ スループット: {throughput:.2f}MB/s")
            print(f"      ⏱️ 圧縮時間: {compress_time:.3f}秒")
            print(f"      💾 圧縮前: {len(original_data):,} bytes")
            print(f"      💾 圧縮後: {len(compressed_data):,} bytes")
            
            # === 可逆性テスト ===
            print("   🔄 可逆性テスト中...")
            decompress_start = time.perf_counter()
            decompressed_data = simulate_nexus_decompression(compressed_data)
            decompress_time = time.perf_counter() - decompress_start
            
            # 検証
            size_match = len(original_data) == len(decompressed_data)
            decompressed_hash = hashlib.sha256(decompressed_data).hexdigest()
            hash_match = original_hash == decompressed_hash
            bytes_match = original_data == decompressed_data
            is_reversible = size_match and hash_match and bytes_match
            
            print(f"   📋 可逆性結果:")
            print(f"      📏 サイズ一致: {'✅' if size_match else '❌'}")
            print(f"      🔑 ハッシュ一致: {'✅' if hash_match else '❌'}")
            print(f"      🔢 バイト一致: {'✅' if bytes_match else '❌'}")
            print(f"      🏆 総合判定: {'✅ 完全可逆' if is_reversible else '❌ 不可逆'}")
            print(f"      ⏱️ 解凍時間: {decompress_time:.3f}秒")
            
            # === 他の圧縮方式との比較 ===
            print("   📊 他形式との比較...")
            comparison_results = compare_with_other_formats(original_data, file_info['name'])
            
            # 結果保存
            result = {
                'name': file_info['name'],
                'type': file_info['type'],
                'original_size': len(original_data),
                'compressed_size': len(compressed_data),
                'compression_ratio': compression_ratio,
                'compress_time': compress_time,
                'decompress_time': decompress_time,
                'throughput': throughput,
                'is_reversible': is_reversible,
                'quality': quality,
                'comparison': comparison_results
            }
            
            results.append(result)
            
            # 比較結果表示
            print(f"   📈 圧縮比較:")
            for method, comp_result in comparison_results.items():
                if comp_result:
                    ratio_diff = compression_ratio - comp_result['ratio']
                    print(f"      {method:>8}: {comp_result['ratio']:6.2f}% (NEXUS{ratio_diff:+.2f}%)")
            
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
    
    # === 総合レポート ===
    print(f"\n{'='*80}")
    print(f"📊 NEXUS実ファイル総合レポート")
    print(f"{'='*80}")
    
    if results:
        successful_results = [r for r in results if r.get('is_reversible', False)]
        failed_results = [r for r in results if not r.get('is_reversible', False)]
        
        print(f"🎯 テスト結果サマリー:")
        print(f"   📊 総テスト数: {len(results)}")
        print(f"   ✅ 成功: {len(successful_results)}")
        print(f"   ❌ 失敗: {len(failed_results)}")
        print(f"   📈 成功率: {len(successful_results)/len(results)*100:.1f}%")
        
        if successful_results:
            print(f"\n🏆 成功ケース詳細:")
            
            # ファイルタイプ別統計
            type_stats = {}
            for result in successful_results:
                file_type = result['type']
                if file_type not in type_stats:
                    type_stats[file_type] = []
                type_stats[file_type].append(result)
            
            for file_type, type_results in type_stats.items():
                avg_ratio = sum(r['compression_ratio'] for r in type_results) / len(type_results)
                avg_throughput = sum(r['throughput'] for r in type_results) / len(type_results)
                
                print(f"\n   📁 {file_type} ファイル ({len(type_results)} 個):")
                print(f"      📈 平均圧縮率: {avg_ratio:.2f}%")
                print(f"      ⚡ 平均スループット: {avg_throughput:.2f}MB/s")
                
                for result in type_results:
                    efficiency = result['compression_ratio'] * 0.7 + (10/max(result['compress_time'], 0.001)) * 0.3
                    print(f"         • {result['name'][:40]:40} | {result['compression_ratio']:6.2f}% | {result['throughput']:6.2f}MB/s | 効率:{efficiency:5.1f}")
        
        # 最高パフォーマンス
        if successful_results:
            best_compression = max(successful_results, key=lambda r: r['compression_ratio'])
            best_speed = max(successful_results, key=lambda r: r['throughput'])
            
            print(f"\n🥇 最高パフォーマンス:")
            print(f"   📈 最高圧縮率: {best_compression['name']} ({best_compression['compression_ratio']:.2f}%)")
            print(f"   ⚡ 最高速度: {best_speed['name']} ({best_speed['throughput']:.2f}MB/s)")
        
        if failed_results:
            print(f"\n❌ 失敗ケース:")
            for result in failed_results:
                print(f"   • {result['name']}: {result.get('error', '詳細不明')}")
    
    print(f"\n🎉 NEXUS実ファイルテスト完了!")
    
    return results


def get_file_type(file_path: Path) -> str:
    """ファイルタイプ判定"""
    suffix = file_path.suffix.lower()
    
    type_mapping = {
        '.txt': 'テキスト',
        '.mp4': '動画',
        '.mp3': '音楽',
        '.wav': '音楽', 
        '.jpg': '画像',
        '.jpeg': '画像',
        '.png': '画像',
        '.pdf': 'ドキュメント',
        '.docx': 'ドキュメント',
        '.xlsx': 'スプレッドシート',
        '.py': 'プログラム',
        '.js': 'プログラム',
        '.html': 'ウェブ',
        '.css': 'ウェブ'
    }
    
    return type_mapping.get(suffix, 'その他')


def determine_quality_for_file_type(file_type: str) -> str:
    """ファイルタイプ別品質設定"""
    quality_mapping = {
        'テキスト': 'max',      # テキストは高圧縮
        '動画': 'fast',         # 動画は既に圧縮済み
        '音楽': 'fast',         # 音楽も既に圧縮済み
        '画像': 'balanced',     # 画像はバランス
        'ドキュメント': 'max',   # ドキュメントは高圧縮
        'プログラム': 'max',     # プログラムは高圧縮
        'ウェブ': 'max',        # ウェブファイルは高圧縮
        'その他': 'balanced'     # その他はバランス
    }
    
    return quality_mapping.get(file_type, 'balanced')


def compare_with_other_formats(data: bytes, filename: str) -> dict:
    """他の圧縮形式との比較"""
    results = {}
    
    # LZMA (7-Zip相当)
    try:
        import lzma
        start_time = time.perf_counter()
        lzma_compressed = lzma.compress(data, preset=6)
        lzma_time = time.perf_counter() - start_time
        lzma_ratio = (1 - len(lzma_compressed) / len(data)) * 100
        
        results['LZMA'] = {
            'size': len(lzma_compressed),
            'ratio': lzma_ratio,
            'time': lzma_time
        }
    except:
        results['LZMA'] = None
    
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


def simulate_nexus_decompression(compressed_data: bytes) -> bytes:
    """NEXUS解凍シミュレーション（簡易版）"""
    try:
        # ヘッダー解析
        if len(compressed_data) < 128:
            return compressed_data
        
        header = compressed_data[:128]
        
        # マジックナンバー確認
        if header[:8] != b'NXPAR002':
            return compressed_data
        
        import struct
        
        # 基本情報抽出
        original_size = struct.unpack('<Q', header[8:16])[0]
        chunk_count = struct.unpack('<I', header[16:20])[0]
        
        # チャンクデータ解凍
        decompressed_chunks = []
        current_pos = 128
        
        for chunk_idx in range(chunk_count):
            if current_pos + 64 > len(compressed_data):
                break
            
            # チャンクヘッダー
            chunk_header = compressed_data[current_pos:current_pos + 64]
            chunk_id = struct.unpack('<I', chunk_header[0:4])[0]
            chunk_size = struct.unpack('<I', chunk_header[4:8])[0]
            
            current_pos += 64
            
            # チャンクデータ
            if current_pos + chunk_size > len(compressed_data):
                chunk_size = len(compressed_data) - current_pos
            
            chunk_data = compressed_data[current_pos:current_pos + chunk_size]
            current_pos += chunk_size
            
            # チャンク解凍
            try:
                import lzma
                decompressed_chunk = lzma.decompress(chunk_data)
            except:
                decompressed_chunk = chunk_data
            
            decompressed_chunks.append((chunk_id, decompressed_chunk))
        
        # 結合
        decompressed_chunks.sort(key=lambda x: x[0])
        result = b''.join(chunk[1] for chunk in decompressed_chunks)
        
        return result
        
    except Exception:
        return compressed_data


if __name__ == "__main__":
    test_nexus_with_real_files()
