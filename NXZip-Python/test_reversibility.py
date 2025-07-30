#!/usr/bin/env python3
"""
NEXUS可逆性テスト - 圧縮・解凍の完全一致性確認
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'nxzip', 'engine'))

import numpy as np
import time
import hashlib
from nexus_parallel_engine_clean import NEXUSParallelEngine, ParallelConfig


def test_nexus_reversibility():
    """NEXUS可逆性テスト"""
    print("🔄 NEXUS可逆性・完全性テスト")
    print("=" * 80)
    
    # テスト用エンジン
    config = ParallelConfig(
        use_gpu=False,  # 可逆性テストでは一貫性重視
        use_multiprocessing=False,
        use_threading=True,
        max_threads=4,
        chunk_size_mb=1,
        memory_limit_gb=4.0
    )
    
    engine = NEXUSParallelEngine(config)
    
    # 多様なテストデータ
    test_cases = [
        {
            'name': 'テキストデータ（ASCII）',
            'data': b"Hello NEXUS World! " * 1000 + b"Testing reversibility and data integrity.",
            'description': '基本テキストパターン'
        },
        {
            'name': 'バイナリデータ（完全ランダム）',
            'data': np.random.randint(0, 256, 50000, dtype=np.uint8).tobytes(),
            'description': '高エントロピーランダムデータ'
        },
        {
            'name': 'パターンデータ（反復構造）',
            'data': (b"PATTERN-123-ABC-" * 2000 + 
                    bytes(range(256)) * 100 + 
                    b"END-MARKER" * 500),
            'description': '構造化反復パターン'
        },
        {
            'name': 'ゼロデータ（極端圧縮）',
            'data': b"\x00" * 100000,
            'description': '同一バイト反復'
        },
        {
            'name': '混合データ（リアル想定）',
            'data': (b"Header-Section:" + 
                    np.random.randint(0, 256, 20000, dtype=np.uint8).tobytes() +
                    b"Middle-Structured-Data:" * 1000 +
                    bytes(range(256)) * 200 +
                    b"Footer-End"),
            'description': '実際のファイル構造模擬'
        },
        {
            'name': 'Unicode文字列',
            'data': "こんにちはNEXUS！🚀 テスト用データです。".encode('utf-8') * 2000,
            'description': 'マルチバイト文字含有'
        }
    ]
    
    print(f"🔬 テストケース数: {len(test_cases)}")
    
    # 各テストケースで可逆性確認
    all_results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*70}")
        print(f"🧪 テストケース {i+1}: {test_case['name']}")
        print(f"   📝 説明: {test_case['description']}")
        print(f"   📊 データサイズ: {len(test_case['data']):,} bytes ({len(test_case['data'])/1024:.1f}KB)")
        
        original_data = test_case['data']
        
        # 元データのハッシュ計算
        original_hash = hashlib.sha256(original_data).hexdigest()
        print(f"   🔑 元データハッシュ: {original_hash[:16]}...")
        
        try:
            # === 圧縮フェーズ ===
            print("   🔄 圧縮実行中...")
            compress_start = time.perf_counter()
            compressed_data = engine.parallel_compress(original_data, 'balanced')
            compress_time = time.perf_counter() - compress_start
            
            compression_ratio = (1 - len(compressed_data) / len(original_data)) * 100
            print(f"   ✅ 圧縮完了")
            print(f"      📈 圧縮率: {compression_ratio:.2f}%")
            print(f"      ⏱️ 圧縮時間: {compress_time:.3f}秒")
            print(f"      💾 圧縮後サイズ: {len(compressed_data):,} bytes")
            
            # === 解凍フェーズ ===
            print("   🔄 解凍実行中...")
            decompress_start = time.perf_counter()
            decompressed_data = simulate_nexus_decompression(compressed_data)
            decompress_time = time.perf_counter() - decompress_start
            
            print(f"   ✅ 解凍完了")
            print(f"      ⏱️ 解凍時間: {decompress_time:.3f}秒")
            print(f"      💾 解凍後サイズ: {len(decompressed_data):,} bytes")
            
            # === 可逆性検証 ===
            print("   🔍 可逆性検証中...")
            
            # 1. サイズ比較
            size_match = len(original_data) == len(decompressed_data)
            
            # 2. ハッシュ比較
            decompressed_hash = hashlib.sha256(decompressed_data).hexdigest()
            hash_match = original_hash == decompressed_hash
            
            # 3. バイト単位比較
            bytes_match = original_data == decompressed_data
            
            # 4. 詳細不一致分析（不一致時）
            mismatch_details = None
            if not bytes_match and len(original_data) == len(decompressed_data):
                mismatch_details = analyze_data_mismatch(original_data, decompressed_data)
            
            # 結果判定
            is_reversible = size_match and hash_match and bytes_match
            
            result = {
                'name': test_case['name'],
                'original_size': len(original_data),
                'compressed_size': len(compressed_data),
                'decompressed_size': len(decompressed_data),
                'compression_ratio': compression_ratio,
                'compress_time': compress_time,
                'decompress_time': decompress_time,
                'size_match': size_match,
                'hash_match': hash_match,
                'bytes_match': bytes_match,
                'is_reversible': is_reversible,
                'original_hash': original_hash,
                'decompressed_hash': decompressed_hash,
                'mismatch_details': mismatch_details
            }
            
            all_results.append(result)
            
            # 結果表示
            print(f"   📋 可逆性検証結果:")
            print(f"      📏 サイズ一致: {'✅' if size_match else '❌'}")
            print(f"      🔑 ハッシュ一致: {'✅' if hash_match else '❌'}")
            print(f"      🔢 バイト一致: {'✅' if bytes_match else '❌'}")
            print(f"      🏆 総合判定: {'✅ 完全可逆' if is_reversible else '❌ 不可逆'}")
            
            if not is_reversible:
                print(f"      ⚠️ 不一致詳細:")
                if not size_match:
                    print(f"         サイズ: {len(original_data)} → {len(decompressed_data)}")
                if not hash_match:
                    print(f"         ハッシュ変化: {original_hash[:16]}... → {decompressed_hash[:16]}...")
                if mismatch_details:
                    print(f"         不一致箇所: {mismatch_details['mismatch_count']} 位置")
                    print(f"         最初の不一致: 位置 {mismatch_details['first_mismatch']} (0x{original_data[mismatch_details['first_mismatch']]:02x} → 0x{decompressed_data[mismatch_details['first_mismatch']]:02x})")
            
        except Exception as e:
            print(f"   ❌ テストエラー: {str(e)}")
            import traceback
            traceback.print_exc()
            
            all_results.append({
                'name': test_case['name'],
                'is_reversible': False,
                'error': str(e)
            })
    
    # 総合レポート
    print(f"\n{'='*80}")
    print(f"📊 NEXUS可逆性総合レポート")
    print(f"{'='*80}")
    
    if all_results:
        successful_tests = [r for r in all_results if r.get('is_reversible', False)]
        failed_tests = [r for r in all_results if not r.get('is_reversible', False)]
        
        print(f"🎯 テスト結果サマリー:")
        print(f"   📊 総テスト数: {len(all_results)}")
        print(f"   ✅ 成功（完全可逆）: {len(successful_tests)}")
        print(f"   ❌ 失敗（不可逆）: {len(failed_tests)}")
        print(f"   📈 可逆性成功率: {len(successful_tests)/len(all_results)*100:.1f}%")
        
        if successful_tests:
            avg_compression_ratio = sum(r.get('compression_ratio', 0) for r in successful_tests) / len(successful_tests)
            avg_compress_time = sum(r.get('compress_time', 0) for r in successful_tests) / len(successful_tests)
            avg_decompress_time = sum(r.get('decompress_time', 0) for r in successful_tests) / len(successful_tests)
            
            print(f"\n🏆 成功ケース統計:")
            print(f"   📈 平均圧縮率: {avg_compression_ratio:.2f}%")
            print(f"   ⏱️ 平均圧縮時間: {avg_compress_time:.3f}秒")
            print(f"   ⏱️ 平均解凍時間: {avg_decompress_time:.3f}秒")
            print(f"   ⚡ 解凍効率: {avg_decompress_time/avg_compress_time:.2f}x")
        
        if failed_tests:
            print(f"\n❌ 失敗ケース詳細:")
            for result in failed_tests:
                print(f"   • {result['name']}: {result.get('error', '詳細不明')}")
        
        # 圧縮性能別分析
        print(f"\n📊 圧縮性能分析:")
        for result in successful_tests:
            efficiency_score = (result.get('compression_ratio', 0) * 0.7 + 
                               (10/max(result.get('compress_time', 0.001), 0.001)) * 0.3)
            print(f"   • {result['name'][:30]:30} | 圧縮率:{result.get('compression_ratio', 0):6.2f}% | 効率:{efficiency_score:6.1f}")
    
    # 理論的考察
    print(f"\n🧠 NEXUS理論的可逆性考察:")
    print(f"   🔬 理論フレームワーク: AEU分解 + HDSC + 順列正規化")
    print(f"   🔄 可逆性保証機構: 数学的双射変換 + メタデータ保存")
    print(f"   🛡️ 整合性検証: SHA256ハッシュ + バイト単位比較")
    print(f"   ⚡ 実装品質: {'優秀' if len(successful_tests) == len(all_results) else '改善要'}")
    
    print(f"\n🎉 NEXUS可逆性テスト完了!")
    
    return all_results


def simulate_nexus_decompression(compressed_data: bytes) -> bytes:
    """
    NEXUS解凍シミュレーション
    注意: 現在は圧縮の逆変換として簡易実装
    """
    print("      🔄 NEXUS解凍処理中...")
    
    try:
        # ヘッダー解析
        if len(compressed_data) < 128:
            raise ValueError("圧縮データが短すぎます")
        
        header = compressed_data[:128]
        
        # マジックナンバー確認
        if header[:8] != b'NXPAR002':
            raise ValueError("無効なNEXUSヘッダー")
        
        import struct
        
        # 基本情報抽出
        original_size = struct.unpack('<Q', header[8:16])[0]
        chunk_count = struct.unpack('<I', header[16:20])[0]
        quality_code = struct.unpack('<I', header[20:24])[0]
        
        print(f"         📊 元サイズ: {original_size:,} bytes")
        print(f"         🔷 チャンク数: {chunk_count}")
        print(f"         🎯 品質: {quality_code}")
        
        # チャンクデータ解凍
        decompressed_chunks = []
        current_pos = 128  # ヘッダー後から開始
        
        for chunk_idx in range(chunk_count):
            if current_pos + 64 > len(compressed_data):
                break
                
            # チャンクヘッダー読み取り
            chunk_header = compressed_data[current_pos:current_pos + 64]
            chunk_id = struct.unpack('<I', chunk_header[0:4])[0]
            chunk_size = struct.unpack('<I', chunk_header[4:8])[0]
            
            current_pos += 64
            
            # チャンクデータ読み取り
            if current_pos + chunk_size > len(compressed_data):
                chunk_size = len(compressed_data) - current_pos
            
            chunk_data = compressed_data[current_pos:current_pos + chunk_size]
            current_pos += chunk_size
            
            # チャンク解凍
            decompressed_chunk = decompress_single_chunk(chunk_data, quality_code)
            decompressed_chunks.append((chunk_id, decompressed_chunk))
        
        # チャンクID順でソート
        decompressed_chunks.sort(key=lambda x: x[0])
        
        # 結合
        result = b''.join(chunk[1] for chunk in decompressed_chunks)
        
        print(f"         ✅ 解凍完了: {len(result):,} bytes")
        
        return result
        
    except Exception as e:
        print(f"         ❌ 解凍エラー: {e}")
        # フォールバック: 元データサイズで切り詰め
        if len(compressed_data) > 128:
            import struct
            try:
                original_size = struct.unpack('<Q', compressed_data[8:16])[0]
                fallback_data = compressed_data[128:128+min(original_size, len(compressed_data)-128)]
                return fallback_data
            except:
                pass
        
        return compressed_data[:len(compressed_data)//2]  # 最後の手段


def decompress_single_chunk(chunk_data: bytes, quality_code: int) -> bytes:
    """単一チャンク解凍"""
    try:
        # 品質別解凍
        if quality_code == 1:  # fast
            preset = 0
        elif quality_code == 2:  # balanced
            preset = 3
        else:  # max
            preset = 6
        
        # LZMA解凍試行
        try:
            import lzma
            return lzma.decompress(chunk_data)
        except:
            # GPU加速圧縮の逆変換試行
            return reverse_gpu_compression(chunk_data)
            
    except Exception as e:
        # フォールバック: 元データとして返却
        return chunk_data


def reverse_gpu_compression(compressed_data: bytes) -> bytes:
    """GPU圧縮の逆変換"""
    try:
        # デルタ符号化の逆変換
        if len(compressed_data) > 0:
            deltas = np.frombuffer(compressed_data, dtype=np.int8)
            # 累積和でデルタを復元
            if len(deltas) > 0:
                restored = np.cumsum(np.concatenate([[0], deltas.astype(np.int16)]))
                restored = np.clip(restored, 0, 255).astype(np.uint8)
                return restored.tobytes()
        
        return compressed_data
        
    except:
        return compressed_data


def analyze_data_mismatch(original: bytes, decompressed: bytes) -> dict:
    """データ不一致分析"""
    mismatch_positions = []
    min_length = min(len(original), len(decompressed))
    
    for i in range(min_length):
        if original[i] != decompressed[i]:
            mismatch_positions.append(i)
    
    return {
        'mismatch_count': len(mismatch_positions),
        'first_mismatch': mismatch_positions[0] if mismatch_positions else None,
        'mismatch_positions': mismatch_positions[:10],  # 最初の10個
        'mismatch_rate': len(mismatch_positions) / min_length if min_length > 0 else 0
    }


if __name__ == "__main__":
    test_nexus_reversibility()
