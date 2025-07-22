#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 NEXUS Ultra Efficient - 効率化エンジン
時間がかかりすぎる問題を解決する超高速圧縮エンジン

🎯 効率化戦略:
1. 事前解析による最適アルゴリズム選択
2. 並列処理による高速化
3. 適応的ファイルサイズ処理
4. メモリ効率最適化
5. 早期終了条件による処理時間短縮
"""

import os
import sys
import time
import zlib
import bz2
import lzma
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

class NexusUltraEfficient:
    """超効率化圧縮エンジン"""
    
    def __init__(self):
        self.max_workers = min(4, os.cpu_count())
        # ファイルサイズ別の処理戦略
        self.size_thresholds = {
            'tiny': 1024,           # 1KB未満: シンプル処理
            'small': 1024 * 100,    # 100KB未満: 標準処理
            'medium': 1024 * 1024 * 10,  # 10MB未満: 並列処理
            'large': 1024 * 1024 * 100   # 100MB以上: 超並列処理
        }
        
    def quick_file_analysis(self, data: bytes) -> dict:
        """超高速ファイル解析 - 最小限の情報で最適戦略決定"""
        size = len(data)
        
        # サンプリング解析（大きなファイルは一部のみ解析）
        if size > 1024 * 1024:  # 1MB以上
            sample_size = min(8192, size // 100)  # 1%または8KBのサンプル
            sample = data[:sample_size]
        else:
            sample = data
            
        # 高速エントロピー推定
        byte_counts = [0] * 256
        for byte in sample:
            byte_counts[byte] += 1
        
        # 圧縮率予測（簡易版）
        unique_bytes = sum(1 for count in byte_counts if count > 0)
        repetition_ratio = max(byte_counts) / len(sample) if sample else 0
        
        # 戦略決定
        if repetition_ratio > 0.7:  # 高繰り返し
            strategy = 'bz2_fast'
        elif unique_bytes < 128:    # 低多様性
            strategy = 'lzma_fast'
        else:                       # 一般的
            strategy = 'zlib_fast'
            
        return {
            'size': size,
            'strategy': strategy,
            'sample_entropy': unique_bytes / 256,
            'repetition': repetition_ratio
        }
    
    def compress_tiny(self, data: bytes) -> tuple:
        """超小ファイル用高速圧縮"""
        # 複数アルゴリズムを並列実行し、最初に完了したものを採用
        def try_compress(algorithm):
            if algorithm == 'zlib':
                return zlib.compress(data, 6), 'zlib_6'
            elif algorithm == 'bz2':
                return bz2.compress(data, 6), 'bz2_6'
            elif algorithm == 'lzma':
                return lzma.compress(data, preset=3), 'lzma_3'
        
        # 並列実行
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(try_compress, algo): algo 
                for algo in ['zlib', 'bz2', 'lzma']
            }
            
            # 最初に完了したものを採用
            for future in as_completed(futures):
                try:
                    result, method = future.result(timeout=0.1)  # 100ms制限
                    # 他のタスクをキャンセル
                    for f in futures:
                        if f != future and not f.done():
                            f.cancel()
                    return result, method
                except:
                    continue
                    
        # フォールバック
        return zlib.compress(data, 1), 'zlib_1'
    
    def compress_chunk(self, chunk: bytes, chunk_id: int, strategy: str) -> tuple:
        """チャンク単位の並列圧縮"""
        try:
            if strategy == 'bz2_fast':
                return bz2.compress(chunk, 3), chunk_id, 'bz2_3'
            elif strategy == 'lzma_fast':
                return lzma.compress(chunk, preset=2), chunk_id, 'lzma_2'
            else:  # zlib_fast
                return zlib.compress(chunk, 4), chunk_id, 'zlib_4'
        except Exception as e:
            # エラー時はzlib圧縮でフォールバック
            return zlib.compress(chunk, 1), chunk_id, 'zlib_1'
    
    def compress_large_parallel(self, data: bytes, strategy: str) -> tuple:
        """大容量ファイル用並列圧縮"""
        size = len(data)
        
        # チャンクサイズ決定
        if size > 100 * 1024 * 1024:  # 100MB以上
            chunk_size = 2 * 1024 * 1024  # 2MB chunks
        elif size > 10 * 1024 * 1024:   # 10MB以上
            chunk_size = 1 * 1024 * 1024  # 1MB chunks
        else:
            chunk_size = 512 * 1024       # 512KB chunks
        
        # データを分割
        chunks = []
        for i in range(0, size, chunk_size):
            chunks.append(data[i:i + chunk_size])
        
        # 並列圧縮
        compressed_chunks = [None] * len(chunks)
        methods = [None] * len(chunks)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # チャンクを並列で圧縮
            futures = {
                executor.submit(self.compress_chunk, chunk, i, strategy): i
                for i, chunk in enumerate(chunks)
            }
            
            for future in as_completed(futures):
                try:
                    compressed_data, chunk_id, method = future.result(timeout=30)  # 30秒制限
                    compressed_chunks[chunk_id] = compressed_data
                    methods[chunk_id] = method
                except Exception as e:
                    chunk_id = futures[future]
                    print(f"⚠️ チャンク {chunk_id} 圧縮失敗: {e}")
                    # フォールバック
                    compressed_chunks[chunk_id] = zlib.compress(chunks[chunk_id], 1)
                    methods[chunk_id] = 'zlib_1'
        
        # 結果をまとめる
        header = f"NEXUS_PARALLEL_V1:{len(chunks)}:{strategy}:".encode()
        result = header
        
        for i, (compressed_chunk, method) in enumerate(zip(compressed_chunks, methods)):
            chunk_header = f"{len(compressed_chunk)}:{method}:".encode()
            result += chunk_header + compressed_chunk
        
        return result, f"parallel_{strategy}"
    
    def compress_file(self, filepath: str) -> dict:
        """効率化ファイル圧縮"""
        start_time = time.time()
        
        try:
            # ファイル読み込み
            file_path = Path(filepath)
            if not file_path.exists():
                return {'success': False, 'error': f'ファイルが見つかりません: {filepath}'}
            
            with open(file_path, 'rb') as f:
                data = f.read()
            
            original_size = len(data)
            print(f"📁 処理開始: {file_path.name} ({original_size:,} bytes)")
            
            # 事前解析
            analysis = self.quick_file_analysis(data)
            print(f"🔍 解析結果: {analysis['strategy']} (エントロピー: {analysis['sample_entropy']:.3f})")
            
            # サイズ別処理戦略
            if original_size <= self.size_thresholds['tiny']:
                # 超小ファイル: 複数アルゴリズム並列実行
                compressed_data, method = self.compress_tiny(data)
                print(f"⚡ 超高速処理: {method}")
                
            elif original_size <= self.size_thresholds['medium']:
                # 中小ファイル: 最適アルゴリズム単体実行
                strategy = analysis['strategy']
                if strategy == 'bz2_fast':
                    compressed_data = bz2.compress(data, 6)
                    method = 'bz2_6'
                elif strategy == 'lzma_fast':
                    compressed_data = lzma.compress(data, preset=4)
                    method = 'lzma_4'
                else:
                    compressed_data = zlib.compress(data, 6)
                    method = 'zlib_6'
                print(f"🚀 最適化処理: {method}")
                
            else:
                # 大容量ファイル: 並列処理
                compressed_data, method = self.compress_large_parallel(data, analysis['strategy'])
                print(f"🔧 並列処理: {method}")
            
            # 結果保存
            output_path = file_path.with_suffix(file_path.suffix + '.nxue')
            with open(output_path, 'wb') as f:
                f.write(compressed_data)
            
            # 統計計算
            compressed_size = len(compressed_data)
            compression_ratio = (1 - compressed_size / original_size) * 100
            processing_time = time.time() - start_time
            speed = (original_size / 1024 / 1024) / processing_time if processing_time > 0 else float('inf')
            
            result = {
                'success': True,
                'original_size': original_size,
                'compressed_size': compressed_size,
                'compression_ratio': compression_ratio,
                'processing_time': processing_time,
                'speed_mbps': speed,
                'method': method,
                'output_file': str(output_path)
            }
            
            print(f"✅ 圧縮完了: {compression_ratio:.1f}% ({compressed_size:,} bytes)")
            print(f"⏱️ 処理時間: {processing_time:.2f}s ({speed:.1f} MB/s)")
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def decompress_file(self, filepath: str) -> dict:
        """効率化ファイル展開"""
        # 展開処理（実装は圧縮逆算）
        return {'success': False, 'error': '展開機能は次回実装予定'}

def run_efficiency_test():
    """効率化テスト実行"""
    print("🚀 NEXUS Ultra Efficient - 効率化テスト")
    print("=" * 60)
    
    engine = NexusUltraEfficient()
    
    # テストファイルリスト（sampleフォルダのみ、処理時間順）
    sample_dir = "NXZip-Python/sample"
    test_files = [
        f"{sample_dir}/陰謀論.mp3",                              # 小ファイル（音声）
        f"{sample_dir}/COT-001.jpg",                            # 中ファイル（画像）
        f"{sample_dir}/generated-music-1752042054079.wav",      # 中ファイル（音声）
        f"{sample_dir}/Python基礎講座3_4月26日-3.mp4",          # 大ファイル（動画）
        f"{sample_dir}/COT-012.png",                           # 超大画像（時間がかかる）
        f"{sample_dir}/出庫実績明細_202412.txt",                # 超大テキスト（時間がかかる）
    ]
    
    results = []
    total_start = time.time()
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\n📁 テスト: {test_file}")
            result = engine.compress_file(test_file)
            if result['success']:
                results.append(result)
            else:
                print(f"❌ エラー: {result.get('error', '不明')}")
        else:
            print(f"⚠️ ファイルが見つかりません: {test_file}")
    
    total_time = time.time() - total_start
    
    # 統計表示
    if results:
        print(f"\n📊 効率化テスト結果 ({len(results)}ファイル)")
        print("=" * 60)
        
        total_original = sum(r['original_size'] for r in results)
        total_compressed = sum(r['compressed_size'] for r in results)
        avg_compression = (1 - total_compressed / total_original) * 100 if total_original > 0 else 0
        avg_speed = sum(r['speed_mbps'] for r in results) / len(results)
        
        print(f"📈 総合統計:")
        print(f"   平均圧縮率: {avg_compression:.1f}%")
        print(f"   平均処理速度: {avg_speed:.1f} MB/s")
        print(f"   総処理時間: {total_time:.1f}s")
        
        print(f"\n📋 個別結果:")
        for i, result in enumerate(results, 1):
            filename = Path(result['output_file']).stem.replace('.nxue', '')
            print(f"   {i}. {filename}: {result['compression_ratio']:.1f}% "
                  f"({result['processing_time']:.1f}s, {result['speed_mbps']:.1f} MB/s)")

def main():
    """メイン関数"""
    if len(sys.argv) < 2:
        print("🚀 NEXUS Ultra Efficient - 効率化エンジン")
        print("使用方法:")
        print("  python nexus_ultra_efficient.py test                    # 効率化テスト")
        print("  python nexus_ultra_efficient.py compress <file>         # ファイル圧縮")
        print("  python nexus_ultra_efficient.py decompress <file>       # ファイル展開")
        return
    
    command = sys.argv[1].lower()
    engine = NexusUltraEfficient()
    
    if command == "test":
        run_efficiency_test()
    elif command == "compress" and len(sys.argv) >= 3:
        input_file = sys.argv[2]
        result = engine.compress_file(input_file)
        if not result['success']:
            print(f"❌ 圧縮失敗: {result.get('error', '不明なエラー')}")
    elif command == "decompress" and len(sys.argv) >= 3:
        input_file = sys.argv[2]
        result = engine.decompress_file(input_file)
        if not result['success']:
            print(f"❌ 展開失敗: {result.get('error', '不明なエラー')}")
    else:
        print("❌ 無効なコマンドまたは引数です")

if __name__ == "__main__":
    main()
