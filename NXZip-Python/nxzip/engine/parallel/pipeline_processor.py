"""
NEXUS TMC Engine - Parallel Processing Module

This module provides parallel processing capabilities for the TMC engine.
"""

import asyncio
import time
import threading
import queue
import psutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from ..core.data_types import AsyncTask

__all__ = ['ParallelPipelineProcessor']

# Configuration constants
MAX_WORKERS = min(psutil.cpu_count() or 4, 16)
PIPELINE_QUEUE_SIZE = 1000
ASYNC_BATCH_SIZE = 8
DEFAULT_CHUNK_SIZE = 64 * 1024


class ParallelPipelineProcessor:
    """
    TMC v9.0 革新的並列パイプライン処理エンジン
    真の並列処理 (ProcessPoolExecutor) + 非同期I/O + インテリジェントスケジューリング
    """
    
    def __init__(self, max_workers: int = MAX_WORKERS):
        self.max_workers = max_workers
        self.pipeline_queue = queue.Queue(maxsize=PIPELINE_QUEUE_SIZE)
        self.result_queue = queue.Queue()
        self.active_tasks = {}
        self.performance_stats = {
            'total_processed': 0,
            'average_throughput': 0.0,
            'pipeline_efficiency': 0.0
        }
        
        # 真の並列処理プール初期化（CPUバウンドタスク用）
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
        # I/Oバウンドタスク用（軽量ワーカー）
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # パイプライン制御
        self.pipeline_active = True
        self.pipeline_thread = None
        
        print(f"🚀 並列パイプライン初期化完了: {max_workers}ワーカー (Process+Thread Hybrid)")
    
    async def process_data_async(self, data_chunks: List[bytes], transform_type: str) -> List[Tuple[bytes, Dict]]:
        """
        CPUの全コアを活用した真の並列データ処理パイプライン
        ProcessPoolExecutorによりGIL制約を突破
        """
        print(f"  [並列パイプライン] 真の並列処理開始: {len(data_chunks)}チャンク")
        
        try:
            # タスクバッチ生成（プロセス間通信の最適化）
            task_batches = self._create_optimized_task_batches(data_chunks, transform_type)
            
            # 真の並列実行（プロセスベース）
            parallel_futures = []
            loop = asyncio.get_event_loop()
            
            for i, batch in enumerate(task_batches):
                # CPUバウンドタスクをプロセスプールで実行
                future = loop.run_in_executor(
                    self.process_pool, 
                    self._process_batch_in_subprocess, 
                    batch, i
                )
                parallel_futures.append(future)
            
            # 結果収集（非同期）
            all_results = []
            completed_batches = 0
            
            for batch_future in asyncio.as_completed(parallel_futures):
                try:
                    batch_data = await batch_future
                    all_results.extend(batch_data)
                    completed_batches += 1
                    
                    progress = (completed_batches / len(task_batches)) * 100
                    print(f"    [パイプライン] バッチ {completed_batches}/{len(task_batches)} 完了 ({progress:.1f}%)")
                    
                except Exception as e:
                    print(f"    [パイプライン] バッチ処理エラー: {e}")
            
            # 結果順序復元
            sorted_results = sorted(all_results, key=lambda x: x[1].get('chunk_id', 0))
            
            print(f"  [並列パイプライン] 真の並列処理完了: {len(sorted_results)}結果")
            return sorted_results
            
        except Exception as e:
            print(f"  [並列パイプライン] 並列処理エラー: {e}")
            return [(chunk, {'error': str(e)}) for chunk in data_chunks]
    
    def _create_optimized_task_batches(self, data_chunks: List[bytes], transform_type: str) -> List[List]:
        """メモリ効率最適化されたタスクバッチ生成"""
        batches = []
        current_batch = []
        current_batch_size = 0
        
        # 動的バッチサイズ決定（利用可能メモリに基づく）
        if psutil:
            available_memory = psutil.virtual_memory().available
            optimal_batch_size = min(8 * 1024 * 1024, available_memory // (self.max_workers * 4))  # 8MB上限
        else:
            optimal_batch_size = 4 * 1024 * 1024  # デフォルト4MB
        
        for i, chunk in enumerate(data_chunks):
            # 軽量タスクデータ構造（メモリ削減）
            task_data = {
                'chunk_data': chunk,
                'chunk_id': i,
                'transform_type': transform_type,
                'size': len(chunk)  # timestampを削除してメモリ節約
            }
            
            current_batch.append(task_data)
            current_batch_size += len(chunk)
            
            # 動的バッチ分割（メモリ効率重視）
            if (current_batch_size >= optimal_batch_size or 
                len(current_batch) >= self.max_workers * 2):  # ワーカー数の2倍まで
                batches.append(current_batch)
                current_batch = []
                current_batch_size = 0
        
        # 残りのタスクをバッチに追加
        if current_batch:
            batches.append(current_batch)
        
        total_chunks = sum(len(b) for b in batches)
        avg_batch_size = total_chunks / len(batches) if batches else 0
        print(f"    [最適化パイプライン] バッチ生成完了: {len(batches)}バッチ, 平均{avg_batch_size:.1f}チャンク, 最適サイズ: {optimal_batch_size//1024//1024}MB")
        return batches
    
    def _process_batch_in_subprocess(self, batch_data: List[Dict], batch_id: int) -> List[Tuple[bytes, Dict]]:
        """
        サブプロセス内でバッチ処理を実行
        GILに制約されない真の並列処理
        """
        import os
        import time
        
        process_id = os.getpid()
        start_time = time.time()
        
        try:
            results = []
            
            for task_data in batch_data:
                chunk_data = task_data['chunk_data']
                chunk_id = task_data['chunk_id']
                transform_type = task_data['transform_type']
                
                # 基本的な変換処理（軽量化）
                try:
                    # この部分では、重いオブジェクト（TMCEngineなど）の再作成を避け、
                    # 基本的な圧縮・変換のみを実行
                    if transform_type == 'basic_compression':
                        processed_chunk = self._subprocess_basic_compression(chunk_data)
                    elif transform_type == 'leco_transform':
                        processed_chunk = self._subprocess_leco_transform(chunk_data)
                    else:
                        # デフォルト処理
                        processed_chunk = chunk_data
                    
                    result_info = {
                        'chunk_id': chunk_id,
                        'original_size': len(chunk_data),
                        'processed_size': len(processed_chunk),
                        'process_id': process_id,
                        'processing_time': time.time() - start_time
                    }
                    
                    results.append((processed_chunk, result_info))
                    
                except Exception as e:
                    # エラー時は元データを返す
                    error_info = {
                        'chunk_id': chunk_id,
                        'error': str(e),
                        'process_id': process_id
                    }
                    results.append((chunk_data, error_info))
            
            batch_processing_time = time.time() - start_time
            print(f"    [プロセス {process_id}] バッチ{batch_id} 完了: {len(results)}チャンク, {batch_processing_time:.3f}秒")
            
            return results
            
        except Exception as e:
            print(f"    [プロセス {process_id}] バッチ{batch_id} エラー: {e}")
            # エラー時は元データをそのまま返す
            return [(task['chunk_data'], {'chunk_id': task.get('chunk_id', i), 'error': str(e)}) 
                   for i, task in enumerate(batch_data)]
    
    def _subprocess_basic_compression(self, data: bytes) -> bytes:
        """サブプロセス用の基本圧縮（軽量）"""
        try:
            import zlib
            return zlib.compress(data, level=6)
        except:
            return data
    
    def _subprocess_leco_transform(self, data: bytes) -> bytes:
        """サブプロセス用のLeCo変換（改良版）"""
        try:
            # より効果的な数値変換
            if len(data) >= 8:
                # 複数のエンコーディング方式を試行
                best_result = data
                best_ratio = 1.0
                
                # 1. 4バイト整数差分エンコーディング
                if len(data) % 4 == 0:
                    result_4byte = self._differential_encoding_4byte(data)
                    ratio_4byte = len(result_4byte) / len(data)
                    if ratio_4byte < best_ratio:
                        best_result = result_4byte
                        best_ratio = ratio_4byte
                
                # 2. 2バイト整数差分エンコーディング
                if len(data) % 2 == 0:
                    result_2byte = self._differential_encoding_2byte(data)
                    ratio_2byte = len(result_2byte) / len(data)
                    if ratio_2byte < best_ratio:
                        best_result = result_2byte
                        best_ratio = ratio_2byte
                
                # 3. 1バイト差分エンコーディング
                result_1byte = self._differential_encoding_1byte(data)
                ratio_1byte = len(result_1byte) / len(data)
                if ratio_1byte < best_ratio:
                    best_result = result_1byte
                    best_ratio = ratio_1byte
                
                return best_result
            
            return data
        except Exception as e:
            return data
    
    def _differential_encoding_4byte(self, data: bytes) -> bytes:
        """4バイト整数差分エンコーディング"""
        try:
            values = []
            for i in range(0, len(data), 4):
                val = int.from_bytes(data[i:i+4], 'little', signed=True)
                values.append(val)
            
            if len(values) > 1:
                # 適応的差分計算
                differences = [values[0]]  # 最初の値
                for i in range(1, len(values)):
                    diff = values[i] - values[i-1]
                    differences.append(diff)
                
                # 小さな差分をより効率的にエンコード
                result = bytearray()
                for diff in differences:
                    # 小さな差分は可変長エンコーディング
                    if -127 <= diff <= 127:
                        result.append(0)  # フラグ: 1バイト
                        result.append(diff & 0xFF)
                    else:
                        result.append(1)  # フラグ: 4バイト
                        result.extend(diff.to_bytes(4, 'little', signed=True))
                
                return bytes(result)
            
            return data
        except:
            return data
    
    def _differential_encoding_2byte(self, data: bytes) -> bytes:
        """2バイト整数差分エンコーディング"""
        try:
            values = []
            for i in range(0, len(data), 2):
                val = int.from_bytes(data[i:i+2], 'little', signed=True)
                values.append(val)
            
            if len(values) > 1:
                differences = [values[0]]
                for i in range(1, len(values)):
                    differences.append(values[i] - values[i-1])
                
                return b''.join(val.to_bytes(2, 'little', signed=True) for val in differences)
            
            return data
        except:
            return data
    
    def _differential_encoding_1byte(self, data: bytes) -> bytes:
        """1バイト差分エンコーディング"""
        try:
            if len(data) > 1:
                result = bytearray([data[0]])  # 最初のバイト
                for i in range(1, len(data)):
                    diff = (data[i] - data[i-1]) & 0xFF
                    result.append(diff)
                return bytes(result)
            
            return data
        except:
            return data
    
    def _create_task_batches(self, data_chunks: List[bytes], transform_type: str) -> List[List[AsyncTask]]:
        """タスクバッチ生成（負荷分散最適化）"""
        batches = []
        current_batch = []
        current_batch_size = 0
        
        for i, chunk in enumerate(data_chunks):
            task = AsyncTask(
                task_id=i,
                task_type=transform_type,
                data=chunk,
                priority=self._calculate_task_priority(chunk),
                created_time=time.time()
            )
            
            current_batch.append(task)
            current_batch_size += len(chunk)
            
            # バッチサイズ制限チェック
            if (len(current_batch) >= ASYNC_BATCH_SIZE or 
                current_batch_size >= DEFAULT_CHUNK_SIZE * 2):
                batches.append(current_batch)
                current_batch = []
                current_batch_size = 0
        
        # 残りのタスクを最終バッチに
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _calculate_task_priority(self, data: bytes) -> int:
        """タスク優先度計算（データサイズとエントロピーベース）"""
        size_factor = min(len(data) // 1024, 10)  # サイズファクター（最大10）
        
        # 簡易エントロピー計算
        try:
            byte_counts = {}
            for byte in data[:1024]:  # 先頭1KBサンプル
                byte_counts[byte] = byte_counts.get(byte, 0) + 1
            
            entropy = 0.0
            total = len(data[:1024])
            for count in byte_counts.values():
                prob = count / total
                entropy -= prob * (prob.bit_length() - 1) if prob > 0 else 0
            
            entropy_factor = min(int(entropy), 8)
        except:
            entropy_factor = 4
        
        # 高エントロピー（圧縮困難）なデータを低優先度に
        return max(1, 10 - entropy_factor + size_factor)
    
    async def _process_batch_async(self, task_batch: List[AsyncTask], batch_id: int) -> List[Tuple[bytes, Dict]]:
        """非同期バッチ処理"""
        try:
            loop = asyncio.get_event_loop()
            
            # バッチ内タスクの並列実行
            batch_futures = []
            for task in task_batch:
                future = loop.run_in_executor(
                    self.thread_pool,
                    self._process_single_task,
                    task
                )
                batch_futures.append(future)
            
            # バッチ内並列完了待機
            batch_results = await asyncio.gather(*batch_futures, return_exceptions=True)
            
            # エラーハンドリング
            processed_results = []
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    print(f"    [バッチ {batch_id}] タスク{i}エラー: {result}")
                    processed_results.append((task_batch[i].data, {'error': str(result)}))
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            print(f"    [バッチ {batch_id}] バッチ処理エラー: {e}")
            return [(task.data, {'error': str(e)}) for task in task_batch]
    
    def _process_single_task(self, task: AsyncTask) -> Tuple[bytes, Dict]:
        """単一タスク処理（ワーカースレッド内実行）"""
        try:
            start_time = time.time()
            thread_id = threading.get_ident()
            
            # ダミー処理（実際の変換ロジックをここに実装）
            processed_data = task.data  # プレースホルダー
            
            processing_time = time.time() - start_time
            
            result_info = {
                'task_id': task.task_id,
                'chunk_id': task.task_id,  # 互換性のため
                'processing_time': processing_time,
                'thread_id': thread_id,
                'task_type': task.task_type,
                'priority': task.priority,
                'original_size': len(task.data),
                'processed_size': len(processed_data)
            }
            
            return processed_data, result_info
            
        except Exception as e:
            return task.data, {'error': str(e), 'task_id': task.task_id}
    
    def start_pipeline(self):
        """パイプライン開始"""
        if not self.pipeline_active:
            self.pipeline_active = True
            self.pipeline_thread = threading.Thread(target=self._pipeline_worker, daemon=True)
            self.pipeline_thread.start()
            print("  [並列パイプライン] パイプラインワーカー開始")
    
    def stop_pipeline(self):
        """パイプライン停止"""
        self.pipeline_active = False
        if self.pipeline_thread and self.pipeline_thread.is_alive():
            self.pipeline_thread.join(timeout=1.0)
        print("  [並列パイプライン] パイプライン停止")
    
    def _pipeline_worker(self):
        """パイプラインワーカー（バックグラウンド処理）"""
        while self.pipeline_active:
            try:
                # キューからタスク取得（タイムアウト付き）
                task = self.pipeline_queue.get(timeout=0.1)
                
                # タスク処理
                result = self._process_single_task(task)
                self.result_queue.put(result)
                
                # 統計更新
                self.performance_stats['total_processed'] += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"    [パイプラインワーカー] エラー: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計取得"""
        return self.performance_stats.copy()
    
    def __del__(self):
        """デストラクタ（リソース解放）"""
        try:
            self.stop_pipeline()
            self.thread_pool.shutdown(wait=False)
            self.process_pool.shutdown(wait=False)
        except:
            pass
