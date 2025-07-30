#!/usr/bin/env python3
"""
NEXUS Parallel Processing Engine - 並列処理エンジン
Multi-threading + Multi-processing + GPU Acceleration + Distributed Processing

並列処理機能:
1. インテリジェントデータ分割
2. 負荷分散並列圧縮  
3. GPU加速処理
4. 非同期I/O処理
5. メモリ効率最適化
6. 分散処理サポート
7. 動的負荷分散
"""

import numpy as np
import time
import threading
import multiprocessing
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import queue
import ctypes
import sys
import os
import psutil
import gc
from pathlib import Path

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

try:
    from numba import cuda, jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    cuda = None
    jit = lambda nopython=True, parallel=False: lambda f: f  # フォールバック
    prange = range

# プロジェクトパス追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from nexus_theory_core import NEXUSTheoryCore, AdaptiveElementalUnit, PolyominoShape
    from nexus_advanced_optimizer import NEXUSAdvancedOptimizer
    NEXUS_ENGINES_AVAILABLE = True
except ImportError:
    NEXUS_ENGINES_AVAILABLE = False
    print("⚠️ NEXUS理論エンジンが利用できません - スタンドアロンモードで動作")


@dataclass
class SystemResources:
    """システムリソース情報"""
    cpu_count: int
    memory_gb: float
    gpu_available: bool
    gpu_memory_gb: float = 0.0
    load_average: float = 0.0
    disk_io_speed: float = 0.0  # MB/s


@dataclass
class ParallelConfig:
    """並列処理設定"""
    use_gpu: bool = True
    use_multiprocessing: bool = True
    use_threading: bool = True
    use_distributed: bool = False
    max_threads: int = field(default_factory=lambda: min(16, multiprocessing.cpu_count() * 2))
    max_processes: int = field(default_factory=lambda: min(8, multiprocessing.cpu_count()))
    chunk_size_mb: int = 4
    gpu_memory_limit_mb: int = 2048
    memory_limit_gb: float = 8.0
    io_buffer_size_mb: int = 64
    
    # 品質別設定
    fast_settings: Dict[str, Any] = field(default_factory=lambda: {
        'chunk_size_mb': 2,
        'max_threads': 4,
        'use_gpu': False
    })
    
    balanced_settings: Dict[str, Any] = field(default_factory=lambda: {
        'chunk_size_mb': 4,
        'max_threads': 8,
        'use_gpu': True
    })
    
    max_settings: Dict[str, Any] = field(default_factory=lambda: {
        'chunk_size_mb': 8,
        'max_threads': 16,
        'use_gpu': True,
        'use_distributed': True
    })


@dataclass
class ProcessingChunk:
    """処理チャンク"""
    chunk_id: int
    data: bytes
    start_offset: int
    end_offset: int
    priority: int = 0
    processing_method: str = "auto"  # auto, cpu, gpu, distributed
    complexity_score: float = 0.0
    entropy: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingResult:
    """処理結果"""
    chunk_id: int
    compressed_data: bytes
    compression_ratio: float
    processing_time: float
    method_used: str
    energy_efficiency: float = 0.0
    quality_score: float = 0.0
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceProfile:
    """パフォーマンスプロファイル"""
    throughput_mb_s: float = 0.0
    compression_ratio: float = 0.0
    cpu_utilization: float = 0.0
    gpu_utilization: float = 0.0
    memory_efficiency: float = 0.0
    power_consumption: float = 0.0
    adaptive_score: float = 0.0


class AdvancedMemoryManager:
    """高度メモリ管理器"""
    
    def __init__(self, limit_mb: int = 4096):
        self.limit_bytes = limit_mb * 1024 * 1024
        self.allocated_memory = {}
        self.memory_pools = {
            'small': [],    # < 1MB
            'medium': [],   # 1-10MB  
            'large': []     # > 10MB
        }
        self.current_usage = 0
        self.peak_usage = 0
        self.lock = threading.Lock()
        self.gc_threshold = 0.8  # 80%でGC実行
        
        print(f"🧠 高度メモリ管理器初期化 - 制限: {limit_mb}MB")
    
    def allocate(self, name: str, size: int, pool_type: str = "auto") -> bool:
        """スマートメモリ割り当て"""
        with self.lock:
            # 利用率チェック
            if self.current_usage + size > self.limit_bytes:
                # 自動ガベージコレクション
                if self._auto_garbage_collect():
                    if self.current_usage + size > self.limit_bytes:
                        return False
                else:
                    return False
            
            # プール決定
            if pool_type == "auto":
                if size < 1024 * 1024:
                    pool_type = "small"
                elif size < 10 * 1024 * 1024:
                    pool_type = "medium"
                else:
                    pool_type = "large"
            
            self.allocated_memory[name] = {
                'size': size,
                'pool': pool_type,
                'timestamp': time.time()
            }
            self.current_usage += size
            self.peak_usage = max(self.peak_usage, self.current_usage)
            
            return True
    
    def deallocate(self, name: str):
        """メモリ解放"""
        with self.lock:
            if name in self.allocated_memory:
                info = self.allocated_memory[name]
                self.current_usage -= info['size']
                del self.allocated_memory[name]
    
    def _auto_garbage_collect(self) -> bool:
        """自動ガベージコレクション"""
        if self.current_usage / self.limit_bytes > self.gc_threshold:
            # 古いメモリブロックを解放
            current_time = time.time()
            to_remove = []
            
            for name, info in self.allocated_memory.items():
                if current_time - info['timestamp'] > 300:  # 5分以上古い
                    to_remove.append(name)
            
            for name in to_remove:
                self.deallocate(name)
            
            # システムGC実行
            gc.collect()
            
            return len(to_remove) > 0
        
        return False
    
    def get_advanced_usage(self) -> Dict[str, Any]:
        """詳細使用量取得"""
        with self.lock:
            pool_usage = {pool: 0 for pool in self.memory_pools.keys()}
            
            for info in self.allocated_memory.values():
                pool_usage[info['pool']] += info['size']
            
            return {
                'current_mb': self.current_usage // (1024 * 1024),
                'peak_mb': self.peak_usage // (1024 * 1024),
                'limit_mb': self.limit_bytes // (1024 * 1024),
                'utilization': self.current_usage / self.limit_bytes,
                'allocated_objects': len(self.allocated_memory),
                'pool_usage_mb': {k: v // (1024 * 1024) for k, v in pool_usage.items()}
            }


class IntelligentGPUAccelerator:
    """インテリジェントGPU加速器"""
    
    def __init__(self):
        self.available = GPU_AVAILABLE and self._check_gpu_availability()
        self.device_count = 0
        self.gpu_memory = 0
        self.compute_capability = 0.0
        self.performance_profile = {}
        
        if self.available and cp is not None:
            try:
                self.device_count = cp.cuda.runtime.getDeviceCount()
                
                # GPU情報取得
                device = cp.cuda.Device(0)
                device.use()
                
                # メモリ情報
                free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                self.gpu_memory = total_mem // (1024 * 1024)  # MB
                
                print(f"🚀 GPU加速利用可能: {self.device_count} デバイス")
                print(f"   💾 GPUメモリ: {self.gpu_memory}MB")
            except Exception as e:
                print(f"⚠️ GPU初期化エラー: {e}")
                self.available = False
        else:
            print(f"⚠️ GPU加速利用不可")
    
    def _check_gpu_availability(self) -> bool:
        """GPU利用可能性チェック"""
        if not GPU_AVAILABLE or cp is None:
            return False
            
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            return device_count > 0
        except:
            return False
    
    def benchmark_gpu_performance(self, data_size: int = 1024*1024) -> Dict[str, float]:
        """GPUパフォーマンスベンチマーク"""
        if not self.available:
            return {'throughput': 0.0, 'latency': float('inf')}
        
        try:
            # テストデータ生成
            test_data = np.random.randint(0, 256, data_size, dtype=np.uint8)
            
            # GPU転送＋処理ベンチマーク
            start_time = time.perf_counter()
            
            gpu_data = cp.asarray(test_data)
            result = self._gpu_test_kernel(gpu_data)
            cp.cuda.Stream.null.synchronize()
            
            end_time = time.perf_counter()
            
            processing_time = end_time - start_time
            throughput = data_size / (1024 * 1024) / processing_time  # MB/s
            
            return {
                'throughput': throughput,
                'latency': processing_time * 1000,  # ms
                'efficiency': min(100.0, throughput / 1000 * 100)  # %
            }
            
        except Exception as e:
            print(f"⚠️ GPUベンチマークエラー: {e}")
            return {'throughput': 0.0, 'latency': float('inf')}
    
    def _gpu_test_kernel(self, data: Any) -> Any:
        """GPUテストカーネル"""
        if cp is None:
            return data
            
        # 簡単な並列処理テスト
        result = cp.zeros_like(data)
        
        # エレメント単位処理（並列）
        result = data * 2 + 1
        result = cp.roll(result, 1)
        result = cp.sort(result)
        
        return result
    
    @jit(nopython=True, parallel=True) if NUMBA_AVAILABLE else lambda func: func
    def parallel_pattern_detection(self, data_array: np.ndarray, pattern_size: int = 4) -> np.ndarray:
        """並列パターン検出"""
        result = np.zeros(len(data_array) - pattern_size + 1, dtype=np.int32)
        
        for i in prange(len(result)):
            pattern_hash = 0
            for j in range(pattern_size):
                pattern_hash = pattern_hash * 31 + data_array[i + j]
            result[i] = pattern_hash & 0x7FFFFFFF
        
        return result
    
    def gpu_accelerated_compression(self, data: bytes, algorithm: str = "hybrid") -> bytes:
        """GPU加速圧縮"""
        if not self.available:
            return self._cpu_fallback_compression(data)
        
        try:
            # データをnumpy配列に変換
            data_array = np.frombuffer(data, dtype=np.uint8)
            
            if algorithm == "hybrid":
                return self._hybrid_gpu_compression(data_array)
            elif algorithm == "pattern":
                return self._pattern_gpu_compression(data_array)
            else:
                return self._simple_gpu_compression(data_array)
                
        except Exception as e:
            print(f"⚠️ GPU圧縮エラー: {e}")
            return self._cpu_fallback_compression(data)
    
    def _hybrid_gpu_compression(self, data_array: np.ndarray) -> bytes:
        """ハイブリッドGPU圧縮"""
        # GPU並列パターン検出
        patterns = self.parallel_pattern_detection(data_array)
        
        # パターン基づく圧縮
        compressed = self._compress_with_patterns(data_array, patterns)
        
        return compressed.tobytes()
    
    def _pattern_gpu_compression(self, data_array: np.ndarray) -> bytes:
        """パターンベースGPU圧縮"""
        if cp is None:
            return data_array.tobytes()
        
        # GPU配列に転送
        gpu_data = cp.asarray(data_array)
        
        # パターン分析
        compressed_gpu = self._gpu_pattern_compress(gpu_data)
        
        # CPU配列に戻す
        compressed = cp.asnumpy(compressed_gpu)
        
        return compressed.tobytes()
    
    def _simple_gpu_compression(self, data_array: np.ndarray) -> bytes:
        """簡易GPU圧縮"""
        # 変換による圧縮効果（デルタ符号化等）
        if len(data_array) > 1:
            delta = np.diff(data_array.astype(np.int16))
            return delta.astype(np.int8).tobytes()
        
        return data_array.tobytes()
    
    def _gpu_pattern_compress(self, gpu_data: Any) -> Any:
        """GPUパターン圧縮カーネル"""
        if cp is None:
            return gpu_data
        
        # 差分計算（GPU並列）
        diff = cp.diff(gpu_data.astype(cp.int16))
        
        # 閾値処理
        compressed = cp.where(cp.abs(diff) < 5, 0, diff)
        
        return compressed.astype(cp.int8)
    
    def _compress_with_patterns(self, data_array: np.ndarray, patterns: np.ndarray) -> np.ndarray:
        """パターンベース圧縮"""
        # パターン頻度分析
        unique_patterns, counts = np.unique(patterns, return_counts=True)
        
        # 頻出パターンを短い符号に置換
        result = np.copy(data_array)
        
        # 上位10パターンを圧縮
        top_patterns = unique_patterns[np.argsort(counts)[-10:]]
        
        for i, pattern in enumerate(top_patterns):
            mask = patterns == pattern
            if np.sum(mask) > 5:  # 十分な頻度
                # 簡易圧縮: パターンを一意値に置換
                result[mask] = 200 + i  # 特別な値域を使用
        
        return result
    
    def _cpu_fallback_compression(self, data: bytes) -> bytes:
        """CPUフォールバック圧縮"""
        import lzma
        return lzma.compress(data, preset=1)


class AdaptiveLoadBalancer:
    """適応的負荷分散器"""
    
    def __init__(self, system_resources: SystemResources):
        self.resources = system_resources
        self.worker_performance = {}  # ワーカー別パフォーマンス履歴
        self.current_loads = {}       # 現在の負荷
        self.balancing_history = []   # 分散履歴
        self.lock = threading.Lock()
        
        print(f"⚖️ 適応的負荷分散器初期化")
    
    def calculate_optimal_chunks(self, data_size: int, quality: str) -> Dict[str, Any]:
        """最適チャンク分割計算"""
        base_chunk_size = 4 * 1024 * 1024  # 4MB基準
        
        # 品質別調整
        quality_multipliers = {
            'fast': 0.5,
            'balanced': 1.0,
            'max': 2.0
        }
        
        chunk_size = int(base_chunk_size * quality_multipliers.get(quality, 1.0))
        
        # システムリソース別調整
        if self.resources.memory_gb < 4:
            chunk_size = min(chunk_size, 2 * 1024 * 1024)  # 2MB制限
        elif self.resources.memory_gb > 16:
            chunk_size = max(chunk_size, 8 * 1024 * 1024)  # 8MB最小
        
        chunk_count = max(1, (data_size + chunk_size - 1) // chunk_size)
        
        # 並列度決定
        optimal_threads = min(
            self.resources.cpu_count,
            chunk_count,
            8 if quality == 'fast' else 16
        )
        
        optimal_processes = min(
            self.resources.cpu_count // 2,
            chunk_count // 2,
            4
        )
        
        return {
            'chunk_size': chunk_size,
            'chunk_count': chunk_count,
            'optimal_threads': optimal_threads,
            'optimal_processes': optimal_processes,
            'memory_per_worker': max(512, self.resources.memory_gb * 1024 // optimal_threads),
            'use_gpu': self.resources.gpu_available and data_size > 10 * 1024 * 1024
        }
    
    def distribute_chunks(self, chunks: List[ProcessingChunk], workers: int) -> List[List[ProcessingChunk]]:
        """チャンク分散"""
        if not chunks:
            return []
        
        # ワーカー数調整
        actual_workers = min(workers, len(chunks))
        
        # 複雑度ベース分散
        chunks_sorted = sorted(chunks, key=lambda c: c.complexity_score, reverse=True)
        
        # Round-robin with complexity balancing
        worker_assignments = [[] for _ in range(actual_workers)]
        worker_loads = [0.0] * actual_workers
        
        for chunk in chunks_sorted:
            # 最も負荷の少ないワーカーを選択
            min_load_worker = min(range(actual_workers), key=lambda i: worker_loads[i])
            
            worker_assignments[min_load_worker].append(chunk)
            worker_loads[min_load_worker] += chunk.complexity_score
        
        return worker_assignments
    
    def update_worker_performance(self, worker_id: str, processing_time: float, data_size: int):
        """ワーカーパフォーマンス更新"""
        with self.lock:
            if worker_id not in self.worker_performance:
                self.worker_performance[worker_id] = {
                    'total_time': 0.0,
                    'total_data': 0,
                    'task_count': 0,
                    'avg_throughput': 0.0
                }
            
            perf = self.worker_performance[worker_id]
            perf['total_time'] += processing_time
            perf['total_data'] += data_size
            perf['task_count'] += 1
            
            # 移動平均スループット
            current_throughput = data_size / max(processing_time, 0.001)
            alpha = 0.3
            perf['avg_throughput'] = (
                alpha * current_throughput + 
                (1 - alpha) * perf['avg_throughput']
            )
    
    def get_load_balancing_report(self) -> Dict[str, Any]:
        """負荷分散レポート"""
        with self.lock:
            total_throughput = sum(
                perf['avg_throughput'] 
                for perf in self.worker_performance.values()
            )
            
            worker_efficiency = {}
            for worker_id, perf in self.worker_performance.items():
                if total_throughput > 0:
                    efficiency = perf['avg_throughput'] / total_throughput * 100
                else:
                    efficiency = 0.0
                
                worker_efficiency[worker_id] = {
                    'efficiency_percent': efficiency,
                    'total_tasks': perf['task_count'],
                    'avg_throughput_mb_s': perf['avg_throughput'] / (1024 * 1024)
                }
            
            return {
                'total_workers': len(self.worker_performance),
                'total_throughput_mb_s': total_throughput / (1024 * 1024),
                'worker_efficiency': worker_efficiency,
                'balancing_history_count': len(self.balancing_history)
            }


class NEXUSParallelEngine:
    """
    NEXUS並列処理エンジン - 次世代並列圧縮システム
    
    機能:
    1. インテリジェント負荷分散
    2. 適応的並列戦略
    3. GPU/CPU ハイブリッド処理
    4. 高度メモリ管理
    5. リアルタイム性能監視
    6. 分散処理サポート
    """
    
    def __init__(self, config: Optional[ParallelConfig] = None):
        """初期化"""
        self.config = config or ParallelConfig()
        
        # システムリソース分析
        self.system_resources = self._analyze_system_resources()
        
        # 高度コンポーネント初期化
        self.memory_manager = AdvancedMemoryManager(
            int(self.config.memory_limit_gb * 1024)
        )
        self.gpu_accelerator = IntelligentGPUAccelerator()
        self.load_balancer = AdaptiveLoadBalancer(self.system_resources)
        
        # 並列処理プール
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.max_threads,
            thread_name_prefix="NEXUS-Thread"
        )
        self.process_pool = ProcessPoolExecutor(
            max_workers=self.config.max_processes
        )
        
        # パフォーマンスプロファイル
        self.performance_profile = PerformanceProfile()
        
        # ベースエンジン（利用可能な場合）
        if NEXUS_ENGINES_AVAILABLE:
            try:
                self.base_engine = NEXUSTheoryCore()
                self.optimizer = NEXUSAdvancedOptimizer()
                print(f"✅ NEXUS理論エンジン統合完了")
            except Exception as e:
                print(f"⚠️ NEXUS理論エンジン統合失敗: {e}")
                self.base_engine = None
                self.optimizer = None
        else:
            self.base_engine = None
            self.optimizer = None
        
        # 統計
        self.processing_stats = {
            'total_chunks_processed': 0,
            'total_data_processed': 0,
            'average_chunk_time': 0.0,
            'peak_throughput': 0.0,
            'energy_efficiency': 0.0
        }
        
        print(f"🚀 NEXUS並列処理エンジン初期化完了")
        print(f"   💻 CPU: {self.system_resources.cpu_count} コア")
        print(f"   🧠 メモリ: {self.system_resources.memory_gb:.1f}GB")
        print(f"   🚀 GPU: {'有効' if self.gpu_accelerator.available else '無効'}")
        print(f"   🧵 最大スレッド: {self.config.max_threads}")
        print(f"   ⚡ 最大プロセス: {self.config.max_processes}")
    
    def _analyze_system_resources(self) -> SystemResources:
        """システムリソース分析"""
        # CPU情報
        cpu_count = multiprocessing.cpu_count()
        
        # メモリ情報
        memory_info = psutil.virtual_memory()
        memory_gb = memory_info.total / (1024**3)
        
        # GPU情報
        gpu_available = GPU_AVAILABLE
        gpu_memory_gb = 0.0
        
        if gpu_available and cp is not None:
            try:
                free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                gpu_memory_gb = total_mem / (1024**3)
            except:
                gpu_available = False
        
        # 負荷平均
        try:
            load_average = psutil.getloadavg()[0]
        except:
            load_average = psutil.cpu_percent() / 100.0
        
        # ディスクI/O速度（簡易測定）
        disk_io_speed = self._measure_disk_speed()
        
        return SystemResources(
            cpu_count=cpu_count,
            memory_gb=memory_gb,
            gpu_available=gpu_available,
            gpu_memory_gb=gpu_memory_gb,
            load_average=load_average,
            disk_io_speed=disk_io_speed
        )
    
    def _measure_disk_speed(self) -> float:
        """ディスクI/O速度測定"""
        try:
            test_data = b"0" * (1024 * 1024)  # 1MB
            
            import tempfile
            with tempfile.NamedTemporaryFile() as tmp:
                start_time = time.perf_counter()
                tmp.write(test_data)
                tmp.flush()
                os.fsync(tmp.fileno())
                write_time = time.perf_counter() - start_time
                
                tmp.seek(0)
                start_time = time.perf_counter()
                _ = tmp.read()
                read_time = time.perf_counter() - start_time
            
            # 書き込み速度をベースに計算
            speed = len(test_data) / (1024 * 1024) / max(write_time, 0.001)
            return speed
            
        except:
            return 100.0  # デフォルト値
    
    def parallel_compress(self, data: bytes, quality: str = 'balanced') -> bytes:
        """
        メイン並列圧縮関数
        
        Args:
            data: 圧縮対象データ
            quality: 圧縮品質 ('fast', 'balanced', 'max')
            
        Returns:
            圧縮済みデータ
        """
        print(f"🔄 NEXUS並列圧縮開始")
        print(f"   📊 データサイズ: {len(data) / 1024 / 1024:.2f}MB")
        print(f"   🎯 品質モード: {quality}")
        
        compression_start = time.perf_counter()
        
        try:
            # 1. 動的設定最適化
            self._optimize_config_for_quality(quality)
            
            # 2. 最適チャンク分割計算
            chunk_config = self.load_balancer.calculate_optimal_chunks(
                len(data), quality
            )
            print(f"   🔷 最適チャンク: {chunk_config['chunk_count']} 個")
            
            # 3. インテリジェントデータ分割
            chunks = self._intelligent_data_chunking(data, chunk_config)
            
            # 4. 並列処理戦略決定
            strategy = self._determine_processing_strategy(chunks, quality)
            print(f"   📋 処理戦略: {strategy}")
            
            # 5. 並列圧縮実行
            if strategy == 'gpu_hybrid':
                results = self._gpu_hybrid_compression(chunks, quality)
            elif strategy == 'multiprocess_advanced':
                results = self._advanced_multiprocess_compression(chunks, quality)
            elif strategy == 'multithread_optimized':
                results = self._optimized_multithread_compression(chunks, quality)
            elif strategy == 'distributed':
                results = self._distributed_compression(chunks, quality)
            else:
                results = self._sequential_compression(chunks, quality)
            
            # 6. 高度結果統合
            compressed_data = self._advanced_merge_results(results, data, quality)
            
            # 7. パフォーマンス分析
            total_time = time.perf_counter() - compression_start
            self._update_performance_profile(data, compressed_data, total_time, strategy)
            
            compression_ratio = (1 - len(compressed_data) / len(data)) * 100
            throughput = len(data) / 1024 / 1024 / total_time
            
            print(f"✅ 並列圧縮完了!")
            print(f"   📈 圧縮率: {compression_ratio:.2f}%")
            print(f"   ⚡ スループット: {throughput:.1f}MB/s")
            print(f"   ⏱️ 処理時間: {total_time:.3f}秒")
            
            return compressed_data
            
        except Exception as e:
            print(f"❌ 並列圧縮エラー: {str(e)}")
            # 安全なフォールバック
            return self._fallback_compression(data)
    
    def _optimize_config_for_quality(self, quality: str):
        """品質別設定最適化"""
        quality_configs = {
            'fast': self.config.fast_settings,
            'balanced': self.config.balanced_settings,
            'max': self.config.max_settings
        }
        
        if quality in quality_configs:
            optimization = quality_configs[quality]
            
            # 動的設定適用
            for key, value in optimization.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
    
    def _intelligent_data_chunking(self, data: bytes, config: Dict[str, Any]) -> List[ProcessingChunk]:
        """インテリジェントデータ分割"""
        chunks = []
        chunk_size = config['chunk_size']
        
        # データ特性分析
        entropy_profile = self._analyze_data_characteristics(data)
        
        current_pos = 0
        chunk_id = 0
        
        while current_pos < len(data):
            # 適応的チャンクサイズ決定
            local_entropy = entropy_profile.get('local_entropy', [5.0])[
                min(chunk_id, len(entropy_profile.get('local_entropy', [])) - 1)
            ]
            
            # エントロピーベース調整
            if local_entropy > 7.0:  # 高エントロピー
                adaptive_size = int(chunk_size * 0.7)
            elif local_entropy < 3.0:  # 低エントロピー
                adaptive_size = int(chunk_size * 1.3)
            else:
                adaptive_size = chunk_size
            
            end_pos = min(current_pos + adaptive_size, len(data))
            
            # スマート境界調整
            end_pos = self._smart_boundary_adjustment(data, current_pos, end_pos)
            
            chunk_data = data[current_pos:end_pos]
            
            # チャンク複雑度計算
            complexity = self._calculate_chunk_complexity(chunk_data, local_entropy)
            
            chunk = ProcessingChunk(
                chunk_id=chunk_id,
                data=chunk_data,
                start_offset=current_pos,
                end_offset=end_pos,
                complexity_score=complexity,
                entropy=local_entropy,
                processing_method="auto",
                metadata={
                    'size_kb': len(chunk_data) // 1024,
                    'entropy': local_entropy,
                    'complexity': complexity
                }
            )
            
            chunks.append(chunk)
            current_pos = end_pos
            chunk_id += 1
        
        return chunks
    
    def _analyze_data_characteristics(self, data: bytes) -> Dict[str, Any]:
        """データ特性分析"""
        sample_size = min(len(data), 64 * 1024)  # 64KB サンプル
        sample = data[:sample_size]
        
        # 基本統計
        byte_array = np.frombuffer(sample, dtype=np.uint8)
        
        characteristics = {
            'size': len(data),
            'entropy': self._calculate_entropy(sample),
            'compression_estimate': self._estimate_compression_ratio(sample),
            'pattern_density': self._analyze_pattern_density(sample),
            'local_entropy': self._calculate_local_entropy_profile(data)
        }
        
        return characteristics
    
    def _calculate_entropy(self, data: bytes) -> float:
        """エントロピー計算"""
        if not data:
            return 0.0
        
        byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        probabilities = byte_counts / len(data)
        
        # エントロピー計算（0による除算回避）
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _estimate_compression_ratio(self, sample: bytes) -> float:
        """圧縮率推定"""
        try:
            import lzma
            compressed = lzma.compress(sample, preset=1)
            return len(compressed) / len(sample)
        except:
            return 0.8  # デフォルト推定値
    
    def _analyze_pattern_density(self, data: bytes) -> float:
        """パターン密度分析"""
        if len(data) < 4:
            return 0.0
        
        # 4バイトパターンの重複率
        patterns = {}
        for i in range(len(data) - 3):
            pattern = data[i:i+4]
            patterns[pattern] = patterns.get(pattern, 0) + 1
        
        total_patterns = len(data) - 3
        unique_patterns = len(patterns)
        
        if total_patterns == 0:
            return 0.0
        
        return 1 - (unique_patterns / total_patterns)
    
    def _calculate_local_entropy_profile(self, data: bytes) -> List[float]:
        """局所エントロピープロファイル"""
        window_size = 4096  # 4KB ウィンドウ
        profile = []
        
        for i in range(0, len(data), window_size):
            window = data[i:i + window_size]
            if window:
                entropy = self._calculate_entropy(window)
                profile.append(entropy)
        
        return profile
    
    def _calculate_chunk_complexity(self, chunk_data: bytes, entropy: float) -> float:
        """チャンク複雑度計算"""
        # サイズファクター
        size_factor = min(1.0, len(chunk_data) / (4 * 1024 * 1024))
        
        # エントロピーファクター
        entropy_factor = entropy / 8.0
        
        # 組み合わせ複雑度
        complexity = (size_factor * 0.3 + entropy_factor * 0.7) * 100
        
        return complexity
    
    def _smart_boundary_adjustment(self, data: bytes, start: int, end: int) -> int:
        """スマート境界調整"""
        if end >= len(data):
            return len(data)
        
        # パターン境界検出範囲
        search_range = min(256, len(data) - end)
        
        # 反復パターン終了検出
        for i in range(search_range):
            pos = end + i
            if pos < len(data) - 4:
                # 4バイト境界での切断最適化
                if (data[pos:pos+2] == data[start:start+2] and 
                    data[pos+2:pos+4] != data[start+2:start+4]):
                    return pos + 2
        
        return end
    
    def _determine_processing_strategy(self, chunks: List[ProcessingChunk], quality: str) -> str:
        """処理戦略決定"""
        total_size = sum(len(chunk.data) for chunk in chunks)
        chunk_count = len(chunks)
        avg_complexity = sum(chunk.complexity_score for chunk in chunks) / max(chunk_count, 1)
        
        # GPU ハイブリッド戦略
        if (self.gpu_accelerator.available and 
            total_size > 20 * 1024 * 1024 and  # 20MB以上
            avg_complexity > 50 and
            quality in ['balanced', 'max']):
            return 'gpu_hybrid'
        
        # 高度マルチプロセス戦略
        elif (chunk_count >= 4 and 
              total_size > 50 * 1024 * 1024 and  # 50MB以上
              self.system_resources.memory_gb > 8 and
              quality == 'max'):
            return 'multiprocess_advanced'
        
        # 最適化マルチスレッド戦略
        elif (chunk_count >= 2 and 
              total_size > 5 * 1024 * 1024):  # 5MB以上
            return 'multithread_optimized'
        
        # 分散処理戦略
        elif (self.config.use_distributed and
              total_size > 100 * 1024 * 1024 and  # 100MB以上
              chunk_count >= 8):
            return 'distributed'
        
        # シーケンシャル戦略
        else:
            return 'sequential'
    
    def _fallback_compression(self, data: bytes) -> bytes:
        """フォールバック圧縮"""
        try:
            import lzma
            return lzma.compress(data, preset=1)
        except:
            return data  # 最後の手段：非圧縮
    
    def _update_performance_profile(self, original_data: bytes, compressed_data: bytes, 
                                  processing_time: float, strategy: str):
        """パフォーマンスプロファイル更新"""
        data_size_mb = len(original_data) / 1024 / 1024
        
        # 基本メトリクス
        self.performance_profile.throughput_mb_s = data_size_mb / processing_time
        self.performance_profile.compression_ratio = len(compressed_data) / len(original_data)
        
        # システムメトリクス
        self.performance_profile.cpu_utilization = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        self.performance_profile.memory_efficiency = (1 - memory_info.percent / 100) * 100
        
        # 統計更新
        self.processing_stats['total_chunks_processed'] += 1
        self.processing_stats['total_data_processed'] += len(original_data)
        
        # 移動平均更新
        alpha = 0.2
        self.processing_stats['average_chunk_time'] = (
            alpha * processing_time + 
            (1 - alpha) * self.processing_stats.get('average_chunk_time', processing_time)
        )
        
        # ピークスループット更新
        self.processing_stats['peak_throughput'] = max(
            self.processing_stats.get('peak_throughput', 0),
            self.performance_profile.throughput_mb_s
        )
    
    # 簡易実装版の並列処理メソッド（プレースホルダー）
    def _gpu_hybrid_compression(self, chunks: List[ProcessingChunk], quality: str) -> List[ProcessingResult]:
        print(f"🚀 GPU ハイブリッド圧縮実行")
        return self._optimized_multithread_compression(chunks, quality)
    
    def _advanced_multiprocess_compression(self, chunks: List[ProcessingChunk], quality: str) -> List[ProcessingResult]:
        print(f"⚡ 高度マルチプロセス圧縮実行")
        return self._optimized_multithread_compression(chunks, quality)
    
    def _optimized_multithread_compression(self, chunks: List[ProcessingChunk], quality: str) -> List[ProcessingResult]:
        print(f"🧵 最適化マルチスレッド圧縮実行")
        
        results = []
        futures = []
        
        # 負荷分散チャンク配布
        worker_assignments = self.load_balancer.distribute_chunks(
            chunks, self.config.max_threads
        )
        
        with self.thread_pool as executor:
            for worker_id, worker_chunks in enumerate(worker_assignments):
                if worker_chunks:  # 空でない場合のみ
                    future = executor.submit(
                        self._process_worker_chunks, 
                        worker_chunks, 
                        f"worker_{worker_id}",
                        quality
                    )
                    futures.append(future)
            
            # 結果収集
            for future in as_completed(futures, timeout=600):
                try:
                    worker_results = future.result()
                    results.extend(worker_results)
                except Exception as e:
                    print(f"⚠️ ワーカーエラー: {e}")
        
        return results
    
    def _distributed_compression(self, chunks: List[ProcessingChunk], quality: str) -> List[ProcessingResult]:
        print(f"🌐 分散処理圧縮実行（フォールバック）")
        return self._optimized_multithread_compression(chunks, quality)
    
    def _sequential_compression(self, chunks: List[ProcessingChunk], quality: str) -> List[ProcessingResult]:
        print(f"📝 シーケンシャル圧縮実行")
        
        results = []
        for chunk in chunks:
            result = self._compress_single_chunk(chunk, "sequential", quality)
            results.append(result)
        
        return results
    
    def _process_worker_chunks(self, chunks: List[ProcessingChunk], worker_id: str, quality: str) -> List[ProcessingResult]:
        """ワーカーチャンク処理"""
        results = []
        
        for chunk in chunks:
            start_time = time.perf_counter()
            result = self._compress_single_chunk(chunk, worker_id, quality)
            processing_time = time.perf_counter() - start_time
            
            # パフォーマンス追跡
            self.load_balancer.update_worker_performance(
                worker_id, processing_time, len(chunk.data)
            )
            
            results.append(result)
        
        return results
    
    def _compress_single_chunk(self, chunk: ProcessingChunk, method: str, quality: str) -> ProcessingResult:
        """単一チャンク圧縮"""
        start_time = time.perf_counter()
        
        try:
            # メモリ確保
            memory_name = f"chunk_{chunk.chunk_id}_{method}"
            memory_needed = len(chunk.data) * 2  # 安全マージン
            
            if not self.memory_manager.allocate(memory_name, memory_needed):
                raise MemoryError(f"メモリ不足: {memory_needed // 1024 // 1024}MB 必要")
            
            # 圧縮実行
            if self.base_engine and quality == 'max':
                # NEXUS理論エンジン使用
                compressed_data = self.base_engine.compress(chunk.data)
            elif self.gpu_accelerator.available and len(chunk.data) > 1024*1024:
                # GPU加速圧縮
                compressed_data = self.gpu_accelerator.gpu_accelerated_compression(
                    chunk.data, "hybrid"
                )
            else:
                # 標準圧縮
                compressed_data = self._standard_compression(chunk.data, quality)
            
            processing_time = time.perf_counter() - start_time
            compression_ratio = len(compressed_data) / len(chunk.data)
            
            # メモリ解放
            self.memory_manager.deallocate(memory_name)
            
            return ProcessingResult(
                chunk_id=chunk.chunk_id,
                compressed_data=compressed_data,
                compression_ratio=compression_ratio,
                processing_time=processing_time,
                method_used=method,
                quality_score=self._calculate_quality_score(
                    compression_ratio, processing_time
                ),
                metrics={
                    'original_size': len(chunk.data),
                    'compressed_size': len(compressed_data),
                    'complexity': chunk.complexity_score,
                    'entropy': chunk.entropy
                }
            )
            
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            
            return ProcessingResult(
                chunk_id=chunk.chunk_id,
                compressed_data=chunk.data,  # 非圧縮
                compression_ratio=1.0,
                processing_time=processing_time,
                method_used=method,
                error=str(e)
            )
    
    def _standard_compression(self, data: bytes, quality: str) -> bytes:
        """標準圧縮"""
        try:
            import lzma
            
            # 品質別プリセット
            presets = {
                'fast': 0,
                'balanced': 3,
                'max': 6
            }
            
            preset = presets.get(quality, 3)
            return lzma.compress(data, preset=preset)
            
        except Exception:
            return data
    
    def _calculate_quality_score(self, compression_ratio: float, processing_time: float) -> float:
        """品質スコア計算"""
        # 圧縮率と処理時間のバランススコア
        compression_score = (1 - compression_ratio) * 100  # 圧縮率（高いほど良い）
        speed_score = min(100, 10 / max(processing_time, 0.001))  # 速度（速いほど良い）
        
        # 重み付け平均
        quality_score = compression_score * 0.7 + speed_score * 0.3
        
        return min(100, max(0, quality_score))
    
    def _advanced_merge_results(self, results: List[ProcessingResult], 
                              original_data: bytes, quality: str) -> bytes:
        """高度結果統合"""
        # チャンクIDでソート
        results.sort(key=lambda r: r.chunk_id)
        
        # ヘッダー作成
        header = self._create_advanced_header(results, original_data, quality)
        
        # データ結合
        merged_data = header
        for result in results:
            chunk_header = self._create_enhanced_chunk_header(result)
            merged_data += chunk_header + result.compressed_data
        
        return merged_data
    
    def _create_advanced_header(self, results: List[ProcessingResult], 
                              original_data: bytes, quality: str) -> bytes:
        """高度ヘッダー作成"""
        import struct
        
        header = bytearray(128)  # 拡張ヘッダー
        
        # マジックナンバー
        header[0:8] = b'NXPAR002'  # NEXUS Parallel v2
        
        # 基本情報
        header[8:16] = struct.pack('<Q', len(original_data))
        header[16:20] = struct.pack('<I', len(results))
        
        # 品質設定
        quality_code = {'fast': 1, 'balanced': 2, 'max': 3}.get(quality, 2)
        header[20:24] = struct.pack('<I', quality_code)
        
        # パフォーマンス統計
        total_time = sum(r.processing_time for r in results)
        avg_compression_ratio = sum(r.compression_ratio for r in results) / len(results)
        avg_quality_score = sum(r.quality_score for r in results) / len(results)
        
        header[24:32] = struct.pack('<d', total_time)
        header[32:40] = struct.pack('<d', avg_compression_ratio)
        header[40:48] = struct.pack('<d', avg_quality_score)
        
        # システム情報
        header[48:52] = struct.pack('<I', self.system_resources.cpu_count)
        header[52:56] = struct.pack('<f', self.system_resources.memory_gb)
        header[56:60] = struct.pack('<I', 1 if self.gpu_accelerator.available else 0)
        
        # 圧縮統計
        total_original = sum(r.metrics.get('original_size', 0) for r in results)
        total_compressed = sum(r.metrics.get('compressed_size', 0) for r in results)
        
        header[60:68] = struct.pack('<Q', total_original)
        header[68:76] = struct.pack('<Q', total_compressed)
        
        # チェックサム
        import hashlib
        checksum = hashlib.blake2b(header[8:76], digest_size=16).digest()
        header[76:92] = checksum
        
        return bytes(header)
    
    def _create_enhanced_chunk_header(self, result: ProcessingResult) -> bytes:
        """拡張チャンクヘッダー作成"""
        import struct
        
        header = bytearray(64)  # 拡張チャンクヘッダー
        
        # 基本情報
        header[0:4] = struct.pack('<I', result.chunk_id)
        header[4:8] = struct.pack('<I', len(result.compressed_data))
        header[8:16] = struct.pack('<d', result.compression_ratio)
        header[16:24] = struct.pack('<d', result.processing_time)
        header[24:32] = struct.pack('<d', result.quality_score)
        
        # メソッド情報
        method_bytes = result.method_used.encode('ascii')[:16]
        header[32:32+len(method_bytes)] = method_bytes
        
        # エラー情報
        if result.error:
            header[48:52] = struct.pack('<I', 1)  # エラーフラグ
        
        return bytes(header)
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """包括的レポート取得"""
        return {
            'system_resources': {
                'cpu_count': self.system_resources.cpu_count,
                'memory_gb': self.system_resources.memory_gb,
                'gpu_available': self.system_resources.gpu_available,
                'gpu_memory_gb': self.system_resources.gpu_memory_gb,
                'load_average': self.system_resources.load_average,
                'disk_io_speed': self.system_resources.disk_io_speed
            },
            'performance_profile': {
                'throughput_mb_s': self.performance_profile.throughput_mb_s,
                'compression_ratio': self.performance_profile.compression_ratio,
                'cpu_utilization': self.performance_profile.cpu_utilization,
                'gpu_utilization': self.performance_profile.gpu_utilization,
                'memory_efficiency': self.performance_profile.memory_efficiency
            },
            'processing_stats': self.processing_stats,
            'memory_usage': self.memory_manager.get_advanced_usage(),
            'load_balancing': self.load_balancer.get_load_balancing_report(),
            'gpu_performance': self.gpu_accelerator.benchmark_gpu_performance() if self.gpu_accelerator.available else {},
            'engine_integration': {
                'nexus_theory_available': NEXUS_ENGINES_AVAILABLE,
                'base_engine_active': self.base_engine is not None,
                'optimizer_active': self.optimizer is not None
            }
        }
    
    def __del__(self):
        """デストラクタ"""
        try:
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=False)
            if hasattr(self, 'process_pool'):
                self.process_pool.shutdown(wait=False)
        except:
            pass
    
    def __init__(self, config: Optional[ParallelConfig] = None):
        self.config = config or ParallelConfig()
        
        # システムリソース分析
        self.system_resources = self._analyze_system_resources()
        
        # 高度コンポーネント初期化
        self.memory_manager = AdvancedMemoryManager(
            int(self.config.memory_limit_gb * 1024)
        )
        self.gpu_accelerator = IntelligentGPUAccelerator()
        self.load_balancer = AdaptiveLoadBalancer(self.system_resources)
        
        # 並列処理プール
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.max_threads,
            thread_name_prefix="NEXUS-Thread"
        )
        self.process_pool = ProcessPoolExecutor(
            max_workers=self.config.max_processes
        )
        
        # パフォーマンスプロファイル
        self.performance_profile = PerformanceProfile()
        
        # ベースエンジン（利用可能な場合）
        if NEXUS_ENGINES_AVAILABLE:
            try:
                self.base_engine = NEXUSTheoryCore()
                self.optimizer = NEXUSAdvancedOptimizer()
                print(f"✅ NEXUS理論エンジン統合完了")
            except Exception as e:
                print(f"⚠️ NEXUS理論エンジン統合失敗: {e}")
                self.base_engine = None
                self.optimizer = None
        else:
            self.base_engine = None
            self.optimizer = None
        
        # 統計
        self.processing_stats = {
            'total_chunks_processed': 0,
            'total_data_processed': 0,
            'average_chunk_time': 0.0,
            'peak_throughput': 0.0,
            'energy_efficiency': 0.0
        }
        
        print(f"🚀 NEXUS並列処理エンジン初期化完了")
        print(f"   💻 CPU: {self.system_resources.cpu_count} コア")
        print(f"   🧠 メモリ: {self.system_resources.memory_gb:.1f}GB")
        print(f"   🚀 GPU: {'有効' if self.gpu_accelerator.available else '無効'}")
        print(f"   🧵 最大スレッド: {self.config.max_threads}")
        print(f"   ⚡ 最大プロセス: {self.config.max_processes}")
    
    def parallel_compress(self, data: bytes, quality: str = 'balanced') -> bytes:
        """並列圧縮"""
        print(f"🔧 NEXUS並列圧縮開始 - サイズ: {len(data)//1024//1024:.1f}MB")
        
        start_time = time.perf_counter()
        
        # 1. データ分析とチャンク分割
        chunks = self._split_data_intelligently(data)
        print(f"📊 チャンク分割: {len(chunks)} チャンク")
        
        # 2. 並列処理戦略決定
        strategy = self._determine_parallel_strategy(chunks, quality)
        print(f"🎯 並列戦略: {strategy}")
        
        # 3. 並列圧縮実行
        if strategy == 'gpu_accelerated':
            results = self._gpu_parallel_compress(chunks, quality)
        elif strategy == 'multiprocess':
            results = self._multiprocess_compress(chunks, quality)
        elif strategy == 'multithread':
            results = self._multithread_compress(chunks, quality)
        else:
            results = self._sequential_compress(chunks, quality)
        
        # 4. 結果統合
        final_result = self._merge_compression_results(results, data)
        
        total_time = time.perf_counter() - start_time
        compression_ratio = (1 - len(final_result) / len(data)) * 100
        
        print(f"✅ 並列圧縮完了: {compression_ratio:.2f}% ({total_time:.2f}s)")
        self._update_performance_stats(len(chunks), total_time)
        
        return final_result
    
    def parallel_decompress(self, compressed_data: bytes) -> bytes:
        """並列展開"""
        print(f"🔓 NEXUS並列展開開始")
        
        start_time = time.perf_counter()
        
        # 1. 圧縮データ解析
        chunk_info = self._analyze_compressed_data(compressed_data)
        
        # 2. 並列展開戦略決定
        strategy = self._determine_decompression_strategy(chunk_info)
        
        # 3. 並列展開実行
        if strategy == 'multiprocess':
            results = self._multiprocess_decompress(chunk_info)
        elif strategy == 'multithread':
            results = self._multithread_decompress(chunk_info)
        else:
            results = self._sequential_decompress(chunk_info)
        
        # 4. 結果統合
        final_result = self._merge_decompression_results(results)
        
        total_time = time.perf_counter() - start_time
        print(f"✅ 並列展開完了: {len(final_result)//1024//1024:.1f}MB ({total_time:.2f}s)")
        
        return final_result
    
    def _split_data_intelligently(self, data: bytes) -> List[ProcessingChunk]:
        """インテリジェントデータ分割"""
        chunks = []
        chunk_size = self.config.chunk_size_mb * 1024 * 1024
        
        # データ特徴分析
        entropy_profile = self._analyze_entropy_profile(data)
        
        # 適応的分割
        current_pos = 0
        chunk_id = 0
        
        while current_pos < len(data):
            # エントロピーに基づく動的チャンクサイズ決定
            local_entropy = entropy_profile[min(chunk_id, len(entropy_profile) - 1)]
            
            if local_entropy > 7.0:  # 高エントロピー
                adaptive_size = int(chunk_size * 0.8)  # 小さなチャンク
            elif local_entropy < 3.0:  # 低エントロピー
                adaptive_size = int(chunk_size * 1.5)  # 大きなチャンク
            else:
                adaptive_size = chunk_size
            
            end_pos = min(current_pos + adaptive_size, len(data))
            
            # 境界調整（パターン境界を考慮）
            end_pos = self._adjust_chunk_boundary(data, current_pos, end_pos)
            
            chunk_data = data[current_pos:end_pos]
            
            chunk = ProcessingChunk(
                chunk_id=chunk_id,
                data=chunk_data,
                start_offset=current_pos,
                end_offset=end_pos,
                metadata={'entropy': local_entropy}
            )
            chunks.append(chunk)
            
            current_pos = end_pos
            chunk_id += 1
        
        return chunks
    
    def _analyze_entropy_profile(self, data: bytes, window_size: int = 1024) -> List[float]:
        """エントロピープロファイル分析"""
        profile = []
        
        for i in range(0, len(data), window_size):
            window = data[i:i + window_size]
            if window:
                entropy = self._calculate_local_entropy(window)
                profile.append(entropy)
        
        return profile
    
    def _calculate_local_entropy(self, window: bytes) -> float:
        """局所エントロピー計算"""
        if not window:
            return 0.0
        
        byte_counts = [0] * 256
        for byte in window:
            byte_counts[byte] += 1
        
        entropy = 0.0
        total = len(window)
        
        for count in byte_counts:
            if count > 0:
                prob = count / total
                entropy -= prob * np.log2(prob)
        
        return entropy
    
    def _adjust_chunk_boundary(self, data: bytes, start: int, end: int) -> int:
        """チャンク境界調整"""
        if end >= len(data):
            return len(data)
        
        # パターン境界検出（簡易版）
        search_range = min(100, len(data) - end)
        
        for i in range(search_range):
            # 反復パターンの終了を検出
            if end + i < len(data) - 1:
                if data[end + i] == data[start] and data[end + i + 1] != data[start + 1]:
                    return end + i + 1
        
        return end
    
    def _determine_parallel_strategy(self, chunks: List[ProcessingChunk], quality: str) -> str:
        """並列戦略決定"""
        total_size = sum(len(chunk.data) for chunk in chunks)
        chunk_count = len(chunks)
        
        # GPU戦略
        if (self.gpu_accelerator.available and 
            total_size > 50 * 1024 * 1024 and  # 50MB以上
            quality in ['balanced', 'max']):
            return 'gpu_accelerated'
        
        # マルチプロセス戦略
        if (chunk_count >= 4 and 
            total_size > 10 * 1024 * 1024 and  # 10MB以上
            quality != 'fast'):
            return 'multiprocess'
        
        # マルチスレッド戦略
        if chunk_count >= 2:
            return 'multithread'
        
        # シーケンシャル戦略
        return 'sequential'
    
    def _gpu_parallel_compress(self, chunks: List[ProcessingChunk], quality: str) -> List[ProcessingResult]:
        """GPU並列圧縮"""
        print(f"🔥 GPU並列圧縮実行")
        
        results = []
        
        # GPU用データ準備
        gpu_tasks = self._prepare_gpu_tasks(chunks)
        
        # バッチ処理
        batch_size = min(4, len(chunks))
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_results = self._process_gpu_batch(batch_chunks, quality)
            results.extend(batch_results)
        
        return results
    
    def _prepare_gpu_tasks(self, chunks: List[ProcessingChunk]) -> List[Dict]:
        """GPU タスク準備"""
        tasks = []
        
        for chunk in chunks:
            # データをnumpy配列に変換
            data_array = np.frombuffer(chunk.data, dtype=np.uint8)
            
            task = {
                'chunk_id': chunk.chunk_id,
                'data_array': data_array,
                'metadata': chunk.metadata
            }
            tasks.append(task)
        
        return tasks
    
    def _process_gpu_batch(self, chunks: List[ProcessingChunk], quality: str) -> List[ProcessingResult]:
        """GPUバッチ処理"""
        results = []
        
        for chunk in chunks:
            start_time = time.perf_counter()
            
            try:
                # GPU加速圧縮
                data_array = np.frombuffer(chunk.data, dtype=np.uint8)
                
                # パターン検出（GPU加速）
                patterns = self._gpu_detect_patterns(data_array)
                
                # 圧縮実行
                compressed = self._compress_with_gpu_patterns(chunk.data, patterns)
                
                processing_time = time.perf_counter() - start_time
                compression_ratio = len(compressed) / len(chunk.data)
                
                result = ProcessingResult(
                    chunk_id=chunk.chunk_id,
                    compressed_data=compressed,
                    compression_ratio=compression_ratio,
                    processing_time=processing_time
                )
                results.append(result)
                
            except Exception as e:
                # フォールバック
                result = ProcessingResult(
                    chunk_id=chunk.chunk_id,
                    compressed_data=chunk.data,
                    compression_ratio=1.0,
                    processing_time=0.0,
                    error=str(e)
                )
                results.append(result)
        
        return results
    
    def _gpu_detect_patterns(self, data_array: np.ndarray) -> List[np.ndarray]:
        """GPU パターン検出"""
        if not self.gpu_accelerator.available:
            return []
        
        # 簡易パターン検出
        patterns = []
        
        # 長さ別パターン検出
        for pattern_len in [2, 4, 8, 16]:
            if len(data_array) >= pattern_len * 2:
                detected = self._detect_patterns_of_length(data_array, pattern_len)
                patterns.extend(detected)
        
        return patterns
    
    def _detect_patterns_of_length(self, data_array: np.ndarray, length: int) -> List[np.ndarray]:
        """指定長パターン検出"""
        patterns = []
        pattern_counts = {}
        
        for i in range(len(data_array) - length + 1):
            pattern = data_array[i:i + length]
            pattern_key = pattern.tobytes()
            
            pattern_counts[pattern_key] = pattern_counts.get(pattern_key, 0) + 1
        
        # 頻出パターンを選択
        for pattern_bytes, count in pattern_counts.items():
            if count >= 3:  # 最低3回出現
                pattern = np.frombuffer(pattern_bytes, dtype=np.uint8)
                patterns.append(pattern)
        
        return patterns
    
    def _compress_with_gpu_patterns(self, data: bytes, patterns: List[np.ndarray]) -> bytes:
        """GPUパターンによる圧縮"""
        # パターン情報を活用した最適化圧縮
        # 実装簡略化のため、ベースエンジンを使用
        return self.base_engine.compress(data)
    
    def _multiprocess_compress(self, chunks: List[ProcessingChunk], quality: str) -> List[ProcessingResult]:
        """マルチプロセス圧縮"""
        print(f"🔄 マルチプロセス圧縮実行")
        
        futures = []
        results = []
        
        with self.process_pool as executor:
            # チャンクを並列処理に投入
            for chunk in chunks:
                future = executor.submit(
                    _compress_chunk_worker, 
                    chunk.data, 
                    chunk.chunk_id, 
                    quality
                )
                futures.append(future)
            
            # 結果収集
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=300)  # 5分タイムアウト
                    results.append(result)
                except Exception as e:
                    print(f"❌ プロセス圧縮エラー: {e}")
        
        return results
    
    def _multithread_compress(self, chunks: List[ProcessingChunk], quality: str) -> List[ProcessingResult]:
        """マルチスレッド圧縮"""
        print(f"🧵 マルチスレッド圧縮実行")
        
        futures = []
        results = []
        
        with self.thread_pool as executor:
            # チャンクを並列処理に投入
            for chunk in chunks:
                future = executor.submit(
                    self._compress_chunk_thread, 
                    chunk, 
                    quality
                )
                futures.append(future)
            
            # 結果収集
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=120)  # 2分タイムアウト
                    results.append(result)
                except Exception as e:
                    print(f"❌ スレッド圧縮エラー: {e}")
        
        return results
    
    def _compress_chunk_thread(self, chunk: ProcessingChunk, quality: str) -> ProcessingResult:
        """スレッドチャンク圧縮"""
        start_time = time.perf_counter()
        
        try:
            # メモリ確保
            memory_name = f"chunk_{chunk.chunk_id}"
            if not self.memory_manager.allocate(memory_name, len(chunk.data) * 2):
                raise MemoryError("Memory allocation failed")
            
            # 圧縮実行
            compressed = self.optimizer.optimize_compression(chunk.data, quality)
            
            processing_time = time.perf_counter() - start_time
            compression_ratio = len(compressed) / len(chunk.data)
            
            # メモリ解放
            self.memory_manager.deallocate(memory_name)
            
            return ProcessingResult(
                chunk_id=chunk.chunk_id,
                compressed_data=compressed,
                compression_ratio=compression_ratio,
                processing_time=processing_time
            )
            
        except Exception as e:
            return ProcessingResult(
                chunk_id=chunk.chunk_id,
                compressed_data=chunk.data,
                compression_ratio=1.0,
                processing_time=0.0,
                error=str(e)
            )
    
    def _sequential_compress(self, chunks: List[ProcessingChunk], quality: str) -> List[ProcessingResult]:
        """シーケンシャル圧縮"""
        print(f"📝 シーケンシャル圧縮実行")
        
        results = []
        
        for chunk in chunks:
            result = self._compress_chunk_thread(chunk, quality)
            results.append(result)
        
        return results
    
    def _merge_compression_results(self, results: List[ProcessingResult], original_data: bytes) -> bytes:
        """圧縮結果統合"""
        # チャンクIDでソート
        results.sort(key=lambda r: r.chunk_id)
        
        # 統合ヘッダー作成
        header = self._create_parallel_header(results, len(original_data))
        
        # データ結合
        combined_data = header
        for result in results:
            # チャンク情報
            chunk_header = self._create_chunk_header(result)
            combined_data += chunk_header + result.compressed_data
        
        return combined_data
    
    def _create_parallel_header(self, results: List[ProcessingResult], original_size: int) -> bytes:
        """並列ヘッダー作成"""
        header = bytearray(64)
        
        # マジックナンバー
        header[0:8] = b'NXPAR001'  # NEXUS Parallel v1
        
        # 基本情報
        header[8:16] = original_size.to_bytes(8, 'little')
        header[16:20] = len(results).to_bytes(4, 'little')
        
        # パフォーマンス統計
        total_time = sum(r.processing_time for r in results)
        avg_ratio = sum(r.compression_ratio for r in results) / len(results)
        
        header[20:24] = int(total_time * 1000).to_bytes(4, 'little')  # ms
        header[24:28] = int(avg_ratio * 10000).to_bytes(4, 'little')  # 0.01%単位
        
        return bytes(header)
    
    def _create_chunk_header(self, result: ProcessingResult) -> bytes:
        """チャンクヘッダー作成"""
        header = bytearray(16)
        
        header[0:4] = result.chunk_id.to_bytes(4, 'little')
        header[4:8] = len(result.compressed_data).to_bytes(4, 'little')
        header[8:12] = int(result.compression_ratio * 10000).to_bytes(4, 'little')
        header[12:16] = int(result.processing_time * 1000).to_bytes(4, 'little')
        
        return bytes(header)
    
    def _analyze_compressed_data(self, compressed_data: bytes) -> Dict:
        """圧縮データ解析"""
        if len(compressed_data) < 64:
            raise ValueError("Invalid compressed data")
        
        header = compressed_data[:64]
        
        if header[0:8] != b'NXPAR001':
            raise ValueError("Invalid parallel format")
        
        original_size = int.from_bytes(header[8:16], 'little')
        chunk_count = int.from_bytes(header[16:20], 'little')
        
        return {
            'original_size': original_size,
            'chunk_count': chunk_count,
            'data_offset': 64
        }
    
    def _determine_decompression_strategy(self, chunk_info: Dict) -> str:
        """展開戦略決定"""
        chunk_count = chunk_info['chunk_count']
        
        if chunk_count >= 4:
            return 'multiprocess'
        elif chunk_count >= 2:
            return 'multithread'
        else:
            return 'sequential'
    
    def _multiprocess_decompress(self, chunk_info: Dict) -> List[bytes]:
        """マルチプロセス展開"""
        print(f"🔄 マルチプロセス展開実行")
        return []  # 実装簡略化
    
    def _multithread_decompress(self, chunk_info: Dict) -> List[bytes]:
        """マルチスレッド展開"""
        print(f"🧵 マルチスレッド展開実行")
        return []  # 実装簡略化
    
    def _sequential_decompress(self, chunk_info: Dict) -> List[bytes]:
        """シーケンシャル展開"""
        print(f"📝 シーケンシャル展開実行")
        return []  # 実装簡略化
    
    def _merge_decompression_results(self, results: List[bytes]) -> bytes:
        """展開結果統合"""
        return b"".join(results)
    
    def _update_performance_stats(self, chunk_count: int, total_time: float):
        """パフォーマンス統計更新"""
        self.performance_stats['total_chunks_processed'] += chunk_count
        
        if chunk_count > 0:
            avg_time = total_time / chunk_count
            current_avg = self.performance_stats['average_chunk_time']
            total_chunks = self.performance_stats['total_chunks_processed']
            
            # 指数平滑移動平均
            alpha = 0.1
            self.performance_stats['average_chunk_time'] = (
                alpha * avg_time + (1 - alpha) * current_avg
            )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """パフォーマンスレポート取得"""
        memory_usage = self.memory_manager.get_usage()
        
        return {
            'parallel_stats': self.performance_stats,
            'memory_usage': memory_usage,
            'gpu_available': self.gpu_accelerator.available,
            'gpu_devices': self.gpu_accelerator.device_count,
            'config': self.config
        }
    
    def __del__(self):
        """デストラクタ"""
        try:
            self.thread_pool.shutdown(wait=False)
            self.process_pool.shutdown(wait=False)
        except:
            pass


def _compress_chunk_worker(data: bytes, chunk_id: int, quality: str) -> ProcessingResult:
    """ワーカープロセス用チャンク圧縮関数"""
    start_time = time.perf_counter()
    
    try:
        # 標準圧縮を使用（プロセス間では共有できないため）
        import lzma
        
        # 品質別プリセット
        presets = {
            'fast': 0,
            'balanced': 3,
            'max': 6
        }
        
        preset = presets.get(quality, 3)
        compressed = lzma.compress(data, preset=preset)
        
        processing_time = time.perf_counter() - start_time
        compression_ratio = len(compressed) / len(data)
        
        return ProcessingResult(
            chunk_id=chunk_id,
            compressed_data=compressed,
            compression_ratio=compression_ratio,
            processing_time=processing_time,
            method_used="process_worker",
            quality_score=(1 - compression_ratio) * 100,
            metrics={
                'original_size': len(data),
                'compressed_size': len(compressed)
            }
        )
        
    except Exception as e:
        return ProcessingResult(
            chunk_id=chunk_id,
            compressed_data=data,
            compression_ratio=1.0,
            processing_time=0.0,
            method_used="fallback",
            error=str(e)
        )


def test_nexus_parallel_engine():
    """NEXUS並列エンジンテスト"""
    print("⚡ NEXUS次世代並列エンジンテスト")
    print("=" * 80)
    
    # 並列設定
    config = ParallelConfig(
        use_gpu=True,
        use_multiprocessing=True,
        use_threading=True,
        max_threads=6,
        max_processes=3,
        chunk_size_mb=2,
        memory_limit_gb=4.0
    )
    
    # エンジン作成
    engine = NEXUSParallelEngine(config)
    
    # テストデータ生成（よりリアルなデータ）
    test_datasets = [
        {
            'name': '小データ（テキスト）',
            'data': (
                b"NEXUS Parallel Engine Test " * 100 +
                b"Compression Performance Evaluation " * 200 +
                b"Multi-threading and GPU Acceleration " * 150
            ),
            'quality': 'fast'
        },
        {
            'name': '中データ（混合パターン）',
            'data': (
                b"Pattern-123-ABC" * 2000 +
                b"\x00\x01\x02\x03\x04\x05" * 3000 +
                b"Repetitive-Data-Sequence" * 1500 +
                bytes(range(256)) * 100
            ),
            'quality': 'balanced'
        },
        {
            'name': '大データ（高エントロピー）',
            'data': (
                np.random.randint(0, 256, 500000, dtype=np.uint8).tobytes() +
                b"Structured-Section" * 5000 +
                np.random.randint(0, 256, 300000, dtype=np.uint8).tobytes()
            ),
            'quality': 'max'
        }
    ]
    
    print(f"🔬 テストデータセット: {len(test_datasets)} 種類")
    
    # 各データセットでテスト
    total_results = []
    
    for i, dataset in enumerate(test_datasets):
        print(f"\n{'='*60}")
        print(f"🧪 テストケース {i+1}: {dataset['name']}")
        print(f"   📊 データサイズ: {len(dataset['data']) / 1024:.1f}KB")
        print(f"   🎯 品質モード: {dataset['quality']}")
        
        try:
            # 並列圧縮実行
            start_time = time.perf_counter()
            compressed = engine.parallel_compress(dataset['data'], dataset['quality'])
            total_time = time.perf_counter() - start_time
            
            # 結果分析
            original_size = len(dataset['data'])
            compressed_size = len(compressed)
            compression_ratio = (1 - compressed_size / original_size) * 100
            throughput = original_size / 1024 / 1024 / total_time  # MB/s
            
            result = {
                'name': dataset['name'],
                'quality': dataset['quality'],
                'original_size': original_size,
                'compressed_size': compressed_size,
                'compression_ratio': compression_ratio,
                'processing_time': total_time,
                'throughput': throughput
            }
            
            total_results.append(result)
            
            print(f"   ✅ 圧縮成功!")
            print(f"      📈 圧縮率: {compression_ratio:.2f}%")
            print(f"      ⚡ スループット: {throughput:.2f}MB/s")
            print(f"      ⏱️ 処理時間: {total_time:.3f}秒")
            print(f"      💾 圧縮前: {original_size:,} bytes")
            print(f"      💾 圧縮後: {compressed_size:,} bytes")
            
        except Exception as e:
            print(f"   ❌ テストエラー: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 総合パフォーマンスレポート
    print(f"\n{'='*80}")
    print(f"📈 総合パフォーマンスレポート")
    print(f"{'='*80}")
    
    if total_results:
        avg_compression_ratio = sum(r['compression_ratio'] for r in total_results) / len(total_results)
        avg_throughput = sum(r['throughput'] for r in total_results) / len(total_results)
        total_data_processed = sum(r['original_size'] for r in total_results)
        total_processing_time = sum(r['processing_time'] for r in total_results)
        
        print(f"🎯 テスト結果サマリー:")
        print(f"   📊 テスト数: {len(total_results)}")
        print(f"   📈 平均圧縮率: {avg_compression_ratio:.2f}%")
        print(f"   ⚡ 平均スループット: {avg_throughput:.2f}MB/s")
        print(f"   💾 総処理データ: {total_data_processed / 1024 / 1024:.2f}MB")
        print(f"   ⏱️ 総処理時間: {total_processing_time:.3f}秒")
        
        # 品質別分析
        print(f"\n🎯 品質別パフォーマンス:")
        for quality in ['fast', 'balanced', 'max']:
            quality_results = [r for r in total_results if r['quality'] == quality]
            if quality_results:
                q_avg_ratio = sum(r['compression_ratio'] for r in quality_results) / len(quality_results)
                q_avg_throughput = sum(r['throughput'] for r in quality_results) / len(quality_results)
                print(f"   {quality:>8}: 圧縮率 {q_avg_ratio:.2f}%, スループット {q_avg_throughput:.2f}MB/s")
    
    # システムレポート
    print(f"\n🖥️ システム詳細レポート:")
    report = engine.get_comprehensive_report()
    
    # システムリソース
    sys_res = report['system_resources']
    print(f"   💻 CPU: {sys_res['cpu_count']} コア, 負荷 {sys_res['load_average']:.2f}")
    print(f"   🧠 メモリ: {sys_res['memory_gb']:.1f}GB")
    print(f"   🚀 GPU: {'有効' if sys_res['gpu_available'] else '無効'}")
    if sys_res['gpu_available']:
        print(f"       GPU メモリ: {sys_res['gpu_memory_gb']:.1f}GB")
    print(f"   💽 ディスクI/O: {sys_res['disk_io_speed']:.1f}MB/s")
    
    # パフォーマンスプロファイル
    perf = report['performance_profile']
    print(f"\n📊 パフォーマンスプロファイル:")
    print(f"   ⚡ スループット: {perf['throughput_mb_s']:.2f}MB/s")
    print(f"   📈 圧縮率: {perf['compression_ratio']:.3f}")
    print(f"   💻 CPU利用率: {perf['cpu_utilization']:.1f}%")
    print(f"   🧠 メモリ効率: {perf['memory_efficiency']:.1f}%")
    
    # メモリ使用量
    memory = report['memory_usage']
    print(f"\n🧠 メモリ使用量:")
    print(f"   📊 現在: {memory['current_mb']}MB / {memory['limit_mb']}MB")
    print(f"   📈 ピーク: {memory['peak_mb']}MB")
    print(f"   📊 利用率: {memory['utilization']*100:.1f}%")
    print(f"   🗂️ 割り当てオブジェクト: {memory['allocated_objects']}")
    
    # 負荷分散
    load_bal = report['load_balancing']
    print(f"\n⚖️ 負荷分散:")
    print(f"   👥 ワーカー数: {load_bal['total_workers']}")
    print(f"   ⚡ 総スループット: {load_bal['total_throughput_mb_s']:.2f}MB/s")
    
    # GPU パフォーマンス
    if report['gpu_performance']:
        gpu_perf = report['gpu_performance']
        print(f"\n🚀 GPU パフォーマンス:")
        print(f"   ⚡ スループット: {gpu_perf.get('throughput', 0):.2f}MB/s")
        print(f"   ⏱️ レイテンシ: {gpu_perf.get('latency', 0):.2f}ms")
        print(f"   🎯 効率: {gpu_perf.get('efficiency', 0):.1f}%")
    
    # エンジン統合
    engine_int = report['engine_integration']
    print(f"\n🔧 エンジン統合:")
    print(f"   🧠 NEXUS理論エンジン: {'利用可能' if engine_int['nexus_theory_available'] else '利用不可'}")
    print(f"   ⚙️ ベースエンジン: {'アクティブ' if engine_int['base_engine_active'] else '非アクティブ'}")
    print(f"   🎯 オプティマイザー: {'アクティブ' if engine_int['optimizer_active'] else '非アクティブ'}")
    
    print(f"\n🎉 NEXUS次世代並列エンジンテスト完了!")
    print(f"=" * 80)


if __name__ == "__main__":
    test_nexus_parallel_engine()
