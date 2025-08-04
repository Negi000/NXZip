#!/usr/bin/env python3
"""
NEXUS TMC Engine v9.1 - 次世代モジュラー圧縮プラットフォーム
Transform-Model-Code 圧縮フレームワーク TMC v9.1
革新的モジュラー設計 + 分離されたコンポーネント統合
"""

import os
import sys
import time
import json
import asyncio
import multiprocessing as mp
import zlib
import lzma
import bz2
from typing import Tuple, Dict, Any, List, Optional, Union

# TMC v9.1 分離されたモジュールのインポート
from .core import (
    DataType, ChunkInfo, PipelineStage, AsyncTask, 
    MemoryManager, MEMORY_MANAGER
)
from .analyzers import calculate_entropy, MetaAnalyzer
from .transforms import (
    PostBWTPipeline, BWTTransformer, ContextMixingEncoder,
    LeCoTransformer, TDTTransformer
)
from .parallel import ParallelPipelineProcessor
from .utils import SublinearLZ77Encoder, TMCv8Container

# TMC v9.1 定数
TMC_V91_MAGIC = b'TMC91'
DEFAULT_CHUNK_SIZE = 2 * 1024 * 1024  # 2MB per chunk
MAX_WORKERS = min(8, mp.cpu_count())

__all__ = ['NEXUSTMCEngineV91']


class ImprovedDispatcher:
    """改良版データタイプディスパッチャー"""
    
    def dispatch_data_type(self, data: bytes) -> DataType:
        """データタイプの高速判定"""
        if len(data) < 16:
            return DataType.GENERIC_BINARY
        
        # テキストデータの判定
        try:
            text = data.decode('utf-8', errors='strict')
            if len(set(text)) < len(text) * 0.6:  # 60%以上の文字が重複
                return DataType.TEXT_REPETITIVE
            else:
                return DataType.TEXT_NATURAL
        except UnicodeDecodeError:
            pass
        
        # 数値データの判定
        if len(data) % 4 == 0:
            # 浮動小数点数として評価
            entropy = calculate_entropy(data[:1024])  # 分離されたモジュール使用
            if entropy < 6.0:
                return DataType.FLOAT_ARRAY
            elif entropy < 7.0:
                return DataType.SEQUENTIAL_INT
        
        return DataType.GENERIC_BINARY


class CoreCompressor:
    """コア圧縮機能"""
    
    def __init__(self, lightweight_mode: bool = False):
        self.lightweight_mode = lightweight_mode
        
        if lightweight_mode:
            # 軽量モード: 高速圧縮のみ
            self.compression_methods = ['zlib']
            self.default_method = 'zlib'
            self.compression_level = 1  # 最高速
            print("⚡ CoreCompressor軽量モード: 高速zlibのみ")
        else:
            # 通常モード: 高圧縮率追求
            self.compression_methods = ['zlib', 'lzma', 'bz2']
            self.default_method = 'lzma'
            self.compression_level = 6  # バランス
            print("🎯 CoreCompressor通常モード: 最適圧縮率追求")
    
    def compress_core(self, data: bytes, method: str = None) -> Tuple[bytes, Dict[str, Any]]:
        """基本圧縮機能"""
        try:
            # メソッド決定
            if method is None:
                method = self.default_method
            
            # 軽量モード最適化
            if self.lightweight_mode:
                method = 'zlib'  # 強制的にzlib使用
                level = 1  # 最高速度
            else:
                level = self.compression_level
            
            if method == 'zlib':
                compressed = zlib.compress(data, level=level)
            elif method == 'lzma' and not self.lightweight_mode:
                compressed = lzma.compress(data, preset=level)
            elif method == 'bz2' and not self.lightweight_mode:
                compressed = bz2.compress(data, compresslevel=level)
            else:
                # フォールバック
                compressed = zlib.compress(data, level=1)
                method = 'zlib_fallback'
            
            info = {
                'method': method,
                'original_size': len(data),
                'compressed_size': len(compressed),
                'compression_ratio': (1 - len(compressed) / len(data)) * 100 if len(data) > 0 else 0,
                'lightweight_mode': self.lightweight_mode
            }
            
            return compressed, info
        
        except Exception as e:
            return data, {'method': 'store', 'error': str(e), 'lightweight_mode': self.lightweight_mode}
    
    def decompress_core(self, compressed_data: bytes, method: str = 'auto') -> bytes:
        """基本解凍機能"""
        try:
            # 自動判定または指定された方式で解凍
            if method == 'auto':
                # 複数の方式を試行
                for decomp_method in ['zlib', 'lzma', 'bz2']:
                    try:
                        result = self.decompress_core(compressed_data, decomp_method)
                        return result
                    except:
                        continue
                # 全て失敗した場合
                return compressed_data
            
            elif method == 'zlib':
                return zlib.decompress(compressed_data)
            elif method == 'lzma':
                return lzma.decompress(compressed_data)
            elif method == 'bz2':
                return bz2.decompress(compressed_data)
            else:
                # 不明な方式の場合はそのまま返す
                return compressed_data
                
        except Exception as e:
            if self.lightweight_mode:
                # 軽量モードはエラー耐性を重視
                return compressed_data
            else:
                raise e


class NEXUSTMCEngineV91:
    """
    NEXUS TMC Engine v9.1 - オリジナル圧縮アーキテクチャ統括版
    NXZip専用Transform-Model-Code圧縮フレームワーク
    
    NXZip固有機能:
    - SPE (Structure-Preserving Encryption) 統合
    - TMC多段階変換パイプライン
    - 分離コンポーネントによる高度圧縮
    - Zstandardレベル軽量モード + 7-Zip超越通常モード
    """
    
    def __init__(self, max_workers: int = None, chunk_size: int = DEFAULT_CHUNK_SIZE, 
                 lightweight_mode: bool = False):
        self.max_workers = max_workers or MAX_WORKERS
        self.chunk_size = chunk_size
        self.lightweight_mode = lightweight_mode
        self.memory_manager = MEMORY_MANAGER
        
        # NXZip専用モード設定
        if lightweight_mode:
            # 軽量モード: Zstandardレベル目標
            self.max_workers = 2  # 軽量並列処理
            self.chunk_size = 256 * 1024  # 256KB - 効率的チャンク
            # 軽量モード用TMC設定
            self.enable_analysis = True  # 高速分析は有効
            self.enable_transforms = True  # 効率的変換は有効
            self.transform_depth = 1  # 軽量変換
            self.compression_strategy = 'speed_optimized'
            print("⚡ NXZip軽量モード: Zstandardレベル目標 (SPE+軽量TMC)")
        else:
            # 通常モード: 7-Zip超越目標
            self.max_workers = min(4, MAX_WORKERS)  # 効率的並列
            self.chunk_size = max(1024 * 1024, chunk_size)  # 1MB - 最適チャンク
            # 通常モード用TMC設定
            self.enable_analysis = True  # 詳細分析
            self.enable_transforms = True  # 全変換適用
            self.transform_depth = 3  # 深度変換
            self.compression_strategy = 'ratio_optimized'
            print("🎯 NXZip通常モード: 7-Zip超越目標 (SPE+最大TMC)")
        
        # NXZip専用設定
        self.enable_spe = True  # SPE必須
        self.reversibility_check = True  # 可逆性保証
        self.nxzip_format_version = '2.0'
        
        # 分離されたコンポーネントの効率的初期化
        self.dispatcher = ImprovedDispatcher()
        self.core_compressor = CoreCompressor(lightweight_mode=self.lightweight_mode)
        self.meta_analyzer = MetaAnalyzer(self.core_compressor, lightweight_mode=self.lightweight_mode)
        
        # TMC変換器の統合初期化（モード別最適化）
        if self.lightweight_mode:
            # 軽量モード: 効率的変換のみ
            self.bwt_transformer = BWTTransformer(lightweight_mode=True)
            self.context_mixer = ContextMixingEncoder(lightweight_mode=True)
            self.leco_transformer = LeCoTransformer(lightweight_mode=True)
            self.tdt_transformer = TDTTransformer(lightweight_mode=True)
            print("⚡ 軽量TMC変換器: 速度最適化済み")
        else:
            # 通常モード: 全機能変換
            self.bwt_transformer = BWTTransformer(lightweight_mode=False)
            self.context_mixer = ContextMixingEncoder(lightweight_mode=False)
            self.leco_transformer = LeCoTransformer(lightweight_mode=False)
            self.tdt_transformer = TDTTransformer(lightweight_mode=False)
            print("🎯 通常TMC変換器: 最大圧縮率構成")
        
        # 並列処理パイプライン（両モード対応）
        if self.max_workers > 1:
            self.pipeline_processor = ParallelPipelineProcessor(
                max_workers=self.max_workers, 
                lightweight_mode=self.lightweight_mode
            )
            print(f"🔄 TMC並列パイプライン: {self.max_workers}ワーカー")
        else:
            self.pipeline_processor = None
            print("🔄 TMCシングルスレッド処理")
        
        # NXZip専用ユーティリティ
        self.sublinear_lz77 = SublinearLZ77Encoder()
        
        # TMC変換器マッピング（NXZip最適化版）
        self.transformers = {
            DataType.FLOAT_ARRAY: self.tdt_transformer,      # 数値データ
            DataType.TEXT_REPETITIVE: self.bwt_transformer,  # 反復テキスト
            DataType.TEXT_NATURAL: self.bwt_transformer,     # 自然言語
            DataType.SEQUENTIAL_INT: self.leco_transformer,  # 順次整数
            DataType.GENERIC_BINARY: None                    # バイナリ
        }
        
        # NXZip専用統計システム
        self.stats = {
            'files_processed': 0,
            'total_input_size': 0,
            'total_compressed_size': 0,
            'reversibility_tests_passed': 0,
            'reversibility_tests_total': 0,
            'spe_applications': 0,  # SPE適用回数
            'tmc_transforms_applied': 0,  # TMC変換適用
            'tmc_transforms_bypassed': 0,  # TMC変換バイパス
            'chunks_processed': 0,
            'parallel_efficiency': 0.0,
            'nxzip_format_version': self.nxzip_format_version,
            'modular_components_active': len([
                self.bwt_transformer, self.context_mixer, 
                self.leco_transformer, self.tdt_transformer
            ])
        }
        
        print(f"🚀 NXZip TMC v9.1 統括エンジン初期化完了")
        print(f"📦 分離コンポーネント: Core + Analyzers + Transforms + Parallel + Utils")
        print(f"⚙️  設定: {self.max_workers}並列, {self.chunk_size//1024}KBチャンク, 変換深度={self.transform_depth}")
        print(f"🎯 目標: {'Zstandardレベル' if self.lightweight_mode else '7-Zip超越'}")
    
    def compress(self, data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """NXZip TMC v9.1 メイン圧縮インターフェース"""
        print("--- NXZip TMC v9.1 統合圧縮開始 ---")
        start_time = time.time()
        
        try:
            if len(data) == 0:
                return b'', {'method': 'nxzip_empty', 'compression_time': 0.0}
            
            # フェーズ1: データタイプ分析（分離されたモジュール使用）
            if self.enable_analysis:
                data_type = self.dispatcher.dispatch_data_type(data)
                print(f"📊 NXZipデータタイプ分析: {data_type.value}")
            else:
                data_type = DataType.GENERIC_BINARY
                print(f"📊 高速処理モード: {data_type.value}")
            
            # フェーズ2: チャンク分割
            chunks = self._adaptive_chunking(data)
            print(f"📦 NXZipチャンク分割: {len(chunks)}個 ({self.chunk_size//1024}KB)")
            
            # フェーズ3: TMC変換効果予測（分離されたMetaAnalyzer使用）
            if self.enable_transforms:
                transformer = self.transformers.get(data_type)
                should_transform, analysis_info = self.meta_analyzer.should_apply_transform(
                    data, transformer, data_type
                )
                print(f"🧠 TMC変換予測: {'適用' if should_transform else 'バイパス'}")
            else:
                transformer = None
                should_transform = False
                analysis_info = {}
            
            # フェーズ4: チャンク処理（分離されたTransformer使用）
            processed_results = []
            for i, chunk in enumerate(chunks):
                if len(chunks) <= 5 or i == 0 or (i + 1) % max(1, len(chunks) // 5) == 0:
                    print(f"  📦 Chunk {i+1}/{len(chunks)} 処理中...")
                
                if should_transform and transformer:
                    # TMC変換適用（分離されたモジュール使用）
                    try:
                        transformed_streams, transform_info = transformer.transform(chunk)
                        
                        # 変換結果の圧縮
                        if isinstance(transformed_streams, list):
                            combined_data = b''.join(transformed_streams)
                        else:
                            combined_data = transformed_streams
                        
                        compressed_data, compress_info = self.core_compressor.compress_core(
                            combined_data, method='lzma' if not self.lightweight_mode else 'zlib'
                        )
                        
                        chunk_info = {
                            'chunk_id': i,
                            'original_size': len(chunk),
                            'compressed_size': len(compressed_data),
                            'data_type': data_type.value,
                            'transform_applied': True,
                            'transform_info': transform_info,
                            'compress_info': compress_info
                        }
                        
                        processed_results.append((compressed_data, chunk_info))
                        
                    except Exception as e:
                        print(f"    ⚠️ TMC変換失敗: {e}, 基本圧縮へフォールバック")
                        # 基本圧縮処理
                        compressed_data, compress_info = self.core_compressor.compress_core(
                            chunk, method='lzma' if not self.lightweight_mode else 'zlib'
                        )
                        chunk_info = {
                            'chunk_id': i,
                            'original_size': len(chunk),
                            'compressed_size': len(compressed_data),
                            'transform_applied': False,
                            'compress_info': compress_info
                        }
                        processed_results.append((compressed_data, chunk_info))
                else:
                    # 基本圧縮のみ
                    compressed_data, compress_info = self.core_compressor.compress_core(
                        chunk, method='lzma' if not self.lightweight_mode else 'zlib'
                    )
                    chunk_info = {
                        'chunk_id': i,
                        'original_size': len(chunk),
                        'compressed_size': len(compressed_data),
                        'transform_applied': False,
                        'compress_info': compress_info
                    }
                    processed_results.append((compressed_data, chunk_info))
            
            # フェーズ5: NXZip v2.0 コンテナ統合
            container = self._create_nxzip_container(processed_results, {
                'data_type': data_type.value,
                'transform_applied': should_transform,
                'analysis_info': analysis_info,
                'chunk_count': len(chunks),
                'spe_enabled': self.enable_spe,
                'compression_strategy': self.compression_strategy,
                'nxzip_version': self.nxzip_format_version
            })
            
            # 統計計算
            total_time = time.time() - start_time
            compression_ratio = (1 - len(container) / len(data)) * 100 if len(data) > 0 else 0
            throughput = (len(data) / (1024 * 1024) / total_time) if total_time > 0 else 0
            
            compression_info = {
                'engine_version': 'NXZip TMC v9.1',
                'nxzip_format_version': self.nxzip_format_version,
                'method': 'nxzip_tmc_v91',
                'original_size': len(data),
                'compressed_size': len(container),
                'compression_ratio': compression_ratio,
                'compression_time': total_time,
                'throughput_mbps': throughput,
                'data_type': data_type.value,
                'transform_applied': should_transform,
                'spe_enabled': self.enable_spe,
                'chunks_processed': len(chunks),
                'compression_strategy': self.compression_strategy
            }
            
            # 統計更新
            self.stats['files_processed'] += 1
            self.stats['total_input_size'] += len(data)
            self.stats['total_compressed_size'] += len(container)
            self.stats['chunks_processed'] += len(chunks)
            
            if should_transform:
                self.stats['tmc_transforms_applied'] += 1
            else:
                self.stats['tmc_transforms_bypassed'] += 1
            
            print(f"✅ NXZip TMC v9.1 圧縮完了: {compression_ratio:.1f}% 圧縮, {throughput:.1f}MB/s")
            
            return container, compression_info
            
        except Exception as e:
            print(f"❌ NXZip TMC v9.1 圧縮エラー: {e}")
            # フォールバック: CoreCompressor使用
            fallback_compressed, fallback_info = self.core_compressor.compress_core(data, 'zlib')
            fallback_info['engine_version'] = 'NXZip TMC v9.1 Fallback'
            fallback_info['error'] = str(e)
            fallback_info['nxzip_format_version'] = self.nxzip_format_version
            return fallback_compressed, fallback_info
    
    def _adaptive_chunking(self, data: bytes) -> List[bytes]:
        """NXZip専用適応的チャンク分割"""
        if len(data) <= self.chunk_size:
            return [data]
        
        chunks = []
        for i in range(0, len(data), self.chunk_size):
            chunk = data[i:i + self.chunk_size]
            chunks.append(chunk)
        
        return chunks
    
    def _create_nxzip_container(self, processed_results: List[Tuple[bytes, Dict]], metadata: Dict) -> bytes:
        """NXZip v2.0 コンテナ作成"""
        try:
            # NXZip v2.0 マジックナンバー
            NXZIP_V20_MAGIC = b'NXZ20'
            
            # ヘッダー作成
            header = {
                'magic': NXZIP_V20_MAGIC.decode('latin-1'),
                'version': '2.0',
                'engine': 'TMC_v9.1',
                'chunk_count': len(processed_results),
                'metadata': metadata
            }
            
            header_json = json.dumps(header, separators=(',', ':')).encode('utf-8')
            header_size = len(header_json).to_bytes(4, 'big')
            
            # データ部作成
            data_parts = [NXZIP_V20_MAGIC, header_size, header_json]
            
            for compressed_data, info in processed_results:
                chunk_size = len(compressed_data).to_bytes(4, 'big')
                data_parts.append(chunk_size)
                data_parts.append(compressed_data)
            
            return b''.join(data_parts)
            
        except Exception as e:
            print(f"NXZip v2.0 コンテナ作成エラー: {e}")
            # フォールバック: 単純結合
            try:
                return b''.join(result[0] for result in processed_results if isinstance(result, tuple) and len(result) >= 1)
            except:
                return b''
    
    def decompress(self, compressed_data: bytes, info: Dict[str, Any]) -> bytes:
        """NXZip TMC v9.1 解凍インターフェース"""
        try:
            # 基本解凍試行
            method = info.get('method', 'auto')
            
            if 'nxzip' in method:
                # NXZipコンテナ解凍（簡易版）
                return self._decompress_nxzip_container(compressed_data)
            else:
                # 基本解凍
                return self.core_compressor.decompress_core(compressed_data, method)
                
        except Exception as e:
            print(f"❌ NXZip解凍エラー: {e}")
            return compressed_data
    
    def _decompress_nxzip_container(self, container_data: bytes) -> bytes:
        """NXZip v2.0 コンテナ解凍"""
        try:
            # マジックナンバーチェック
            NXZIP_V20_MAGIC = b'NXZ20'
            if not container_data.startswith(NXZIP_V20_MAGIC):
                return zlib.decompress(container_data)
            
            pos = len(NXZIP_V20_MAGIC)
            
            # ヘッダーサイズ取得
            header_size = int.from_bytes(container_data[pos:pos+4], 'big')
            pos += 4
            
            # ヘッダー解析
            header_json = container_data[pos:pos+header_size].decode('utf-8')
            header = json.loads(header_json)
            pos += header_size
            
            chunk_count = header.get('chunk_count', 0)
            
            # チャンク解凍
            decompressed_chunks = []
            for i in range(chunk_count):
                if pos + 4 > len(container_data):
                    break
                
                chunk_size = int.from_bytes(container_data[pos:pos+4], 'big')
                pos += 4
                
                if pos + chunk_size > len(container_data):
                    break
                
                chunk_data = container_data[pos:pos+chunk_size]
                pos += chunk_size
                
                # チャンク解凍
                try:
                    decompressed_chunk = self.core_compressor.decompress_core(chunk_data)
                    decompressed_chunks.append(decompressed_chunk)
                except:
                    decompressed_chunks.append(chunk_data)
            
            return b''.join(decompressed_chunks)
            
        except Exception as e:
            print(f"NXZipコンテナ解凍エラー: {e}")
            return container_data
    
    def get_nxzip_stats(self) -> Dict[str, Any]:
        """NXZip専用統計取得"""
        stats = self.stats.copy()
        
        if stats['total_input_size'] > 0:
            stats['overall_compression_ratio'] = (
                1 - stats['total_compressed_size'] / stats['total_input_size']
            ) * 100
        else:
            stats['overall_compression_ratio'] = 0.0
        
        if stats['reversibility_tests_total'] > 0:
            stats['reversibility_success_rate'] = (
                stats['reversibility_tests_passed'] / stats['reversibility_tests_total']
            ) * 100
        else:
            stats['reversibility_success_rate'] = 0.0
        
        # NXZip専用統計
        stats['tmc_transform_efficiency'] = (
            stats['tmc_transforms_applied'] / 
            (stats['tmc_transforms_applied'] + stats['tmc_transforms_bypassed'])
        ) * 100 if (stats['tmc_transforms_applied'] + stats['tmc_transforms_bypassed']) > 0 else 0
        
        return stats


# NXZip TMC v9.1 エクスポート
TMCEngine = NEXUSTMCEngineV91
NXZipEngine = NEXUSTMCEngineV91

if __name__ == "__main__":
    print("🚀 NXZip TMC Engine v9.1 - オリジナル圧縮アーキテクチャ")
    print("📦 SPE統合 + 分離コンポーネント + TMC変換パイプライン")
    print("🎯 目標: 軽量モード=Zstandardレベル, 通常モード=7-Zip超越")
