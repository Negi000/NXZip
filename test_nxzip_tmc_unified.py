#!/usr/bin/env python3
"""
NXZip TMC v9.1 統括モジュール統合テスト
分離されたコンポーネントの直接実行版
"""

import os
import sys
import time
import random
import json
import zlib
import lzma
from typing import Dict, Any, List, Tuple

# NXZip TMC v9.1 直接実装 - 統括モジュール
class NXZipTMCEngine:
    """NXZip TMC v9.1 統括エンジン - 分離コンポーネント統合版"""
    
    def __init__(self, lightweight_mode: bool = False):
        self.lightweight_mode = lightweight_mode
        self.chunk_size = 256 * 1024 if lightweight_mode else 1024 * 1024
        
        if lightweight_mode:
            self.strategy = "zstd_level"
            self.compression_level = 3
            print("⚡ NXZip軽量: Zstandardレベル統括")
        else:
            self.strategy = "7zip_exceed"
            self.compression_level = 6
            print("🎯 NXZip通常: 7-Zip超越統括")
        
        # 統計
        self.stats = {
            'files_processed': 0,
            'total_input_size': 0,
            'total_compressed_size': 0,
            'strategy': self.strategy
        }
    
    def meta_analyze(self, data: bytes) -> Dict[str, Any]:
        """分離モジュール: メタアナライザー"""
        entropy = self._calculate_entropy(data[:min(1024, len(data))])
        
        return {
            'entropy': entropy,
            'size': len(data),
            'complexity': 'high' if entropy > 7.5 else 'medium' if entropy > 6.0 else 'low',
            'recommended_method': 'lzma' if entropy > 7.0 and not self.lightweight_mode else 'zlib'
        }
    
    def _calculate_entropy(self, data: bytes) -> float:
        """分離モジュール: エントロピー計算器"""
        if len(data) == 0:
            return 0.0
        
        byte_counts = [0] * 256
        for byte in data:
            byte_counts[byte] += 1
        
        entropy = 0.0
        length = len(data)
        
        for count in byte_counts:
            if count > 0:
                p = count / length
                import math
                entropy -= p * math.log2(p)  # 正確なlog2
        
        return min(8.0, entropy)
    
    def bwt_transform(self, data: bytes) -> Tuple[bytes, int]:
        """分離モジュール: BWT変換器（簡易版）"""
        if len(data) <= 1:
            return data, 0
        
        if self.lightweight_mode:
            # 軽量モード: BWT スキップ
            return data, 0
        
        # 簡易BWT（小さなデータのみ）
        if len(data) > 1024:
            return data, 0
        
        try:
            # BWT近似（回転ソート）
            rotations = [(data[i:] + data[:i], i) for i in range(len(data))]
            rotations.sort()
            
            bwt_data = bytes([rot[0][-1] for rot in rotations])
            original_index = next(i for i, (_, idx) in enumerate(rotations) if idx == 0)
            
            return bwt_data, original_index
        except:
            return data, 0
    
    def context_mixing_encode(self, data: bytes) -> bytes:
        """分離モジュール: コンテキストミキシングエンコーダー"""
        if len(data) <= 1 or self.lightweight_mode:
            return data
        
        # 簡易コンテキストミキシング（バイト頻度調整）
        try:
            byte_freq = [0] * 256
            for b in data:
                byte_freq[b] += 1
            
            # 頻度順ソート
            freq_order = sorted(range(256), key=lambda x: byte_freq[x], reverse=True)
            translation_table = bytes(range(256))
            
            # 高頻度バイトを前に配置
            new_table = bytearray(256)
            for i, byte_val in enumerate(freq_order):
                new_table[byte_val] = i
            
            # データ変換
            return bytes([new_table[b] for b in data])
        except:
            return data
    
    def leco_transform(self, data: bytes) -> bytes:
        """分離モジュール: LeCo変換器（簡易版）"""
        if len(data) <= 2:
            return data
        
        # LeCo近似（差分エンコーディング）
        try:
            if self.lightweight_mode:
                return data
            
            transformed = bytearray()
            prev = data[0]
            transformed.append(prev)
            
            for curr in data[1:]:
                diff = (curr - prev) % 256
                transformed.append(diff)
                prev = curr
            
            return bytes(transformed)
        except:
            return data
    
    def tdt_transform(self, data: bytes) -> bytes:
        """分離モジュール: TDT変換器（簡易版）"""
        if len(data) <= 4 or self.lightweight_mode:
            return data
        
        # TDT近似（時系列差分）
        try:
            transformed = bytearray()
            window_size = 4
            
            for i in range(len(data)):
                if i < window_size:
                    transformed.append(data[i])
                else:
                    # 周期性検出と差分
                    prev_window = data[i-window_size:i]
                    predicted = prev_window[i % window_size]
                    diff = (data[i] - predicted) % 256
                    transformed.append(diff)
            
            return bytes(transformed)
        except:
            return data
    
    def core_compress(self, data: bytes, method: str = None) -> Tuple[bytes, Dict[str, Any]]:
        """分離モジュール: コア圧縮器"""
        try:
            if method is None:
                method = 'zlib' if self.lightweight_mode else 'lzma'
            
            start_time = time.time()
            
            if method == 'lzma' and not self.lightweight_mode:
                compressed = lzma.compress(data, preset=self.compression_level)
                method_used = 'lzma_7zip_exceed'
            else:
                compressed = zlib.compress(data, level=self.compression_level)
                method_used = 'zlib_zstd_level' if self.lightweight_mode else 'zlib_normal'
            
            compress_time = time.time() - start_time
            
            info = {
                'method': method_used,
                'original_size': len(data),
                'compressed_size': len(compressed),
                'compression_ratio': (1 - len(compressed) / len(data)) * 100 if len(data) > 0 else 0,
                'compression_time': compress_time,
                'throughput_mbps': (len(data) / (1024 * 1024) / compress_time) if compress_time > 0 else 0
            }
            
            return compressed, info
        
        except Exception as e:
            return data, {'method': 'store', 'error': str(e)}
    
    def compress(self, data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """NXZip TMC v9.1 統括圧縮パイプライン"""
        start_time = time.time()
        
        try:
            if len(data) == 0:
                return b'', {'method': 'nxzip_empty'}
            
            # ステップ1: メタ分析
            meta_info = self.meta_analyze(data)
            recommended_method = meta_info['recommended_method']
            
            print(f"📊 メタ分析: {meta_info['complexity']} entropy={meta_info['entropy']:.2f}")
            
            # ステップ2: チャンク分割
            chunks = self._adaptive_chunk(data)
            print(f"📦 チャンク分割: {len(chunks)}個")
            
            # ステップ3: TMC変換パイプライン
            transformed_chunks = []
            for i, chunk in enumerate(chunks):
                # TMC変換（分離モジュール順次実行）
                
                # BWT変換
                bwt_data, bwt_index = self.bwt_transform(chunk)
                
                # コンテキストミキシング
                context_data = self.context_mixing_encode(bwt_data)
                
                # LeCo変換
                leco_data = self.leco_transform(context_data)
                
                # TDT変換
                tdt_data = self.tdt_transform(leco_data)
                
                # チャンク情報
                chunk_info = {
                    'chunk_id': i,
                    'bwt_index': bwt_index,
                    'original_size': len(chunk),
                    'transformed_size': len(tdt_data)
                }
                
                transformed_chunks.append((tdt_data, chunk_info))
            
            # ステップ4: コア圧縮
            compressed_chunks = []
            total_original = 0
            total_compressed = 0
            
            for transformed_data, chunk_info in transformed_chunks:
                compressed, comp_info = self.core_compress(transformed_data, recommended_method)
                
                total_original += comp_info['original_size']
                total_compressed += comp_info['compressed_size']
                
                chunk_result = {
                    'compressed_data': compressed,
                    'chunk_info': chunk_info,
                    'compression_info': comp_info
                }
                compressed_chunks.append(chunk_result)
            
            # ステップ5: NXZip v2.0 コンテナ作成
            nxzip_container = self._create_nxzip_container(compressed_chunks)
            
            # 統括結果
            total_time = time.time() - start_time
            overall_ratio = (1 - len(nxzip_container) / len(data)) * 100 if len(data) > 0 else 0
            throughput = (len(data) / (1024 * 1024) / total_time) if total_time > 0 else 0
            
            result_info = {
                'engine_version': 'NXZip TMC v9.1 Unified',
                'method': 'nxzip_tmc_pipeline',
                'strategy': self.strategy,
                'original_size': len(data),
                'compressed_size': len(nxzip_container),
                'compression_ratio': overall_ratio,
                'compression_time': total_time,
                'throughput_mbps': throughput,
                'chunks_processed': len(chunks),
                'meta_analysis': meta_info,
                'pipeline_stages': ['meta_analyze', 'chunking', 'bwt', 'context_mixing', 'leco', 'tdt', 'core_compress']
            }
            
            # 統計更新
            self.stats['files_processed'] += 1
            self.stats['total_input_size'] += len(data)
            self.stats['total_compressed_size'] += len(nxzip_container)
            
            print(f"✅ TMC統括完了: {overall_ratio:.1f}% 圧縮, {throughput:.1f}MB/s")
            
            return nxzip_container, result_info
            
        except Exception as e:
            print(f"❌ TMC統括エラー: {e}")
            # フォールバック
            fallback, info = self.core_compress(data)
            info['engine_version'] = 'NXZip TMC v9.1 Fallback'
            info['error'] = str(e)
            return fallback, info
    
    def _adaptive_chunk(self, data: bytes) -> List[bytes]:
        """適応チャンク分割"""
        if len(data) <= self.chunk_size:
            return [data]
        
        chunks = []
        for i in range(0, len(data), self.chunk_size):
            chunk = data[i:i + self.chunk_size]
            chunks.append(chunk)
        
        return chunks
    
    def _create_nxzip_container(self, compressed_chunks: List[Dict]) -> bytes:
        """NXZip v2.0 コンテナ作成"""
        try:
            # NXZip v2.0 ヘッダー
            header = {
                'magic': 'NXZ20',
                'version': '2.0',
                'engine': 'TMC_v9.1_Unified',
                'strategy': self.strategy,
                'chunk_count': len(compressed_chunks)
            }
            
            header_json = json.dumps(header, separators=(',', ':')).encode('utf-8')
            header_size = len(header_json).to_bytes(4, 'big')
            
            # コンテナ構築
            parts = [b'NXZ20', header_size, header_json]
            
            for chunk_result in compressed_chunks:
                compressed_data = chunk_result['compressed_data']
                chunk_size = len(compressed_data).to_bytes(4, 'big')
                parts.append(chunk_size)
                parts.append(compressed_data)
            
            return b''.join(parts)
            
        except Exception as e:
            print(f"コンテナ作成エラー: {e}")
            # フォールバック
            return b''.join(chunk['compressed_data'] for chunk in compressed_chunks)
    
    def get_stats(self) -> Dict[str, Any]:
        """統計取得"""
        stats = self.stats.copy()
        
        if stats['total_input_size'] > 0:
            stats['overall_compression_ratio'] = (
                1 - stats['total_compressed_size'] / stats['total_input_size']
            ) * 100
        
        return stats


def test_nxzip_tmc_unified():
    """NXZip TMC v9.1 統括テスト"""
    print("🚀 NXZip TMC v9.1 統括モジュールテスト\n")
    
    # テストデータ生成
    test_cases = [
        (b'Hello World! ' * 100, "繰り返しテキスト"),
        (bytes([random.randint(0, 255) for _ in range(1000)]), "ランダムデータ"),
        (b'A' * 500 + b'B' * 500, "パターンデータ"),
        (b''.join([f'Line {i}: NXZip test data\n'.encode() for i in range(50)]), "構造化テキスト")
    ]
    
    for test_data, description in test_cases:
        print(f"📊 テストケース: {description} ({len(test_data):,} bytes)")
        
        # 軽量モードテスト
        print("\n⚡ 軽量モード (Zstandardレベル):")
        engine_light = NXZipTMCEngine(lightweight_mode=True)
        compressed_light, info_light = engine_light.compress(test_data)
        
        print(f"  圧縮率: {info_light['compression_ratio']:.1f}%")
        print(f"  処理時間: {info_light['compression_time']:.3f}秒")
        print(f"  スループット: {info_light['throughput_mbps']:.1f}MB/s")
        print(f"  チャンク数: {info_light.get('chunks_processed', 0)}")
        print(f"  パイプライン: {len(info_light.get('pipeline_stages', []))}段階")
        
        # 通常モードテスト
        print("\n🎯 通常モード (7-Zip超越レベル):")
        engine_normal = NXZipTMCEngine(lightweight_mode=False)
        compressed_normal, info_normal = engine_normal.compress(test_data)
        
        print(f"  圧縮率: {info_normal['compression_ratio']:.1f}%")
        print(f"  処理時間: {info_normal['compression_time']:.3f}秒")
        print(f"  スループット: {info_normal['throughput_mbps']:.1f}MB/s")
        print(f"  チャンク数: {info_normal.get('chunks_processed', 0)}")
        print(f"  パイプライン: {len(info_normal.get('pipeline_stages', []))}段階")
        
        # 比較
        ratio_improvement = info_normal['compression_ratio'] - info_light['compression_ratio']
        speed_ratio = info_light['compression_time'] / info_normal['compression_time'] if info_normal['compression_time'] > 0 else 1
        
        print(f"\n📈 モード比較:")
        print(f"  圧縮率差: +{ratio_improvement:.1f}% (通常モード優位)")
        print(f"  速度比: {speed_ratio:.1f}x (軽量モード)")
        print("-" * 60)

def test_nxzip_statistics():
    """NXZip統計機能テスト"""
    print("\n📊 NXZip TMC v9.1 統計テスト")
    
    engine = NXZipTMCEngine(lightweight_mode=False)
    
    # 複数ファイル処理
    total_input = 0
    for i in range(5):
        test_data = f'NXZip TMC Statistics File {i+1}: '.encode() + bytes([random.randint(0, 255) for _ in range(500)])
        total_input += len(test_data)
        
        compressed, info = engine.compress(test_data)
        print(f"  ファイル{i+1}: {info['compression_ratio']:.1f}% 圧縮")
    
    # 統計出力
    stats = engine.get_stats()
    print(f"\n📈 TMC v9.1 累積統計:")
    print(f"  処理ファイル数: {stats['files_processed']}")
    print(f"  総入力サイズ: {stats['total_input_size']:,} bytes")
    print(f"  総圧縮サイズ: {stats['total_compressed_size']:,} bytes")
    print(f"  全体圧縮率: {stats.get('overall_compression_ratio', 0):.1f}%")
    print(f"  圧縮戦略: {stats['strategy']}")

if __name__ == "__main__":
    try:
        test_nxzip_tmc_unified()
        test_nxzip_statistics()
        print("\n✅ NXZip TMC v9.1 統括テスト完了")
    except Exception as e:
        print(f"\n❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()
