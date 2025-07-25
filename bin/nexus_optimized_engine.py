#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🚀 NEXUS OPTIMIZED ENGINE V2.0 🚀
Ultra-High Performance Adaptive Compression Engine

最適化戦略を実装したNEXUS次世代エンジン:
- サイズベース適応圧縮
- スマート統合アルゴリズム  
- メタデータ圧縮パイプライン
- ハイブリッド圧縮戦略
"""

import numpy as np
import os
import sys
import time
import hashlib
import lzma
import gzip
import zlib
from collections import Counter, defaultdict
from itertools import combinations, product
import pickle
import json
from pathlib import Path

class NEXUSOptimizedEngine:
    """NEXUS最適化圧縮エンジン - V2.0"""
    
    def __init__(self):
        """初期化"""
        self.version = "2.0-OPTIMIZED"
        self.shapes = {
            'I-1': [(0, 0)],
            'I-2': [(0, 0), (0, 1)],
            'I-3': [(0, 0), (0, 1), (0, 2)],
            'I-4': [(0, 0), (0, 1), (0, 2), (0, 3)],
            'I-5': [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
            'L-3': [(0, 0), (0, 1), (1, 0)],
            'L-4': [(0, 0), (0, 1), (0, 2), (1, 0)],
            'T-4': [(0, 0), (0, 1), (0, 2), (1, 1)],
            'T-5': [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)],
            'H-3': [(0, 0), (1, 0), (2, 0)],
            'H-5': [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
            'H-7': [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 1)],
            'S-4': [(0, 0), (0, 1), (1, 1), (1, 2)],
            'Z-4': [(0, 1), (0, 2), (1, 0), (1, 1)],
            'O-4': [(0, 0), (0, 1), (1, 0), (1, 1)]
        }
        print(f"🔥 NEXUS Optimized Engine V{self.version} initialized")
        print(f"   [Optimization] Size-adaptive compression enabled")
        print(f"   [Optimization] Smart consolidation algorithms loaded")
        print(f"   [Optimization] Metadata compression pipeline ready")
        print(f"   [Optimization] Hybrid compression strategy active")
    
    def get_file_category(self, data_size):
        """ファイルサイズによるカテゴリ分類"""
        if data_size < 100:
            return "micro"  # マイクロファイル
        elif data_size < 1024:
            return "tiny"   # 小ファイル
        elif data_size < 10240:
            return "small"  # 中小ファイル
        elif data_size < 102400:
            return "medium" # 中ファイル
        else:
            return "large"  # 大ファイル
    
    def standard_compression_fallback(self, data):
        """標準圧縮アルゴリズムへのフォールバック"""
        print("   [Fallback] Testing standard compression algorithms...")
        
        # 複数の標準アルゴリズムを試行
        results = {}
        
        # LZMA
        try:
            lzma_compressed = lzma.compress(data, preset=9)
            results['lzma'] = len(lzma_compressed)
            print(f"   [Fallback] LZMA: {len(data)} -> {len(lzma_compressed)} bytes ({len(lzma_compressed)/len(data)*100:.1f}%)")
        except:
            results['lzma'] = float('inf')
        
        # Gzip
        try:
            gzip_compressed = gzip.compress(data, compresslevel=9)
            results['gzip'] = len(gzip_compressed)
            print(f"   [Fallback] Gzip: {len(data)} -> {len(gzip_compressed)} bytes ({len(gzip_compressed)/len(data)*100:.1f}%)")
        except:
            results['gzip'] = float('inf')
            
        # Zlib
        try:
            zlib_compressed = zlib.compress(data, level=9)
            results['zlib'] = len(zlib_compressed)
            print(f"   [Fallback] Zlib: {len(data)} -> {len(zlib_compressed)} bytes ({len(zlib_compressed)/len(data)*100:.1f}%)")
        except:
            results['zlib'] = float('inf')
        
        # 最良の結果を選択
        best_algo = min(results, key=results.get)
        best_size = results[best_algo]
        
        print(f"   [Fallback] Best standard algorithm: {best_algo.upper()} ({best_size} bytes)")
        
        # 最良のアルゴリズムで圧縮
        if best_algo == 'lzma':
            compressed_data = lzma.compress(data, preset=9)
            return compressed_data, {'algorithm': 'lzma', 'original_size': len(data)}
        elif best_algo == 'gzip':
            compressed_data = gzip.compress(data, compresslevel=9)
            return compressed_data, {'algorithm': 'gzip', 'original_size': len(data)}
        elif best_algo == 'zlib':
            compressed_data = zlib.compress(data, level=9)
            return compressed_data, {'algorithm': 'zlib', 'original_size': len(data)}
    
    def standard_decompression_fallback(self, compressed_data, metadata):
        """標準圧縮の展開"""
        algo = metadata['algorithm']
        
        if algo == 'lzma':
            return lzma.decompress(compressed_data)
        elif algo == 'gzip':
            return gzip.decompress(compressed_data)
        elif algo == 'zlib':
            return zlib.decompress(compressed_data)
        else:
            raise ValueError(f"Unknown fallback algorithm: {algo}")
    
    def smart_consolidation(self, groups, category, tolerance_factor=1.0):
        """スマート統合アルゴリズム - カテゴリ別最適化"""
        if category == "micro":
            # マイクロファイル: 激的統合
            tolerance = 50 * tolerance_factor  # 非常に高い許容度
            print(f"   [Smart Consolidation] Micro file mode: high tolerance ({tolerance})")
        elif category == "tiny":
            # 小ファイル: 高統合
            tolerance = 30 * tolerance_factor
            print(f"   [Smart Consolidation] Tiny file mode: elevated tolerance ({tolerance})")
        elif category == "small":
            # 中小ファイル: 中統合
            tolerance = 15 * tolerance_factor
            print(f"   [Smart Consolidation] Small file mode: moderate tolerance ({tolerance})")
        elif category == "medium":
            # 中ファイル: 低統合
            tolerance = 5 * tolerance_factor
            print(f"   [Smart Consolidation] Medium file mode: low tolerance ({tolerance})")
        else:
            # 大ファイル: 精密統合
            tolerance = 1 * tolerance_factor
            print(f"   [Smart Consolidation] Large file mode: precision tolerance ({tolerance})")
        
        # 統合前のグループ数
        original_count = len(groups)
        
        # 距離ベース統合アルゴリズム
        consolidated_groups = []
        used_indices = set()
        
        for i, group1 in enumerate(groups):
            if i in used_indices:
                continue
                
            merged_group = group1.copy()
            merge_candidates = [i]
            
            for j, group2 in enumerate(groups[i+1:], i+1):
                if j in used_indices:
                    continue
                
                # グループ間の距離計算（L2ノルム）
                distance = np.linalg.norm(np.array(group1) - np.array(group2))
                
                if distance <= tolerance:
                    # 統合候補として追加
                    merge_candidates.append(j)
                    # 重心で統合
                    merged_group = ((np.array(merged_group) + np.array(group2)) / 2).astype(int)
            
            # 統合実行
            for idx in merge_candidates:
                used_indices.add(idx)
            
            consolidated_groups.append(tuple(merged_group))
        
        consolidated_count = len(consolidated_groups)
        reduction_rate = (1 - consolidated_count / original_count) * 100
        
        print(f"   [Smart Consolidation] {original_count} -> {consolidated_count} groups ({reduction_rate:.1f}% reduction)")
        
        return consolidated_groups
    
    def compress_metadata(self, metadata):
        """メタデータ圧縮パイプライン"""
        print("   [Metadata Compression] Applying advanced compression...")
        
        # メタデータをJSON化
        metadata_json = json.dumps(metadata, separators=(',', ':')).encode('utf-8')
        original_size = len(metadata_json)
        
        # 複数のアルゴリズムで試行
        compression_results = {}
        
        # LZMA圧縮
        try:
            lzma_compressed = lzma.compress(metadata_json, preset=9)
            compression_results['lzma'] = lzma_compressed
        except:
            pass
        
        # Zlib圧縮  
        try:
            zlib_compressed = zlib.compress(metadata_json, level=9)
            compression_results['zlib'] = zlib_compressed
        except:
            pass
        
        # 最小のものを選択
        if compression_results:
            best_algo = min(compression_results, key=lambda k: len(compression_results[k]))
            best_compressed = compression_results[best_algo]
            compressed_size = len(best_compressed)
            
            print(f"   [Metadata Compression] {original_size} -> {compressed_size} bytes ({compressed_size/original_size*100:.1f}%, {best_algo.upper()})")
            
            return {
                'compressed_metadata': best_compressed,
                'compression_algo': best_algo,
                'original_size': original_size
            }
        else:
            # 圧縮失敗時は生データ
            print(f"   [Metadata Compression] No compression benefit, using raw data")
            return {
                'compressed_metadata': metadata_json,
                'compression_algo': 'none',
                'original_size': original_size
            }
    
    def decompress_metadata(self, compressed_metadata_info):
        """メタデータ展開"""
        compressed_data = compressed_metadata_info['compressed_metadata']
        algo = compressed_metadata_info['compression_algo']
        
        if algo == 'lzma':
            metadata_json = lzma.decompress(compressed_data)
        elif algo == 'zlib':
            metadata_json = zlib.decompress(compressed_data)
        elif algo == 'none':
            metadata_json = compressed_data
        else:
            raise ValueError(f"Unknown metadata compression algorithm: {algo}")
        
        return json.loads(metadata_json.decode('utf-8'))
    
    def adaptive_compress(self, data):
        """適応的圧縮 - サイズ別最適化戦略"""
        data_size = len(data)
        category = self.get_file_category(data_size)
        
        print(f"🔥 NEXUS OPTIMIZED COMPRESSION STARTING...")
        print(f"   [Adaptive Strategy] File size: {data_size} bytes, category: {category.upper()}")
        
        start_time = time.time()
        
        # マイクロファイル: 標準圧縮のみ
        if category == "micro":
            print("   [Adaptive Strategy] Micro file detected: using standard compression only")
            compressed_data, metadata = self.standard_compression_fallback(data)
            
            result = {
                'compressed_data': compressed_data,
                'metadata': metadata,
                'compression_type': 'standard',
                'algorithm': metadata['algorithm'],
                'original_size': data_size,
                'compressed_size': len(compressed_data),
                'compression_time': time.time() - start_time
            }
            
            print(f"✅ OPTIMIZED COMPRESSION COMPLETE!")
            print(f"⏱️  Compression time: {result['compression_time']:.3f}s")
            print(f"📦 Compressed size: {len(compressed_data)} bytes")
            print(f"📊 Compression ratio: {len(compressed_data)/data_size:.4f} ({len(compressed_data)/data_size*100:.2f}%)")
            
            return result
        
        # 小ファイル: 簡素化NEXUS vs 標準圧縮
        elif category in ["tiny", "small"]:
            print(f"   [Adaptive Strategy] {category.capitalize()} file: simplified NEXUS vs standard compression")
            
            # 標準圧縮を試行
            standard_compressed, standard_metadata = self.standard_compression_fallback(data)
            standard_size = len(standard_compressed)
            
            # 簡素化NEXUS圧縮を試行
            try:
                nexus_result = self.simplified_nexus_compress(data, category)
                nexus_size = nexus_result['compressed_size']
                
                print(f"   [Adaptive Strategy] Standard: {standard_size} bytes vs NEXUS: {nexus_size} bytes")
                
                # より小さい方を選択
                if standard_size <= nexus_size:
                    print(f"   [Adaptive Strategy] Selected: STANDARD compression ({standard_size} bytes)")
                    result = {
                        'compressed_data': standard_compressed,
                        'metadata': standard_metadata,
                        'compression_type': 'standard',
                        'algorithm': standard_metadata['algorithm'],
                        'original_size': data_size,
                        'compressed_size': standard_size,
                        'compression_time': time.time() - start_time
                    }
                else:
                    print(f"   [Adaptive Strategy] Selected: NEXUS compression ({nexus_size} bytes)")
                    nexus_result['compression_time'] = time.time() - start_time
                    result = nexus_result
                    
            except Exception as e:
                print(f"   [Adaptive Strategy] NEXUS failed ({e}), using standard compression")
                result = {
                    'compressed_data': standard_compressed,
                    'metadata': standard_metadata,
                    'compression_type': 'standard',
                    'algorithm': standard_metadata['algorithm'],
                    'original_size': data_size,
                    'compressed_size': standard_size,
                    'compression_time': time.time() - start_time
                }
        
        # 中～大ファイル: 最適化NEXUS
        else:
            print(f"   [Adaptive Strategy] {category.capitalize()} file: optimized NEXUS compression")
            try:
                result = self.optimized_nexus_compress(data, category)
                result['compression_time'] = time.time() - start_time
            except Exception as e:
                print(f"   [Adaptive Strategy] Optimized NEXUS failed ({e}), fallback to standard")
                compressed_data, metadata = self.standard_compression_fallback(data)
                result = {
                    'compressed_data': compressed_data,
                    'metadata': metadata,
                    'compression_type': 'standard',
                    'algorithm': metadata['algorithm'],
                    'original_size': data_size,
                    'compressed_size': len(compressed_data),
                    'compression_time': time.time() - start_time
                }
        
        print(f"✅ OPTIMIZED COMPRESSION COMPLETE!")
        print(f"⏱️  Compression time: {result['compression_time']:.3f}s")
        print(f"📦 Compressed size: {result['compressed_size']} bytes")
        print(f"📊 Compression ratio: {result['compressed_size']/data_size:.4f} ({result['compressed_size']/data_size*100:.2f}%)")
        print(f"💾 Space saved: {(1-result['compressed_size']/data_size)*100:.2f}%")
        
        return result
    
    def simplified_nexus_compress(self, data, category):
        """簡素化NEXUS圧縮 - 小ファイル用"""
        print("   [Simplified NEXUS] Starting simplified compression...")
        
        # 最も単純な形状のみを使用
        shape_name = 'I-1'
        shape = self.shapes[shape_name]
        
        # パディングとグリッド設定
        shape_width = max(coord[1] for coord in shape) + 1
        shape_height = max(coord[0] for coord in shape) + 1
        grid_width = int(np.ceil(np.sqrt(len(data))))
        
        # パディング
        total_size = grid_width * grid_width
        padded_data = np.pad(data, (0, max(0, total_size - len(data))), mode='constant', constant_values=0)
        
        # ブロック正規化（簡素化）
        blocks = []
        for start_row in range(0, grid_width, shape_height):
            for start_col in range(0, grid_width, shape_width):
                if start_row + shape_height <= grid_width and start_col + shape_width <= grid_width:
                    block_data = []
                    for rel_row, rel_col in shape:
                        abs_row = start_row + rel_row
                        abs_col = start_col + rel_col
                        if abs_row < grid_width and abs_col < grid_width:
                            idx = abs_row * grid_width + abs_col
                            if idx < len(padded_data):
                                block_data.append(int(padded_data[idx]))
                    if block_data:
                        blocks.append(tuple(block_data))
        
        # 統合（激的）
        unique_groups = list(set(blocks))
        if category == "tiny":
            # 小ファイルは50%統合を目標
            target_groups = max(1, len(unique_groups) // 2)
            unique_groups = self.smart_consolidation(unique_groups, category, tolerance_factor=2.0)[:target_groups]
        
        print(f"   [Simplified NEXUS] Using {len(unique_groups)} groups from {len(blocks)} blocks")
        
        # グループIDストリーム生成
        group_mapping = {group: idx for idx, group in enumerate(unique_groups)}
        group_id_stream = []
        for block in blocks:
            # 最も近いグループを検索
            best_group = min(unique_groups, key=lambda g: np.linalg.norm(np.array(g) - np.array(block)))
            group_id_stream.append(group_mapping[best_group])
        
        # 簡素化エンコーディング
        # RLE圧縮
        rle_encoded = []
        if group_id_stream:
            current_id = group_id_stream[0]
            count = 1
            for next_id in group_id_stream[1:]:
                if next_id == current_id:
                    count += 1
                else:
                    rle_encoded.extend([current_id, count])
                    current_id = next_id
                    count = 1
            rle_encoded.extend([current_id, count])
        
        # メタデータ
        metadata = {
            'version': self.version,
            'compression_type': 'simplified_nexus',
            'shape': shape_name,
            'grid_width': grid_width,
            'original_size': len(data),
            'groups': unique_groups,
            'rle_stream': rle_encoded,
            'blocks_count': len(blocks)
        }
        
        # メタデータ圧縮
        compressed_metadata_info = self.compress_metadata(metadata)
        
        # 最終パッケージング
        final_package = pickle.dumps(compressed_metadata_info)
        
        return {
            'compressed_data': final_package,
            'metadata': metadata,
            'compression_type': 'simplified_nexus',
            'original_size': len(data),
            'compressed_size': len(final_package)
        }
    
    def optimized_nexus_compress(self, data, category):
        """最適化NEXUS圧縮 - 中～大ファイル用"""
        print("   [Optimized NEXUS] Starting optimized compression...")
        
        # 形状選択（従来のロジック）
        if len(data) < 10000:
            shape_name = 'I-2'
        else:
            shape_name = 'I-4'
        
        shape = self.shapes[shape_name]
        
        # グリッド計算
        shape_width = max(coord[1] for coord in shape) + 1
        shape_height = max(coord[0] for coord in shape) + 1
        grid_width = max(shape_width, int(np.ceil(np.sqrt(len(data) / len(shape)))))
        
        # パディング
        rows = int(np.ceil(len(data) / grid_width))
        total_size = rows * grid_width
        padded_data = np.pad(data, (0, max(0, total_size - len(data))), mode='constant', constant_values=0)
        
        # ブロック正規化
        blocks = []
        for start_row in range(0, rows, shape_height):
            for start_col in range(0, grid_width, shape_width):
                if start_row + shape_height <= rows and start_col + shape_width <= grid_width:
                    block_data = []
                    for rel_row, rel_col in shape:
                        abs_row = start_row + rel_row
                        abs_col = start_col + rel_col
                        if abs_row < rows and abs_col < grid_width:
                            idx = abs_row * grid_width + abs_col
                            if idx < len(padded_data):
                                block_data.append(int(padded_data[idx]))
                    if block_data:
                        blocks.append(tuple(block_data))
        
        # ハッシュベース重複除去
        unique_groups = list(set(blocks))
        print(f"   [Optimized NEXUS] Found {len(unique_groups)} unique groups from {len(blocks)} blocks")
        
        # カテゴリ別スマート統合
        if category == "medium":
            # 中ファイル: 適度な統合
            unique_groups = self.smart_consolidation(unique_groups, category, tolerance_factor=1.5)
        # 大ファイルは統合なしで精密性を保持
        
        # グループIDストリーム生成
        group_mapping = {group: idx for idx, group in enumerate(unique_groups)}
        group_id_stream = []
        for block in blocks:
            # 最も近いグループを検索
            best_group = min(unique_groups, key=lambda g: np.linalg.norm(np.array(g) - np.array(block)))
            group_id_stream.append(group_mapping[best_group])
        
        # 最適化エンコーディング
        # グループIDストリームのLZMA圧縮
        group_id_bytes = bytes(group_id_stream)
        compressed_stream = lzma.compress(group_id_bytes, preset=6)
        
        # メタデータ
        metadata = {
            'version': self.version,
            'compression_type': 'optimized_nexus',
            'shape': shape_name,
            'grid_width': grid_width,
            'rows': rows,
            'original_size': len(data),
            'groups': unique_groups,
            'compressed_stream': compressed_stream,
            'blocks_count': len(blocks)
        }
        
        # メタデータ圧縮
        compressed_metadata_info = self.compress_metadata(metadata)
        
        # 最終パッケージング
        final_package = pickle.dumps(compressed_metadata_info)
        
        return {
            'compressed_data': final_package,
            'metadata': metadata,
            'compression_type': 'optimized_nexus',
            'original_size': len(data),
            'compressed_size': len(final_package)
        }
    
    def adaptive_decompress(self, compressed_result):
        """適応的展開"""
        print(f"🔥 NEXUS OPTIMIZED DECOMPRESSION STARTING...")
        
        start_time = time.time()
        
        compression_type = compressed_result.get('compression_type', 'unknown')
        
        if compression_type == 'standard':
            # 標準圧縮の展開
            print("   [Adaptive Decompression] Standard algorithm decompression")
            data = self.standard_decompression_fallback(
                compressed_result['compressed_data'],
                compressed_result['metadata']
            )
        elif compression_type == 'simplified_nexus':
            # 簡素化NEXUS展開
            print("   [Adaptive Decompression] Simplified NEXUS decompression")
            data = self.simplified_nexus_decompress(compressed_result['compressed_data'])
        elif compression_type == 'optimized_nexus':
            # 最適化NEXUS展開
            print("   [Adaptive Decompression] Optimized NEXUS decompression")
            data = self.optimized_nexus_decompress(compressed_result['compressed_data'])
        else:
            raise ValueError(f"Unknown compression type: {compression_type}")
        
        decompression_time = time.time() - start_time
        
        print(f"✅ OPTIMIZED DECOMPRESSION COMPLETE!")
        print(f"⏱️  Decompression time: {decompression_time:.3f}s")
        print(f"📄 Decompressed size: {len(data)} bytes")
        
        return data
    
    def simplified_nexus_decompress(self, compressed_data):
        """簡素化NEXUS展開"""
        # メタデータ展開
        compressed_metadata_info = pickle.loads(compressed_data)
        metadata = self.decompress_metadata(compressed_metadata_info)
        
        # RLEストリーム展開
        rle_stream = metadata['rle_stream']
        group_id_stream = []
        for i in range(0, len(rle_stream), 2):
            group_id = rle_stream[i]
            count = rle_stream[i + 1]
            group_id_stream.extend([group_id] * count)
        
        # ブロック再構築
        groups = metadata['groups']
        grid_width = metadata['grid_width']
        original_size = metadata['original_size']
        
        # データ配列初期化
        reconstructed_data = np.zeros(grid_width * grid_width, dtype=int)
        
        # ブロックの配置
        shape = self.shapes[metadata['shape']]
        shape_width = max(coord[1] for coord in shape) + 1
        shape_height = max(coord[0] for coord in shape) + 1
        
        block_idx = 0
        for start_row in range(0, grid_width, shape_height):
            for start_col in range(0, grid_width, shape_width):
                if start_row + shape_height <= grid_width and start_col + shape_width <= grid_width:
                    if block_idx < len(group_id_stream):
                        group_id = group_id_stream[block_idx]
                        block_data = groups[group_id]
                        
                        for data_idx, (rel_row, rel_col) in enumerate(shape):
                            if data_idx < len(block_data):
                                abs_row = start_row + rel_row
                                abs_col = start_col + rel_col
                                if abs_row < grid_width and abs_col < grid_width:
                                    idx = abs_row * grid_width + abs_col
                                    reconstructed_data[idx] = block_data[data_idx]
                        
                        block_idx += 1
        
        # 元のサイズにトリミング
        return reconstructed_data[:original_size].astype(np.uint8)
    
    def optimized_nexus_decompress(self, compressed_data):
        """最適化NEXUS展開"""
        # メタデータ展開
        compressed_metadata_info = pickle.loads(compressed_data)
        metadata = self.decompress_metadata(compressed_metadata_info)
        
        # 圧縮ストリーム展開
        compressed_stream = metadata['compressed_stream']
        group_id_bytes = lzma.decompress(compressed_stream)
        group_id_stream = list(group_id_bytes)
        
        # ブロック再構築
        groups = metadata['groups']
        grid_width = metadata['grid_width']
        rows = metadata['rows']
        original_size = metadata['original_size']
        
        # データ配列初期化
        reconstructed_data = np.zeros(rows * grid_width, dtype=int)
        
        # ブロックの配置
        shape = self.shapes[metadata['shape']]
        shape_width = max(coord[1] for coord in shape) + 1
        shape_height = max(coord[0] for coord in shape) + 1
        
        block_idx = 0
        for start_row in range(0, rows, shape_height):
            for start_col in range(0, grid_width, shape_width):
                if start_row + shape_height <= rows and start_col + shape_width <= grid_width:
                    if block_idx < len(group_id_stream):
                        group_id = group_id_stream[block_idx]
                        if group_id < len(groups):
                            block_data = groups[group_id]
                            
                            for data_idx, (rel_row, rel_col) in enumerate(shape):
                                if data_idx < len(block_data):
                                    abs_row = start_row + rel_row
                                    abs_col = start_col + rel_col
                                    if abs_row < rows and abs_col < grid_width:
                                        idx = abs_row * grid_width + abs_col
                                        if idx < len(reconstructed_data):
                                            reconstructed_data[idx] = block_data[data_idx]
                        
                        block_idx += 1
        
        # 元のサイズにトリミング
        return reconstructed_data[:original_size].astype(np.uint8)

def test_optimized_nexus():
    """最適化NEXUSエンジンのテスト"""
    print("🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥")
    print("🔥 NEXUS OPTIMIZED ENGINE TEST 🔥")
    print("🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥")
    
    engine = NEXUSOptimizedEngine()
    
    # テストファイルパス
    test_files = [
        r"c:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\sample\element_test_small.bin",
        r"c:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\sample\test_small.txt"
    ]
    
    results = []
    
    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"\n🔥 TESTING: {os.path.basename(file_path)}")
            print("=" * 60)
            
            # ファイル読み込み
            with open(file_path, 'rb') as f:
                original_data = f.read()
            
            original_md5 = hashlib.md5(original_data).hexdigest()
            print(f"📄 File size: {len(original_data)} bytes")
            print(f"🔍 Original MD5: {original_md5}")
            
            # 圧縮
            compressed_result = engine.adaptive_compress(original_data)
            
            # 展開
            decompressed_data = engine.adaptive_decompress(compressed_result)
            
            # 検証
            decompressed_md5 = hashlib.md5(decompressed_data).hexdigest()
            print(f"🔍 Decompressed MD5: {decompressed_md5}")
            
            if original_md5 == decompressed_md5:
                print("🎯 ✅ PERFECT MATCH - OPTIMIZATION SUCCESSFUL!")
                status = "SUCCESS"
            else:
                print("❌ MD5 MISMATCH - OPTIMIZATION FAILED!")
                status = "FAILED"
            
            # 結果記録
            results.append({
                'filename': os.path.basename(file_path),
                'original_size': len(original_data),
                'compressed_size': compressed_result['compressed_size'],
                'ratio': compressed_result['compressed_size'] / len(original_data),
                'compression_type': compressed_result['compression_type'],
                'status': status,
                'time': compressed_result['compression_time']
            })
    
    # 結果サマリー
    print("\n" + "🔥" * 60)
    print("🔥 OPTIMIZATION TEST RESULTS 🔥")
    print("🔥" * 60)
    
    for result in results:
        print(f"📁 {result['filename']}")
        print(f"   📄 Size: {result['original_size']} -> {result['compressed_size']} bytes")
        print(f"   📊 Ratio: {result['ratio']:.4f} ({result['ratio']*100:.2f}%)")
        print(f"   🔧 Method: {result['compression_type'].upper()}")
        print(f"   ⏱️  Time: {result['time']:.3f}s")
        print(f"   🎯 Status: {result['status']}")
        print()
    
    # 改善率計算
    success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
    print(f"🎯 SUCCESS RATE: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    
    if results:
        avg_ratio = sum(r['ratio'] for r in results) / len(results)
        print(f"📊 AVERAGE COMPRESSION RATIO: {avg_ratio:.4f} ({avg_ratio*100:.2f}%)")
        
        # 改善度評価
        if avg_ratio < 1.0:
            improvement = (1 - avg_ratio) * 100
            print(f"🚀 COMPRESSION ACHIEVED: {improvement:.1f}% size reduction!")
        else:
            expansion = (avg_ratio - 1) * 100
            print(f"📊 Current status: {expansion:.1f}% expansion (target: reduction)")
    
    print("🔥 NEXUS OPTIMIZATION TESTING COMPLETE! 🔥")

if __name__ == "__main__":
    test_optimized_nexus()
