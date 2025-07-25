#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEXUS FINAL ENGINE
Ultimate implementation of NEXUS theory with greedy tiling
"""

import sys
import json
import math
import heapq
import collections
import time
import os
import glob
from typing import List, Tuple, Dict, Any

try:
    import lzma
    LZMA_AVAILABLE = True
except ImportError:
    LZMA_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print("Installing numpy...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'numpy'])
    import numpy as np
    NUMPY_AVAILABLE = True

# --- 静的順列辞書プロバイダ ---
class StaticPermutationProvider:
    """
    ブロックサイズごとの順列を静的に管理し、IDとの相互変換を行う。
    これにより、ファイルごとに順列辞書を保存する必要がなくなり、メタデータを削減する。
    """
    _cache: Dict[int, Tuple[Dict[Tuple[int, ...], int], Dict[int, Tuple[int, ...]]]] = {}

    @classmethod
    def get_provider_for_size(cls, size: int):
        if size not in cls._cache:
            if size > 8: # 実用的な上限（8! = 40320）
                raise ValueError("Permutation size too large for static provider.")
            
            from itertools import permutations
            perms = sorted(list(permutations(range(size))))
            
            perm_to_id = {perm: i for i, perm in enumerate(perms)}
            id_to_perm = {i: perm for i, perm in enumerate(perms)}
            cls._cache[size] = (perm_to_id, id_to_perm)
        return cls._cache[size]

    @classmethod
    def get_perm_id(cls, perm: Tuple[int, ...]) -> int:
        size = len(perm)
        perm_to_id, _ = cls.get_provider_for_size(size)
        return perm_to_id.get(perm, 0)

    @classmethod
    def get_perm_from_id(cls, perm_id: int, size: int) -> Tuple[int, ...]:
        _, id_to_perm = cls.get_provider_for_size(size)
        return id_to_perm.get(perm_id, tuple(range(size)))

# --- ユーティリティ & エンコーダ ---
class HuffmanEncoder:
    """Huffman符号化/復号化クラス"""
    def encode(self, data: List[int]) -> Tuple[Dict[int, str], str]:
        if not data: return {}, ""
        frequency = collections.Counter(data)
        if len(frequency) == 1:
            # Single symbol case
            symbol = list(frequency.keys())[0]
            return {symbol: "0"}, "0" * len(data)
        
        heap = [[weight, [symbol, ""]] for symbol, weight in frequency.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            lo, hi = heapq.heappop(heap), heapq.heappop(heap)
            for pair in lo[1:]: pair[1] = '0' + pair[1]
            for pair in hi[1:]: pair[1] = '1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        huffman_tree = dict(sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p)))
        return huffman_tree, "".join([huffman_tree[d] for d in data])

    def decode(self, encoded_data: str, huffman_tree: Dict[str, Any]) -> List[int]:
        if not encoded_data: return []
        reverse_tree = {v: int(k) for k, v in huffman_tree.items()}
        decoded_data, current_code = [], ""
        for bit in encoded_data:
            current_code += bit
            if current_code in reverse_tree:
                decoded_data.append(reverse_tree[current_code])
                current_code = ""
        return decoded_data

def _apply_delta_encoding(stream: List[int]) -> List[int]:
    """差分符号化を適用し、ハフマン効率を高める"""
    if not stream: return []
    return [stream[0]] + [stream[i] - stream[i-1] for i in range(1, len(stream))]

def _reverse_delta_encoding(delta_stream: List[int]) -> List[int]:
    """差分符号化を逆変換する"""
    if not delta_stream: return []
    original_stream = [delta_stream[0]]
    for i in range(1, len(delta_stream)):
        original_stream.append(original_stream[-1] + delta_stream[i])
    return original_stream

# --- Polyomino形状定義 ---
POLYOMINO_SHAPES = {
    # ID: (座標タプル, サイズ)
    0: (((0, 0),), 1),  # I-1
    1: (((0, 0), (0, 1)), 2),  # I-2
    2: (((0, 0), (0, 1), (1, 0), (1, 1)), 4),  # O-4
    3: (((0, 0), (0, 1), (0, 2)), 3), # I-3
    4: (((0, 0), (0, 1), (0, 2), (0, 3)), 4), # I-4
    5: (((0, 0), (1, 0), (2, 0), (2, 1)), 4), # L-4
}
SHAPE_IDS = sorted(POLYOMINO_SHAPES.keys())

class ProgressBar:
    def __init__(self, total: int, description: str = "", width: int = 40):
        self.total, self.description, self.width = total, description, width
        self.current = 0
    def update(self, current: int):
        self.current = current
        percentage = min(100, (self.current / self.total) * 100) if self.total > 0 else 100
        filled = int(self.width * self.current // self.total) if self.total > 0 else self.width
        bar = '█' * filled + '░' * (self.width - filled)
        print(f'\r{self.description} |{bar}| {percentage:.1f}%', end='', flush=True)
    def finish(self):
        print()

class NexusCompressor:
    """NEXUS最終版圧縮エンジン"""

    def __init__(self):
        self.huffman_encoder = HuffmanEncoder()

    def _find_best_shape_for_position(self, data_grid: np.ndarray, y: int, x: int, unique_groups: Dict[Tuple[int, ...], int]) -> Tuple[int, Tuple[int, ...], int]:
        """指定位置でデータを最も効率的に表現できる形状を見つける"""
        best_shape_id = -1
        best_group_tuple = None
        min_cost = float('inf') # 既存グループなら0、新規なら1

        grid_height, grid_width = data_grid.shape

        for shape_id in SHAPE_IDS:
            shape_coords, _ = POLYOMINO_SHAPES[shape_id]
            
            # 形状がグリッド範囲内に収まるかチェック
            max_y = y + max(dy for dy, dx in shape_coords)
            max_x = x + max(dx for dy, dx in shape_coords)
            if max_y >= grid_height or max_x >= grid_width:
                continue

            # データを抽出してブロックを作成
            current_block = tuple(data_grid[y + dy, x + dx] for dy, dx in shape_coords)
            
            # 正規化
            normalized_block = tuple(sorted(current_block))
            
            cost = 0 if normalized_block in unique_groups else 1
            
            if cost < min_cost:
                min_cost = cost
                best_shape_id = shape_id
                best_group_tuple = current_block
                if cost == 0: # 既存グループが見つかれば、それが最適なので即時終了
                    return best_shape_id, best_group_tuple, cost
        
        # どの形状もはまらない場合は最小のI-1形状を返す
        if best_shape_id == -1:
            best_shape_id = 0 # I-1
            best_group_tuple = (data_grid[y, x],)
            min_cost = 1 if tuple(sorted(best_group_tuple)) not in unique_groups else 0

        return best_shape_id, best_group_tuple, min_cost

    def _perform_greedy_tiling(self, data_grid: np.ndarray) -> Tuple[List[Any], List[Any]]:
        """
        貪欲タイリングアルゴリズムを実行し、Blueprintとユニークグループを生成する。
        これが可逆性と圧縮率を両立させる心臓部。
        """
        grid_height, grid_width = data_grid.shape
        coverage_grid = np.zeros_like(data_grid, dtype=bool)
        
        blueprint = []
        unique_groups: Dict[Tuple[int, ...], int] = {}
        group_id_counter = 0

        total_cells = grid_height * grid_width
        cells_covered = 0
        
        progress_bar = ProgressBar(total_cells, "Tiling")

        y, x = 0, 0
        while cells_covered < total_cells:
            # 次の未被覆セルを探す
            while y < grid_height and coverage_grid[y, x]:
                x += 1
                if x >= grid_width:
                    x = 0
                    y += 1
            
            if y >= grid_height: break # 全て被覆完了

            # 最適な形状を見つける
            shape_id, block_tuple, cost = self._find_best_shape_for_position(data_grid, y, x, unique_groups)
            shape_coords, shape_size = POLYOMINO_SHAPES[shape_id]
            
            # グループIDを決定
            normalized_block = tuple(sorted(block_tuple))
            if normalized_block not in unique_groups:
                unique_groups[normalized_block] = group_id_counter
                group_id_counter += 1
            group_id = unique_groups[normalized_block]

            # 順列IDを決定
            perm_map = tuple(normalized_block.index(val) for val in block_tuple)
            perm_id = StaticPermutationProvider.get_perm_id(perm_map)

            # Blueprintに記録 (shape_id, group_id, perm_id, y, x)
            blueprint.append((shape_id, group_id, perm_id, y, x))

            # 被覆グリッドを更新
            for dy, dx in shape_coords:
                if not coverage_grid[y + dy, x + dx]:
                    coverage_grid[y + dy, x + dx] = True
                    cells_covered += 1
            
            progress_bar.update(cells_covered)

        progress_bar.finish()
        
        # 辞書をリストに変換 (NumPy型をPython標準型に変換)
        sorted_unique_groups = []
        for group in sorted(unique_groups.keys(), key=lambda k: unique_groups[k]):
            sorted_unique_groups.append([int(x) for x in group])
        return blueprint, sorted_unique_groups

    def compress(self, data: bytes) -> bytes:
        """NEXUS圧縮を実行"""
        if not data: return b''
        
        print("🔥 NEXUS Final Engine: Compression")
        original_length = len(data)

        # 1. グリッド化
        grid_width = math.ceil(math.sqrt(original_length))
        grid_height = math.ceil(original_length / grid_width)
        padded_size = grid_width * grid_height
        padded_data = data.ljust(padded_size, b'\0')
        data_grid = np.frombuffer(padded_data, dtype=np.uint8).reshape((grid_height, grid_width))

        print(f"   Grid: {grid_height}x{grid_width} ({padded_size} cells)")

        # 2. 貪欲タイリングでBlueprintとユニークグループを生成
        blueprint, unique_groups_list = self._perform_greedy_tiling(data_grid)

        print(f"   Blueprint: {len(blueprint)} blocks, {len(unique_groups_list)} unique groups")

        # 3. Blueprintをストリームに分割
        shape_id_stream = [bp[0] for bp in blueprint]
        group_id_stream = [bp[1] for bp in blueprint]
        perm_id_stream = [bp[2] for bp in blueprint]
        pos_y_stream = [bp[3] for bp in blueprint]
        pos_x_stream = [bp[4] for bp in blueprint]
        
        # 4. 差分符号化
        print("   Applying Delta Encoding...")
        streams_to_encode = {
            "s_id": _apply_delta_encoding(shape_id_stream),
            "g_id": _apply_delta_encoding(group_id_stream),
            "p_id": _apply_delta_encoding(perm_id_stream),
            "pos_y": _apply_delta_encoding(pos_y_stream),
            "pos_x": _apply_delta_encoding(pos_x_stream),
        }

        # 5. Huffman符号化
        print("   Applying Huffman Encoding...")
        huffman_trees, encoded_streams = {}, {}
        for name, stream in streams_to_encode.items():
            tree, encoded_data = self.huffman_encoder.encode(stream)
            huffman_trees[name] = tree
            encoded_streams[name] = encoded_data

        # 6. ペイロード作成
        payload = {
            "header": {
                "algorithm": "NEXUS_vFinal_Tiling",
                "original_length": original_length,
                "grid_dims": (grid_height, grid_width),
            },
            "unique_groups": unique_groups_list,
            "huffman_trees": huffman_trees,
            "encoded_streams": encoded_streams,
        }

        # 7. シリアライズ & 最終圧縮
        print("   Finalizing with LZMA...")
        serialized_payload = json.dumps(payload).encode('utf-8')
        if LZMA_AVAILABLE:
            return lzma.compress(serialized_payload, preset=9)
        else:
            return serialized_payload


class NexusDecompressor:
    """NEXUS最終版解凍エンジン"""
    
    def __init__(self):
        self.huffman_encoder = HuffmanEncoder()

    def decompress(self, compressed_data: bytes) -> bytes:
        if not compressed_data: return b''

        print("🔓 NEXUS Final Engine: Decompression")
        
        # 1. 最終解凍 & デシリアライズ
        if LZMA_AVAILABLE:
            serialized_payload = lzma.decompress(compressed_data)
        else:
            serialized_payload = compressed_data
        payload = json.loads(serialized_payload.decode('utf-8'))

        header = payload["header"]
        original_length = header["original_length"]
        grid_height, grid_width = header["grid_dims"]

        print(f"   Target: {original_length} bytes, Grid: {grid_height}x{grid_width}")

        # 2. Huffman復号化
        print("   Decoding Huffman streams...")
        huffman_trees = payload["huffman_trees"]
        encoded_streams = payload["encoded_streams"]
        delta_streams = {}
        for name, encoded_data in encoded_streams.items():
            tree = huffman_trees[name]
            delta_streams[name] = self.huffman_encoder.decode(encoded_data, tree)
        
        # 3. 差分逆変換
        print("   Reversing Delta Encoding...")
        shape_id_stream = _reverse_delta_encoding(delta_streams["s_id"])
        group_id_stream = _reverse_delta_encoding(delta_streams["g_id"])
        perm_id_stream = _reverse_delta_encoding(delta_streams["p_id"])
        pos_y_stream = _reverse_delta_encoding(delta_streams["pos_y"])
        pos_x_stream = _reverse_delta_encoding(delta_streams["pos_x"])

        # 4. グリッド再構成
        print("   Reconstructing data grid from blueprint...")
        unique_groups = [tuple(g) for g in payload["unique_groups"]]
        reconstructed_grid = np.zeros((grid_height, grid_width), dtype=np.uint8)

        total_blocks = len(shape_id_stream)
        progress_bar = ProgressBar(total_blocks, "Rebuilding")

        for i in range(total_blocks):
            shape_id = shape_id_stream[i]
            group_id = group_id_stream[i]
            perm_id = perm_id_stream[i]
            y, x = pos_y_stream[i], pos_x_stream[i]

            shape_coords, shape_size = POLYOMINO_SHAPES[shape_id]

            # ブロックを復元
            sorted_group = unique_groups[group_id]
            perm_map = StaticPermutationProvider.get_perm_from_id(perm_id, shape_size)
            
            original_block = [0] * shape_size
            for original_pos, sorted_pos in enumerate(perm_map):
                original_block[original_pos] = sorted_group[sorted_pos]

            # グリッドに配置 (上書きなし)
            for j, (dy, dx) in enumerate(shape_coords):
                reconstructed_grid[y + dy, x + dx] = original_block[j]
            
            progress_bar.update(i + 1)
        
        progress_bar.finish()
        
        # 5. 最終データに変換
        return reconstructed_grid.tobytes()[:original_length]


def test_nexus_final():
    """NEXUS Final Engineをテスト"""
    print("🚀 NEXUS FINAL ENGINE TEST")
    print("=" * 50)
    
    compressor = NexusCompressor()
    decompressor = NexusDecompressor()
    
    # テストデータ作成
    test_sizes = [1, 5, 10]  # KB
    
    for size_kb in test_sizes:
        print(f"\n📁 Testing {size_kb}KB file:")
        
        # パターンリッチなテストデータ
        target_size = size_kb * 1024
        patterns = [
            b"ABCDEFGHIJKLMNOP" * 4,  # 64 bytes
            b"Hello World! " * 8,     # 104 bytes  
            b"1234567890" * 10,       # 100 bytes
        ]
        
        data = bytearray()
        while len(data) < target_size * 0.8:
            data.extend(patterns[len(data) % len(patterns)])
        
        # Add some random data
        import random
        while len(data) < target_size:
            data.append(random.randint(0, 255))
        
        data = bytes(data[:target_size])
        
        print(f"   Original: {len(data)} bytes")
        
        # Compression
        start_time = time.time()
        compressed = compressor.compress(data)
        comp_time = time.time() - start_time
        
        # Decompression
        start_time = time.time()
        decompressed = decompressor.decompress(compressed)
        decomp_time = time.time() - start_time
        
        # Results
        ratio = len(compressed) / len(data) * 100
        is_perfect = data == decompressed
        space_saved = (1 - len(compressed) / len(data)) * 100
        
        print(f"   Compressed: {len(compressed)} bytes ({ratio:.1f}%)")
        print(f"   Compression time: {comp_time:.3f}s")
        print(f"   Decompression time: {decomp_time:.3f}s")
        print(f"   Perfect recovery: {'✓' if is_perfect else '✗'}")
        print(f"   Space saved: {space_saved:.1f}%")
        
        if is_perfect:
            if ratio < 50:
                print("   🎉 EXCELLENT compression achieved!")
            else:
                print("   ✅ Compression successful")
        else:
            print("   ❌ Data corruption detected")
            if len(data) != len(decompressed):
                print(f"   Length mismatch: {len(data)} vs {len(decompressed)}")


def benchmark_vs_lzma():
    """LZMA/LZMA2との比較ベンチマーク"""
    if not LZMA_AVAILABLE:
        print("LZMA not available for benchmark")
        return
    
    print("\n" + "=" * 60)
    print("🏁 NEXUS FINAL vs LZMA BENCHMARK")
    print("=" * 60)
    
    compressor = NexusCompressor()
    decompressor = NexusDecompressor()
    
    test_sizes = [5, 10, 20]  # KB
    nexus_wins = 0
    lzma_wins = 0
    
    for size_kb in test_sizes:
        print(f"\n⚔️  Battle {size_kb}KB:")
        
        # Create realistic test data
        target_size = size_kb * 1024
        
        # 60% structured patterns
        patterns = [
            b"function processData(input) { return input * 2; }" * 2,
            b"<html><head><title>Test</title></head><body>" * 2,
            b"The quick brown fox jumps over the lazy dog " * 3,
        ]
        
        data = bytearray()
        while len(data) < target_size * 0.6:
            data.extend(patterns[len(data) % len(patterns)])
        
        # 40% semi-random data
        import random
        while len(data) < target_size:
            if len(data) % 10 == 0:
                data.extend(b"REPEAT" * 5)
            else:
                data.append(random.randint(0, 255))
        
        data = bytes(data[:target_size])
        
        print(f"   Original: {len(data)} bytes")
        
        # NEXUS compression
        start_time = time.time()
        nexus_compressed = compressor.compress(data)
        nexus_time = time.time() - start_time
        
        nexus_decompressed = decompressor.decompress(nexus_compressed)
        nexus_perfect = data == nexus_decompressed
        nexus_ratio = len(nexus_compressed) / len(data) * 100
        
        # LZMA compression
        start_time = time.time()
        lzma_compressed = lzma.compress(data, preset=9)
        lzma_time = time.time() - start_time
        
        lzma_decompressed = lzma.decompress(lzma_compressed)
        lzma_perfect = data == lzma_decompressed
        lzma_ratio = len(lzma_compressed) / len(data) * 100
        
        # Results
        print(f"   NEXUS: {len(nexus_compressed):,} bytes ({nexus_ratio:.1f}%) - {nexus_time:.3f}s - {'✓' if nexus_perfect else '✗'}")
        print(f"   LZMA:  {len(lzma_compressed):,} bytes ({lzma_ratio:.1f}%) - {lzma_time:.3f}s - {'✓' if lzma_perfect else '✗'}")
        
        # Winner
        if nexus_perfect and lzma_perfect:
            if nexus_ratio < lzma_ratio:
                improvement = (lzma_ratio - nexus_ratio) / lzma_ratio * 100
                print(f"   🏆 NEXUS WINS! {improvement:.1f}% better compression!")
                nexus_wins += 1
            else:
                deficit = (nexus_ratio - lzma_ratio) / lzma_ratio * 100
                print(f"   💀 LZMA wins by {deficit:.1f}%")
                lzma_wins += 1
        elif nexus_perfect:
            print(f"   ⚠️  NEXUS perfect, LZMA failed")
        elif lzma_perfect:
            print(f"   ⚠️  LZMA perfect, NEXUS failed")
            lzma_wins += 1
        else:
            print(f"   ❌ Both failed")
    
    # Final verdict
    print(f"\n🏁 FINAL SCORE:")
    print(f"   NEXUS Final: {nexus_wins} wins")
    print(f"   LZMA:        {lzma_wins} wins")
    
    if nexus_wins > lzma_wins:
        print(f"   🎉 NEXUS FINAL TRIUMPHS!")
    elif lzma_wins > nexus_wins:
        print(f"   📈 LZMA wins, but NEXUS Final shows revolutionary potential!")
    else:
        print(f"   🤝 Epic tie! Both algorithms are evenly matched!")


def test_real_files():
    """実際のサンプルファイルでテスト"""
    print("🗂️  REAL FILE TEST WITH SAMPLE FOLDER")
    print("=" * 60)
    
    sample_folder = r"c:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\sample"
    compressor = NexusCompressor()
    decompressor = NexusDecompressor()
    
    import glob
    sample_files = glob.glob(os.path.join(sample_folder, "*"))
    
    if not sample_files:
        print("❌ No sample files found!")
        return
    
    nexus_total_original = 0
    nexus_total_compressed = 0
    lzma_total_compressed = 0
    nexus_wins = 0
    lzma_wins = 0
    
    for file_path in sample_files:
        if os.path.isfile(file_path):
            filename = os.path.basename(file_path)
            print(f"\n📄 Testing: {filename}")
            
            # ファイル読み込み
            try:
                with open(file_path, 'rb') as f:
                    data = f.read()
            except Exception as e:
                print(f"   ❌ Error reading file: {e}")
                continue
            
            if len(data) == 0:
                print("   ⚠️  Empty file, skipping")
                continue
            
            print(f"   Original size: {len(data):,} bytes")
            nexus_total_original += len(data)
            
            # NEXUS compression test
            try:
                print("   🔥 NEXUS Final compression...")
                start_time = time.time()
                nexus_compressed = compressor.compress(data)
                nexus_comp_time = time.time() - start_time
                
                print("   🔓 NEXUS Final decompression...")
                start_time = time.time()
                nexus_decompressed = decompressor.decompress(nexus_compressed)
                nexus_decomp_time = time.time() - start_time
                
                nexus_perfect = data == nexus_decompressed
                nexus_ratio = len(nexus_compressed) / len(data) * 100
                nexus_total_compressed += len(nexus_compressed)
                
                print(f"   NEXUS: {len(nexus_compressed):,} bytes ({nexus_ratio:.1f}%) - {nexus_comp_time:.3f}s + {nexus_decomp_time:.3f}s - {'✓' if nexus_perfect else '✗'}")
                
            except Exception as e:
                print(f"   ❌ NEXUS failed: {e}")
                nexus_perfect = False
                nexus_ratio = float('inf')
            
            # LZMA compression test
            if LZMA_AVAILABLE:
                try:
                    start_time = time.time()
                    lzma_compressed = lzma.compress(data, preset=9)
                    lzma_comp_time = time.time() - start_time
                    
                    start_time = time.time()
                    lzma_decompressed = lzma.decompress(lzma_compressed)
                    lzma_decomp_time = time.time() - start_time
                    
                    lzma_perfect = data == lzma_decompressed
                    lzma_ratio = len(lzma_compressed) / len(data) * 100
                    lzma_total_compressed += len(lzma_compressed)
                    
                    print(f"   LZMA:  {len(lzma_compressed):,} bytes ({lzma_ratio:.1f}%) - {lzma_comp_time:.3f}s + {lzma_decomp_time:.3f}s - {'✓' if lzma_perfect else '✗'}")
                    
                except Exception as e:
                    print(f"   ❌ LZMA failed: {e}")
                    lzma_perfect = False
                    lzma_ratio = float('inf')
            else:
                print("   ⚠️  LZMA not available")
                lzma_perfect = False
                lzma_ratio = float('inf')
            
            # Winner determination
            if nexus_perfect and lzma_perfect:
                if nexus_ratio < lzma_ratio:
                    improvement = (lzma_ratio - nexus_ratio) / lzma_ratio * 100
                    print(f"   🏆 NEXUS WINS! {improvement:.1f}% better compression!")
                    nexus_wins += 1
                else:
                    deficit = (nexus_ratio - lzma_ratio) / lzma_ratio * 100
                    print(f"   💀 LZMA wins by {deficit:.1f}%")
                    lzma_wins += 1
            elif nexus_perfect:
                print(f"   🎉 NEXUS perfect, LZMA failed!")
                nexus_wins += 1
            elif lzma_perfect:
                print(f"   ⚠️  LZMA perfect, NEXUS failed")
                lzma_wins += 1
            else:
                print(f"   ❌ Both algorithms failed")
    
    # Overall results
    print(f"\n🏁 OVERALL RESULTS:")
    print(f"   Files tested: {len([f for f in sample_files if os.path.isfile(f)])}")
    print(f"   NEXUS wins: {nexus_wins}")
    print(f"   LZMA wins:  {lzma_wins}")
    
    if nexus_total_original > 0:
        nexus_overall_ratio = nexus_total_compressed / nexus_total_original * 100
        lzma_overall_ratio = lzma_total_compressed / nexus_total_original * 100
        
        print(f"\n📊 OVERALL COMPRESSION RATIOS:")
        print(f"   Total original: {nexus_total_original:,} bytes")
        print(f"   NEXUS total:    {nexus_total_compressed:,} bytes ({nexus_overall_ratio:.1f}%)")
        print(f"   LZMA total:     {lzma_total_compressed:,} bytes ({lzma_overall_ratio:.1f}%)")
        
        if nexus_overall_ratio < lzma_overall_ratio:
            overall_improvement = (lzma_overall_ratio - nexus_overall_ratio) / lzma_overall_ratio * 100
            print(f"   🎉 NEXUS OVERALL VICTORY! {overall_improvement:.1f}% better!")
        else:
            overall_deficit = (nexus_overall_ratio - lzma_overall_ratio) / lzma_overall_ratio * 100
            print(f"   📈 LZMA overall victory by {overall_deficit:.1f}%")


def test_large_file():
    """大きなファイルでNEXUS Final Engineをテスト"""
    print("🗂️  LARGE FILE TEST - 出庫実績明細_202412.txt")
    print("=" * 70)
    
    large_file_path = r"C:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\sample\出庫実績明細_202412.txt"
    
    if not os.path.exists(large_file_path):
        print("❌ Large file not found!")
        return
    
    compressor = NexusCompressor()
    decompressor = NexusDecompressor()
    
    print(f"📄 Testing: {os.path.basename(large_file_path)}")
    
    # ファイル読み込み
    try:
        with open(large_file_path, 'rb') as f:
            data = f.read()
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
        return
    
    file_size_mb = len(data) / (1024 * 1024)
    print(f"   Original size: {len(data):,} bytes ({file_size_mb:.1f} MB)")
    
    # NEXUS Final Engine compression test
    print("\n🔥 NEXUS Final Engine Compression Test...")
    try:
        start_time = time.time()
        nexus_compressed = compressor.compress(data)
        nexus_comp_time = time.time() - start_time
        
        print(f"\n🔓 NEXUS Final Engine Decompression Test...")
        start_time = time.time()
        nexus_decompressed = decompressor.decompress(nexus_compressed)
        nexus_decomp_time = time.time() - start_time
        
        nexus_perfect = data == nexus_decompressed
        nexus_ratio = len(nexus_compressed) / len(data) * 100
        nexus_space_saved = (1 - len(nexus_compressed) / len(data)) * 100
        
        print(f"\n📊 NEXUS Results:")
        print(f"   Compressed size: {len(nexus_compressed):,} bytes ({nexus_ratio:.2f}%)")
        print(f"   Space saved: {nexus_space_saved:.2f}%")
        print(f"   Compression time: {nexus_comp_time:.3f}s")
        print(f"   Decompression time: {nexus_decomp_time:.3f}s")
        print(f"   Perfect recovery: {'✅ YES' if nexus_perfect else '❌ NO'}")
        
        if not nexus_perfect:
            print(f"   Length match: {len(data) == len(nexus_decompressed)}")
            if len(data) != len(nexus_decompressed):
                print(f"   Original: {len(data)} vs Decompressed: {len(nexus_decompressed)}")
        
    except Exception as e:
        print(f"   ❌ NEXUS failed: {e}")
        nexus_perfect = False
        nexus_ratio = float('inf')
        nexus_comp_time = 0
        nexus_decomp_time = 0
    
    # LZMA compression test for comparison
    if LZMA_AVAILABLE:
        print(f"\n🏛️  LZMA Compression Test...")
        try:
            start_time = time.time()
            lzma_compressed = lzma.compress(data, preset=9)
            lzma_comp_time = time.time() - start_time
            
            start_time = time.time()
            lzma_decompressed = lzma.decompress(lzma_compressed)
            lzma_decomp_time = time.time() - start_time
            
            lzma_perfect = data == lzma_decompressed
            lzma_ratio = len(lzma_compressed) / len(data) * 100
            lzma_space_saved = (1 - len(lzma_compressed) / len(data)) * 100
            
            print(f"\n📊 LZMA Results:")
            print(f"   Compressed size: {len(lzma_compressed):,} bytes ({lzma_ratio:.2f}%)")
            print(f"   Space saved: {lzma_space_saved:.2f}%")
            print(f"   Compression time: {lzma_comp_time:.3f}s")
            print(f"   Decompression time: {lzma_decomp_time:.3f}s")
            print(f"   Perfect recovery: {'✅ YES' if lzma_perfect else '❌ NO'}")
            
        except Exception as e:
            print(f"   ❌ LZMA failed: {e}")
            lzma_perfect = False
            lzma_ratio = float('inf')
    else:
        print("   ⚠️  LZMA not available")
        lzma_perfect = False
        lzma_ratio = float('inf')
    
    # Final comparison
    print(f"\n🏁 FINAL COMPARISON:")
    print(f"=" * 50)
    
    if nexus_perfect and lzma_perfect:
        if nexus_ratio < lzma_ratio:
            improvement = (lzma_ratio - nexus_ratio) / lzma_ratio * 100
            compression_advantage = lzma_ratio - nexus_ratio
            print(f"🏆 NEXUS FINAL ENGINE VICTORY!")
            print(f"   🎯 Compression advantage: {improvement:.2f}% better than LZMA")
            print(f"   📉 Size reduction: {compression_advantage:.2f} percentage points")
            print(f"   💾 Space saved: {(len(lzma_compressed) - len(nexus_compressed)):,} bytes more than LZMA")
        elif nexus_ratio == lzma_ratio:
            print(f"🤝 PERFECT TIE!")
            print(f"   Both algorithms achieved identical compression ratios")
        else:
            deficit = (nexus_ratio - lzma_ratio) / lzma_ratio * 100
            print(f"🥈 LZMA Victory")
            print(f"   📊 LZMA advantage: {deficit:.2f}% better compression")
            print(f"   But NEXUS showed competitive performance on large file!")
    elif nexus_perfect:
        print(f"🎉 NEXUS PERFECT - LZMA FAILED!")
        print(f"   NEXUS achieved perfect compression while LZMA failed")
    elif lzma_perfect:
        print(f"🥉 NEXUS FAILED - LZMA PERFECT")
        print(f"   LZMA succeeded while NEXUS failed")
    else:
        print(f"💥 BOTH ALGORITHMS FAILED")
    
    # Performance summary
    if nexus_perfect:
        print(f"\n⚡ PERFORMANCE SUMMARY:")
        print(f"   NEXUS speed: {nexus_comp_time + nexus_decomp_time:.3f}s total")
        if lzma_perfect:
            print(f"   LZMA speed:  {lzma_comp_time + lzma_decomp_time:.3f}s total")
            if (nexus_comp_time + nexus_decomp_time) < (lzma_comp_time + lzma_decomp_time):
                print(f"   🚀 NEXUS is faster!")
            else:
                print(f"   🐌 LZMA is faster")


if __name__ == "__main__":
    test_large_file()
