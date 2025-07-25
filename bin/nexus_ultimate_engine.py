#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEXUS HYPER ENGINE
The ultimate evolution, achieving native C/Fortran speeds via Numba JIT compilation
for the core tiling algorithm, shattering previous performance bottlenecks.
"""

import sys
import json
import math
import heapq
import collections
import time
import os
import glob
import random
from typing import List, Tuple, Dict, Any, Optional
import concurrent.futures

# --- 依存ライブラリのチェックとインストール ---
def install_package(package):
    """指定されたパッケージのインストールを試みる"""
    print(f"Info: {package} not found. Attempting to install...")
    try:
        import subprocess
        # ユーザーに確認を求めるプロンプト
        response = input(f"   > This script requires '{package}'. Do you want to install it now? (y/n): ").lower()
        if response == 'y':
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"Successfully installed {package}.")
            return True
        else:
            print(f"Installation cancelled by user.")
            return False
    except Exception as e:
        print(f"Error: Failed to install {package}. Please install it manually: pip install {package}")
        print(f"Details: {e}")
        return False

try:
    import lzma
    LZMA_AVAILABLE = True
except ImportError:
    LZMA_AVAILABLE = False
    print("Warning: lzma module not found. LZMA compression will be disabled.")

try:
    import numpy as np
except ImportError:
    if not install_package('numpy'): sys.exit(1)
    import numpy as np

try:
    import numba
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    if not install_package('numba'): sys.exit(1)
    import numba
    from numba import jit
    NUMBA_AVAILABLE = True

# --- 静的順列辞書プロバイダ ---
class StaticPermutationProvider:
    """ブロックサイズごとの順列を静的に管理し、IDとの相互変換を行う"""
    _cache: Dict[int, Tuple[Dict[Tuple[int, ...], int], Dict[int, Tuple[int, ...]]]] = {}
    @classmethod
    def get_provider_for_size(cls, size: int):
        if size not in cls._cache:
            if size > 8: raise ValueError("Permutation size too large for static provider.")
            from itertools import permutations
            # 順列を生成し、ソートして決定的な順序にする
            perms = sorted(list(permutations(range(size))))
            perm_to_id = {perm: i for i, perm in enumerate(perms)}
            id_to_perm = {i: perm for i, perm in enumerate(perms)}
            cls._cache[size] = (perm_to_id, id_to_perm)
        return cls._cache[size]
    @classmethod
    def get_perm_id(cls, perm: Tuple[int, ...]) -> int:
        perm_to_id, _ = cls.get_provider_for_size(len(perm))
        return perm_to_id.get(perm, 0)
    @classmethod
    def get_perm_from_id(cls, perm_id: int, size: int) -> Tuple[int, ...]:
        _, id_to_perm = cls.get_provider_for_size(size)
        return id_to_perm.get(perm_id, tuple(range(size)))

# --- Huffmanエンコーダ ---
class HuffmanEncoder:
    """BlueprintのIDストリームをHuffman符号化/復号化するクラス"""
    def encode(self, data: List[int]) -> Tuple[Dict[str, str], str]:
        if not data: return {}, ""
        frequency = collections.Counter(data)
        if len(frequency) == 1:
            symbol = list(frequency.keys())[0]
            return {str(symbol): "0"}, "0" * len(data)
        heap = [[weight, [symbol, ""]] for symbol, weight in frequency.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            lo, hi = heapq.heappop(heap), heapq.heappop(heap)
            for pair in lo[1:]: pair[1] = '0' + pair[1]
            for pair in hi[1:]: pair[1] = '1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        # JSON互換性のためにキーを文字列に変換
        huffman_tree = {str(p[0]): p[1] for p in sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))}
        return huffman_tree, "".join([huffman_tree[str(d)] for d in data])

    def decode(self, encoded_data: str, huffman_tree: Dict[str, Any]) -> List[int]:
        if not encoded_data: return []
        # JSONから読み込んだキー（文字列）を整数に変換して逆引きツリーを作成
        reverse_tree = {v: int(k) for k, v in huffman_tree.items()}
        decoded_data, current_code = [], ""
        for bit in encoded_data:
            current_code += bit
            if current_code in reverse_tree:
                decoded_data.append(reverse_tree[current_code])
                current_code = ""
        return decoded_data

# --- 差分符号化 ---
def _apply_delta_encoding(stream: List[int]) -> List[int]:
    """差分符号化を適用"""
    if not stream: return []
    return [stream[0]] + [stream[i] - stream[i-1] for i in range(1, len(stream))]

def _reverse_delta_encoding(delta_stream: List[int]) -> List[int]:
    """差分符号化を逆変換"""
    if not delta_stream: return []
    original_stream = [delta_stream[0]]
    for i in range(1, len(delta_stream)):
        original_stream.append(original_stream[-1] + delta_stream[i])
    return original_stream

# --- プログレスバー ---
class ProgressBar:
    """コンソールに進捗バーを表示するクラス"""
    def __init__(self, total: int, description: str = "", width: int = 40):
        self.total = total
        self.description = description
        self.width = width
        self.current = 0
        self.start_time = time.time()
    def update(self, current: int):
        self.current = current
        percentage = min(100, (self.current / self.total) * 100) if self.total > 0 else 100
        filled = int(self.width * self.current // self.total) if self.total > 0 else self.width
        bar = '█' * filled + '░' * (self.width - filled)
        elapsed = time.time() - self.start_time
        eta_str = f" ETA: {elapsed * (self.total - self.current) / self.current:.1f}s" if self.current > 0 else ""
        print(f'\r{self.description} |{bar}| {percentage:.1f}%{eta_str}', end='', flush=True)
    def finish(self):
        elapsed = time.time() - self.start_time
        print(f'\r{self.description} |{"█" * self.width}| 100.0% Complete in {elapsed:.2f}s')

# --- ⚙️ 圧縮率向上の核：形状の拡充と回転・反転の自動生成 ---
def _generate_transformed_shapes(base_shapes):
    """基本形状から回転・反転バリエーションを自動生成する"""
    transformed_shapes = {}
    shape_id_counter = 0
    for _, (coords, size) in base_shapes.items():
        seen_coords = set()
        current_coords = list(coords)
        for _ in range(4): # 4方向回転
            # 反転なし
            max_x_val = max(x for y, x in current_coords) if current_coords else 0
            normalized_coords = tuple(sorted(tuple(c) for c in current_coords))
            if normalized_coords not in seen_coords:
                transformed_shapes[shape_id_counter] = (normalized_coords, size)
                seen_coords.add(normalized_coords)
                shape_id_counter += 1
            # X軸反転
            flipped_coords = tuple(sorted(tuple((y, max_x_val - x)) for y, x in current_coords))
            if flipped_coords not in seen_coords:
                transformed_shapes[shape_id_counter] = (flipped_coords, size)
                seen_coords.add(flipped_coords)
                shape_id_counter += 1
            # 90度回転: (y, x) -> (x, -y)
            current_coords = [(x, -y) for y, x in current_coords]
            # 座標を正規化(左上が0,0になるように)
            min_x = min(x for y, x in current_coords) if current_coords else 0
            min_y = min(y for y, x in current_coords) if current_coords else 0
            current_coords = sorted([(y - min_y, x - min_x) for y, x in current_coords])
    return transformed_shapes

BASE_POLYOMINO_SHAPES = {
    0: (((0, 0),), 1),
    1: (((0, 0), (0, 1)), 2),
    2: (((0, 0), (0, 1), (1, 0), (1, 1)), 4),
    3: (((0, 0), (0, 1), (0, 2)), 3),
    4: (((0, 0), (0, 1), (0, 2), (0, 3)), 4),
    5: (((0, 0), (1, 0), (2, 0), (2, 1)), 4), # L-4
    6: (((0,0), (0,1), (0,2), (1,1)), 4), # T-4
    7: (((0,0), (0,1), (0,2), (0,3), (0,4)), 5), # I-5 (Penta)
    8: (((0,0), (0,1), (1,0), (1,1), (2,0)), 5), # P-5
}
POLYOMINO_SHAPES = _generate_transformed_shapes(BASE_POLYOMINO_SHAPES)
SHAPE_IDS = sorted(POLYOMINO_SHAPES.keys())

# Numbaが扱いやすいように形状データをタプルのタプルに変換
SHAPES_FOR_NUMBA = tuple(
    # 各座標もタプルに変換
    (tuple(numba.types.UniTuple(numba.types.int64, 2)(c) for c in shape[0]), shape[1])
    for shape in POLYOMINO_SHAPES.values()
)

# --- 🚀 高速化の核：Numba JITコンパイルされたタイリングカーネル ---
@jit(nopython=True, fastmath=True, cache=True)
def _tiling_kernel_numba(data_chunk: np.ndarray, initial_known_groups: numba.typed.Dict) -> List[Tuple[int, Tuple[int, ...], int, int]]:
    """超高速タイリングカーネル。Numbaによってマシンコードにコンパイルされる"""
    grid_height, grid_width = data_chunk.shape
    coverage_grid = np.zeros((grid_height, grid_width), dtype=numba.boolean)
    
    raw_blueprint = []
    
    y, x = 0, 0
    while True:
        if y >= grid_height: break
        if coverage_grid[y, x]:
            x += 1
            if x >= grid_width:
                x = 0; y += 1
            continue
        
        best_shape_id, min_cost = -1, 2
        best_block_tuple = (np.uint8(0),) 
        
        for shape_id in range(len(SHAPES_FOR_NUMBA)):
            shape_coords, _ = SHAPES_FOR_NUMBA[shape_id]
            
            # 形状のバウンディングボックスを計算
            max_y_offset = 0
            max_x_offset = 0
            for dy, dx in shape_coords:
                if dy > max_y_offset: max_y_offset = dy
                if dx > max_x_offset: max_x_offset = dx

            if y + max_y_offset >= grid_height or x + max_x_offset >= grid_width: continue
            
            is_occupied = False
            for dy, dx in shape_coords:
                if coverage_grid[y + dy, x + dx]:
                    is_occupied = True
                    break
            if is_occupied: continue

            block_list = [data_chunk[y + dy, x + dx] for dy, dx in shape_coords]
            current_block = tuple(block_list)
            
            normalized_block = tuple(sorted(current_block))
            cost = 0 if normalized_block in initial_known_groups else 1
            
            if cost < min_cost:
                min_cost = cost
                best_shape_id = shape_id
                best_block_tuple = current_block
                if cost == 0: break
        
        if best_shape_id == -1:
            best_shape_id = 0
            best_block_tuple = (data_chunk[y, x],)
        
        raw_blueprint.append((best_shape_id, best_block_tuple, y, x))
        
        shape_coords, _ = SHAPES_FOR_NUMBA[best_shape_id]
        for dy, dx in shape_coords:
            coverage_grid[y + dy, x + dx] = True
            
    return raw_blueprint

def _tile_chunk_worker(args: Tuple[int, np.ndarray, Dict[Tuple[int, ...], int]]) -> Tuple[int, List, List]:
    """Python側のワーカー関数。Numbaカーネルを呼び出し、結果をPythonオブジェクトに変換する"""
    chunk_id, data_chunk, global_unique_groups = args
    
    numba_dict = numba.typed.Dict.empty(
        key_type=numba.types.containers.Tuple(numba.types.uint8, -1),
        value_type=numba.types.int64
    )
    for k, v in global_unique_groups.items():
        # キーをuint8のタプルに変換
        numba_dict[k] = v

    raw_blueprint = _tiling_kernel_numba(data_chunk, numba_dict)
    
    local_new_groups = []
    for _, block_tuple, _, _ in raw_blueprint:
        normalized = tuple(sorted(block_tuple))
        if normalized not in global_unique_groups and normalized not in [g for g,s in local_new_groups]:
            local_new_groups.append((normalized, len(block_tuple)))

    return chunk_id, raw_blueprint, local_new_groups

class NexusCompressor:
    """NEXUSハイパーエンジン圧縮クラス"""
    def __init__(self, num_workers: Optional[int] = None):
        self.num_workers = num_workers if num_workers else (os.cpu_count() or 1)
        print("Initializing and JIT-compiling NEXUS kernel...")
        dummy_array = np.zeros((32, 32), dtype=np.uint8)
        dummy_dict = numba.typed.Dict.empty(
            key_type=numba.types.containers.Tuple(numba.types.uint8, -1),
            value_type=numba.types.int64
        )
        _tiling_kernel_numba(dummy_array, dummy_dict)
        print("Kernel compiled and ready.")

    def compress(self, data: bytes) -> bytes:
        if not data: return b''
        
        print(f"🔥 NEXUS Hyper Engine: Compression (using {self.num_workers} workers)")
        original_length = len(data)

        grid_width = math.ceil(math.sqrt(original_length))
        grid_height = math.ceil(original_length / grid_width)
        padded_data = data.ljust(grid_width * grid_height, b'\0')
        data_grid = np.frombuffer(padded_data, dtype=np.uint8).reshape((grid_height, grid_width))
        print(f"   Grid: {grid_height}x{grid_width}, Shapes: {len(POLYOMINO_SHAPES)} variations")

        chunk_size = 256
        chunks_y = (grid_height + chunk_size - 1) // chunk_size
        chunks_x = (grid_width + chunk_size - 1) // chunk_size
        
        global_unique_groups: Dict[Tuple[np.uint8, ...], int] = {}
        
        print("   Sampling initial unique groups...")
        sample_regions = min(chunks_y * chunks_x, 32)
        for _ in range(sample_regions):
            if grid_height > 16 and grid_width > 16:
                sy, sx = random.randint(0, grid_height-16), random.randint(0, grid_width-16)
                sample = data_grid[sy:sy+16, sx:sx+16].flatten()
                if len(sample) > 0:
                    normalized = tuple(sorted(sample))
                    if normalized not in global_unique_groups:
                        global_unique_groups[normalized] = len(global_unique_groups)
        
        print(f"   Splitting into {chunks_y * chunks_x} chunks for parallel processing...")
        tasks = []
        for cy in range(chunks_y):
            for cx in range(chunks_x):
                y_start, y_end = cy * chunk_size, min((cy + 1) * chunk_size, grid_height)
                x_start, x_end = cx * chunk_size, min((cx + 1) * chunk_size, grid_width)
                tasks.append(((cy * chunks_x) + cx, data_grid[y_start:y_end, x_start:x_end], global_unique_groups.copy()))

        blueprint_chunks = [None] * len(tasks)
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(_tile_chunk_worker, task) for task in tasks]
            progress = ProgressBar(len(futures), "Parallel Tiling")
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                chunk_id, local_bp, local_new = future.result()
                blueprint_chunks[chunk_id] = (local_bp, local_new)
                progress.update(i + 1)
            progress.finish()

        print("   Merging parallel results...")
        full_blueprint = []
        current_group_id = len(global_unique_groups)
        
        for _, local_new_groups in blueprint_chunks:
            for normalized, size in local_new_groups:
                if normalized not in global_unique_groups:
                    global_unique_groups[normalized] = current_group_id
                    current_group_id += 1
        
        for chunk_id, (local_bp, _) in enumerate(blueprint_chunks):
            cy, cx = chunk_id // chunks_x, chunk_id % chunks_x
            offset_y, offset_x = cy * chunk_size, cx * chunk_size
            for shape_id, block_tuple, y, x in local_bp:
                normalized = tuple(sorted(block_tuple))
                group_id = global_unique_groups[normalized]
                perm_map = tuple(normalized.index(val) for val in block_tuple)
                perm_id = StaticPermutationProvider.get_perm_id(perm_map)
                full_blueprint.append((shape_id, group_id, perm_id, y + offset_y, x + offset_x))

        unique_groups_list = sorted(global_unique_groups.keys(), key=lambda k: global_unique_groups[k])
        print(f"   Blueprint: {len(full_blueprint)} blocks, {len(unique_groups_list)} unique groups")

        print("   Applying Context-Mixed Encoding...")
        streams_by_context: Dict[int, Dict[str, List]] = collections.defaultdict(lambda: {"g_id": [], "p_id": []})
        pos_y_stream, pos_x_stream = [], []

        for shape_id, group_id, perm_id, y, x in full_blueprint:
            streams_by_context[shape_id]["g_id"].append(group_id)
            streams_by_context[shape_id]["p_id"].append(perm_id)
            pos_y_stream.append(y)
            pos_x_stream.append(x)
        
        delta_pos_y = _apply_delta_encoding(pos_y_stream)
        delta_pos_x = _apply_delta_encoding(pos_x_stream)

        huffman_trees, encoded_streams = {}, {}
        h_encoder = HuffmanEncoder()
        huffman_trees['pos_y'], encoded_streams['pos_y'] = h_encoder.encode(delta_pos_y)
        huffman_trees['pos_x'], encoded_streams['pos_x'] = h_encoder.encode(delta_pos_x)
        
        huffman_trees['context'], encoded_streams['context'] = {}, {}
        for context_id, streams in streams_by_context.items():
            delta_g_id = _apply_delta_encoding(streams['g_id'])
            delta_p_id = _apply_delta_encoding(streams['p_id'])
            g_tree, g_data = h_encoder.encode(delta_g_id)
            p_tree, p_data = h_encoder.encode(delta_p_id)
            huffman_trees['context'][str(context_id)] = {'g': g_tree, 'p': p_tree}
            encoded_streams['context'][str(context_id)] = {'g': g_data, 'p': p_data}

        payload = {
            "header": {"algo": "NEXUS_vHyper", "orig_len": original_length, "grid_dims": (grid_height, grid_width)},
            "unique_groups": [[int(v) for v in g] for g in unique_groups_list],
            "blueprint_order": [bp[0] for bp in full_blueprint],
            "huffman_trees": huffman_trees, "encoded_streams": encoded_streams,
        }
        
        print("   Finalizing with LZMA...")
        serialized_payload = json.dumps(payload, separators=(',', ':')).encode('utf-8')
        if LZMA_AVAILABLE:
            return lzma.compress(serialized_payload, preset=9, check=lzma.CHECK_NONE)
        return serialized_payload

class NexusDecompressor:
    """NEXUSハイパーエンジン解凍クラス"""
    def __init__(self):
        self.huffman_encoder = HuffmanEncoder()

    def decompress(self, compressed_data: bytes) -> bytes:
        if not compressed_data: return b''
        print("🔓 NEXUS Hyper Engine: Decompression")
        
        if LZMA_AVAILABLE:
            serialized_payload = lzma.decompress(compressed_data)
        else:
            serialized_payload = compressed_data
        payload = json.loads(serialized_payload.decode('utf-8'))

        header = payload["header"]
        original_length = header["orig_len"]
        grid_height, grid_width = header["grid_dims"]
        print(f"   Target: {original_length} bytes, Grid: {grid_height}x{grid_width}")

        print("   Decoding streams with Context-Mixing...")
        h_trees = payload["huffman_trees"]
        e_streams = payload["encoded_streams"]
        h_decoder = self.huffman_encoder

        delta_pos_y = h_decoder.decode(e_streams['pos_y'], h_trees['pos_y'])
        delta_pos_x = h_decoder.decode(e_streams['pos_x'], h_trees['pos_x'])
        
        streams_by_context = collections.defaultdict(lambda: {"g_id": [], "p_id": []})
        for str_context_id, streams in e_streams['context'].items():
            context_id = int(str_context_id)
            trees = h_trees['context'][str_context_id]
            streams_by_context[context_id]['g_id'] = h_decoder.decode(streams['g'], trees['g'])
            streams_by_context[context_id]['p_id'] = h_decoder.decode(streams['p'], trees['p'])
        
        pos_y_stream = _reverse_delta_encoding(delta_pos_y)
        pos_x_stream = _reverse_delta_encoding(delta_pos_x)
        
        context_streams = {}
        for context_id, streams in streams_by_context.items():
            context_streams[context_id] = {
                'g_id': _reverse_delta_encoding(streams['g_id']),
                'p_id': _reverse_delta_encoding(streams['p_id'])
            }
        
        print("   Reconstructing data grid from blueprint...")
        unique_groups = [tuple(g) for g in payload["unique_groups"]]
        shape_id_order = payload["blueprint_order"]
        reconstructed_grid = np.zeros((grid_height, grid_width), dtype=np.uint8)
        context_counters = collections.defaultdict(int)
        
        total_blocks = len(shape_id_order)
        progress = ProgressBar(total_blocks, "Rebuilding")

        for i in range(total_blocks):
            shape_id = shape_id_order[i]
            counter = context_counters[shape_id]
            group_id = context_streams[shape_id]['g_id'][counter]
            perm_id = context_streams[shape_id]['p_id'][counter]
            y, x = pos_y_stream[i], pos_x_stream[i]

            _, shape_size = POLYOMINO_SHAPES[shape_id]
            shape_coords, _ = SHAPES_FOR_NUMBA[shape_id]
            sorted_group = unique_groups[group_id]
            perm_map = StaticPermutationProvider.get_perm_from_id(perm_id, shape_size)
            
            original_block = [0] * shape_size
            # sorted_groupからperm_mapを使ってoriginal_blockを復元
            # This is the inverse of perm_map = tuple(normalized_block.index(val) for val in block_tuple)
            temp_sorted = list(sorted_group)
            for original_pos in range(shape_size):
                sorted_pos = perm_map.index(original_pos)
                original_block[original_pos] = temp_sorted[sorted_pos]

            for j, (dy, dx) in enumerate(shape_coords):
                reconstructed_grid[y + dy, x + dx] = original_block[j]
                
            context_counters[shape_id] += 1
            if (i + 1) % 10000 == 0: progress.update(i + 1)
        progress.finish()
        
        return reconstructed_grid.tobytes()[:original_length]

def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  Compress:   python nexus_hyper_engine.py c <input_file> <output.nxz>")
        print("  Decompress: python nexus_hyper_engine.py d <input.nxz> <output_file>")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == 'c':
        input_path, output_path = sys.argv[2], sys.argv[3]
        try:
            with open(input_path, 'rb') as f_in: data = f_in.read()
        except FileNotFoundError:
            print(f"Error: Input file not found at '{input_path}'")
            sys.exit(1)
            
        start_time = time.time()
        compressor = NexusCompressor()
        compressed = compressor.compress(data)
        end_time = time.time()
        
        with open(output_path, 'wb') as f_out: f_out.write(compressed)
        
        print(f"\n✅ Compression complete in {end_time - start_time:.2f}s")
        print(f"   Original:      {len(data):,} bytes")
        print(f"   Compressed:    {len(compressed):,} bytes")
        if len(data) > 0:
            print(f"   Ratio:         {len(compressed)/len(data):.2%}")
            speed_mbps = (len(data) / (1024*1024)) / (end_time - start_time) if end_time > start_time else 0
            print(f"   Speed:         {speed_mbps:.2f} MB/s")

    elif command == 'd':
        input_path, output_path = sys.argv[2], sys.argv[3]
        try:
            with open(input_path, 'rb') as f_in: data = f_in.read()
        except FileNotFoundError:
            print(f"Error: Input file not found at '{input_path}'")
            sys.exit(1)

        start_time = time.time()
        decompressor = NexusDecompressor()
        decompressed = decompressor.decompress(data)
        end_time = time.time()
        
        with open(output_path, 'wb') as f_out: f_out.write(decompressed)
        
        print(f"\n✅ Decompression complete in {end_time - start_time:.2f}s")
        print(f"   Decompressed: {len(decompressed):,} bytes")
        speed_mbps = (len(decompressed) / (1024*1024)) / (end_time - start_time) if end_time > start_time else 0
        print(f"   Speed:        {speed_mbps:.2f} MB/s")
    else:
        print(f"Error: Unknown command '{command}'")

if __name__ == "__main__":
    # Numba JITコンパイルは最初の実行時に時間がかかるため、
    # __main__ブロックを保護してマルチプロセッシングの再帰実行を防ぐ
    main()