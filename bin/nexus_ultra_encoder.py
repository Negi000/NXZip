"""
NEXUS超高効率圧縮エンコーダー
差分圧縮、統計圧縮、適応的アルゴリズムを統合
"""

import struct
import hashlib
import zlib
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter


class UltraCompactNEXUSEncoder:
    """NEXUS超高効率エンコーダー"""
    
    MAGIC_HEADER = b'NXU1'  # NEXUS Ultra v1
    VERSION = 1
    
    @staticmethod
    def encode_nexus_state(nexus_state) -> bytes:
        """超高効率NEXUS状態エンコード（タイムアウト対策）"""
        import time
        start_time = time.time()
        timeout_limit = 60.0  # 60秒でタイムアウト
        
        print("📦 超高効率NEXUS状態エンコード中...")
        
        try:
            # 最適化された中間形式の作成
            optimized_data = UltraCompactNEXUSEncoder._create_optimized_representation(nexus_state)
            
            # タイムアウトチェック
            if time.time() - start_time > timeout_limit:
                raise TimeoutError("エンコード処理タイムアウト")
            
            # 複数の圧縮手法を試行し、最良の結果を選択
            candidates = []
            
            # 1. 差分圧縮 + zlib（高速）
            try:
                if time.time() - start_time < timeout_limit:
                    diff_compressed = UltraCompactNEXUSEncoder._differential_encoding(optimized_data)
                    final_compressed = zlib.compress(diff_compressed, level=6)  # レベル下げて高速化
                    candidates.append((1, final_compressed))
                    print(f"  差分+zlib: {len(final_compressed)} bytes")
            except Exception as e:
                print(f"  差分圧縮失敗: {e}")
            
            # 2. 統計圧縮（条件付き）
            try:
                if time.time() - start_time < timeout_limit and len(optimized_data) < 1024 * 1024:  # 1MB未満のみ
                    stats_compressed = UltraCompactNEXUSEncoder._statistical_encoding(optimized_data)
                    candidates.append((2, stats_compressed))
                    print(f"  統計圧縮: {len(stats_compressed)} bytes")
            except Exception as e:
                print(f"  統計圧縮失敗: {e}")
            
            # 3. ハイブリッド圧縮（小ファイルのみ）
            try:
                if time.time() - start_time < timeout_limit and len(optimized_data) < 512 * 1024:  # 512KB未満のみ
                    hybrid_compressed = UltraCompactNEXUSEncoder._hybrid_encoding(optimized_data)
                    candidates.append((3, hybrid_compressed))
                    print(f"  ハイブリッド: {len(hybrid_compressed)} bytes")
            except Exception as e:
                print(f"  ハイブリッド圧縮失敗: {e}")
            
            # 最良の結果を選択
            if not candidates:
                # フォールバック：基本zlib圧縮
                fallback_compressed = zlib.compress(optimized_data, level=6)
                candidates.append((1, fallback_compressed))
                print(f"  フォールバック圧縮: {len(fallback_compressed)} bytes")
            
            best_method, best_compressed = min(candidates, key=lambda x: len(x[1]))
            
            # 最終エンコード
            encoded = bytearray()
            encoded.extend(UltraCompactNEXUSEncoder.MAGIC_HEADER)
            encoded.append(UltraCompactNEXUSEncoder.VERSION)
            encoded.append(best_method)  # 圧縮手法
            encoded.extend(struct.pack('<I', len(best_compressed)))
            encoded.extend(best_compressed)
            
            processing_time = time.time() - start_time
            print(f"  最適手法: {best_method}, 最終サイズ: {len(encoded)} bytes, 処理時間: {processing_time:.3f}s")
            return bytes(encoded)
            
        except TimeoutError:
            print("⏰ エンコード処理タイムアウト - 緊急フォールバック")
            # 緊急フォールバック：基本圧縮のみ
            import pickle
            basic_data = pickle.dumps(nexus_state)
            fallback_compressed = zlib.compress(basic_data, level=1)
            
            encoded = bytearray()
            encoded.extend(b'NXU_TIMEOUT')  # 特別ヘッダー
            encoded.extend(struct.pack('<I', len(fallback_compressed)))
            encoded.extend(fallback_compressed)
            
            return bytes(encoded)
    
    @staticmethod
    def _create_optimized_representation(nexus_state) -> bytes:
        """最適化された中間表現の作成"""
        data = bytearray()
        
        # メタデータ（最小限）
        original_size = nexus_state.compression_metadata.get('original_size', 0)
        width, height = nexus_state.grid_dimensions
        data.extend(struct.pack('<I', original_size))
        data.extend(struct.pack('<I', width))  # H -> I に変更（範囲拡張）
        data.extend(struct.pack('<I', height))  # H -> I に変更（範囲拡張）
        
        # グループ統計の圧縮
        groups = nexus_state.original_groups
        
        # 大量グループの場合はサンプリング
        if len(groups) > 32768:  # 32K以上の場合
            print(f"  大量グループ検出: {len(groups)}個 -> サンプリング実行")
            # 重要なグループのみ抽出（頻出パターン優先）
            groups = UltraCompactNEXUSEncoder._sample_important_groups(groups)
            print(f"  サンプリング後: {len(groups)}個")
        
        unique_elements = set()
        for group in groups:
            unique_elements.update(group.elements)
        
        # 要素辞書作成（頻出要素を短いインデックスに）
        element_counts = Counter()
        for group in groups:
            element_counts.update(group.elements)
        
        sorted_elements = [elem for elem, _ in element_counts.most_common()]
        element_to_index = {elem: i for i, elem in enumerate(sorted_elements)}
        
        # 辞書サイズ（制限）
        dict_size = min(len(sorted_elements), 255)
        data.append(dict_size)
        for elem in sorted_elements[:dict_size]:
            data.append(elem)
        
        # グループデータ（範囲制限）
        group_count = min(len(groups), 65535)  # 16bit範囲内
        data.extend(struct.pack('<H', group_count))
        
        for i, group in enumerate(groups[:group_count]):
            # 形状（圧縮）
            shape_code = UltraCompactNEXUSEncoder._encode_shape_compact(group.shape)
            data.append(shape_code)
            
            # 要素（インデックス使用、制限）
            elements_count = min(len(group.elements), 255)
            data.append(elements_count)
            for elem in group.elements[:elements_count]:
                index = element_to_index.get(elem, dict_size)
                data.append(min(index, 255))
            
            # 位置（差分エンコード、制限）
            positions_count = min(len(group.positions), 255)
            data.append(positions_count)
            last_row, last_col = 0, 0
            for row, col in group.positions[:positions_count]:
                # 差分計算（オーバーフロー対策）
                dr = max(0, min(255, (row - last_row + 128) % 256))
                dc = max(0, min(255, (col - last_col + 128) % 256))
                data.extend([dr, dc])
                last_row, last_col = row, col
        
        return bytes(data)
    
    @staticmethod
    def _sample_important_groups(groups: list) -> list:
        """重要なグループのサンプリング（高速化・無限ループ対策）"""
        if len(groups) <= 32768:
            return groups
        
        print(f"  大量グループ高速サンプリング: {len(groups)}個")
        
        # 高速サンプリング（重要度計算簡略化）
        sampled_groups = []
        
        # 1. 先頭から一定間隔でサンプリング（確実に終了）
        step = max(1, len(groups) // 16384)  # 16K個を目標
        for i in range(0, len(groups), step):
            if len(sampled_groups) >= 16384:
                break
            sampled_groups.append(groups[i])
        
        # 2. ランダムサンプリングで残りを補完
        import random
        remaining = 32768 - len(sampled_groups)
        if remaining > 0 and len(groups) > len(sampled_groups):
            available = [g for g in groups if g not in sampled_groups]
            if available:
                additional = random.sample(available, min(remaining, len(available)))
                sampled_groups.extend(additional)
        
        print(f"  サンプリング完了: {len(sampled_groups)}個")
        return sampled_groups[:32768]  # 確実に制限内
    
    @staticmethod
    def _differential_encoding(data: bytes) -> bytes:
        """差分エンコード"""
        if len(data) < 2:
            return data
        
        result = bytearray([data[0]])  # 最初の値はそのまま
        
        for i in range(1, len(data)):
            diff = (data[i] - data[i-1] + 256) % 256
            result.append(diff)
        
        return bytes(result)
    
    @staticmethod
    def _statistical_encoding(data: bytes) -> bytes:
        """統計ベース圧縮"""
        if len(data) < 4:
            return data
        
        # 頻度分析
        freq = Counter(data)
        sorted_by_freq = [byte for byte, _ in freq.most_common()]
        
        # ハフマン風の可変長エンコード（簡易版）
        encoded = bytearray()
        
        # 頻度テーブル
        encoded.append(min(len(sorted_by_freq), 256))
        for byte in sorted_by_freq[:256]:
            encoded.append(byte)
        
        # データを頻度順インデックスでエンコード
        byte_to_index = {byte: i for i, byte in enumerate(sorted_by_freq)}
        
        for byte in data:
            index = byte_to_index.get(byte, 255)
            encoded.append(index)
        
        return bytes(encoded)
    
    @staticmethod
    def _hybrid_encoding(data: bytes) -> bytes:
        """ハイブリッド圧縮（RLE + 差分 + zlib）"""
        # Step 1: RLE
        rle_data = UltraCompactNEXUSEncoder._simple_rle(data)
        
        # Step 2: 差分エンコード
        diff_data = UltraCompactNEXUSEncoder._differential_encoding(rle_data)
        
        # Step 3: zlib圧縮
        return zlib.compress(diff_data, level=6)
    
    @staticmethod
    def _simple_rle(data: bytes) -> bytes:
        """シンプルRLE圧縮"""
        if len(data) < 2:
            return data
        
        result = bytearray()
        i = 0
        
        while i < len(data):
            current = data[i]
            count = 1
            
            while i + count < len(data) and data[i + count] == current and count < 127:
                count += 1
            
            if count >= 3:
                result.extend([128 + count, current])  # RLEマーカー (128+count)
            else:
                for _ in range(count):
                    if current >= 128:
                        result.extend([127, current])  # エスケープ
                    else:
                        result.append(current)
            
            i += count
        
        return bytes(result)
    
    @staticmethod
    def _encode_shape_compact(shape) -> int:
        """形状の超コンパクトエンコード"""
        shape_map = {
            'I': 0, 'O': 1, 'T': 2, 'J': 3, 'L': 4, 'S': 5, 'Z': 6,
            '1': 7, '2': 8, '3': 9
        }
        return shape_map.get(shape.value, 7)  # デフォルトは SINGLE

# デコード処理も追加予定
if __name__ == "__main__":
    print("🚀 NEXUS超高効率エンコーダー準備完了")
    print("   - 差分圧縮最適化")
    print("   - 統計圧縮統合") 
    print("   - ハイブリッド手法")
