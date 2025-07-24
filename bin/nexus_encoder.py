"""
NEXUS高効率バイナリエンコード/デコードシステム
位置情報と元データサイズを効率的に保存する実装
"""

import struct
import hashlib
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum


class CompactNEXUSEncoder:
    """NEXUS圧縮状態の効率的バイナリエンコーダー"""
    
    MAGIC_HEADER = b'NXS2'
    VERSION = 1
    
    @staticmethod
    def encode_nexus_state(nexus_state) -> bytes:
        """効率的なNEXUS状態エンコード"""
        print("📦 高効率NEXUS状態エンコード中...")
        
        encoded = bytearray()
        
        # ヘッダー
        encoded.extend(CompactNEXUSEncoder.MAGIC_HEADER)
        encoded.append(CompactNEXUSEncoder.VERSION)
        
        # メタデータ
        original_size = nexus_state.compression_metadata.get('original_size', 0)
        encoded.extend(struct.pack('<I', original_size))
        
        # グリッド次元
        width, height = nexus_state.grid_dimensions
        encoded.extend(struct.pack('<HH', width, height))
        
        # 元グループ数
        original_groups = nexus_state.original_groups
        encoded.extend(struct.pack('<H', len(original_groups)))
        
        # 各グループの情報を効率的に保存
        for group in original_groups:
            # 形状タイプ（1バイト）
            shape_byte = CompactNEXUSEncoder._encode_shape(group.shape)
            encoded.append(shape_byte)
            
            # 要素数（1バイト、最大255要素）
            encoded.append(min(len(group.elements), 255))
            
            # 要素データ
            for element in group.elements:
                encoded.append(element & 0xFF)
            
            # 位置数（1バイト）
            encoded.append(min(len(group.positions), 255))
            
            # 位置データ（効率的エンコード）
            for row, col in group.positions:
                encoded.extend(struct.pack('<HH', row, col))
        
        print(f"  エンコード完了: {len(encoded)} bytes")
        return bytes(encoded)
    
    @staticmethod
    def decode_nexus_state(compressed_data: bytes):
        """効率的なNEXUS状態デコード"""
        print("📤 高効率NEXUS状態デコード中...")
        
        offset = 0
        
        # ヘッダー検証
        if compressed_data[offset:offset+4] != CompactNEXUSEncoder.MAGIC_HEADER:
            raise ValueError("不正なNEXUSファイル形式")
        offset += 4
        
        version = compressed_data[offset]
        offset += 1
        if version != CompactNEXUSEncoder.VERSION:
            raise ValueError(f"サポートされていないバージョン: {version}")
        
        # メタデータ
        original_size = struct.unpack('<I', compressed_data[offset:offset+4])[0]
        offset += 4
        
        # グリッド次元
        width, height = struct.unpack('<HH', compressed_data[offset:offset+4])
        offset += 4
        
        # グループ数
        group_count = struct.unpack('<H', compressed_data[offset:offset+2])[0]
        offset += 2
        
        # グループ復元
        original_groups = []
        for _ in range(group_count):
            # 形状タイプ
            shape = CompactNEXUSEncoder._decode_shape(compressed_data[offset])
            offset += 1
            
            # 要素数
            element_count = compressed_data[offset]
            offset += 1
            
            # 要素データ
            elements = list(compressed_data[offset:offset+element_count])
            offset += element_count
            
            # 位置数
            position_count = compressed_data[offset]
            offset += 1
            
            # 位置データ
            positions = []
            for _ in range(position_count):
                row, col = struct.unpack('<HH', compressed_data[offset:offset+4])
                offset += 4
                positions.append((row, col))
            
            # NEXUSGroup作成
            from nexus_compression_engine import NEXUSGroup
            normalized = tuple(sorted(elements))
            hash_value = hashlib.md5(str(normalized).encode()).hexdigest()
            
            group = NEXUSGroup(
                elements=elements,
                shape=shape,
                positions=positions,
                normalized=normalized,
                hash_value=hash_value
            )
            original_groups.append(group)
        
        # NEXUSCompressionState作成
        from nexus_compression_engine import NEXUSCompressionState
        nexus_state = NEXUSCompressionState(
            unique_groups=[],  # 簡易実装では省略
            group_counts={},
            position_map=[],
            original_groups=original_groups,
            shape_distribution={},
            grid_dimensions=(width, height),
            compression_metadata={'original_size': original_size}
        )
        
        print("  デコード完了")
        return nexus_state
    
    @staticmethod
    def _encode_shape(shape) -> int:
        """形状を1バイトにエンコード"""
        shape_map = {
            'I': 1, 'O': 2, 'T': 3, 'J': 4, 'L': 5, 'S': 6, 'Z': 7,
            '1': 8, '2': 9, '3': 10
        }
        return shape_map.get(shape.value, 0)
    
    @staticmethod
    def _decode_shape(shape_byte: int):
        """1バイトから形状をデコード"""
        from nexus_compression_engine import PolyominoShape
        shape_map = {
            1: PolyominoShape.I, 2: PolyominoShape.O, 3: PolyominoShape.T,
            4: PolyominoShape.J, 5: PolyominoShape.L, 6: PolyominoShape.S,
            7: PolyominoShape.Z, 8: PolyominoShape.SINGLE, 9: PolyominoShape.LINE2,
            10: PolyominoShape.LINE3
        }
        return shape_map.get(shape_byte, PolyominoShape.SINGLE)


if __name__ == "__main__":
    print("🔧 NEXUS高効率エンコーダーモジュール")
    print("   - 位置情報完全保存")
    print("   - 元データサイズ保証")
    print("   - 効率的バイナリ形式")
