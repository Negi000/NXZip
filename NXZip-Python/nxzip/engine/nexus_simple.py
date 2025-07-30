#!/usr/bin/env python3
"""
NEXUS理論 - 簡易実装版
動作する基本的なNEXUS理論実装
"""

import struct
import time
import lzma
import hashlib
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import sys
import pickle

# プロジェクトパス追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 直接SPEクラスを定義
class SPECoreJIT:
    """簡易SPE暗号化クラス"""
    def apply_transform(self, data):
        if not data:
            return data
        # 簡易XOR変換
        return bytes(b ^ 0x42 for b in data)
    
    def reverse_transform(self, data):
        if not data:
            return data
        # XOR逆変換
        return bytes(b ^ 0x42 for b in data)


@dataclass
class ElementalUnit:
    """要素単位"""
    data: bytes
    unit_type: str
    size: int
    
    def __post_init__(self):
        self.hash_value = hash(self.data)


@dataclass
class GroupInfo:
    """グループ情報"""
    elements: List[bytes]
    frequency: int = 1
    
    def __post_init__(self):
        self.normalized_form = b"".join(sorted(self.elements))
        self.group_hash = hash(self.normalized_form)


class NEXUSSimpleEngine:
    """
    NEXUS理論 - 簡易実装
    基本的な理論要素を実装した動作確認版
    """
    
    def __init__(self):
        self.spe = SPECoreJIT()
        
    def compress(self, data: bytes) -> bytes:
        """圧縮"""
        if not data:
            return self._create_empty_header()
        
        print(f"🔬 NEXUS簡易圧縮開始 - サイズ: {len(data)} bytes")
        
        # 1. データ形式分析
        data_format = self._analyze_format(data)
        print(f"📊 形式: {data_format}")
        
        # 2. 要素分解
        units = self._decompose_elements(data)
        print(f"🔧 要素分解: {len(units)} 要素")
        
        # 3. グループ化
        groups = self._create_groups(units)
        print(f"🔷 グループ化: {len(groups)} グループ")
        
        # 4. ユニークテーブル構築
        unique_groups = self._build_unique_table(groups)
        print(f"📋 ユニークテーブル: {len(unique_groups)} エントリ")
        
        # 5. エンコード
        encoded_data = self._encode_data(unique_groups, data_format, len(data))
        
        # 6. SPE暗号化
        encrypted_data = self.spe.apply_transform(encoded_data)
        
        # 7. ヘッダー作成
        header = self._create_header(len(data), len(encoded_data), len(encrypted_data))
        
        result = header + encrypted_data
        compression_ratio = (1 - len(result) / len(data)) * 100
        print(f"✅ 圧縮完了: {compression_ratio:.1f}%")
        
        return result
    
    def decompress(self, compressed_data: bytes) -> bytes:
        """展開"""
        if not compressed_data:
            return b""
        
        print(f"🔓 NEXUS簡易展開開始")
        
        # 1. ヘッダー解析
        if len(compressed_data) < 48:
            raise ValueError("Invalid compressed data")
        
        header_info = self._parse_header(compressed_data[:48])
        encrypted_data = compressed_data[48:]
        
        # 2. SPE復号化
        encoded_data = self.spe.reverse_transform(encrypted_data)
        
        # 3. デコード
        original_data = self._decode_data(encoded_data, header_info)
        
        print(f"✅ 展開完了: {len(original_data)} bytes")
        return original_data
    
    def _analyze_format(self, data: bytes) -> str:
        """データ形式分析"""
        if len(data) < 16:
            return "binary"
        
        # テキスト判定
        try:
            sample = data[:min(1024, len(data))]
            sample.decode('utf-8')
            text_ratio = sum(1 for b in sample if 32 <= b <= 126 or b in [9, 10, 13]) / len(sample)
            if text_ratio > 0.8:
                return "text"
        except:
            pass
        
        # バイナリ形式チェック
        if data.startswith(b'\xFF\xD8'):
            return "image"
        elif data.startswith(b'RIFF'):
            return "audio"
        
        return "binary"
    
    def _decompose_elements(self, data: bytes) -> List[ElementalUnit]:
        """要素分解"""
        units = []
        
        # 簡易実装：固定長4バイト単位
        unit_size = 4
        
        for i in range(0, len(data), unit_size):
            unit_data = data[i:i + unit_size]
            
            if len(unit_data) < unit_size:
                # パディング
                unit_data = unit_data + b'\x00' * (unit_size - len(unit_data))
            
            unit = ElementalUnit(
                data=unit_data,
                unit_type="fixed_4",
                size=len(unit_data)
            )
            units.append(unit)
        
        return units
    
    def _create_groups(self, units: List[ElementalUnit]) -> List[GroupInfo]:
        """グループ作成"""
        groups = []
        
        # 簡易実装：8要素ずつグループ化
        group_size = 8
        
        for i in range(0, len(units), group_size):
            group_units = units[i:i + group_size]
            
            # 要素データ抽出
            elements = [unit.data for unit in group_units]
            
            group = GroupInfo(elements=elements)
            groups.append(group)
        
        return groups
    
    def _build_unique_table(self, groups: List[GroupInfo]) -> List[GroupInfo]:
        """ユニークテーブル構築"""
        unique_table = {}
        
        for group in groups:
            group_key = group.group_hash
            
            if group_key in unique_table:
                unique_table[group_key].frequency += 1
            else:
                unique_table[group_key] = group
        
        return list(unique_table.values())
    
    def _encode_data(self, unique_groups: List[GroupInfo], data_format: str, original_size: int) -> bytes:
        """データエンコード"""
        # ユニークグループをシリアライズ
        groups_data = []
        
        for group in unique_groups:
            group_data = {
                'elements': group.elements,
                'frequency': group.frequency,
                'hash': group.group_hash
            }
            groups_data.append(group_data)
        
        # Pickleでシリアライズ
        serialized = pickle.dumps({
            'groups': groups_data,
            'format': data_format,
            'original_size': original_size
        })
        
        # LZMA圧縮
        compressed = lzma.compress(serialized, preset=6)
        
        return compressed
    
    def _decode_data(self, encoded_data: bytes, header_info: Dict) -> bytes:
        """データデコード"""
        # LZMA展開
        serialized = lzma.decompress(encoded_data)
        
        # Pickleデシリアライズ
        data_dict = pickle.loads(serialized)
        
        # グループから元データ復元
        result_data = b""
        
        for group_data in data_dict['groups']:
            elements = group_data['elements']
            frequency = group_data['frequency']
            
            # 頻度分だけ要素を復元
            for _ in range(frequency):
                for element in elements:
                    result_data += element
        
        # 元サイズに切り詰め
        original_size = data_dict['original_size']
        result_data = result_data[:original_size]
        
        return result_data
    
    def _create_header(self, original_size: int, encoded_size: int, encrypted_size: int) -> bytes:
        """ヘッダー作成"""
        header = bytearray(48)
        
        # マジックナンバー
        header[0:8] = b'NXSIMP01'  # NEXUS Simple v1
        
        # サイズ情報
        header[8:16] = struct.pack('<Q', original_size)
        header[16:24] = struct.pack('<Q', encoded_size)
        header[24:32] = struct.pack('<Q', encrypted_size)
        
        # チェックサム
        checksum = hashlib.md5(header[8:32]).digest()[:16]
        header[32:48] = checksum
        
        return bytes(header)
    
    def _parse_header(self, header: bytes) -> Dict:
        """ヘッダー解析"""
        if len(header) < 48:
            raise ValueError("Invalid header size")
        
        magic = header[0:8]
        if magic != b'NXSIMP01':
            raise ValueError("Invalid magic number")
        
        original_size = struct.unpack('<Q', header[8:16])[0]
        encoded_size = struct.unpack('<Q', header[16:24])[0]
        encrypted_size = struct.unpack('<Q', header[24:32])[0]
        
        return {
            'original_size': original_size,
            'encoded_size': encoded_size,
            'encrypted_size': encrypted_size
        }
    
    def _create_empty_header(self) -> bytes:
        """空ヘッダー作成"""
        return self._create_header(0, 0, 0)


def test_nexus_simple():
    """NEXUS簡易実装テスト"""
    print("🧪 NEXUS簡易実装テスト")
    print("=" * 60)
    
    engine = NEXUSSimpleEngine()
    
    # テストケース
    test_cases = [
        {
            'name': 'テキストデータ',
            'data': b'Hello NEXUS! This is a simple test. ' * 100
        },
        {
            'name': 'バイナリパターン',
            'data': b'\x00\x01\x02\x03\xFF\xFE\xFD\xFC' * 500
        },
        {
            'name': '反復データ',
            'data': b'PATTERN' * 1000
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🔬 テスト: {test_case['name']}")
        print(f"📊 データサイズ: {len(test_case['data'])} bytes")
        
        try:
            # 圧縮テスト
            start_time = time.perf_counter()
            compressed = engine.compress(test_case['data'])
            compress_time = time.perf_counter() - start_time
            
            # 展開テスト
            start_time = time.perf_counter()
            decompressed = engine.decompress(compressed)
            decomp_time = time.perf_counter() - start_time
            
            # 結果評価
            is_correct = test_case['data'] == decompressed
            compression_ratio = (1 - len(compressed) / len(test_case['data'])) * 100
            
            print(f"✅ 圧縮: {compression_ratio:.1f}% ({compress_time:.3f}s)")
            print(f"✅ 展開: {decomp_time:.3f}s")
            print(f"🔍 正確性: {'✅' if is_correct else '❌'}")
            
            if not is_correct:
                print(f"❌ サイズ不一致: 原本{len(test_case['data'])} vs 復元{len(decompressed)}")
            
        except Exception as e:
            print(f"❌ エラー: {str(e)}")
    
    print(f"\n🎯 NEXUS簡易実装テスト完了")


if __name__ == "__main__":
    test_nexus_simple()
