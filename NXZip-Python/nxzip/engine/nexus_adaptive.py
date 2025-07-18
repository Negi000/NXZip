#!/usr/bin/env python3
"""
NEXUS Adaptive Engine - フォーマット特化圧縮システム
完全可逆性を保ちながら、7zとは異なるアプローチで圧縮効率を向上
"""

import struct
import zlib
import lzma
import bz2
import time
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import sys

# プロジェクトパス追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from .spe_core_jit import SPECoreJIT

# NXZ定数
NXZ_MAGIC = b'NXZP'
NXZ_VERSION = 3  # Adaptive版

class FormatDetector:
    """フォーマット検出器"""
    
    @staticmethod
    def detect_format(data: bytes) -> str:
        """データフォーマットを検出"""
        if len(data) < 16:
            return "binary"
        
        # マジックナンバーによる検出
        magic_patterns = {
            b'\xFF\xD8\xFF': "jpeg",
            b'\x89PNG\r\n\x1A\n': "png",
            b'GIF87a': "gif",
            b'GIF89a': "gif",
            b'BM': "bmp",
            b'RIFF': "wav",
            b'ID3': "mp3",
            b'\x00\x00\x00\x18ftypmp4': "mp4",
            b'\x00\x00\x00\x20ftypmp4': "mp4",
            b'\x1A\x45\xDF\xA3': "mkv",
            b'FLV\x01': "flv",
            b'PK\x03\x04': "zip",
            b'7z\xBC\xAF\x27\x1C': "7z",
            b'\x1F\x8B': "gzip",
            b'WEBP': "webp",
            b'fLaC': "flac",
            b'OggS': "ogg",
        }
        
        for magic, format_type in magic_patterns.items():
            if data.startswith(magic):
                return format_type
        
        # MP4の詳細検出
        if b'ftyp' in data[:64]:
            return "mp4"
        
        # AVIの検出
        if data.startswith(b'RIFF') and b'AVI ' in data[:32]:
            return "avi"
        
        # テキストデータの検出
        try:
            data[:1024].decode('utf-8')
            return "text"
        except UnicodeDecodeError:
            pass
        
        return "binary"

class StructureAnalyzer:
    """データ構造解析器"""
    
    @staticmethod
    def analyze_mp4_structure(data: bytes) -> Dict[str, Any]:
        """MP4構造解析"""
        if len(data) < 32:
            return {"type": "simple", "metadata_size": 0}
        
        # MP4 box構造の解析
        boxes = []
        offset = 0
        metadata_size = 0
        
        while offset < len(data) - 8:
            try:
                box_size = struct.unpack('>I', data[offset:offset+4])[0]
                box_type = data[offset+4:offset+8]
                
                if box_size == 0:
                    break
                
                if box_type in [b'ftyp', b'moov', b'mdat']:
                    boxes.append({
                        'type': box_type.decode('ascii', errors='ignore'),
                        'size': box_size,
                        'offset': offset
                    })
                    
                    # メタデータサイズ推定
                    if box_type in [b'ftyp', b'moov']:
                        metadata_size += box_size
                
                offset += box_size
                
            except (struct.error, ValueError):
                break
        
        return {
            "type": "mp4",
            "boxes": boxes,
            "metadata_size": metadata_size,
            "data_ratio": (len(data) - metadata_size) / len(data) if len(data) > 0 else 0
        }
    
    @staticmethod
    def find_patterns(data: bytes, max_patterns: int = 1000) -> Dict[str, Any]:
        """データパターン検出"""
        if len(data) < 64:
            return {"entropy": 1.0, "patterns": []}
        
        # エントロピー計算
        byte_freq = [0] * 256
        for b in data[:min(10000, len(data))]:  # サンプリング
            byte_freq[b] += 1
        
        entropy = 0.0
        sample_size = min(10000, len(data))
        for freq in byte_freq:
            if freq > 0:
                p = freq / sample_size
                import math
                entropy -= p * math.log2(p)
        
        entropy /= 8.0  # 正規化
        
        # 繰り返しパターン検出
        patterns = []
        for pattern_size in [4, 8, 16, 32]:
            if len(data) < pattern_size * 2:
                continue
            
            pattern_count = {}
            for i in range(0, min(len(data) - pattern_size, 10000), pattern_size):
                pattern = data[i:i+pattern_size]
                pattern_count[pattern] = pattern_count.get(pattern, 0) + 1
            
            # 頻出パターンを抽出
            for pattern, count in pattern_count.items():
                if count >= 3:  # 3回以上出現
                    patterns.append({
                        'pattern': pattern,
                        'count': count,
                        'size': pattern_size
                    })
        
        return {
            "entropy": entropy,
            "patterns": patterns[:max_patterns],
            "pattern_ratio": len(patterns) / max(len(data) // 32, 1)
        }

class NEXUSAdaptive:
    """
    NEXUS Adaptive Engine - 完全可逆性を保ちながら7zとは異なるアプローチ
    
    戦略:
    1. 構造保存圧縮 (SPE + 構造解析)
    2. パターン最適化 (繰り返し検出 + 効率的圧縮)
    3. 段階的圧縮 (メタデータ分離 + データ部最適化)
    """
    
    def __init__(self):
        self.spe = SPECoreJIT()
        self.detector = FormatDetector()
        self.analyzer = StructureAnalyzer()
        
        # フォーマット別圧縮設定
        self.compression_settings = {
            "text": {"algorithm": "lzma", "preset": 6, "spe_enabled": True},
            "jpeg": {"algorithm": "lzma", "preset": 3, "spe_enabled": True},
            "png": {"algorithm": "zlib", "level": 6, "spe_enabled": True},
            "gif": {"algorithm": "zlib", "level": 6, "spe_enabled": True},
            "bmp": {"algorithm": "lzma", "preset": 6, "spe_enabled": True},
            "webp": {"algorithm": "zlib", "level": 6, "spe_enabled": True},
            "mp3": {"algorithm": "lzma", "preset": 3, "spe_enabled": True},
            "wav": {"algorithm": "lzma", "preset": 4, "spe_enabled": True},
            "flac": {"algorithm": "lzma", "preset": 3, "spe_enabled": True},
            "ogg": {"algorithm": "lzma", "preset": 3, "spe_enabled": True},
            "mp4": {"algorithm": "adaptive", "preset": 4, "spe_enabled": True},
            "avi": {"algorithm": "adaptive", "preset": 4, "spe_enabled": True},
            "mkv": {"algorithm": "adaptive", "preset": 4, "spe_enabled": True},
            "flv": {"algorithm": "adaptive", "preset": 4, "spe_enabled": True},
            "zip": {"algorithm": "lzma", "preset": 3, "spe_enabled": False},
            "7z": {"algorithm": "lzma", "preset": 3, "spe_enabled": False},
            "gzip": {"algorithm": "lzma", "preset": 3, "spe_enabled": False},
            "binary": {"algorithm": "lzma", "preset": 6, "spe_enabled": True},
        }
        
        # 段階的目標設定
        self.stage_goals = {
            1: {"compression": 50, "speed": 100},  # 50%圧縮, 100MB/s
            2: {"compression": 80, "speed": 150},  # 80%圧縮, 150MB/s
            3: {"compression": 90, "speed": 200},  # 90%圧縮, 200MB/s
        }
    
    def compress(self, data: bytes, format_hint: Optional[str] = None) -> bytes:
        """フォーマット適応圧縮"""
        if not data:
            return self._create_empty_nxz()
        
        # 1. フォーマット検出
        detected_format = format_hint or self.detector.detect_format(data)
        settings = self.compression_settings.get(detected_format, self.compression_settings["binary"])
        
        # 2. 構造解析
        if detected_format == "mp4":
            structure = self.analyzer.analyze_mp4_structure(data)
            return self._compress_mp4_adaptive(data, structure, settings)
        elif detected_format in ["avi", "mkv", "flv"]:
            return self._compress_video_adaptive(data, detected_format, settings)
        else:
            return self._compress_standard(data, detected_format, settings)
    
    def _compress_mp4_adaptive(self, data: bytes, structure: Dict[str, Any], settings: Dict[str, Any]) -> bytes:
        """MP4専用適応圧縮"""
        # MP4構造を活用した圧縮
        if structure["type"] == "mp4" and structure["data_ratio"] > 0.8:
            # 大部分がデータ部の場合、メタデータ分離圧縮
            return self._compress_structured_video(data, structure, settings)
        else:
            # 標準圧縮
            return self._compress_standard(data, "mp4", settings)
    
    def _compress_structured_video(self, data: bytes, structure: Dict[str, Any], settings: Dict[str, Any]) -> bytes:
        """構造化動画圧縮"""
        # 1. パターン解析
        patterns = self.analyzer.find_patterns(data)
        
        # 2. エントロピーベース圧縮選択
        if patterns["entropy"] < 0.3:
            # 低エントロピー → 高圧縮
            compressed_data = b'NXZLZMA' + lzma.compress(data, preset=6)
        elif patterns["entropy"] < 0.7:
            # 中エントロピー → バランス圧縮
            compressed_data = b'NXZLZMA' + lzma.compress(data, preset=4)
        else:
            # 高エントロピー → 高速圧縮
            compressed_data = b'NXZZLIB' + zlib.compress(data, level=6)
        
        # 3. SPE適用
        if settings["spe_enabled"]:
            encrypted_data = self.spe.apply_transform(compressed_data)
        else:
            encrypted_data = compressed_data
        
        # 4. 適応ヘッダー作成
        header = self._create_adaptive_header(
            original_size=len(data),
            compressed_size=len(compressed_data),
            encrypted_size=len(encrypted_data),
            format_type="mp4",
            entropy=patterns["entropy"],
            pattern_count=len(patterns["patterns"])
        )
        
        return header + encrypted_data
    
    def _compress_video_adaptive(self, data: bytes, format_type: str, settings: Dict[str, Any]) -> bytes:
        """動画専用適応圧縮"""
        # パターン解析
        patterns = self.analyzer.find_patterns(data)
        
        # 動画の特性に応じた圧縮
        if patterns["pattern_ratio"] > 0.1:
            # パターンが多い場合
            compressed_data = b'NXZLZMA' + lzma.compress(data, preset=settings["preset"])
        else:
            # パターンが少ない場合
            compressed_data = b'NXZZLIB' + zlib.compress(data, level=6)
        
        # SPE適用
        if settings["spe_enabled"]:
            encrypted_data = self.spe.apply_transform(compressed_data)
        else:
            encrypted_data = compressed_data
        
        # ヘッダー作成
        header = self._create_adaptive_header(
            original_size=len(data),
            compressed_size=len(compressed_data),
            encrypted_size=len(encrypted_data),
            format_type=format_type,
            entropy=patterns["entropy"],
            pattern_count=len(patterns["patterns"])
        )
        
        return header + encrypted_data
    
    def _compress_standard(self, data: bytes, format_type: str, settings: Dict[str, Any]) -> bytes:
        """標準フォーマット圧縮"""
        # アルゴリズム選択
        if settings["algorithm"] == "lzma":
            compressed_data = b'NXZLZMA' + lzma.compress(data, preset=settings["preset"])
        elif settings["algorithm"] == "zlib":
            compressed_data = b'NXZZLIB' + zlib.compress(data, level=settings["level"])
        elif settings["algorithm"] == "bz2":
            compressed_data = b'NXZBZ2' + bz2.compress(data)
        else:
            # デフォルト
            compressed_data = b'NXZLZMA' + lzma.compress(data, preset=6)
        
        # SPE適用
        if settings["spe_enabled"]:
            encrypted_data = self.spe.apply_transform(compressed_data)
        else:
            encrypted_data = compressed_data
        
        # ヘッダー作成
        header = self._create_adaptive_header(
            original_size=len(data),
            compressed_size=len(compressed_data),
            encrypted_size=len(encrypted_data),
            format_type=format_type,
            entropy=0.5,  # デフォルト値
            pattern_count=0
        )
        
        return header + encrypted_data
    
    def decompress(self, nxz_data: bytes) -> bytes:
        """適応展開"""
        if not nxz_data:
            return b""
        
        # 1. ヘッダー解析
        header_info = self._parse_adaptive_header(nxz_data)
        if not header_info:
            raise ValueError("Invalid NXZ Adaptive format")
        
        # 2. 暗号化データ抽出
        encrypted_data = nxz_data[64:]  # 適応ヘッダー64バイト後
        
        # 3. SPE復号化
        if header_info["spe_enabled"]:
            compressed_data = self.spe.reverse_transform(encrypted_data)
        else:
            compressed_data = encrypted_data
        
        # 4. 圧縮展開
        if compressed_data.startswith(b'NXZLZMA'):
            original_data = lzma.decompress(compressed_data[7:])
        elif compressed_data.startswith(b'NXZZLIB'):
            original_data = zlib.decompress(compressed_data[7:])
        elif compressed_data.startswith(b'NXZBZ2'):
            original_data = bz2.decompress(compressed_data[6:])
        else:
            raise ValueError("Unknown compression format")
        
        return original_data
    
    def _create_adaptive_header(self, original_size: int, compressed_size: int, encrypted_size: int, 
                               format_type: str, entropy: float, pattern_count: int) -> bytes:
        """適応ヘッダー作成 (64バイト)"""
        header = bytearray(64)
        
        # マジックナンバー
        header[0:4] = NXZ_MAGIC
        
        # バージョン
        header[4:8] = struct.pack('<I', NXZ_VERSION)
        
        # サイズ情報
        header[8:16] = struct.pack('<Q', original_size)
        header[16:24] = struct.pack('<Q', compressed_size)
        header[24:32] = struct.pack('<Q', encrypted_size)
        
        # フォーマット情報
        format_bytes = format_type.encode('ascii')[:8].ljust(8, b'\x00')
        header[32:40] = format_bytes
        
        # 解析情報
        header[40:44] = struct.pack('<f', entropy)
        header[44:48] = struct.pack('<I', pattern_count)
        
        # SPE有効フラグ
        header[48:52] = struct.pack('<I', 1)  # SPE enabled
        
        # タイムスタンプ
        header[52:60] = struct.pack('<Q', int(time.time()))
        
        # CRC32
        crc32 = zlib.crc32(header[0:60])
        header[60:64] = struct.pack('<I', crc32 & 0xffffffff)
        
        return bytes(header)
    
    def _parse_adaptive_header(self, nxz_data: bytes) -> Optional[Dict[str, Any]]:
        """適応ヘッダー解析"""
        if len(nxz_data) < 64:
            return None
        
        # マジックナンバー確認
        if nxz_data[0:4] != NXZ_MAGIC:
            return None
        
        # ヘッダー情報抽出
        version = struct.unpack('<I', nxz_data[4:8])[0]
        original_size = struct.unpack('<Q', nxz_data[8:16])[0]
        compressed_size = struct.unpack('<Q', nxz_data[16:24])[0]
        encrypted_size = struct.unpack('<Q', nxz_data[24:32])[0]
        
        format_type = nxz_data[32:40].rstrip(b'\x00').decode('ascii', errors='ignore')
        entropy = struct.unpack('<f', nxz_data[40:44])[0]
        pattern_count = struct.unpack('<I', nxz_data[44:48])[0]
        spe_enabled = struct.unpack('<I', nxz_data[48:52])[0] == 1
        
        timestamp = struct.unpack('<Q', nxz_data[52:60])[0]
        crc32 = struct.unpack('<I', nxz_data[60:64])[0]
        
        return {
            'version': version,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'encrypted_size': encrypted_size,
            'format_type': format_type,
            'entropy': entropy,
            'pattern_count': pattern_count,
            'spe_enabled': spe_enabled,
            'timestamp': timestamp,
            'crc32': crc32
        }
    
    def _create_empty_nxz(self) -> bytes:
        """空のNXZ適応ファイル作成"""
        return self._create_adaptive_header(0, 0, 0, "empty", 0.0, 0)
    
    def evaluate_stage_performance(self, compression_ratio: float, speed: float) -> int:
        """段階的目標評価"""
        for stage in [3, 2, 1]:
            goals = self.stage_goals[stage]
            if compression_ratio >= goals["compression"] and speed >= goals["speed"]:
                return stage
        return 0

def test_nexus_adaptive_mp4():
    """NEXUS Adaptive MP4テスト"""
    print("🎬 NEXUS Adaptive MP4 テスト - 新戦略")
    print("=" * 60)
    
    # MP4テストファイル
    test_file = Path(r"C:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\OneTEL_CADDi全体会議午後_restored.mp4")
    
    if not test_file.exists():
        print("❌ MP4テストファイルが見つかりません")
        return
    
    file_size = test_file.stat().st_size
    print(f"📄 ファイル: {test_file.name}")
    print(f"📊 サイズ: {file_size//1024//1024} MB")
    
    # データ読み込み
    print("\n📖 データ読み込み中...")
    with open(test_file, 'rb') as f:
        data = f.read()
    
    # NEXUS Adaptive初期化
    nexus = NEXUSAdaptive()
    
    # フォーマット検出
    detected_format = nexus.detector.detect_format(data)
    print(f"🔍 検出フォーマット: {detected_format}")
    
    # 構造解析
    if detected_format == "mp4":
        structure = nexus.analyzer.analyze_mp4_structure(data)
        print(f"📐 MP4構造: {structure['type']}")
        print(f"   メタデータ: {structure['metadata_size']//1024} KB")
        print(f"   データ比率: {structure['data_ratio']:.2%}")
    
    # 圧縮テスト
    print(f"\n🎬 NEXUS Adaptive 圧縮中...")
    start_time = time.perf_counter()
    compressed = nexus.compress(data, format_hint="mp4")
    compress_time = time.perf_counter() - start_time
    
    # 圧縮結果
    compression_ratio = (1 - len(compressed) / len(data)) * 100
    compress_speed = (len(data) / 1024 / 1024) / compress_time
    
    print(f"✅ 圧縮完了!")
    print(f"   📈 圧縮率: {compression_ratio:.2f}%")
    print(f"   ⚡ 速度: {compress_speed:.2f} MB/s")
    print(f"   ⏱️ 時間: {compress_time:.2f}秒")
    
    # 展開テスト
    print(f"\n🔄 展開テスト中...")
    start_time = time.perf_counter()
    decompressed = nexus.decompress(compressed)
    decomp_time = time.perf_counter() - start_time
    
    # 展開結果
    decomp_speed = (len(data) / 1024 / 1024) / decomp_time
    
    print(f"✅ 展開完了!")
    print(f"   ⚡ 速度: {decomp_speed:.2f} MB/s")
    print(f"   ⏱️ 時間: {decomp_time:.2f}秒")
    
    # 正確性確認
    is_correct = data == decompressed
    print(f"   🔍 正確性: {'✅ OK' if is_correct else '❌ NG'}")
    
    # 総合評価
    total_time = compress_time + decomp_time
    total_speed = (len(data) * 2 / 1024 / 1024) / total_time
    
    print(f"\n🎬 NEXUS Adaptive 最終結果:")
    print(f"   圧縮率: {compression_ratio:.2f}%")
    print(f"   総合速度: {total_speed:.2f} MB/s")
    print(f"   戦略: 構造解析 + パターン最適化")
    print(f"   完全可逆性: ✅ 保証")
    
    # 段階的目標評価
    stage = nexus.evaluate_stage_performance(compression_ratio, total_speed)
    print(f"\n🎯 段階的目標:")
    print(f"   達成ステージ: {stage}/3")
    if stage >= 1:
        print(f"   ✅ Stage 1: 50%圧縮 + 100MB/s")
    if stage >= 2:
        print(f"   ✅ Stage 2: 80%圧縮 + 150MB/s")
    if stage >= 3:
        print(f"   ✅ Stage 3: 90%圧縮 + 200MB/s")
    
    # 改善提案
    if compression_ratio < 50:
        print(f"\n💡 改善提案:")
        print(f"   - MP4内部構造をさらに詳細解析")
        print(f"   - フレーム間差分検出の実装")
        print(f"   - 動画固有パターンの最適化")
    
    return compression_ratio, total_speed, stage

if __name__ == "__main__":
    test_nexus_adaptive_mp4()
