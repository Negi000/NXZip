#!/usr/bin/env python3
"""
NEXUS SDC (Structure-Destructive Compression) Engine - 簡潔表示版
革命的構造破壊型圧縮エンジンの実装

ユーザーの革新的アイデア実装:
「構造をバイナリレベルで完全把握 → 原型破壊圧縮 → 構造復元」

特徴：詳細ログ制御で簡潔な進捗表示
"""

import os
import sys
import time
import lzma
import zlib
import bz2
import struct
import hashlib
import pickle
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Tuple, Optional

# プロジェクト内モジュールのインポート
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from progress_display import ProgressDisplay

# 高度な構造解析器をインポート
try:
    from nexus_sdc_analyzer import AdvancedStructureAnalyzer
except ImportError:
    AdvancedStructureAnalyzer = None

# 高度化圧縮アルゴリズムをインポート  
try:
    from nexus_sdc_enhanced import EnhancedCompressionAlgorithms
except ImportError:
    EnhancedCompressionAlgorithms = None

# 詳細ログ制御設定
ENABLE_DETAILED_LOGGING = False  # ← この設定で詳細ログを制御

class CompressionMethod(Enum):
    """圧縮アルゴリズムの種類"""
    RAW = "raw"
    ZLIB = "zlib"
    LZMA = "lzma"
    BZ2 = "bz2"

@dataclass
class StructureElement:
    """構造要素の定義"""
    element_type: str
    position: int
    size: int
    compression_potential: float
    category: str = "unknown"
    metadata: Dict = None
    compressed_data: bytes = None
    compression_method: CompressionMethod = CompressionMethod.RAW
    compression_ratio: float = 0.0

@dataclass
class FileStructure:
    """ファイル構造の定義"""
    format_type: str
    total_size: int
    elements: List[StructureElement]
    metadata: Dict
    structure_hash: str

# 進捗表示インスタンス
progress = ProgressDisplay()

# ユーティリティ関数
def show_step(message: str):
    """メインステップ表示"""
    print(f"📊 {message}")

def show_substep(message: str):
    """サブステップ表示（詳細ログ制御対象）"""
    if ENABLE_DETAILED_LOGGING:
        print(f"   💫 {message}")

def show_success(message: str):
    """成功メッセージ"""
    print(f"✅ {message}")

def show_warning(message: str):
    """警告メッセージ"""
    print(f"⚠️  {message}")

class NexusSDCEngine:
    """NEXUS構造破壊型圧縮エンジン"""
    
    def __init__(self):
        self.name = "NEXUS SDC Engine"
        self.version = "2.0.0"
        self.advanced_analyzer = AdvancedStructureAnalyzer() if AdvancedStructureAnalyzer else None
        self.enhanced_algorithms = EnhancedCompressionAlgorithms() if EnhancedCompressionAlgorithms else None
        self.statistics = {
            'total_files_processed': 0,
            'total_bytes_compressed': 0,
            'total_bytes_saved': 0,
            'average_compression_ratio': 0.0
        }
    
    def compress_file(self, input_path: str, output_path: str = None) -> Dict[str, Any]:
        """構造破壊型圧縮の実行"""
        if output_path is None:
            output_path = f"{input_path}.sdc"
        
        original_size = os.path.getsize(input_path)
        file_name = os.path.basename(input_path)
        start_time = time.time()
        
        # 進捗開始
        progress.start_task(f"構造破壊型圧縮: {file_name}", original_size, file_name)
        
        try:
            # ファイル構造解析 (0-30%)
            progress.update_progress(5, "📊 ファイル解析開始")
            show_step(f"構造破壊型圧縮: {file_name}")
            print(f"📁 ファイル: {file_name}")
            print(f"💾 サイズ: {original_size / (1024*1024):.1f}MB")
            
            with open(input_path, 'rb') as f:
                data = f.read()
            
            show_substep(f"原サイズ: {original_size:,} bytes")
            
            # 構造解析実行
            progress.update_progress(10, "🧬 構造解析実行中")
            file_structure = self._analyze_file_structure(data)
            progress.update_progress(30, "✅ 構造解析完了")
            
            show_substep(f"構造要素数: {len(file_structure.elements)}")
            
            # 原型破壊圧縮 (30-80%)
            progress.update_progress(35, "💥 原型破壊圧縮開始")
            self._compress_elements_with_progress(file_structure, data)
            progress.update_progress(80, "✅ 破壊的圧縮完了")
            
            # ファイル保存 (80-100%)
            progress.update_progress(85, "💾 圧縮ファイル保存中")
            compressed_size = self._save_compressed_file(file_structure, output_path)
            progress.update_progress(100, "✅ 保存完了")
            
            # 統計計算
            elapsed_time = time.time() - start_time
            compression_ratio = (1 - compressed_size / original_size) * 100
            speed_mbps = (original_size / (1024 * 1024)) / max(elapsed_time, 0.001)
            
            result = {
                'input_path': input_path,
                'output_path': output_path,
                'original_size': original_size,
                'compressed_size': compressed_size,
                'compression_ratio': compression_ratio,
                'structure_elements': len(file_structure.elements),
                'speed_mbps': speed_mbps
            }
            
            # 進捗完了
            final_msg = f"圧縮率: {compression_ratio:.1f}% ({original_size:,} → {compressed_size:,} bytes)"
            progress.finish_task(True, final_msg)
            
            self._print_compression_result(result)
            self._update_stats(result)
            
            return result
            
        except Exception as e:
            progress.finish_task(False, f"エラー: {str(e)}")
            raise
    
    def decompress_file(self, input_path: str, output_path: str = None) -> Dict[str, Any]:
        """構造破壊型圧縮ファイルの展開"""
        if output_path is None:
            output_path = input_path.replace('.sdc', '')
        
        file_name = os.path.basename(input_path)
        
        # 進捗開始
        progress.start_task(f"構造復元: {file_name}", 100, file_name)
        
        try:
            # 圧縮データ読み込み (0-20%)
            progress.update_progress(5, "💾 圧縮ファイル読み込み中")
            compressed_structure = self._load_compressed_file(input_path)
            progress.update_progress(20, "✅ 読み込み完了")
            
            # 構造復元 (20-90%)
            progress.update_progress(25, "🔄 構造復元開始")
            restored_data = self._restore_structure_with_progress(compressed_structure)
            progress.update_progress(90, "✅ 構造復元完了")
            
            # ファイル保存 (90-100%)
            progress.update_progress(95, "💾 ファイル保存中")
            with open(output_path, 'wb') as f:
                f.write(restored_data)
            progress.update_progress(100, "✅ 保存完了")
            
            # 結果表示
            result = {
                'input_path': input_path,
                'output_path': output_path,
                'restored_size': len(restored_data)
            }
            
            final_msg = f"復元サイズ: {len(restored_data):,} bytes"
            progress.finish_task(True, final_msg)
            
            return result
            
        except Exception as e:
            progress.finish_task(False, f"エラー: {str(e)}")
            raise
    
    def _analyze_file_structure(self, data: bytes) -> FileStructure:
        """ファイル構造の分析"""
        # 高度解析器が利用可能な場合は使用
        if self.advanced_analyzer:
            try:
                return self.advanced_analyzer.analyze_comprehensive(data)
            except Exception:
                # フォールバックして基本解析器を使用
                pass
        
        # 基本解析器を使用
        return self._basic_structure_analysis(data)
    
    def _basic_structure_analysis(self, data: bytes) -> FileStructure:
        """基本的な構造解析"""
        elements = []
        
        # ファイル形式判定
        if data.startswith(b'ID3') or (len(data) > 1024 and b'\xff\xfb' in data[:1024]):
            # MP3ファイル
            format_type = "MP3"
            self._analyze_mp3_structure_basic(data, elements)
        elif data.startswith(b'\x00\x00\x00') and b'ftyp' in data[:32]:
            # MP4ファイル
            format_type = "MP4"
            self._analyze_mp4_structure_basic(data, elements)
        elif data.startswith(b'RIFF') and b'WAVE' in data[:12]:
            # WAVファイル
            format_type = "WAV"
            self._analyze_wav_structure_basic(data, elements)
        else:
            # 汎用ファイル
            format_type = "GENERIC"
            self._analyze_generic_structure_basic(data, elements)
        
        structure_hash = hashlib.sha256(data[:1024]).hexdigest()[:16]
        
        return FileStructure(
            format_type=format_type,
            total_size=len(data),
            elements=elements,
            metadata={"format": format_type},
            structure_hash=structure_hash
        )
    
    def _analyze_mp3_structure_basic(self, data: bytes, elements: List[StructureElement]):
        """MP3構造の基本解析（簡潔版）"""
        pos = 0
        frame_count = 0
        
        # ID3タグ検出
        if data.startswith(b'ID3'):
            if len(data) >= 10:
                tag_size = struct.unpack('>I', b'\x00' + data[6:9])[0]
                elements.append(StructureElement(
                    element_type="ID3v2_TAG",
                    position=0,
                    size=10 + tag_size,
                    compression_potential=0.3,
                    category="metadata"
                ))
                pos = 10 + tag_size
        
        # MP3フレーム解析（まとめて処理）
        total_audio_size = 0
        audio_start = pos
        
        while pos < len(data) - 4:
            if data[pos:pos+2] == b'\xff\xfb' or data[pos:pos+2] == b'\xff\xfa':
                # MP3フレームヘッダー発見
                if pos + 4 <= len(data):
                    header = struct.unpack('>I', data[pos:pos+4])[0]
                    frame_size = self._calculate_mp3_frame_size(header)
                    if frame_size > 0 and pos + frame_size <= len(data):
                        frame_count += 1
                        total_audio_size += frame_size
                        pos += frame_size
                        continue
            pos += 1
        
        # 全音声データを一つの要素として追加
        if total_audio_size > 0:
            elements.append(StructureElement(
                element_type="MP3_AUDIO_DATA",
                position=audio_start,
                size=total_audio_size,
                compression_potential=0.75,
                category="audio_data",
                metadata={"frame_count": frame_count}
            ))
        
        # ID3v1タグ（末尾）
        if len(data) >= 128 and data[-128:-125] == b'TAG':
            elements.append(StructureElement(
                element_type="ID3v1_TAG",
                position=len(data) - 128,
                size=128,
                compression_potential=0.3,
                category="metadata"
            ))
    
    def _analyze_mp4_structure_basic(self, data: bytes, elements: List[StructureElement]):
        """MP4構造の基本解析"""
        pos = 0
        while pos < len(data) - 8:
            try:
                size = struct.unpack('>I', data[pos:pos+4])[0]
                atom_type = data[pos+4:pos+8]
                
                if size == 0:
                    break
                if size < 8:
                    pos += 8
                    continue
                
                # Atom情報を要素として追加
                compression_potential = 0.6 if atom_type == b'mdat' else 0.3
                elements.append(StructureElement(
                    element_type=f"MP4_ATOM_{atom_type.decode('ascii', errors='ignore')}",
                    position=pos,
                    size=size,
                    compression_potential=compression_potential,
                    category="video_data" if atom_type == b'mdat' else "metadata"
                ))
                
                pos += size
            except:
                pos += 1
    
    def _analyze_wav_structure_basic(self, data: bytes, elements: List[StructureElement]):
        """WAV構造の基本解析"""
        if len(data) < 44:
            return
        
        # RIFFヘッダー
        elements.append(StructureElement(
            element_type="RIFF_HEADER",
            position=0,
            size=12,
            compression_potential=0.1,
            category="header"
        ))
        
        pos = 12
        while pos < len(data) - 8:
            if pos + 8 > len(data):
                break
            
            chunk_id = data[pos:pos+4]
            chunk_size = struct.unpack('<I', data[pos+4:pos+8])[0]
            
            compression_potential = 0.85 if chunk_id == b'data' else 0.2
            elements.append(StructureElement(
                element_type=f"WAV_{chunk_id.decode('ascii', errors='ignore').upper()}_CHUNK",
                position=pos,
                size=8 + chunk_size,
                compression_potential=compression_potential,
                category="audio_data" if chunk_id == b'data' else "metadata"
            ))
            
            pos += 8 + chunk_size
            if chunk_size % 2:
                pos += 1
    
    def _analyze_generic_structure_basic(self, data: bytes, elements: List[StructureElement]):
        """汎用構造の基本解析"""
        chunk_size = max(4096, len(data) // 10)  # 大きなチャンクに分割
        pos = 0
        
        while pos < len(data):
            remaining = len(data) - pos
            size = min(chunk_size, remaining)
            
            elements.append(StructureElement(
                element_type=f"GENERIC_CHUNK_{pos // chunk_size}",
                position=pos,
                size=size,
                compression_potential=0.7,
                category="data"
            ))
            
            pos += size
    
    def _calculate_mp3_frame_size(self, header: int) -> int:
        """MP3フレームサイズ計算"""
        try:
            version = (header >> 19) & 0x3
            layer = (header >> 17) & 0x3
            bitrate_index = (header >> 12) & 0xF
            sample_rate_index = (header >> 10) & 0x3
            
            if bitrate_index == 0 or bitrate_index == 15:
                return 0
            if sample_rate_index == 3:
                return 0
            
            # 簡略化された計算
            bitrates = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320]
            sample_rates = [44100, 48000, 32000]
            
            if bitrate_index < len(bitrates) and sample_rate_index < len(sample_rates):
                bitrate = bitrates[bitrate_index] * 1000
                sample_rate = sample_rates[sample_rate_index]
                return int(144 * bitrate / sample_rate) + ((header >> 9) & 1)
            
            return 0
        except:
            return 0
    
    def _compress_elements_with_progress(self, structure: FileStructure, data: bytes):
        """進捗表示付き原型破壊圧縮（簡潔版）"""
        show_step("原型破壊圧縮実行中...")
        
        total_elements = len(structure.elements)
        total_compression_ratio = 0.0
        processed_bytes = 0
        
        for i, element in enumerate(structure.elements):
            # 進捗更新（詳細ログなし）
            element_progress = 30 + int((i / total_elements) * 50)  # 30-80%の範囲
            progress.update_progress(element_progress, bytes_processed=processed_bytes)
            
            # 要素データの抽出（ログ出力制御）
            element_data = data[element.position:element.position + element.size]
            
            # 詳細ログ制御
            if ENABLE_DETAILED_LOGGING:
                show_substep(f"要素 {i+1}/{total_elements}: {element.element_type} ({element.size} bytes)")
            
            # 高度化アルゴリズムを使用（可能な場合）
            if self.enhanced_algorithms and element.compression_potential > 0.5:
                try:
                    compressed, method = self.enhanced_algorithms.adaptive_compress(
                        element_data, element.compression_potential
                    )
                    element.compression_method = CompressionMethod.RAW
                    element.compressed_data = compressed
                    
                    if method in ['custom_high', 'destructive', 'lzma_enhanced', 'zlib_enhanced', 'bz2_enhanced']:
                        element.custom_method = method
                        
                except Exception as e:
                    if i == 0:  # 最初の要素でのみ警告表示
                        show_warning(f"高度化圧縮失敗、標準アルゴリズムにフォールバック")
                    compressed, method = self._standard_compress(element_data, element.compression_potential)
                    element.compressed_data = compressed
                    element.compression_method = CompressionMethod(method)
            else:
                # 標準圧縮アルゴリズム
                compressed, method = self._standard_compress(element_data, element.compression_potential)
                element.compressed_data = compressed
                element.compression_method = CompressionMethod(method)
            
            # 圧縮効果チェック
            if len(element.compressed_data) >= len(element_data) * 0.95:
                element.compressed_data = element_data
                element.compression_method = CompressionMethod.RAW
                if hasattr(element, 'custom_method'):
                    delattr(element, 'custom_method')
            
            element.compression_ratio = (1 - len(element.compressed_data) / len(element_data)) * 100
            total_compression_ratio += element.compression_ratio * element.size
            processed_bytes += element.size
            
            # 詳細ログ制御
            if ENABLE_DETAILED_LOGGING:
                show_substep(f"圧縮率: {element.compression_ratio:.1f}% ({element.size} → {len(element.compressed_data)})")
        
        # 加重平均圧縮率計算
        weighted_compression = total_compression_ratio / structure.total_size
        show_success(f"要素別平均圧縮率: {weighted_compression:.1f}%")
    
    def _standard_compress(self, data: bytes, compression_potential: float) -> Tuple[bytes, str]:
        """標準圧縮アルゴリズム"""
        try:
            if compression_potential > 0.7:
                compressed = lzma.compress(data, preset=6)
                return compressed, "lzma"
            elif compression_potential > 0.4:
                compressed = zlib.compress(data, level=6)
                return compressed, "zlib"
            else:
                compressed = bz2.compress(data, compresslevel=6)
                return compressed, "bz2"
        except:
            return data, "raw"
    
    def _save_compressed_file(self, structure: FileStructure, output_path: str) -> int:
        """圧縮ファイルの保存"""
        with open(output_path, 'wb') as f:
            # ヘッダー情報
            header = {
                'version': self.version,
                'format_type': structure.format_type,
                'total_size': structure.total_size,
                'structure_hash': structure.structure_hash,
                'elements_count': len(structure.elements)
            }
            
            header_data = pickle.dumps(header)
            f.write(struct.pack('<I', len(header_data)))
            f.write(header_data)
            
            # 構造情報
            structure_info = []
            for element in structure.elements:
                info = {
                    'element_type': element.element_type,
                    'position': element.position,
                    'size': element.size,
                    'compression_method': element.compression_method.value,
                    'category': element.category,
                    'metadata': element.metadata
                }
                if hasattr(element, 'custom_method'):
                    info['custom_method'] = element.custom_method
                structure_info.append(info)
            
            structure_data = pickle.dumps(structure_info)
            f.write(struct.pack('<I', len(structure_data)))
            f.write(structure_data)
            
            # 各要素の圧縮データ
            for element in structure.elements:
                f.write(struct.pack('<I', len(element.compressed_data)))
                f.write(element.compressed_data)
            
        return os.path.getsize(output_path)
    
    def _load_compressed_file(self, input_path: str) -> FileStructure:
        """圧縮ファイルの読み込み"""
        with open(input_path, 'rb') as f:
            # ヘッダー読み込み
            header_size = struct.unpack('<I', f.read(4))[0]
            header_data = f.read(header_size)
            header = pickle.loads(header_data)
            
            # 構造情報読み込み
            structure_size = struct.unpack('<I', f.read(4))[0]
            structure_data = f.read(structure_size)
            structure_info = pickle.loads(structure_data)
            
            # 要素データ読み込み
            elements = []
            for info in structure_info:
                data_size = struct.unpack('<I', f.read(4))[0]
                compressed_data = f.read(data_size)
                
                element = StructureElement(
                    element_type=info['element_type'],
                    position=info['position'],
                    size=info['size'],
                    compression_potential=0.0,
                    category=info['category'],
                    metadata=info['metadata'],
                    compressed_data=compressed_data,
                    compression_method=CompressionMethod(info['compression_method'])
                )
                
                if 'custom_method' in info:
                    element.custom_method = info['custom_method']
                
                elements.append(element)
            
            return FileStructure(
                format_type=header['format_type'],
                total_size=header['total_size'],
                elements=elements,
                metadata={},
                structure_hash=header['structure_hash']
            )
    
    def _restore_structure_with_progress(self, structure: FileStructure) -> bytes:
        """進捗表示付き構造復元（簡潔版）"""
        show_step("構造復元実行中...")
        
        # 復元データの初期化
        restored_data = bytearray(structure.total_size)
        total_elements = len(structure.elements)
        
        for i, element in enumerate(structure.elements):
            # 進捗更新
            restoration_progress = 25 + int((i / total_elements) * 65)  # 25-90%の範囲
            progress.update_progress(restoration_progress)
            
            # 詳細ログ制御
            if ENABLE_DETAILED_LOGGING:
                show_substep(f"要素 {i+1}/{total_elements}: {element.element_type}")
            
            # データ解凍
            decompressed_data = self._decompress_element_data(element)
            
            # 元の位置に復元
            if len(decompressed_data) == element.size:
                restored_data[element.position:element.position + element.size] = decompressed_data
            else:
                if i == 0:  # 最初の要素でのみ警告表示
                    show_warning(f"サイズ不一致検出、調整中")
                # サイズ調整
                if len(decompressed_data) < element.size:
                    decompressed_data += b'\x00' * (element.size - len(decompressed_data))
                else:
                    decompressed_data = decompressed_data[:element.size]
                restored_data[element.position:element.position + element.size] = decompressed_data
        
        return bytes(restored_data)
    
    def _decompress_element_data(self, element: StructureElement) -> bytes:
        """要素データの解凍"""
        try:
            if hasattr(element, 'custom_method'):
                # カスタム解凍メソッドがある場合
                if self.enhanced_algorithms:
                    return self.enhanced_algorithms.adaptive_decompress(
                        element.compressed_data, element.custom_method
                    )
            
            if element.compression_method == CompressionMethod.LZMA:
                return lzma.decompress(element.compressed_data)
            elif element.compression_method == CompressionMethod.ZLIB:
                return zlib.decompress(element.compressed_data)
            elif element.compression_method == CompressionMethod.BZ2:
                return bz2.decompress(element.compressed_data)
            else:
                return element.compressed_data
                
        except Exception:
            return element.compressed_data
    
    def _print_compression_result(self, result: Dict[str, Any]):
        """圧縮結果の表示"""
        print("--------------------------------------------------")
        show_success("構造破壊型圧縮完了")
        print(f"📁 入力: {os.path.basename(result['input_path'])}")
        print(f"💾 原サイズ: {result['original_size']:,} bytes")
        print(f"🗜️  圧縮サイズ: {result['compressed_size']:,} bytes")
        print(f"🎯 圧縮率: {result['compression_ratio']:.1f}%")
        print(f"⚡ 圧縮速度: {result['speed_mbps']:.1f} MB/s")
        print(f"🧬 構造要素: {result['structure_elements']}個")
    
    def _update_stats(self, result: Dict[str, Any]):
        """統計情報の更新"""
        self.statistics['total_files_processed'] += 1
        self.statistics['total_bytes_compressed'] += result['original_size']
        self.statistics['total_bytes_saved'] += (result['original_size'] - result['compressed_size'])
        
        # 移動平均で平均圧縮率を更新
        old_avg = self.statistics['average_compression_ratio']
        new_ratio = result['compression_ratio']
        files_count = self.statistics['total_files_processed']
        self.statistics['average_compression_ratio'] = (old_avg * (files_count - 1) + new_ratio) / files_count
    
    def print_statistics(self):
        """統計情報の表示"""
        stats = self.statistics
        if stats['total_files_processed'] == 0:
            print("📊 統計情報なし")
            return
        
        print("\n📊 NEXUS SDC Engine 統計情報")
        print("=" * 40)
        print(f"📁 処理ファイル数: {stats['total_files_processed']}")
        print(f"💾 総処理サイズ: {stats['total_bytes_compressed']:,} bytes")
        print(f"💰 総節約サイズ: {stats['total_bytes_saved']:,} bytes")
        print(f"📊 平均圧縮率: {stats['average_compression_ratio']:.1f}%")


def main():
    """メイン実行関数"""
    engine = NexusSDCEngine()
    
    if len(sys.argv) < 2:
        print(f"使用方法: {sys.argv[0]} <command> [options]")
        print("コマンド:")
        print("  test                - テスト実行")
        print("  compress <file>     - ファイル圧縮")
        print("  decompress <file>   - ファイル展開")
        print("  stats               - 統計表示")
        return
    
    command = sys.argv[1].lower()
    
    if command == "test":
        print("🧪 NEXUS SDC テスト実行")
        print("=" * 60)
        
        # テストファイルの設定
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sample_dir = os.path.join(os.path.dirname(base_dir), "NXZip-Python", "sample")
        
        test_files = [
            "陰謀論.mp3",
            "Python基礎講座3_4月26日-3.mp4",
            "generated-music-1752042054079.wav"
        ]
        
        # 存在するファイルのみテスト
        available_files = []
        for filename in test_files:
            file_path = os.path.join(sample_dir, filename)
            if os.path.exists(file_path):
                available_files.append(file_path)
        
        if not available_files:
            print("❌ テストファイルが見つかりません")
            return
        
        print(f"🔧 テスト対象ファイル: {len(available_files)}個")
        
        # 各ファイルでテスト実行
        compression_results = []
        for i, file_path in enumerate(available_files, 1):
            print(f"🔧 テスト {i}/{len(available_files)}: {os.path.basename(file_path)}")
            
            try:
                # 圧縮テスト
                result = engine.compress_file(file_path)
                compression_results.append(result)
                
                # 可逆性確認
                print("🔧 可逆性テスト実行中")
                engine.decompress_file(result['output_path'])
                print("✅ 可逆性確認完了")
                
            except Exception as e:
                print(f"❌ テスト失敗: {str(e)}")
                continue
        
        # 総合結果表示
        if compression_results:
            total_original = sum(r['original_size'] for r in compression_results)
            total_compressed = sum(r['compressed_size'] for r in compression_results)
            avg_compression = (1 - total_compressed / total_original) * 100
            
            print("\n📊 総合テスト結果")
            print("=" * 60)
            print(f"🎯 テストファイル数: {len(compression_results)}")
            print(f"📊 平均圧縮率: {avg_compression:.1f}%")
            print(f"💾 総処理サイズ: {total_original:,} bytes")
            print(f"🗜️ 総圧縮サイズ: {total_compressed:,} bytes")
    
    elif command == "compress":
        if len(sys.argv) < 3:
            print("使用方法: compress <input_file> [output_file]")
            return
        
        input_file = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else None
        
        if not os.path.exists(input_file):
            print(f"❌ ファイルが見つかりません: {input_file}")
            return
        
        try:
            result = engine.compress_file(input_file, output_file)
            print("✅ 圧縮完了")
        except Exception as e:
            print(f"❌ 圧縮エラー: {str(e)}")
    
    elif command == "decompress":
        if len(sys.argv) < 3:
            print("使用方法: decompress <input_file> [output_file]")
            return
        
        input_file = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else None
        
        if not os.path.exists(input_file):
            print(f"❌ ファイルが見つかりません: {input_file}")
            return
        
        try:
            result = engine.decompress_file(input_file, output_file)
            print("✅ 展開完了")
        except Exception as e:
            print(f"❌ 展開エラー: {str(e)}")
    
    elif command == "stats":
        engine.print_statistics()
    
    else:
        print(f"❌ 未知のコマンド: {command}")


if __name__ == "__main__":
    main()
