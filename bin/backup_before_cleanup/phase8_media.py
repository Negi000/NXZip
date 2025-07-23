#!/usr/bin/env python3
"""
Phase 8 画像・動画特化エンジン - 高圧縮率実現
画像・動画の構造特性を活用した専用最適化
"""

import os
import sys
import struct
import json
import lzma
import zlib
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Phase 8 可逆エンジンを継承
sys.path.append('bin')
from phase8_reversible import Phase8ReversibleEngine, ReversibleCompressionResult

class MediaSpecificElement:
    """メディア特化要素クラス"""
    def __init__(self, data: bytes, element_type: str, offset: int, size: int):
        self.data = data
        self.type = element_type
        self.offset = offset
        self.size = size
        self.entropy = 0.0
        self.pattern_score = 0.0
        self.compression_hint = "adaptive_optimal"
        self.ai_analysis = {}
        self.media_specific = {}

class Phase8MediaEngine(Phase8ReversibleEngine):
    """Phase 8 画像・動画特化エンジン"""
    
    def __init__(self):
        super().__init__()
        self.version = "8.0-Media"
        self.magic_header = b'NXZ8M'  # Media版マジックナンバー
        
        # メディア特化パラメータ
        self.image_chunk_size = 8192      # 画像チャンクサイズ
        self.video_chunk_size = 65536     # 動画チャンクサイズ
        self.pixel_analysis_enabled = True
        self.frame_analysis_enabled = True
    
    def analyze_media_structure(self, data: bytes, filename: str = "") -> List[MediaSpecificElement]:
        """メディア特化構造解析"""
        print(f"🎥 メディア特化解析開始: {len(data):,} bytes")
        
        # ファイル形式判定
        file_type = self._detect_media_type(data, filename)
        print(f"📋 メディア形式: {file_type}")
        
        if file_type in ['JPEG', 'PNG', 'BMP', 'GIF']:
            return self._analyze_image_structure(data, file_type)
        elif file_type in ['MP4', 'AVI', 'MOV', 'MKV']:
            return self._analyze_video_structure(data, file_type)
        else:
            # フォールバック: 通常解析
            return self._analyze_generic_media(data)
    
    def _detect_media_type(self, data: bytes, filename: str) -> str:
        """メディア形式自動検出"""
        if not data:
            return "UNKNOWN"
        
        # ファイル拡張子チェック
        ext = filename.lower().split('.')[-1] if '.' in filename else ''
        
        # マジックナンバーチェック
        if data.startswith(b'\xFF\xD8\xFF'):
            return "JPEG"
        elif data.startswith(b'\x89PNG\r\n\x1A\n'):
            return "PNG"
        elif data.startswith(b'BM'):
            return "BMP"
        elif data.startswith(b'GIF8'):
            return "GIF"
        elif b'ftyp' in data[:32]:
            return "MP4"
        elif data.startswith(b'RIFF') and b'AVI ' in data[:32]:
            return "AVI"
        
        # 拡張子ベース判定
        if ext in ['jpg', 'jpeg']:
            return "JPEG"
        elif ext in ['png']:
            return "PNG"
        elif ext in ['mp4', 'm4v']:
            return "MP4"
        elif ext in ['avi']:
            return "AVI"
        
        return "UNKNOWN"
    
    def _analyze_image_structure(self, data: bytes, image_type: str) -> List[MediaSpecificElement]:
        """画像特化構造解析"""
        elements = []
        
        if image_type == "JPEG":
            elements = self._analyze_jpeg_structure(data)
        elif image_type == "PNG":
            elements = self._analyze_png_structure(data)
        else:
            elements = self._analyze_generic_image(data)
        
        # 画像特化AI解析
        for element in elements:
            self._enhance_image_element(element, image_type)
        
        print(f"🖼️ 画像解析完了: {len(elements)}要素")
        return elements
    
    def _analyze_jpeg_structure(self, data: bytes) -> List[MediaSpecificElement]:
        """JPEG特化構造解析"""
        elements = []
        offset = 0
        
        while offset < len(data) - 1:
            # JPEGセグメント解析
            if data[offset] == 0xFF and data[offset + 1] != 0xFF:
                marker = data[offset + 1]
                segment_start = offset
                
                # セグメントサイズ計算
                if marker in [0xD8, 0xD9]:  # SOI, EOI
                    segment_size = 2
                elif marker == 0xDA:  # SOS (画像データ開始)
                    # 圧縮画像データを一括処理
                    end_pos = self._find_jpeg_image_data_end(data, offset + 2)
                    segment_size = end_pos - offset
                else:
                    if offset + 3 < len(data):
                        segment_size = struct.unpack('>H', data[offset + 2:offset + 4])[0] + 2
                    else:
                        segment_size = len(data) - offset
                
                # 要素作成
                element_data = data[segment_start:segment_start + segment_size]
                element_type = f"JPEG_SEGMENT_{marker:02X}"
                
                element = MediaSpecificElement(
                    element_data, element_type, segment_start, segment_size
                )
                
                # JPEG特化分析
                element.media_specific = {
                    'marker': marker,
                    'is_image_data': marker == 0xDA,
                    'is_metadata': marker in [0xE0, 0xE1, 0xE2, 0xFE],
                    'is_quantization': marker in [0xDB],
                    'is_huffman': marker in [0xC4]
                }
                
                elements.append(element)
                offset += segment_size
            else:
                offset += 1
        
        return elements
    
    def _find_jpeg_image_data_end(self, data: bytes, start: int) -> int:
        """JPEG画像データ終端検索"""
        pos = start
        while pos < len(data) - 1:
            if data[pos] == 0xFF and data[pos + 1] == 0xD9:  # EOI
                return pos
            pos += 1
        return len(data)
    
    def _analyze_png_structure(self, data: bytes) -> List[MediaSpecificElement]:
        """PNG特化構造解析"""
        elements = []
        offset = 8  # PNG署名をスキップ
        
        while offset < len(data) - 8:
            try:
                # チャンクサイズ
                chunk_size = struct.unpack('>I', data[offset:offset + 4])[0]
                chunk_type = data[offset + 4:offset + 8]
                
                total_chunk_size = chunk_size + 12  # サイズ(4) + タイプ(4) + データ + CRC(4)
                chunk_data = data[offset:offset + total_chunk_size]
                
                element = MediaSpecificElement(
                    chunk_data, f"PNG_CHUNK_{chunk_type.decode('ascii', errors='ignore')}", 
                    offset, total_chunk_size
                )
                
                # PNG特化分析
                element.media_specific = {
                    'chunk_type': chunk_type.decode('ascii', errors='ignore'),
                    'chunk_size': chunk_size,
                    'is_critical': chunk_type[0] & 0x20 == 0,
                    'is_image_data': chunk_type == b'IDAT',
                    'is_metadata': chunk_type in [b'tEXt', b'zTXt', b'iTXt'],
                    'is_palette': chunk_type == b'PLTE'
                }
                
                elements.append(element)
                offset += total_chunk_size
                
            except (struct.error, UnicodeDecodeError):
                # エラー時は残りを一括処理
                remaining_data = data[offset:]
                element = MediaSpecificElement(
                    remaining_data, "PNG_REMAINING", offset, len(remaining_data)
                )
                elements.append(element)
                break
        
        return elements
    
    def _analyze_video_structure(self, data: bytes, video_type: str) -> List[MediaSpecificElement]:
        """動画特化構造解析"""
        elements = []
        
        if video_type == "MP4":
            elements = self._analyze_mp4_structure(data)
        else:
            elements = self._analyze_generic_video(data)
        
        # 動画特化AI解析
        for element in elements:
            self._enhance_video_element(element, video_type)
        
        print(f"🎬 動画解析完了: {len(elements)}要素")
        return elements
    
    def _analyze_mp4_structure(self, data: bytes) -> List[MediaSpecificElement]:
        """MP4特化構造解析"""
        elements = []
        offset = 0
        
        while offset < len(data) - 8:
            try:
                # アトムサイズ
                atom_size = struct.unpack('>I', data[offset:offset + 4])[0]
                atom_type = data[offset + 4:offset + 8]
                
                if atom_size == 0:
                    atom_size = len(data) - offset
                elif atom_size == 1:
                    # 64bit サイズ
                    if offset + 16 <= len(data):
                        atom_size = struct.unpack('>Q', data[offset + 8:offset + 16])[0]
                        atom_data = data[offset:offset + atom_size]
                    else:
                        atom_data = data[offset:]
                        atom_size = len(atom_data)
                else:
                    atom_data = data[offset:offset + atom_size]
                
                element = MediaSpecificElement(
                    atom_data, f"MP4_ATOM_{atom_type.decode('ascii', errors='ignore')}", 
                    offset, atom_size
                )
                
                # MP4特化分析
                element.media_specific = {
                    'atom_type': atom_type.decode('ascii', errors='ignore'),
                    'atom_size': atom_size,
                    'is_container': atom_type in [b'moov', b'trak', b'mdia'],
                    'is_media_data': atom_type == b'mdat',
                    'is_metadata': atom_type in [b'meta', b'udta'],
                    'is_header': atom_type in [b'ftyp', b'mvhd']
                }
                
                elements.append(element)
                offset += atom_size
                
            except (struct.error, UnicodeDecodeError):
                # 残りを一括処理
                remaining_data = data[offset:]
                element = MediaSpecificElement(
                    remaining_data, "MP4_REMAINING", offset, len(remaining_data)
                )
                elements.append(element)
                break
        
        return elements
    
    def _analyze_generic_image(self, data: bytes) -> List[MediaSpecificElement]:
        """汎用画像解析"""
        return self._chunk_analysis(data, self.image_chunk_size, "IMAGE_CHUNK")
    
    def _analyze_generic_video(self, data: bytes) -> List[MediaSpecificElement]:
        """汎用動画解析"""
        return self._chunk_analysis(data, self.video_chunk_size, "VIDEO_CHUNK")
    
    def _analyze_generic_media(self, data: bytes) -> List[MediaSpecificElement]:
        """汎用メディア解析"""
        return self._chunk_analysis(data, 16384, "MEDIA_CHUNK")
    
    def _chunk_analysis(self, data: bytes, chunk_size: int, prefix: str) -> List[MediaSpecificElement]:
        """チャンク分割解析"""
        elements = []
        offset = 0
        chunk_index = 0
        
        while offset < len(data):
            current_chunk_size = min(chunk_size, len(data) - offset)
            chunk_data = data[offset:offset + current_chunk_size]
            
            element = MediaSpecificElement(
                chunk_data, f"{prefix}_{chunk_index:04d}", 
                offset, current_chunk_size
            )
            
            elements.append(element)
            offset += current_chunk_size
            chunk_index += 1
        
        return elements
    
    def _enhance_image_element(self, element: MediaSpecificElement, image_type: str):
        """画像要素AI強化"""
        data = element.data
        
        # エントロピー計算
        element.entropy = self._calculate_entropy(data)
        
        # 画像特化パターン解析
        element.pattern_score = self._analyze_image_patterns(data)
        
        # 圧縮手法推薦
        if element.media_specific.get('is_image_data', False):
            # 画像データ: 高度圧縮
            if element.entropy < 3.0:
                element.compression_hint = "rle_enhanced"
            elif element.entropy < 6.0:
                element.compression_hint = "lzma"
            else:
                element.compression_hint = "structure_destructive"
        elif element.media_specific.get('is_metadata', False):
            # メタデータ: テキスト圧縮
            element.compression_hint = "lzma"
        else:
            # その他: 適応的
            element.compression_hint = "adaptive_optimal"
        
        # AI解析結果
        element.ai_analysis = {
            'media_type': image_type,
            'estimated_redundancy': max(0, 8.0 - element.entropy),
            'compression_potential': self._estimate_compression_potential(data),
            'pixel_patterns': self._detect_pixel_patterns(data) if len(data) > 64 else []
        }
    
    def _enhance_video_element(self, element: MediaSpecificElement, video_type: str):
        """動画要素AI強化"""
        data = element.data
        
        # エントロピー計算
        element.entropy = self._calculate_entropy(data)
        
        # 動画特化パターン解析
        element.pattern_score = self._analyze_video_patterns(data)
        
        # 圧縮手法推薦
        if element.media_specific.get('is_media_data', False):
            # 動画データ: 構造破壊的圧縮
            if element.entropy < 4.0:
                element.compression_hint = "structure_destructive"
            elif element.entropy < 7.0:
                element.compression_hint = "lzma"
            else:
                element.compression_hint = "minimal_processing"
        elif element.media_specific.get('is_metadata', False):
            # メタデータ: 高効率圧縮
            element.compression_hint = "lzma"
        else:
            # その他: 適応的
            element.compression_hint = "adaptive_optimal"
        
        # AI解析結果
        element.ai_analysis = {
            'media_type': video_type,
            'estimated_redundancy': max(0, 8.0 - element.entropy),
            'compression_potential': self._estimate_compression_potential(data),
            'frame_patterns': self._detect_frame_patterns(data) if len(data) > 256 else []
        }
    
    def _analyze_image_patterns(self, data: bytes) -> float:
        """画像パターン解析"""
        if len(data) < 64:
            return 0.0
        
        # ピクセル隣接性解析
        adjacent_similarity = 0.0
        sample_size = min(1024, len(data) - 1)
        
        for i in range(0, sample_size, 4):  # 4バイトずつサンプリング
            if i + 4 < len(data):
                current = data[i:i+4]
                next_pixel = data[i+4:i+8] if i+8 < len(data) else data[i:i+4]
                
                # バイト単位の類似度
                similarity = sum(1 for a, b in zip(current, next_pixel) if abs(a - b) < 16)
                adjacent_similarity += similarity / 4
        
        return adjacent_similarity / (sample_size // 4) if sample_size > 0 else 0.0
    
    def _analyze_video_patterns(self, data: bytes) -> float:
        """動画パターン解析"""
        if len(data) < 256:
            return 0.0
        
        # フレーム間類似性推定
        block_size = 64
        blocks = [data[i:i+block_size] for i in range(0, min(512, len(data)), block_size)]
        
        similarity_score = 0.0
        comparisons = 0
        
        for i in range(len(blocks) - 1):
            block1, block2 = blocks[i], blocks[i + 1]
            if len(block1) == len(block2):
                # ブロック間類似度
                similarity = sum(1 for a, b in zip(block1, block2) if abs(a - b) < 32)
                similarity_score += similarity / len(block1)
                comparisons += 1
        
        return similarity_score / comparisons if comparisons > 0 else 0.0
    
    def _detect_pixel_patterns(self, data: bytes) -> List[str]:
        """ピクセルパターン検出"""
        patterns = []
        
        if len(data) < 32:
            return patterns
        
        # 反復パターン検出
        for pattern_size in [3, 4, 6, 8]:
            if len(data) >= pattern_size * 4:
                pattern = data[:pattern_size]
                repeats = 1
                pos = pattern_size
                
                while pos + pattern_size <= len(data):
                    if data[pos:pos+pattern_size] == pattern:
                        repeats += 1
                        pos += pattern_size
                    else:
                        break
                
                if repeats >= 3:
                    patterns.append(f"repeat_{pattern_size}x{repeats}")
        
        # グラデーションパターン検出
        if len(data) >= 16:
            gradients = 0
            for i in range(0, min(64, len(data) - 1)):
                diff = abs(data[i + 1] - data[i])
                if 1 <= diff <= 8:  # 小さな変化
                    gradients += 1
            
            if gradients > len(data) // 4:
                patterns.append("gradient")
        
        return patterns
    
    def _detect_frame_patterns(self, data: bytes) -> List[str]:
        """フレームパターン検出"""
        patterns = []
        
        if len(data) < 128:
            return patterns
        
        # 周期性検出
        for period in [16, 32, 64, 128]:
            if len(data) >= period * 3:
                matches = 0
                checks = min(period, len(data) // period)
                
                for i in range(checks):
                    byte1 = data[i]
                    byte2 = data[i + period] if i + period < len(data) else 0
                    byte3 = data[i + period * 2] if i + period * 2 < len(data) else 0
                    
                    if abs(byte1 - byte2) < 16 and abs(byte2 - byte3) < 16:
                        matches += 1
                
                if matches > checks // 2:
                    patterns.append(f"period_{period}")
        
        return patterns
    
    def _estimate_compression_potential(self, data: bytes) -> float:
        """圧縮ポテンシャル推定"""
        if len(data) < 16:
            return 0.1
        
        # エントロピーベース推定
        entropy = self._calculate_entropy(data)
        theoretical_max = 8.0 - entropy
        
        # パターンボーナス
        pattern_bonus = 0.0
        
        # 反復検出
        unique_bytes = len(set(data))
        if unique_bytes < 64:
            pattern_bonus += 0.5
        
        # 連続検出
        consecutive = 0
        for i in range(len(data) - 1):
            if data[i] == data[i + 1]:
                consecutive += 1
        
        if consecutive > len(data) // 4:
            pattern_bonus += 0.3
        
        return min(0.95, (theoretical_max + pattern_bonus) / 8.0)
    
    def media_compress(self, data: bytes, filename: str = "media") -> ReversibleCompressionResult:
        """メディア特化圧縮"""
        start_time = time.time()
        original_size = len(data)
        
        print(f"🎭 メディア特化圧縮開始: {filename}")
        print(f"📊 元サイズ: {original_size:,} bytes ({original_size/1024/1024:.1f} MB)")
        
        # メディア特化構造解析
        elements = self.analyze_media_structure(data, filename)
        
        # 可逆圧縮実行（継承メソッド使用）
        result = self.reversible_compress(data, filename)
        
        # メディア特化メトリクス追加
        if elements:
            media_metrics = {
                'media_elements': len(elements),
                'image_data_elements': sum(1 for e in elements if e.media_specific.get('is_image_data', False)),
                'metadata_elements': sum(1 for e in elements if e.media_specific.get('is_metadata', False)),
                'avg_compression_potential': sum(e.ai_analysis.get('compression_potential', 0) for e in elements) / len(elements),
                'detected_patterns': sum(len(e.ai_analysis.get('pixel_patterns', [])) + len(e.ai_analysis.get('frame_patterns', [])) for e in elements)
            }
            
            result.performance_metrics.update(media_metrics)
        
        return result
    
    def compress_file(self, input_path: str, output_path: str = None) -> bool:
        """メディアファイル圧縮"""
        if not os.path.exists(input_path):
            print(f"❌ ファイルが見つかりません: {input_path}")
            return False
        
        if output_path is None:
            output_path = input_path + '.p8m'  # Phase 8 Media
        
        try:
            with open(input_path, 'rb') as f:
                data = f.read()
            
            filename = os.path.basename(input_path)
            result = self.media_compress(data, filename)
            
            with open(output_path, 'wb') as f:
                f.write(result.compressed_data)
            
            print(f"💾 メディア圧縮ファイル保存: {output_path}")
            
            # メディア特化分析結果表示
            metrics = result.performance_metrics
            if 'media_elements' in metrics:
                print(f"🎭 メディア解析結果:")
                print(f"   要素数: {metrics['media_elements']}")
                print(f"   画像データ要素: {metrics.get('image_data_elements', 0)}")
                print(f"   メタデータ要素: {metrics.get('metadata_elements', 0)}")
                print(f"   平均圧縮ポテンシャル: {metrics.get('avg_compression_potential', 0):.1%}")
                print(f"   検出パターン数: {metrics.get('detected_patterns', 0)}")
            
            return True
        
        except Exception as e:
            print(f"❌ 圧縮エラー: {e}")
            return False

def run_media_test():
    """メディア特化テスト"""
    print("🎭 Phase 8 メディア特化テスト")
    print("=" * 60)
    
    engine = Phase8MediaEngine()
    sample_dir = Path("../NXZip-Python/sample")
    
    # メディアファイル重点テスト
    test_files = [
        # 画像ファイル
        ("COT-001.jpg", "JPEG画像"),
        ("COT-012.png", "PNG画像"),
        
        # 動画ファイル  
        ("Python基礎講座3_4月26日-3.mp4", "MP4動画"),
        
        # 音声ファイル（比較用）
        ("陰謀論.mp3", "MP3音声"),
    ]
    
    results = []
    
    for filename, description in test_files:
        filepath = sample_dir / filename
        if not filepath.exists():
            print(f"⚠️ ファイルなし: {filename}")
            continue
        
        print(f"\n🎭 メディア特化テスト: {description}")
        print(f"📁 ファイル: {filename}")
        print("-" * 50)
        
        try:
            # 大容量ファイルは部分テスト
            if filename == "COT-012.png":
                with open(filepath, 'rb') as f:
                    test_data = f.read(2*1024*1024)  # 2MB制限
                print(f"📏 部分テスト: {len(test_data):,} bytes (2MB制限)")
            elif filename == "Python基礎講座3_4月26日-3.mp4":
                with open(filepath, 'rb') as f:
                    test_data = f.read(5*1024*1024)  # 5MB制限
                print(f"📏 部分テスト: {len(test_data):,} bytes (5MB制限)")
            else:
                with open(filepath, 'rb') as f:
                    test_data = f.read()
                print(f"📏 全体テスト: {len(test_data):,} bytes")
            
            # メディア特化圧縮
            result = engine.media_compress(test_data, filename)
            
            # 可逆性検証
            decompressed = engine.reversible_decompress(result.compressed_data)
            is_identical = (test_data == decompressed.original_data)
            
            # 結果保存
            results.append({
                'filename': filename,
                'description': description,
                'original_size': len(test_data),
                'compressed_size': result.compressed_size,
                'compression_ratio': result.compression_ratio,
                'reversible': is_identical,
                'processing_time': result.processing_time,
                'media_metrics': result.performance_metrics
            })
            
            # 個別結果表示
            print(f"✅ 圧縮完了: {result.compression_ratio:.1f}%")
            print(f"🔍 可逆性: {'✅' if is_identical else '❌'}")
            print(f"⏱️ 処理時間: {result.processing_time:.2f}秒")
            
        except Exception as e:
            print(f"❌ テストエラー: {str(e)[:80]}...")
    
    # 総合結果
    if results:
        print("\n" + "=" * 60)
        print("🏆 Phase 8 メディア特化テスト結果")
        print("=" * 60)
        
        total_original = sum(r['original_size'] for r in results)
        total_compressed = sum(r['compressed_size'] for r in results)
        overall_ratio = (1 - total_compressed / total_original) * 100
        reversible_count = sum(1 for r in results if r['reversible'])
        
        print(f"🎭 メディア特化圧縮率: {overall_ratio:.1f}%")
        print(f"🔒 可逆性成功率: {reversible_count}/{len(results)} ({reversible_count/len(results)*100:.1f}%)")
        print(f"📈 テストファイル数: {len(results)}")
        print(f"💾 総データ量: {total_original/1024/1024:.1f} MB")
        
        # メディア種別分析
        print(f"\n📊 メディア種別詳細結果:")
        for result in results:
            name = result['filename'][:30] + ('...' if len(result['filename']) > 30 else '')
            size_mb = result['original_size'] / 1024 / 1024
            rev_icon = '✅' if result['reversible'] else '❌'
            
            print(f"   🎬 {result['description']}: {result['compression_ratio']:.1f}% ({size_mb:.1f}MB) {rev_icon}")
            
            # メディア特化メトリクス
            metrics = result['media_metrics']
            if 'media_elements' in metrics:
                print(f"      📈 解析要素: {metrics['media_elements']}")
                print(f"      🎨 画像要素: {metrics.get('image_data_elements', 0)}")
                print(f"      📋 メタ要素: {metrics.get('metadata_elements', 0)}")
                print(f"      🎯 圧縮ポテンシャル: {metrics.get('avg_compression_potential', 0):.1%}")
        
        # 最適化提案
        low_compression = [r for r in results if r['compression_ratio'] < 20]
        if low_compression:
            print(f"\n⚠️ 低圧縮率メディア ({len(low_compression)}個):")
            for r in low_compression:
                print(f"   🔧 {r['description']}: {r['compression_ratio']:.1f}% - 特化アルゴリズム要開発")
        
        high_compression = [r for r in results if r['compression_ratio'] >= 50]
        if high_compression:
            print(f"\n🏅 高圧縮率達成 ({len(high_compression)}個):")
            for r in high_compression:
                print(f"   🌟 {r['description']}: {r['compression_ratio']:.1f}% - 優秀な結果")

def main():
    """メイン処理"""
    if len(sys.argv) < 2:
        print("🎭 Phase 8 メディア特化エンジン")
        print("使用方法:")
        print("  python phase8_media.py test                     # メディア特化テスト")
        print("  python phase8_media.py compress <file>          # メディア圧縮")
        print("  python phase8_media.py decompress <file.p8m>    # メディア展開")
        return
    
    command = sys.argv[1].lower()
    engine = Phase8MediaEngine()
    
    if command == "test":
        run_media_test()
    elif command == "compress" and len(sys.argv) >= 3:
        input_file = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) >= 4 else None
        engine.compress_file(input_file, output_file)
    elif command == "decompress" and len(sys.argv) >= 3:
        input_file = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) >= 4 else None
        engine.decompress_file(input_file, output_file)
    else:
        print("❌ 無効なコマンドです")

if __name__ == "__main__":
    main()
