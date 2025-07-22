#!/usr/bin/env python3
"""
Phase 8 統合版 - 可逆性保証 + 画像・動画特化
100%可逆性と高圧縮率の両立実現
"""

import os
import sys
import time
import json
import struct
import lzma
import zlib
import hashlib
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class MediaOptimizedEngine:
    """メディア最適化エンジン - 可逆性保証版"""
    
    def __init__(self):
        self.version = "8.0-MediaOptimized"
        self.magic_header = b'NXZ8O'  # Optimized版マジックナンバー
    
    def calculate_entropy(self, data: bytes) -> float:
        """エントロピー計算"""
        if not data:
            return 0.0
        
        byte_counts = [0] * 256
        for byte in data:
            byte_counts[byte] += 1
        
        entropy = 0.0
        data_len = len(data)
        for count in byte_counts:
            if count > 0:
                probability = count / data_len
                entropy -= probability * math.log2(probability)
        
        return min(entropy, 8.0)
    
    def detect_media_type(self, data: bytes, filename: str) -> str:
        """メディア形式検出"""
        if not data:
            return "UNKNOWN"
        
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
        elif data.startswith(b'ID3') or data[0:2] == b'\xFF\xFB':
            return "MP3"
        elif data.startswith(b'RIFF') and b'WAVE' in data[:32]:
            return "WAV"
        
        # 拡張子ベース判定
        ext = filename.lower().split('.')[-1] if '.' in filename else ''
        if ext in ['jpg', 'jpeg']:
            return "JPEG"
        elif ext in ['png']:
            return "PNG"
        elif ext in ['mp4', 'm4v']:
            return "MP4"
        elif ext in ['mp3']:
            return "MP3"
        elif ext in ['wav']:
            return "WAV"
        elif ext in ['txt', 'csv', 'json', 'xml']:
            return "TEXT"
        
        return "UNKNOWN"
    
    def analyze_image_structure(self, data: bytes, image_type: str) -> List[Dict]:
        """画像構造解析"""
        segments = []
        
        if image_type == "JPEG":
            segments = self.analyze_jpeg_segments(data)
        elif image_type == "PNG":
            segments = self.analyze_png_chunks(data)
        else:
            # 汎用画像解析
            segments = self.analyze_generic_chunks(data, 8192)
        
        return segments
    
    def analyze_jpeg_segments(self, data: bytes) -> List[Dict]:
        """JPEG セグメント解析"""
        segments = []
        offset = 0
        
        while offset < len(data) - 1:
            if data[offset] == 0xFF and data[offset + 1] != 0xFF:
                marker = data[offset + 1]
                segment_start = offset
                
                if marker in [0xD8, 0xD9]:  # SOI, EOI
                    segment_size = 2
                elif marker == 0xDA:  # SOS (画像データ)
                    # 画像データ終端まで
                    end_pos = self.find_jpeg_eoi(data, offset + 2)
                    segment_size = end_pos - offset
                else:
                    if offset + 3 < len(data):
                        segment_size = struct.unpack('>H', data[offset + 2:offset + 4])[0] + 2
                    else:
                        segment_size = len(data) - offset
                
                segment_data = data[segment_start:segment_start + segment_size]
                segments.append({
                    'type': f'JPEG_MARKER_{marker:02X}',
                    'data': segment_data,
                    'offset': segment_start,
                    'size': segment_size,
                    'is_image_data': marker == 0xDA,
                    'is_metadata': marker in [0xE0, 0xE1, 0xE2, 0xFE]
                })
                
                offset += segment_size
            else:
                offset += 1
        
        return segments
    
    def find_jpeg_eoi(self, data: bytes, start: int) -> int:
        """JPEG EOI検索"""
        pos = start
        while pos < len(data) - 1:
            if data[pos] == 0xFF and data[pos + 1] == 0xD9:
                return pos + 2
            pos += 1
        return len(data)
    
    def analyze_png_chunks(self, data: bytes) -> List[Dict]:
        """PNG チャンク解析"""
        segments = []
        offset = 8  # PNG署名スキップ
        
        while offset < len(data) - 8:
            try:
                chunk_size = struct.unpack('>I', data[offset:offset + 4])[0]
                chunk_type = data[offset + 4:offset + 8]
                total_size = chunk_size + 12
                
                chunk_data = data[offset:offset + total_size]
                segments.append({
                    'type': f'PNG_CHUNK_{chunk_type.decode("ascii", errors="ignore")}',
                    'data': chunk_data,
                    'offset': offset,
                    'size': total_size,
                    'is_image_data': chunk_type == b'IDAT',
                    'is_metadata': chunk_type in [b'tEXt', b'zTXt', b'iTXt']
                })
                
                offset += total_size
            except:
                # 残りを一括処理
                remaining = data[offset:]
                segments.append({
                    'type': 'PNG_REMAINING',
                    'data': remaining,
                    'offset': offset,
                    'size': len(remaining),
                    'is_image_data': False,
                    'is_metadata': False
                })
                break
        
        return segments
    
    def analyze_video_structure(self, data: bytes, video_type: str) -> List[Dict]:
        """動画構造解析"""
        if video_type == "MP4":
            return self.analyze_mp4_atoms(data)
        else:
            return self.analyze_generic_chunks(data, 65536)
    
    def analyze_mp4_atoms(self, data: bytes) -> List[Dict]:
        """MP4 アトム解析"""
        segments = []
        offset = 0
        
        while offset < len(data) - 8:
            try:
                atom_size = struct.unpack('>I', data[offset:offset + 4])[0]
                atom_type = data[offset + 4:offset + 8]
                
                if atom_size == 0:
                    atom_size = len(data) - offset
                
                atom_data = data[offset:offset + atom_size]
                segments.append({
                    'type': f'MP4_ATOM_{atom_type.decode("ascii", errors="ignore")}',
                    'data': atom_data,
                    'offset': offset,
                    'size': atom_size,
                    'is_media_data': atom_type == b'mdat',
                    'is_metadata': atom_type in [b'meta', b'udta']
                })
                
                offset += atom_size
            except:
                # 残りを一括処理
                remaining = data[offset:]
                segments.append({
                    'type': 'MP4_REMAINING',
                    'data': remaining,
                    'offset': offset,
                    'size': len(remaining),
                    'is_media_data': False,
                    'is_metadata': False
                })
                break
        
        return segments
    
    def analyze_generic_chunks(self, data: bytes, chunk_size: int) -> List[Dict]:
        """汎用チャンク解析"""
        segments = []
        offset = 0
        chunk_index = 0
        
        while offset < len(data):
            current_size = min(chunk_size, len(data) - offset)
            chunk_data = data[offset:offset + current_size]
            
            segments.append({
                'type': f'CHUNK_{chunk_index:04d}',
                'data': chunk_data,
                'offset': offset,
                'size': current_size,
                'is_image_data': False,
                'is_metadata': False
            })
            
            offset += current_size
            chunk_index += 1
        
        return segments
    
    def optimize_compression_strategy(self, segment: Dict, media_type: str) -> str:
        """メディア特化圧縮戦略最適化"""
        data = segment['data']
        entropy = self.calculate_entropy(data)
        
        # 画像特化戦略
        if media_type in ['JPEG', 'PNG', 'BMP', 'GIF']:
            if segment.get('is_image_data', False):
                # 画像データ: エントロピーベース選択
                if entropy < 2.0:
                    return "rle"  # 低エントロピー
                elif entropy < 6.0:
                    return "lzma"  # 中エントロピー
                else:
                    return "delta_lzma"  # 高エントロピー（デルタ圧縮）
            elif segment.get('is_metadata', False):
                return "lzma"  # メタデータ
            else:
                return "zlib"  # その他
        
        # 動画特化戦略
        elif media_type in ['MP4', 'AVI', 'MOV']:
            if segment.get('is_media_data', False):
                # 動画データ: フレーム間差分最適化
                if entropy < 3.0:
                    return "frame_delta"
                elif entropy < 7.0:
                    return "lzma"
                else:
                    return "zlib"
            else:
                return "lzma"
        
        # 音声特化戦略
        elif media_type in ['MP3', 'WAV']:
            if entropy < 2.0:
                return "audio_rle"  # 無音部分等
            elif entropy < 5.0:
                return "lzma"
            else:
                return "zlib"
        
        # テキスト特化戦略
        elif media_type == 'TEXT':
            if entropy < 3.0:
                return "text_rle"
            else:
                return "lzma"
        
        # デフォルト戦略
        if entropy < 2.0:
            return "rle"
        elif entropy < 6.0:
            return "lzma"
        else:
            return "zlib"
    
    def media_optimized_compress(self, data: bytes, filename: str = "data") -> Dict:
        """メディア最適化圧縮"""
        start_time = time.time()
        original_size = len(data)
        
        print(f"🎯 メディア最適化圧縮開始: {filename}")
        print(f"📊 元サイズ: {original_size:,} bytes ({original_size/1024/1024:.1f} MB)")
        
        # メディア形式検出
        media_type = self.detect_media_type(data, filename)
        print(f"🎭 メディア形式: {media_type}")
        
        # 元データハッシュ
        original_hash = hashlib.sha256(data).hexdigest()
        print(f"🔐 原本ハッシュ: {original_hash[:16]}...")
        
        # メディア特化構造解析
        if media_type in ['JPEG', 'PNG', 'BMP', 'GIF']:
            segments = self.analyze_image_structure(data, media_type)
        elif media_type in ['MP4', 'AVI', 'MOV']:
            segments = self.analyze_video_structure(data, media_type)
        else:
            segments = self.analyze_generic_chunks(data, 16384)
        
        print(f"📈 構造解析: {len(segments)}セグメント")
        
        # セグメント別最適化圧縮
        compressed_segments = []
        segment_metadata = []
        total_segments = len(segments)
        
        for i, segment in enumerate(segments):
            # 最適化戦略決定
            strategy = self.optimize_compression_strategy(segment, media_type)
            
            # 圧縮実行
            compressed_data = self.apply_compression_strategy(segment['data'], strategy)
            
            # メタデータ保存
            metadata = {
                'original_size': segment['size'],
                'strategy': strategy,
                'segment_type': segment['type'],
                'offset': segment['offset']
            }
            
            compressed_segments.append(compressed_data)
            segment_metadata.append(metadata)
            
            # 進捗表示
            if (i + 1) % max(1, total_segments // 4) == 0:
                percent = ((i + 1) / total_segments) * 100
                print(f"🎯 最適化進捗: {percent:.0f}%")
        
        # 最終統合
        final_compressed = self.build_optimized_file(
            compressed_segments, segment_metadata, original_hash, media_type, original_size
        )
        
        compressed_size = len(final_compressed)
        compression_ratio = ((original_size - compressed_size) / original_size) * 100
        processing_time = time.time() - start_time
        
        # 戦略統計
        strategy_stats = {}
        for metadata in segment_metadata:
            strategy = metadata['strategy']
            strategy_stats[strategy] = strategy_stats.get(strategy, 0) + 1
        
        print(f"🎯 最適化戦略統計:")
        for strategy, count in strategy_stats.items():
            print(f"   {strategy}: {count}セグメント")
        
        print(f"✅ 最適化完了: {compression_ratio:.1f}% ({original_size:,} → {compressed_size:,})")
        print(f"⏱️ 処理時間: {processing_time:.2f}秒")
        
        return {
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
            'compressed_data': final_compressed,
            'processing_time': processing_time,
            'media_type': media_type,
            'segments_count': len(segments),
            'strategy_stats': strategy_stats,
            'original_hash': original_hash
        }\n    \n    def apply_compression_strategy(self, data: bytes, strategy: str) -> bytes:\n        \"\"\"圧縮戦略適用\"\"\"\n        if not data:\n            return b''\n        \n        try:\n            if strategy == \"rle\" or strategy == \"audio_rle\" or strategy == \"text_rle\":\n                return self.safe_rle_compress(data)\n            elif strategy == \"lzma\":\n                return lzma.compress(data, preset=6, check=lzma.CHECK_CRC64)\n            elif strategy == \"zlib\":\n                return zlib.compress(data, level=6)\n            elif strategy == \"delta_lzma\":\n                return self.delta_lzma_compress(data)\n            elif strategy == \"frame_delta\":\n                return self.frame_delta_compress(data)\n            else:\n                # フォールバック\n                return zlib.compress(data, level=3)\n        except:\n            # エラー時は元データ返却\n            return data\n    \n    def delta_lzma_compress(self, data: bytes) -> bytes:\n        \"\"\"デルタ圧縮 + LZMA\"\"\"\n        if len(data) < 16:\n            return data\n        \n        # 隣接バイト差分計算\n        delta = bytearray([data[0]])  # 最初のバイト\n        for i in range(1, len(data)):\n            diff = (data[i] - data[i-1]) % 256\n            delta.append(diff)\n        \n        # LZMA圧縮\n        try:\n            compressed = lzma.compress(bytes(delta), preset=3)\n            # 圧縮効果チェック\n            if len(compressed) < len(data):\n                return compressed\n        except:\n            pass\n        \n        return data\n    \n    def frame_delta_compress(self, data: bytes) -> bytes:\n        \"\"\"フレーム間差分圧縮\"\"\"\n        if len(data) < 64:\n            return data\n        \n        # 64バイトブロック単位で差分計算\n        block_size = 64\n        compressed = bytearray()\n        \n        # 最初のブロック\n        first_block = data[:block_size]\n        compressed.extend(first_block)\n        \n        # 差分ブロック\n        for i in range(block_size, len(data), block_size):\n            current_block = data[i:i+block_size]\n            prev_block = data[i-block_size:i]\n            \n            # ブロック間差分\n            diff_block = bytearray()\n            for j in range(len(current_block)):\n                if j < len(prev_block):\n                    diff = (current_block[j] - prev_block[j]) % 256\n                    diff_block.append(diff)\n                else:\n                    diff_block.append(current_block[j])\n            \n            compressed.extend(diff_block)\n        \n        # LZMA圧縮\n        try:\n            final_compressed = lzma.compress(bytes(compressed), preset=3)\n            if len(final_compressed) < len(data):\n                return final_compressed\n        except:\n            pass\n        \n        return data\n    \n    def safe_rle_compress(self, data: bytes) -> bytes:\n        \"\"\"安全なRLE圧縮\"\"\"\n        if not data:\n            return b''\n        \n        compressed = bytearray()\n        i = 0\n        \n        while i < len(data):\n            current_byte = data[i]\n            count = 1\n            \n            # 連続カウント\n            while (i + count < len(data) and \n                   data[i + count] == current_byte and \n                   count < 253):\n                count += 1\n            \n            if count >= 3:\n                # RLE: 254 count byte\n                compressed.extend([254, count, current_byte])\n                i += count\n            else:\n                # エスケープ処理\n                if current_byte == 254:\n                    compressed.extend([255, 254])\n                elif current_byte == 255:\n                    compressed.extend([255, 255])\n                else:\n                    compressed.append(current_byte)\n                i += 1\n        \n        return bytes(compressed)\n    \n    def build_optimized_file(self, compressed_segments: List[bytes], \n                           segment_metadata: List[Dict], original_hash: str,\n                           media_type: str, original_size: int) -> bytes:\n        \"\"\"最適化ファイル構築\"\"\"\n        result = bytearray()\n        \n        # ヘッダー\n        result.extend(self.magic_header)\n        \n        # 元データハッシュ\n        result.extend(original_hash.encode('ascii'))\n        \n        # 元サイズ\n        result.extend(struct.pack('<I', original_size))\n        \n        # メディア形式\n        media_type_bytes = media_type.encode('ascii')\n        result.extend(struct.pack('<H', len(media_type_bytes)))\n        result.extend(media_type_bytes)\n        \n        # セグメントメタデータ\n        metadata_json = json.dumps(segment_metadata, separators=(',', ':')).encode('utf-8')\n        metadata_compressed = lzma.compress(metadata_json, preset=9)\n        result.extend(struct.pack('<I', len(metadata_compressed)))\n        result.extend(metadata_compressed)\n        \n        # 圧縮セグメント\n        result.extend(struct.pack('<I', len(compressed_segments)))\n        for segment in compressed_segments:\n            result.extend(struct.pack('<I', len(segment)))\n            result.extend(segment)\n        \n        return bytes(result)\n    \n    def media_optimized_decompress(self, compressed_data: bytes) -> Dict:\n        \"\"\"メディア最適化展開\"\"\"\n        start_time = time.time()\n        \n        print(\"🎯 メディア最適化展開開始\")\n        \n        # ヘッダー検証\n        if not compressed_data.startswith(self.magic_header):\n            raise ValueError(\"❌ メディア最適化形式ではありません\")\n        \n        offset = len(self.magic_header)\n        \n        # 元データハッシュ\n        original_hash = compressed_data[offset:offset+64].decode('ascii')\n        offset += 64\n        print(f\"🔐 原本ハッシュ: {original_hash[:16]}...\")\n        \n        # 元サイズ\n        original_size = struct.unpack('<I', compressed_data[offset:offset+4])[0]\n        offset += 4\n        \n        # メディア形式\n        media_type_len = struct.unpack('<H', compressed_data[offset:offset+2])[0]\n        offset += 2\n        media_type = compressed_data[offset:offset+media_type_len].decode('ascii')\n        offset += media_type_len\n        print(f\"🎭 メディア形式: {media_type}\")\n        \n        # セグメントメタデータ\n        metadata_size = struct.unpack('<I', compressed_data[offset:offset+4])[0]\n        offset += 4\n        metadata_compressed = compressed_data[offset:offset+metadata_size]\n        offset += metadata_size\n        \n        metadata_json = lzma.decompress(metadata_compressed)\n        segment_metadata = json.loads(metadata_json.decode('utf-8'))\n        print(f\"📈 セグメント数: {len(segment_metadata)}\")\n        \n        # 圧縮セグメント数\n        segments_count = struct.unpack('<I', compressed_data[offset:offset+4])[0]\n        offset += 4\n        \n        # セグメント展開\n        decompressed_segments = []\n        for i in range(segments_count):\n            segment_size = struct.unpack('<I', compressed_data[offset:offset+4])[0]\n            offset += 4\n            \n            segment_data = compressed_data[offset:offset+segment_size]\n            offset += segment_size\n            \n            # 展開戦略適用\n            if i < len(segment_metadata):\n                strategy = segment_metadata[i]['strategy']\n                decompressed = self.apply_decompression_strategy(segment_data, strategy)\n            else:\n                decompressed = segment_data\n            \n            decompressed_segments.append(decompressed)\n            \n            # 進捗表示\n            if (i + 1) % max(1, segments_count // 4) == 0:\n                percent = ((i + 1) / segments_count) * 100\n                print(f\"🎯 展開進捗: {percent:.0f}%\")\n        \n        # 元データ復元\n        original_data = b''.join(decompressed_segments)\n        \n        # 可逆性検証\n        restored_hash = hashlib.sha256(original_data).hexdigest()\n        is_identical = (restored_hash == original_hash)\n        \n        processing_time = time.time() - start_time\n        print(f\"✅ 展開完了: {len(original_data):,} bytes ({processing_time:.2f}秒)\")\n        print(f\"🔍 可逆性検証: {'✅ 完全一致' if is_identical else '❌ 不一致'}\")\n        \n        if not is_identical:\n            print(f\"⚠️ 原本: {original_hash[:16]}...\")\n            print(f\"⚠️ 復元: {restored_hash[:16]}...\")\n            raise ValueError(\"❌ 可逆性検証失敗\")\n        \n        return {\n            'original_data': original_data,\n            'decompressed_size': len(original_data),\n            'processing_time': processing_time,\n            'media_type': media_type,\n            'is_reversible': is_identical\n        }\n    \n    def apply_decompression_strategy(self, data: bytes, strategy: str) -> bytes:\n        \"\"\"展開戦略適用\"\"\"\n        try:\n            if strategy in [\"rle\", \"audio_rle\", \"text_rle\"]:\n                return self.safe_rle_decompress(data)\n            elif strategy == \"lzma\":\n                return lzma.decompress(data)\n            elif strategy == \"zlib\":\n                return zlib.decompress(data)\n            elif strategy == \"delta_lzma\":\n                return self.delta_lzma_decompress(data)\n            elif strategy == \"frame_delta\":\n                return self.frame_delta_decompress(data)\n            else:\n                return zlib.decompress(data)\n        except:\n            return data\n    \n    def delta_lzma_decompress(self, data: bytes) -> bytes:\n        \"\"\"デルタ圧縮展開\"\"\"\n        try:\n            # LZMA展開\n            delta_data = lzma.decompress(data)\n            \n            # 差分復元\n            if len(delta_data) > 0:\n                result = bytearray([delta_data[0]])\n                for i in range(1, len(delta_data)):\n                    restored_byte = (result[-1] + delta_data[i]) % 256\n                    result.append(restored_byte)\n                return bytes(result)\n        except:\n            pass\n        \n        return data\n    \n    def frame_delta_decompress(self, data: bytes) -> bytes:\n        \"\"\"フレーム間差分展開\"\"\"\n        try:\n            # LZMA展開\n            diff_data = lzma.decompress(data)\n            \n            # フレーム復元\n            block_size = 64\n            if len(diff_data) >= block_size:\n                result = bytearray(diff_data[:block_size])  # 最初のブロック\n                \n                for i in range(block_size, len(diff_data), block_size):\n                    diff_block = diff_data[i:i+block_size]\n                    prev_block = result[i-block_size:i]\n                    \n                    # 差分復元\n                    for j in range(len(diff_block)):\n                        if j < len(prev_block):\n                            restored_byte = (prev_block[j] + diff_block[j]) % 256\n                            result.append(restored_byte)\n                        else:\n                            result.append(diff_block[j])\n                \n                return bytes(result)\n        except:\n            pass\n        \n        return data\n    \n    def safe_rle_decompress(self, data: bytes) -> bytes:\n        \"\"\"安全なRLE展開\"\"\"\n        if not data:\n            return b''\n        \n        result = bytearray()\n        i = 0\n        \n        while i < len(data):\n            if data[i] == 254 and i + 2 < len(data):\n                # RLE展開\n                count = data[i + 1]\n                byte_value = data[i + 2]\n                result.extend([byte_value] * count)\n                i += 3\n            elif data[i] == 255 and i + 1 < len(data):\n                # エスケープ展開\n                result.append(data[i + 1])\n                i += 2\n            else:\n                result.append(data[i])\n                i += 1\n        \n        return bytes(result)\n    \n    def compress_file(self, input_path: str, output_path: str = None) -> bool:\n        \"\"\"ファイル圧縮\"\"\"\n        if not os.path.exists(input_path):\n            print(f\"❌ ファイルが見つかりません: {input_path}\")\n            return False\n        \n        if output_path is None:\n            output_path = input_path + '.p8o'  # Phase 8 Optimized\n        \n        try:\n            with open(input_path, 'rb') as f:\n                data = f.read()\n            \n            filename = os.path.basename(input_path)\n            result = self.media_optimized_compress(data, filename)\n            \n            with open(output_path, 'wb') as f:\n                f.write(result['compressed_data'])\n            \n            print(f\"💾 最適化圧縮ファイル保存: {output_path}\")\n            return True\n        \n        except Exception as e:\n            print(f\"❌ 圧縮エラー: {e}\")\n            return False\n    \n    def decompress_file(self, input_path: str, output_path: str = None) -> bool:\n        \"\"\"ファイル展開\"\"\"\n        if not os.path.exists(input_path):\n            print(f\"❌ ファイルが見つかりません: {input_path}\")\n            return False\n        \n        if output_path is None:\n            if input_path.endswith('.p8o'):\n                output_path = input_path[:-4]\n            else:\n                output_path = input_path + '.restored'\n        \n        try:\n            with open(input_path, 'rb') as f:\n                compressed_data = f.read()\n            \n            result = self.media_optimized_decompress(compressed_data)\n            \n            with open(output_path, 'wb') as f:\n                f.write(result['original_data'])\n            \n            print(f\"📁 最適化復元ファイル保存: {output_path}\")\n            return True\n        \n        except Exception as e:\n            print(f\"❌ 展開エラー: {e}\")\n            return False\n\ndef run_media_optimization_test():\n    \"\"\"メディア最適化総合テスト\"\"\"\n    print(\"🎯 Phase 8 メディア最適化総合テスト\")\n    print(\"=\" * 60)\n    \n    engine = MediaOptimizedEngine()\n    sample_dir = Path(\"../NXZip-Python/sample\")\n    \n    # 全メディア形式テスト\n    test_files = [\n        # 画像ファイル\n        (\"COT-001.jpg\", 1024*1024, \"JPEG画像 (1MB)\"),\n        (\"COT-012.png\", 2*1024*1024, \"PNG画像 (2MB)\"),\n        \n        # 動画ファイル\n        (\"Python基礎講座3_4月26日-3.mp4\", 3*1024*1024, \"MP4動画 (3MB)\"),\n        \n        # 音声ファイル\n        (\"陰謀論.mp3\", 1024*1024, \"MP3音声 (1MB)\"),\n        \n        # テキストファイル\n        (\"出庫実績明細_202412.txt\", 2*1024*1024, \"テキスト (2MB)\"),\n    ]\n    \n    results = []\n    \n    for filename, size_limit, description in test_files:\n        filepath = sample_dir / filename\n        if not filepath.exists():\n            print(f\"⚠️ ファイルなし: {filename}\")\n            continue\n        \n        print(f\"\\n🎯 最適化テスト: {description}\")\n        print(\"-\" * 50)\n        \n        try:\n            with open(filepath, 'rb') as f:\n                test_data = f.read(size_limit)\n            print(f\"📏 テストサイズ: {len(test_data):,} bytes\")\n            \n            # 最適化圧縮\n            result = engine.media_optimized_compress(test_data, filename)\n            \n            # 最適化展開\n            decompressed = engine.media_optimized_decompress(result['compressed_data'])\n            \n            # 結果保存\n            results.append({\n                'filename': filename,\n                'description': description,\n                'original_size': len(test_data),\n                'compressed_size': result['compressed_size'],\n                'compression_ratio': result['compression_ratio'],\n                'reversible': decompressed['is_reversible'],\n                'processing_time': result['processing_time'],\n                'media_type': result['media_type'],\n                'segments_count': result['segments_count'],\n                'strategy_stats': result['strategy_stats']\n            })\n            \n            print(f\"✅ 最適化成功: 可逆性 {'✅' if decompressed['is_reversible'] else '❌'}\")\n            \n        except Exception as e:\n            print(f\"❌ テストエラー: {str(e)[:60]}...\")\n    \n    # 総合結果\n    if results:\n        print(\"\\n\" + \"=\" * 60)\n        print(\"🏆 Phase 8 メディア最適化総合テスト結果\")\n        print(\"=\" * 60)\n        \n        total_original = sum(r['original_size'] for r in results)\n        total_compressed = sum(r['compressed_size'] for r in results)\n        overall_ratio = (1 - total_compressed / total_original) * 100\n        reversible_count = sum(1 for r in results if r['reversible'])\n        \n        print(f\"🎯 最適化圧縮率: {overall_ratio:.1f}%\")\n        print(f\"🔒 可逆性成功率: {reversible_count}/{len(results)} ({reversible_count/len(results)*100:.1f}%)\")\n        print(f\"📈 テストファイル数: {len(results)}\")\n        print(f\"💾 総データ量: {total_original/1024/1024:.1f} MB\")\n        \n        # メディア別分析\n        print(f\"\\n📊 メディア別最適化結果:\")\n        for result in results:\n            name = result['filename'][:25]\n            size_mb = result['original_size'] / 1024 / 1024\n            rev_icon = '✅' if result['reversible'] else '❌'\n            \n            print(f\"   🎬 {result['description']}: {result['compression_ratio']:.1f}% ({size_mb:.1f}MB) {rev_icon}\")\n            print(f\"      🎭 形式: {result['media_type']}, セグメント: {result['segments_count']}\")\n            print(f\"      🔧 戦略: {', '.join(f'{k}({v})' for k, v in result['strategy_stats'].items())}\")\n        \n        # 改善提案\n        high_compression = [r for r in results if r['compression_ratio'] >= 50]\n        low_compression = [r for r in results if r['compression_ratio'] < 20]\n        \n        if high_compression:\n            print(f\"\\n🏅 高圧縮率達成 ({len(high_compression)}個):\")\n            for r in high_compression:\n                print(f\"   🌟 {r['description']}: {r['compression_ratio']:.1f}% - 優秀\")\n        \n        if low_compression:\n            print(f\"\\n⚠️ 低圧縮率 ({len(low_compression)}個):\")\n            for r in low_compression:\n                print(f\"   🔧 {r['description']}: {r['compression_ratio']:.1f}% - 更なる特化必要\")\n        \n        if reversible_count == len(results):\n            print(\"\\n🎉 全メディア最適化可逆性達成！画像・動画の圧縮率向上実現！\")\n        else:\n            failed_count = len(results) - reversible_count\n            print(f\"\\n⚠️ {failed_count}ファイルで可逆性問題\")\n\ndef main():\n    \"\"\"メイン処理\"\"\"\n    if len(sys.argv) < 2:\n        print(\"🎯 Phase 8 メディア最適化エンジン\")\n        print(\"使用方法:\")\n        print(\"  python phase8_media_optimized.py test                     # 最適化テスト\")\n        print(\"  python phase8_media_optimized.py compress <file>          # 最適化圧縮\")\n        print(\"  python phase8_media_optimized.py decompress <file.p8o>    # 最適化展開\")\n        return\n    \n    command = sys.argv[1].lower()\n    engine = MediaOptimizedEngine()\n    \n    if command == \"test\":\n        run_media_optimization_test()\n    elif command == \"compress\" and len(sys.argv) >= 3:\n        input_file = sys.argv[2]\n        output_file = sys.argv[3] if len(sys.argv) >= 4 else None\n        engine.compress_file(input_file, output_file)\n    elif command == \"decompress\" and len(sys.argv) >= 3:\n        input_file = sys.argv[2]\n        output_file = sys.argv[3] if len(sys.argv) >= 4 else None\n        engine.decompress_file(input_file, output_file)\n    else:\n        print(\"❌ 無効なコマンドです\")\n\nif __name__ == \"__main__\":\n    main()
