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

class MediaOptimizedEngine:
    """メディア最適化エンジン - 可逆性保証版"""
    
    def __init__(self):
        self.version = "8.0-MediaOptimized"
        self.magic_header = b'NXZ8O'
    
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
        elif b'ftyp' in data[:32]:
            return "MP4"
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
    
    def analyze_structure(self, data: bytes, media_type: str) -> list:
        """構造解析"""
        if media_type == "JPEG":
            return self.analyze_jpeg(data)
        elif media_type == "PNG":
            return self.analyze_png(data)
        elif media_type == "MP4":
            return self.analyze_mp4(data)
        else:
            return self.analyze_generic(data)
    
    def analyze_jpeg(self, data: bytes) -> list:
        """JPEG解析"""
        segments = []
        offset = 0
        
        while offset < len(data) - 1:
            if data[offset] == 0xFF and data[offset + 1] != 0xFF:
                marker = data[offset + 1]
                start = offset
                
                if marker in [0xD8, 0xD9]:
                    size = 2
                elif marker == 0xDA:
                    # 画像データ終端まで
                    end = self.find_jpeg_end(data, offset + 2)
                    size = end - offset
                else:
                    if offset + 3 < len(data):
                        size = struct.unpack('>H', data[offset + 2:offset + 4])[0] + 2
                    else:
                        size = len(data) - offset
                
                chunk = data[start:start + size]
                segments.append({
                    'data': chunk,
                    'type': f'JPEG_{marker:02X}',
                    'is_image': marker == 0xDA
                })
                offset += size
            else:
                offset += 1
        
        return segments
    
    def find_jpeg_end(self, data: bytes, start: int) -> int:
        """JPEG終端検索"""
        pos = start
        while pos < len(data) - 1:
            if data[pos] == 0xFF and data[pos + 1] == 0xD9:
                return pos + 2
            pos += 1
        return len(data)
    
    def analyze_png(self, data: bytes) -> list:
        """PNG解析"""
        segments = []
        offset = 8  # PNG署名スキップ
        
        while offset < len(data) - 8:
            try:
                chunk_size = struct.unpack('>I', data[offset:offset + 4])[0]
                chunk_type = data[offset + 4:offset + 8]
                total_size = chunk_size + 12
                
                chunk = data[offset:offset + total_size]
                segments.append({
                    'data': chunk,
                    'type': f'PNG_{chunk_type.decode("ascii", errors="ignore")}',
                    'is_image': chunk_type == b'IDAT'
                })
                offset += total_size
            except:
                remaining = data[offset:]
                segments.append({
                    'data': remaining,
                    'type': 'PNG_REST',
                    'is_image': False
                })
                break
        
        return segments
    
    def analyze_mp4(self, data: bytes) -> list:
        """MP4解析"""
        segments = []
        offset = 0
        
        while offset < len(data) - 8:
            try:
                atom_size = struct.unpack('>I', data[offset:offset + 4])[0]
                atom_type = data[offset + 4:offset + 8]
                
                if atom_size == 0:
                    atom_size = len(data) - offset
                
                atom = data[offset:offset + atom_size]
                segments.append({
                    'data': atom,
                    'type': f'MP4_{atom_type.decode("ascii", errors="ignore")}',
                    'is_media': atom_type == b'mdat'
                })
                offset += atom_size
            except:
                remaining = data[offset:]
                segments.append({
                    'data': remaining,
                    'type': 'MP4_REST',
                    'is_media': False
                })
                break
        
        return segments
    
    def analyze_generic(self, data: bytes) -> list:
        """汎用解析"""
        segments = []
        chunk_size = 16384
        offset = 0
        index = 0
        
        while offset < len(data):
            size = min(chunk_size, len(data) - offset)
            chunk = data[offset:offset + size]
            segments.append({
                'data': chunk,
                'type': f'CHUNK_{index:04d}',
                'is_image': False
            })
            offset += size
            index += 1
        
        return segments
    
    def select_strategy(self, segment: dict, media_type: str) -> str:
        """圧縮戦略選択"""
        data = segment['data']
        entropy = self.calculate_entropy(data)
        
        # 画像データ特化
        if segment.get('is_image', False):
            if entropy < 3.0:
                return "rle"
            elif entropy < 6.0:
                return "lzma"
            else:
                return "delta"
        
        # 動画データ特化
        elif segment.get('is_media', False):
            if entropy < 4.0:
                return "frame_delta"
            else:
                return "lzma"
        
        # 一般戦略
        elif entropy < 2.0:
            return "rle"
        elif entropy < 6.0:
            return "lzma"
        else:
            return "zlib"
    
    def apply_compression(self, data: bytes, strategy: str) -> bytes:
        """圧縮適用"""
        if not data:
            return b''
        
        try:
            if strategy == "rle":
                return self.rle_compress(data)
            elif strategy == "lzma":
                return lzma.compress(data, preset=6, check=lzma.CHECK_CRC64)
            elif strategy == "zlib":
                return zlib.compress(data, level=6)
            elif strategy == "delta":
                return self.delta_compress(data)
            elif strategy == "frame_delta":
                return self.frame_delta_compress(data)
            else:
                return zlib.compress(data, level=3)
        except:
            return data
    
    def rle_compress(self, data: bytes) -> bytes:
        """RLE圧縮"""
        if not data:
            return b''
        
        result = bytearray()
        i = 0
        
        while i < len(data):
            byte = data[i]
            count = 1
            
            while (i + count < len(data) and 
                   data[i + count] == byte and 
                   count < 253):
                count += 1
            
            if count >= 3:
                result.extend([254, count, byte])
                i += count
            else:
                if byte == 254:
                    result.extend([255, 254])
                elif byte == 255:
                    result.extend([255, 255])
                else:
                    result.append(byte)
                i += 1
        
        return bytes(result)
    
    def delta_compress(self, data: bytes) -> bytes:
        """デルタ圧縮"""
        if len(data) < 16:
            return data
        
        delta = bytearray([data[0]])
        for i in range(1, len(data)):
            diff = (data[i] - data[i-1]) % 256
            delta.append(diff)
        
        try:
            compressed = lzma.compress(bytes(delta), preset=3)
            return compressed if len(compressed) < len(data) else data
        except:
            return data
    
    def frame_delta_compress(self, data: bytes) -> bytes:
        """フレーム差分圧縮"""
        if len(data) < 64:
            return data
        
        block_size = 64
        result = bytearray(data[:block_size])  # 最初のブロック
        
        for i in range(block_size, len(data), block_size):
            current = data[i:i+block_size]
            prev = data[i-block_size:i]
            
            diff = bytearray()
            for j in range(len(current)):
                if j < len(prev):
                    d = (current[j] - prev[j]) % 256
                    diff.append(d)
                else:
                    diff.append(current[j])
            
            result.extend(diff)
        
        try:
            compressed = lzma.compress(bytes(result), preset=3)
            return compressed if len(compressed) < len(data) else data
        except:
            return data
    
    def media_compress(self, data: bytes, filename: str = "data") -> dict:
        """メディア圧縮"""
        start_time = time.time()
        original_size = len(data)
        
        print(f"🎯 メディア最適化圧縮: {filename}")
        print(f"📊 サイズ: {original_size:,} bytes")
        
        # 形式検出
        media_type = self.detect_media_type(data, filename)
        print(f"🎭 形式: {media_type}")
        
        # ハッシュ計算
        original_hash = hashlib.sha256(data).hexdigest()
        
        # 構造解析
        segments = self.analyze_structure(data, media_type)
        print(f"📈 セグメント: {len(segments)}")
        
        # 圧縮実行
        compressed_data = []
        metadata = []
        
        for i, segment in enumerate(segments):
            strategy = self.select_strategy(segment, media_type)
            compressed = self.apply_compression(segment['data'], strategy)
            
            compressed_data.append(compressed)
            metadata.append({
                'strategy': strategy,
                'original_size': len(segment['data']),
                'type': segment['type']
            })
            
            if (i + 1) % max(1, len(segments) // 4) == 0:
                progress = (i + 1) / len(segments) * 100
                print(f"🎯 進捗: {progress:.0f}%")
        
        # ファイル構築
        final_data = self.build_file(compressed_data, metadata, original_hash, media_type, original_size)
        
        compressed_size = len(final_data)
        ratio = (1 - compressed_size / original_size) * 100
        time_taken = time.time() - start_time
        
        # 戦略統計
        strategy_count = {}
        for m in metadata:
            s = m['strategy']
            strategy_count[s] = strategy_count.get(s, 0) + 1
        
        print(f"🎯 戦略統計: {strategy_count}")
        print(f"✅ 完了: {ratio:.1f}% ({original_size:,} → {compressed_size:,})")
        print(f"⏱️ 時間: {time_taken:.2f}秒")
        
        return {
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': ratio,
            'compressed_data': final_data,
            'processing_time': time_taken,
            'media_type': media_type,
            'original_hash': original_hash
        }
    
    def build_file(self, compressed_data: list, metadata: list, 
                  original_hash: str, media_type: str, original_size: int) -> bytes:
        """ファイル構築"""
        result = bytearray()
        
        # ヘッダー
        result.extend(self.magic_header)
        result.extend(original_hash.encode('ascii'))
        result.extend(struct.pack('<I', original_size))
        
        # メディア形式
        media_bytes = media_type.encode('ascii')
        result.extend(struct.pack('<H', len(media_bytes)))
        result.extend(media_bytes)
        
        # メタデータ
        meta_json = json.dumps(metadata).encode('utf-8')
        meta_compressed = lzma.compress(meta_json, preset=9)
        result.extend(struct.pack('<I', len(meta_compressed)))
        result.extend(meta_compressed)
        
        # 圧縮データ
        result.extend(struct.pack('<I', len(compressed_data)))
        for data in compressed_data:
            result.extend(struct.pack('<I', len(data)))
            result.extend(data)
        
        return bytes(result)
    
    def media_decompress(self, compressed_data: bytes) -> dict:
        """メディア展開"""
        start_time = time.time()
        
        print("🎯 メディア最適化展開開始")
        
        if not compressed_data.startswith(self.magic_header):
            raise ValueError("❌ 形式エラー")
        
        offset = len(self.magic_header)
        
        # ハッシュ
        original_hash = compressed_data[offset:offset+64].decode('ascii')
        offset += 64
        
        # 元サイズ
        original_size = struct.unpack('<I', compressed_data[offset:offset+4])[0]
        offset += 4
        
        # メディア形式
        media_len = struct.unpack('<H', compressed_data[offset:offset+2])[0]
        offset += 2
        media_type = compressed_data[offset:offset+media_len].decode('ascii')
        offset += media_len
        
        print(f"🎭 形式: {media_type}")
        
        # メタデータ
        meta_size = struct.unpack('<I', compressed_data[offset:offset+4])[0]
        offset += 4
        meta_compressed = compressed_data[offset:offset+meta_size]
        offset += meta_size
        
        meta_json = lzma.decompress(meta_compressed)
        metadata = json.loads(meta_json.decode('utf-8'))
        
        # セグメント数
        segments_count = struct.unpack('<I', compressed_data[offset:offset+4])[0]
        offset += 4
        
        # 展開
        decompressed_segments = []
        for i in range(segments_count):
            size = struct.unpack('<I', compressed_data[offset:offset+4])[0]
            offset += 4
            
            segment_data = compressed_data[offset:offset+size]
            offset += size
            
            if i < len(metadata):
                strategy = metadata[i]['strategy']
                decompressed = self.apply_decompression(segment_data, strategy)
            else:
                decompressed = segment_data
            
            decompressed_segments.append(decompressed)
        
        # 結合
        original_data = b''.join(decompressed_segments)
        
        # 検証
        restored_hash = hashlib.sha256(original_data).hexdigest()
        is_valid = (restored_hash == original_hash)
        
        time_taken = time.time() - start_time
        print(f"✅ 展開完了: {len(original_data):,} bytes")
        print(f"🔍 可逆性: {'✅' if is_valid else '❌'}")
        
        if not is_valid:
            raise ValueError("❌ 可逆性検証失敗")
        
        return {
            'original_data': original_data,
            'processing_time': time_taken,
            'is_reversible': is_valid
        }
    
    def apply_decompression(self, data: bytes, strategy: str) -> bytes:
        """展開適用"""
        try:
            if strategy == "rle":
                return self.rle_decompress(data)
            elif strategy == "lzma":
                return lzma.decompress(data)
            elif strategy == "zlib":
                return zlib.decompress(data)
            elif strategy == "delta":
                return self.delta_decompress(data)
            elif strategy == "frame_delta":
                return self.frame_delta_decompress(data)
            else:
                return zlib.decompress(data)
        except:
            return data
    
    def rle_decompress(self, data: bytes) -> bytes:
        """RLE展開"""
        if not data:
            return b''
        
        result = bytearray()
        i = 0
        
        while i < len(data):
            if data[i] == 254 and i + 2 < len(data):
                count = data[i + 1]
                byte = data[i + 2]
                result.extend([byte] * count)
                i += 3
            elif data[i] == 255 and i + 1 < len(data):
                result.append(data[i + 1])
                i += 2
            else:
                result.append(data[i])
                i += 1
        
        return bytes(result)
    
    def delta_decompress(self, data: bytes) -> bytes:
        """デルタ展開"""
        try:
            delta_data = lzma.decompress(data)
            if len(delta_data) > 0:
                result = bytearray([delta_data[0]])
                for i in range(1, len(delta_data)):
                    byte = (result[-1] + delta_data[i]) % 256
                    result.append(byte)
                return bytes(result)
        except:
            pass
        return data
    
    def frame_delta_decompress(self, data: bytes) -> bytes:
        """フレーム差分展開"""
        try:
            diff_data = lzma.decompress(data)
            block_size = 64
            
            if len(diff_data) >= block_size:
                result = bytearray(diff_data[:block_size])
                
                for i in range(block_size, len(diff_data), block_size):
                    diff_block = diff_data[i:i+block_size]
                    prev_block = result[i-block_size:i]
                    
                    for j in range(len(diff_block)):
                        if j < len(prev_block):
                            byte = (prev_block[j] + diff_block[j]) % 256
                            result.append(byte)
                        else:
                            result.append(diff_block[j])
                
                return bytes(result)
        except:
            pass
        return data
    
    def compress_file(self, input_path: str, output_path: str = None) -> bool:
        """ファイル圧縮"""
        if not os.path.exists(input_path):
            print(f"❌ ファイルなし: {input_path}")
            return False
        
        if output_path is None:
            output_path = input_path + '.p8o'
        
        try:
            with open(input_path, 'rb') as f:
                data = f.read()
            
            filename = os.path.basename(input_path)
            result = self.media_compress(data, filename)
            
            with open(output_path, 'wb') as f:
                f.write(result['compressed_data'])
            
            print(f"💾 保存: {output_path}")
            return True
        
        except Exception as e:
            print(f"❌ エラー: {e}")
            return False
    
    def decompress_file(self, input_path: str, output_path: str = None) -> bool:
        """ファイル展開"""
        if not os.path.exists(input_path):
            print(f"❌ ファイルなし: {input_path}")
            return False
        
        if output_path is None:
            if input_path.endswith('.p8o'):
                output_path = input_path[:-4]
            else:
                output_path = input_path + '.restored'
        
        try:
            with open(input_path, 'rb') as f:
                compressed_data = f.read()
            
            result = self.media_decompress(compressed_data)
            
            with open(output_path, 'wb') as f:
                f.write(result['original_data'])
            
            print(f"📁 復元: {output_path}")
            return True
        
        except Exception as e:
            print(f"❌ エラー: {e}")
            return False

def run_media_test():
    """メディア最適化テスト"""
    print("🎯 Phase 8 メディア最適化テスト")
    print("=" * 50)
    
    engine = MediaOptimizedEngine()
    sample_dir = Path("../NXZip-Python/sample")
    
    test_files = [
        ("COT-001.jpg", 1024*1024, "JPEG画像"),
        ("COT-012.png", 2*1024*1024, "PNG画像"), 
        ("Python基礎講座3_4月26日-3.mp4", 3*1024*1024, "MP4動画"),
        ("陰謀論.mp3", 1024*1024, "MP3音声"),
        ("出庫実績明細_202412.txt", 2*1024*1024, "テキスト"),
    ]
    
    results = []
    
    for filename, limit, desc in test_files:
        filepath = sample_dir / filename
        if not filepath.exists():
            print(f"⚠️ スキップ: {filename}")
            continue
        
        print(f"\n🎯 テスト: {desc}")
        print("-" * 30)
        
        try:
            with open(filepath, 'rb') as f:
                test_data = f.read(limit)
            
            # 圧縮
            comp_result = engine.media_compress(test_data, filename)
            
            # 展開
            decomp_result = engine.media_decompress(comp_result['compressed_data'])
            
            results.append({
                'filename': filename,
                'desc': desc,
                'original_size': len(test_data),
                'compressed_size': comp_result['compressed_size'],
                'ratio': comp_result['compression_ratio'],
                'reversible': decomp_result['is_reversible'],
                'media_type': comp_result['media_type']
            })
            
        except Exception as e:
            print(f"❌ エラー: {str(e)[:50]}...")
    
    # 結果表示
    if results:
        print("\n" + "=" * 50)
        print("🏆 メディア最適化結果")
        print("=" * 50)
        
        total_orig = sum(r['original_size'] for r in results)
        total_comp = sum(r['compressed_size'] for r in results)
        overall_ratio = (1 - total_comp / total_orig) * 100
        reversible_count = sum(1 for r in results if r['reversible'])
        
        print(f"🎯 総合圧縮率: {overall_ratio:.1f}%")
        print(f"🔒 可逆性: {reversible_count}/{len(results)} ({reversible_count/len(results)*100:.1f}%)")
        print(f"📊 テスト数: {len(results)}")
        
        print(f"\n📋 詳細結果:")
        for r in results:
            size_mb = r['original_size'] / 1024 / 1024
            icon = '✅' if r['reversible'] else '❌'
            print(f"   {icon} {r['desc']}: {r['ratio']:.1f}% ({size_mb:.1f}MB, {r['media_type']})")
        
        if reversible_count == len(results):
            print("\n🎉 全メディア最適化成功！")
        else:
            print(f"\n⚠️ {len(results) - reversible_count}ファイルで問題")

def main():
    """メイン処理"""
    if len(sys.argv) < 2:
        print("🎯 Phase 8 メディア最適化エンジン")
        print("使用方法:")
        print("  python phase8_media_final.py test")
        print("  python phase8_media_final.py compress <file>")
        print("  python phase8_media_final.py decompress <file.p8o>")
        return
    
    command = sys.argv[1].lower()
    engine = MediaOptimizedEngine()
    
    if command == "test":
        run_media_test()
    elif command == "compress" and len(sys.argv) >= 3:
        engine.compress_file(sys.argv[2], sys.argv[3] if len(sys.argv) >= 4 else None)
    elif command == "decompress" and len(sys.argv) >= 3:
        engine.decompress_file(sys.argv[2], sys.argv[3] if len(sys.argv) >= 4 else None)
    else:
        print("❌ 無効なコマンド")

if __name__ == "__main__":
    main()
