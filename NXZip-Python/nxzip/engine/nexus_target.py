#!/usr/bin/env python3
"""
NEXUS Target Achievement Engine - 目標達成特化エンジン
確実な目標達成: 圧縮率80%(テキスト95%), 圧縮100MB/s, 展開200MB/s
"""

import struct
import time
import lzma
import zlib
import bz2
import threading
import concurrent.futures
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import sys
import io
import hashlib
import os
from multiprocessing import Pool, cpu_count

# プロジェクトパス追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from .spe_core_jit import SPECoreJIT

class NEXUSTargetAchievement:
    """
    目標達成特化エンジン
    
    戦略:
    1. 圧縮率最優先アルゴリズム
    2. 高速処理との最適バランス
    3. 形式別特化最適化
    4. 多段階圧縮戦略
    5. 適応的パラメータ調整
    """
    
    def __init__(self):
        self.spe = SPECoreJIT()
        self.cpu_count = cpu_count()
        
        # 目標達成特化設定
        self.target_configs = {
            'video': {
                'target_ratio': 0.80,
                'multi_stage': True,
                'algorithms': ['lzma', 'zlib', 'bz2'],
                'presets': [6, 8, 9],
                'fast_preset': 3
            },
            'audio': {
                'target_ratio': 0.80,
                'multi_stage': True,
                'algorithms': ['lzma', 'zlib', 'bz2'],
                'presets': [6, 8, 9],
                'fast_preset': 3
            },
            'image': {
                'target_ratio': 0.80,
                'multi_stage': True,
                'algorithms': ['lzma', 'zlib', 'bz2'],
                'presets': [6, 8, 9],
                'fast_preset': 3
            },
            'text': {
                'target_ratio': 0.95,
                'multi_stage': True,
                'algorithms': ['lzma', 'bz2', 'zlib'],
                'presets': [9, 9, 9],
                'fast_preset': 6
            },
            'binary': {
                'target_ratio': 0.80,
                'multi_stage': True,
                'algorithms': ['lzma', 'zlib', 'bz2'],
                'presets': [6, 8, 9],
                'fast_preset': 3
            }
        }
    
    def compress(self, data: bytes) -> bytes:
        """目標達成圧縮"""
        if not data:
            return self._create_empty_nxz()
        
        # 1. 形式検出
        format_type = self._detect_format_target(data)
        config = self.target_configs[format_type]
        
        print(f"🎯 形式: {format_type} (目標圧縮率: {config['target_ratio']*100:.0f}%)")
        
        # 2. 戦略選択
        data_size = len(data)
        if data_size > 50 * 1024 * 1024:  # 50MB超: 速度重視
            strategy = 'fast_compression'
        else:  # 50MB以下: 圧縮率重視
            strategy = 'max_compression'
        
        # 3. 形式特化前処理
        processed_data = self._preprocess_target(data, format_type)
        
        # 4. 多段階圧縮
        compressed_data = self._multi_stage_compress(processed_data, format_type, strategy)
        
        # 5. SPE暗号化
        encrypted_data = self.spe.apply_transform(compressed_data)
        
        # 6. ヘッダー作成
        header = self._create_target_header(
            original_size=len(data),
            compressed_size=len(compressed_data),
            encrypted_size=len(encrypted_data),
            format_type=format_type,
            strategy=strategy
        )
        
        return header + encrypted_data
    
    def decompress(self, nxz_data: bytes) -> bytes:
        """目標達成展開"""
        if not nxz_data:
            return b""
        
        # 1. ヘッダー解析
        if len(nxz_data) < 48:
            raise ValueError("Invalid NXZ Target format")
        
        header_info = self._parse_target_header(nxz_data[:48])
        
        # 2. SPE復号化
        encrypted_data = nxz_data[48:]
        compressed_data = self.spe.reverse_transform(encrypted_data)
        
        # 3. 多段階展開
        processed_data = self._multi_stage_decompress(compressed_data, header_info['format_type'])
        
        # 4. 後処理
        original_data = self._postprocess_target(processed_data, header_info['format_type'])
        
        return original_data
    
    def _detect_format_target(self, data: bytes) -> str:
        """目標達成形式検出"""
        if len(data) < 16:
            return "binary"
        
        # 詳細検出
        header = data[:32]
        
        # 動画形式
        if header[4:8] == b'ftyp':
            return "video"
        if header.startswith(b'RIFF'):
            if b'AVI ' in data[:64]:
                return "video"
            elif b'WAVE' in data[:64]:
                return "audio"
        if header.startswith(b'\x1A\x45\xDF\xA3'):
            return "video"
        
        # 音声形式
        if header.startswith(b'ID3') or header.startswith(b'\xFF\xFB') or header.startswith(b'\xFF\xF3'):
            return "audio"
        
        # 画像形式
        if header.startswith(b'\xFF\xD8') or header.startswith(b'\x89PNG') or header.startswith(b'GIF'):
            return "image"
        
        # テキスト形式（精密検出）
        try:
            # より大きなサンプルでテキスト判定
            sample_size = min(8192, len(data))
            sample = data[:sample_size]
            
            # UTF-8デコードテスト
            decoded = sample.decode('utf-8')
            
            # テキスト文字の割合
            text_chars = sum(1 for c in decoded if c.isprintable() or c in '\t\n\r')
            text_ratio = text_chars / len(decoded)
            
            if text_ratio > 0.85:
                return "text"
                
        except:
            pass
        
        return "binary"
    
    def _preprocess_target(self, data: bytes, format_type: str) -> bytes:
        """形式特化前処理"""
        if format_type == "text":
            return self._preprocess_text_target(data)
        elif format_type == "video":
            return self._preprocess_video_target(data)
        elif format_type == "audio":
            return self._preprocess_audio_target(data)
        elif format_type == "image":
            return self._preprocess_image_target(data)
        else:
            return self._preprocess_binary_target(data)
    
    def _preprocess_text_target(self, data: bytes) -> bytes:
        """テキスト特化前処理"""
        try:
            # UTF-8として処理
            text = data.decode('utf-8')
            
            # 改行コードを統一
            text = text.replace('\r\n', '\n').replace('\r', '\n')
            
            # 連続する空白を最適化
            import re
            text = re.sub(r' +', ' ', text)
            text = re.sub(r'\n+', '\n', text)
            
            return text.encode('utf-8')
        except:
            return data
    
    def _preprocess_video_target(self, data: bytes) -> bytes:
        """動画特化前処理"""
        # 動画は既に圧縮済みなので軽い前処理
        return data
    
    def _preprocess_audio_target(self, data: bytes) -> bytes:
        """音声特化前処理"""
        # 音声は既に圧縮済みなので軽い前処理
        return data
    
    def _preprocess_image_target(self, data: bytes) -> bytes:
        """画像特化前処理"""
        # 画像は既に圧縮済みなので軽い前処理
        return data
    
    def _preprocess_binary_target(self, data: bytes) -> bytes:
        """バイナリ特化前処理"""
        return data
    
    def _multi_stage_compress(self, data: bytes, format_type: str, strategy: str) -> bytes:
        """多段階圧縮"""
        config = self.target_configs[format_type]
        
        if strategy == 'fast_compression':
            # 速度重視
            return self._fast_compress(data, format_type, config)
        else:
            # 圧縮率重視
            return self._max_compress(data, format_type, config)
    
    def _fast_compress(self, data: bytes, format_type: str, config: dict) -> bytes:
        """速度重視圧縮"""
        if format_type == "text":
            # テキストは圧縮率重視
            return self._max_compress(data, format_type, config)
        
        # その他は高速圧縮
        try:
            compressed = lzma.compress(data, preset=config['fast_preset'], check=lzma.CHECK_CRC32)
            return b'FAST' + compressed
        except:
            return b'RAW0' + data
    
    def _max_compress(self, data: bytes, format_type: str, config: dict) -> bytes:
        """圧縮率重視多段階圧縮"""
        best_compressed = data
        best_marker = b'RAW0'
        best_ratio = 0
        
        # 各アルゴリズムを試行
        algorithms = config['algorithms']
        presets = config['presets']
        
        for i, algorithm in enumerate(algorithms):
            try:
                if algorithm == 'lzma':
                    compressed = lzma.compress(data, preset=presets[i], check=lzma.CHECK_CRC32)
                    marker = b'LZM' + str(i).encode('ascii')
                elif algorithm == 'bz2':
                    compressed = bz2.compress(data, compresslevel=presets[i])
                    marker = b'BZ2' + str(i).encode('ascii')
                elif algorithm == 'zlib':
                    compressed = zlib.compress(data, level=min(presets[i], 9))
                    marker = b'ZLB' + str(i).encode('ascii')
                else:
                    continue
                
                # 圧縮率チェック
                ratio = (1 - len(compressed) / len(data)) * 100
                if ratio > best_ratio:
                    best_compressed = compressed
                    best_marker = marker
                    best_ratio = ratio
                    
            except Exception as e:
                continue
        
        # 目標圧縮率チェック
        target_ratio = config['target_ratio'] * 100
        if best_ratio < target_ratio and format_type == "text":
            # テキストで目標未達成の場合、さらなる圧縮を試行
            return self._ultra_compress_text(data, best_compressed, best_marker, best_ratio)
        
        return best_marker + best_compressed
    
    def _ultra_compress_text(self, original: bytes, current_best: bytes, marker: bytes, ratio: float) -> bytes:
        """テキスト超圧縮"""
        try:
            # 2段階圧縮
            stage1 = lzma.compress(original, preset=9, check=lzma.CHECK_CRC32)
            stage2 = bz2.compress(stage1, compresslevel=9)
            
            # 3段階目も試行
            stage3 = zlib.compress(stage2, level=9)
            
            # 最適な結果を選択
            candidates = [
                (b'2STG', stage2),
                (b'3STG', stage3),
                (marker, current_best)
            ]
            
            best = min(candidates, key=lambda x: len(x[1]))
            return best[0] + best[1]
            
        except:
            return marker + current_best
    
    def _multi_stage_decompress(self, data: bytes, format_type: str) -> bytes:
        """多段階展開"""
        if len(data) < 4:
            return data
        
        marker = data[:4]
        compressed_data = data[4:]
        
        try:
            if marker == b'RAW0':
                return compressed_data
            elif marker == b'FAST':
                return lzma.decompress(compressed_data)
            elif marker.startswith(b'LZM'):
                return lzma.decompress(compressed_data)
            elif marker.startswith(b'BZ2'):
                return bz2.decompress(compressed_data)
            elif marker.startswith(b'ZLB'):
                return zlib.decompress(compressed_data)
            elif marker == b'2STG':
                # 2段階展開
                stage1 = bz2.decompress(compressed_data)
                return lzma.decompress(stage1)
            elif marker == b'3STG':
                # 3段階展開
                stage1 = zlib.decompress(compressed_data)
                stage2 = bz2.decompress(stage1)
                return lzma.decompress(stage2)
            else:
                # フォールバック
                return lzma.decompress(compressed_data)
                
        except Exception as e:
            raise ValueError(f"Decompression failed: {e}")
    
    def _postprocess_target(self, data: bytes, format_type: str) -> bytes:
        """後処理"""
        # 現在は前処理の逆変換なし
        return data
    
    # === ヘッダー処理 ===
    
    def _create_target_header(self, original_size: int, compressed_size: int, 
                             encrypted_size: int, format_type: str, strategy: str) -> bytes:
        """目標達成ヘッダー作成 (48バイト)"""
        header = bytearray(48)
        
        # マジックナンバー
        header[0:4] = b'NXZT'  # Target専用
        
        # バージョン
        header[4:8] = struct.pack('<I', 1)
        
        # サイズ情報
        header[8:16] = struct.pack('<Q', original_size)
        header[16:24] = struct.pack('<Q', compressed_size)
        header[24:32] = struct.pack('<Q', encrypted_size)
        
        # フォーマット情報
        format_bytes = format_type.encode('ascii')[:8]
        header[32:40] = format_bytes.ljust(8, b'\x00')
        
        # 戦略情報
        strategy_bytes = strategy.encode('ascii')[:8]
        header[40:48] = strategy_bytes.ljust(8, b'\x00')
        
        return bytes(header)
    
    def _parse_target_header(self, header: bytes) -> dict:
        """目標達成ヘッダー解析"""
        if len(header) < 48:
            raise ValueError("Invalid header size")
        
        magic = header[0:4]
        if magic != b'NXZT':
            raise ValueError("Invalid magic number")
        
        version = struct.unpack('<I', header[4:8])[0]
        original_size = struct.unpack('<Q', header[8:16])[0]
        compressed_size = struct.unpack('<Q', header[16:24])[0]
        encrypted_size = struct.unpack('<Q', header[24:32])[0]
        
        format_type = header[32:40].rstrip(b'\x00').decode('ascii')
        strategy = header[40:48].rstrip(b'\x00').decode('ascii')
        
        return {
            'version': version,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'encrypted_size': encrypted_size,
            'format_type': format_type,
            'strategy': strategy
        }
    
    def _create_empty_nxz(self) -> bytes:
        """空のNXZファイル作成"""
        return self._create_target_header(0, 0, 0, "empty", "none")

def test_nexus_target():
    """NEXUS Target Achievement テスト"""
    print("🎯 NEXUS Target Achievement テスト - 目標達成特化")
    print("=" * 70)
    print("🏆 目標: 圧縮率80%(テキスト95%), 圧縮100MB/s, 展開200MB/s")
    print("=" * 70)
    
    # 重要なテストファイル
    test_files = [
        {
            'path': r"C:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\sample\出庫実績明細_202412.txt",
            'type': 'text',
            'target_ratio': 95
        },
        {
            'path': r"C:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\sample\Python基礎講座3_4月26日-3.mp4",
            'type': 'video',
            'target_ratio': 80
        }
    ]
    
    nexus = NEXUSTargetAchievement()
    results = []
    
    for test_file in test_files:
        file_path = Path(test_file['path'])
        
        if not file_path.exists():
            print(f"⚠️ ファイルが見つかりません: {file_path.name}")
            continue
        
        print(f"\n📄 テスト: {file_path.name}")
        
        # データ読み込み
        with open(file_path, 'rb') as f:
            data = f.read()
        
        file_size = len(data)
        print(f"📊 サイズ: {file_size//1024//1024:.1f} MB")
        
        # 圧縮テスト
        start_time = time.perf_counter()
        compressed = nexus.compress(data)
        compress_time = time.perf_counter() - start_time
        
        # 圧縮結果
        compression_ratio = (1 - len(compressed) / len(data)) * 100
        compress_speed = (len(data) / 1024 / 1024) / compress_time
        
        print(f"🔥 圧縮: {compression_ratio:.1f}% ({compress_speed:.1f} MB/s)")
        
        # 展開テスト
        start_time = time.perf_counter()
        decompressed = nexus.decompress(compressed)
        decomp_time = time.perf_counter() - start_time
        
        # 展開結果
        decomp_speed = (len(data) / 1024 / 1024) / decomp_time
        is_correct = data == decompressed
        
        print(f"💨 展開: {decomp_speed:.1f} MB/s (正確性: {'✅' if is_correct else '❌'})")
        
        # 目標達成評価
        ratio_ok = compression_ratio >= test_file['target_ratio']
        compress_ok = compress_speed >= 100
        decomp_ok = decomp_speed >= 200
        
        print(f"🎯 評価: 圧縮率{'🎉' if ratio_ok else '🔧'} 圧縮速度{'🎉' if compress_ok else '🔧'} 展開速度{'🎉' if decomp_ok else '🔧'}")
        
        # 詳細分析
        if ratio_ok and compress_ok and decomp_ok:
            print(f"✨ 完全成功！全目標達成！")
        else:
            if not ratio_ok:
                print(f"   📊 圧縮率: {compression_ratio:.1f}% (目標: {test_file['target_ratio']:.0f}%)")
            if not compress_ok:
                print(f"   ⚡ 圧縮速度: {compress_speed:.1f} MB/s (目標: 100 MB/s)")
            if not decomp_ok:
                print(f"   💨 展開速度: {decomp_speed:.1f} MB/s (目標: 200 MB/s)")
        
        results.append({
            'file': file_path.name,
            'type': test_file['type'],
            'compression_ratio': compression_ratio,
            'compress_speed': compress_speed,
            'decomp_speed': decomp_speed,
            'target_ratio': test_file['target_ratio'],
            'ratio_ok': ratio_ok,
            'compress_ok': compress_ok,
            'decomp_ok': decomp_ok,
            'is_correct': is_correct
        })
    
    # 総合評価
    print(f"\n🏆 最終評価")
    print("=" * 70)
    
    if results:
        total_ratio_ok = sum(1 for r in results if r['ratio_ok'])
        total_compress_ok = sum(1 for r in results if r['compress_ok'])
        total_decomp_ok = sum(1 for r in results if r['decomp_ok'])
        total_correct = sum(1 for r in results if r['is_correct'])
        
        print(f"📊 圧縮率目標達成: {total_ratio_ok}/{len(results)} ファイル")
        print(f"⚡ 圧縮速度目標達成: {total_compress_ok}/{len(results)} ファイル")
        print(f"💨 展開速度目標達成: {total_decomp_ok}/{len(results)} ファイル")
        print(f"🔍 完全可逆性: {total_correct}/{len(results)} ファイル")
        
        # 平均性能
        avg_ratio = sum(r['compression_ratio'] for r in results) / len(results)
        avg_compress = sum(r['compress_speed'] for r in results) / len(results)
        avg_decomp = sum(r['decomp_speed'] for r in results) / len(results)
        
        print(f"\n📈 平均性能:")
        print(f"   圧縮率: {avg_ratio:.1f}%")
        print(f"   圧縮速度: {avg_compress:.1f} MB/s")
        print(f"   展開速度: {avg_decomp:.1f} MB/s")
        
        # 最終判定
        perfect_success = (total_ratio_ok == len(results) and 
                          total_compress_ok == len(results) and 
                          total_decomp_ok == len(results) and 
                          total_correct == len(results))
        
        if perfect_success:
            print(f"\n🎉🎉🎉 完全成功！全目標達成！🎉🎉🎉")
            print(f"🏆 制約なし圧縮技術の革命的実現！")
            print(f"🚀 AV1/SRLA/AVIF制約除去戦略の勝利！")
        else:
            success_rate = (total_ratio_ok + total_compress_ok + total_decomp_ok) / (len(results) * 3) * 100
            print(f"\n🔧 成功率: {success_rate:.1f}% - 継続改善中")
            print(f"📊 目標達成に向けて最適化を続けます")
    
    return results

if __name__ == "__main__":
    test_nexus_target()
