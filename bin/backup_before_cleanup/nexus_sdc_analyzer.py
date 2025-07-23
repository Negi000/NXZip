#!/usr/bin/env python3
"""
NEXUS SDC Advanced Structure Analyzer
高度な構造解析機能

MP3, MP4, WAV等の追加フォーマット対応
"""

import struct
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class AudioFormat:
    """音声フォーマット情報"""
    format_tag: int
    channels: int
    sample_rate: int
    bit_rate: int
    block_align: int
    bits_per_sample: int

class AdvancedStructureAnalyzer:
    """高度な構造解析器"""
    
    def analyze_mp3_structure(self, data: bytes) -> Dict:
        """MP3構造の詳細解析"""
        elements = []
        pos = 0
        frame_count = 0
        
        # ID3v2タグチェック
        if pos < len(data) - 10 and data[pos:pos+3] == b'ID3':
            # ID3v2ヘッダー解析
            version = data[pos+3:pos+5]
            flags = data[pos+5]
            size_bytes = data[pos+6:pos+10]
            
            # サイズ計算（synchsafe integer）
            tag_size = 0
            for byte in size_bytes:
                tag_size = (tag_size << 7) | (byte & 0x7F)
            
            elements.append({
                'type': 'ID3v2_TAG',
                'position': pos,
                'size': tag_size + 10,
                'compression_potential': 0.5,
                'category': 'metadata',
                'metadata': {
                    'version': f"{version[0]}.{version[1]}",
                    'flags': flags,
                    'tag_size': tag_size
                }
            })
            
            pos += tag_size + 10
        
        # MP3フレーム解析
        while pos < len(data) - 4 and frame_count < 1000:  # 最大1000フレーム
            if data[pos] == 0xFF and (data[pos+1] & 0xE0) == 0xE0:
                # フレームヘッダー解析
                header = struct.unpack('>I', data[pos:pos+4])[0]
                
                # MPEG version
                version = (header >> 19) & 0x3
                layer = (header >> 17) & 0x3
                bitrate_index = (header >> 12) & 0xF
                sample_rate_index = (header >> 10) & 0x3
                
                # フレームサイズ計算（簡略化）
                if bitrate_index != 0 and sample_rate_index != 3:
                    bitrates = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320]
                    sample_rates = [44100, 48000, 32000]
                    
                    if bitrate_index < len(bitrates) and sample_rate_index < len(sample_rates):
                        bitrate = bitrates[bitrate_index] * 1000
                        sample_rate = sample_rates[sample_rate_index]
                        frame_size = int(144 * bitrate / sample_rate) + ((header >> 9) & 1)
                        
                        elements.append({
                            'type': f'MP3_FRAME_{frame_count}',
                            'position': pos,
                            'size': frame_size,
                            'compression_potential': 0.75,  # 音声データは高圧縮可能
                            'category': 'audio_data',
                            'metadata': {
                                'bitrate': bitrate,
                                'sample_rate': sample_rate,
                                'version': version,
                                'layer': layer
                            }
                        })
                        
                        pos += frame_size
                        frame_count += 1
                        continue
            
            pos += 1
        
        # ID3v1タグ（ファイル末尾）
        if len(data) >= 128 and data[-128:-125] == b'TAG':
            elements.append({
                'type': 'ID3v1_TAG',
                'position': len(data) - 128,
                'size': 128,
                'compression_potential': 0.3,
                'category': 'metadata'
            })
        
        return {
            'format': 'MP3',
            'elements': elements,
            'total_size': len(data),
            'metadata': {
                'frame_count': frame_count,
                'has_id3v2': any(e['type'] == 'ID3v2_TAG' for e in elements),
                'has_id3v1': any(e['type'] == 'ID3v1_TAG' for e in elements)
            }
        }
    
    def analyze_wav_structure(self, data: bytes) -> Dict:
        """WAV構造の詳細解析"""
        elements = []
        
        if len(data) < 12:
            return {'format': 'WAV', 'elements': [], 'total_size': len(data)}
        
        # RIFFヘッダーチェック
        if data[:4] != b'RIFF' or data[8:12] != b'WAVE':
            return {'format': 'WAV', 'elements': [], 'total_size': len(data)}
        
        # RIFFヘッダー
        file_size = struct.unpack('<I', data[4:8])[0]
        elements.append({
            'type': 'RIFF_HEADER',
            'position': 0,
            'size': 12,
            'compression_potential': 0.0,
            'category': 'header'
        })
        
        pos = 12
        audio_format = None
        
        while pos < len(data) - 8:
            chunk_id = data[pos:pos+4]
            chunk_size = struct.unpack('<I', data[pos+4:pos+8])[0]
            
            if chunk_id == b'fmt ':
                # フォーマットチャンク
                fmt_data = data[pos+8:pos+8+chunk_size]
                if len(fmt_data) >= 16:
                    audio_format = AudioFormat(
                        format_tag=struct.unpack('<H', fmt_data[0:2])[0],
                        channels=struct.unpack('<H', fmt_data[2:4])[0],
                        sample_rate=struct.unpack('<I', fmt_data[4:8])[0],
                        bit_rate=struct.unpack('<I', fmt_data[8:12])[0],
                        block_align=struct.unpack('<H', fmt_data[12:14])[0],
                        bits_per_sample=struct.unpack('<H', fmt_data[14:16])[0]
                    )
                
                elements.append({
                    'type': 'WAV_FMT_CHUNK',
                    'position': pos,
                    'size': chunk_size + 8,
                    'compression_potential': 0.1,
                    'category': 'metadata',
                    'metadata': audio_format.__dict__ if audio_format else {}
                })
            
            elif chunk_id == b'data':
                # データチャンク（音声データ）
                compression_potential = 0.8  # 音声データは高圧縮可能
                if audio_format and audio_format.format_tag != 1:  # PCMでない場合
                    compression_potential = 0.6  # 既に圧縮されている可能性
                
                elements.append({
                    'type': 'WAV_DATA_CHUNK',
                    'position': pos,
                    'size': chunk_size + 8,
                    'compression_potential': compression_potential,
                    'category': 'audio_data'
                })
            
            else:
                # その他のチャンク
                elements.append({
                    'type': f'WAV_CHUNK_{chunk_id.decode("ascii", errors="ignore")}',
                    'position': pos,
                    'size': chunk_size + 8,
                    'compression_potential': 0.2,
                    'category': 'metadata'
                })
            
            pos += chunk_size + 8
            
            # チャンクサイズが奇数の場合、1バイトパディング
            if chunk_size % 2 == 1:
                pos += 1
        
        return {
            'format': 'WAV',
            'elements': elements,
            'total_size': len(data),
            'metadata': {
                'audio_format': audio_format.__dict__ if audio_format else {},
                'chunk_count': len(elements)
            }
        }
    
    def analyze_mp4_structure(self, data: bytes) -> Dict:
        """MP4構造の詳細解析"""
        elements = []
        pos = 0
        
        while pos < len(data) - 8:
            # Atomサイズとタイプ
            if pos + 8 > len(data):
                break
            
            atom_size = struct.unpack('>I', data[pos:pos+4])[0]
            atom_type = data[pos+4:pos+8].decode('ascii', errors='ignore')
            
            if atom_size == 0:
                # サイズ0は残り全部
                atom_size = len(data) - pos
            elif atom_size == 1:
                # 64bit サイズ
                if pos + 16 > len(data):
                    break
                atom_size = struct.unpack('>Q', data[pos+8:pos+16])[0]
                pos += 8  # 64bitサイズの分オフセット調整
            
            # 圧縮可能性の判定
            if atom_type == 'mdat':
                compression_potential = 0.8  # メディアデータ
                category = 'video_data'
            elif atom_type in ['moov', 'trak', 'mdia']:
                compression_potential = 0.4  # メタデータ
                category = 'metadata'
            elif atom_type in ['ftyp', 'free', 'skip']:
                compression_potential = 0.1  # ヘッダー情報
                category = 'header'
            else:
                compression_potential = 0.3  # その他
                category = 'other'
            
            elements.append({
                'type': f'MP4_ATOM_{atom_type}',
                'position': pos,
                'size': min(atom_size, len(data) - pos),
                'compression_potential': compression_potential,
                'category': category,
                'metadata': {
                    'atom_type': atom_type,
                    'atom_size': atom_size
                }
            })
            
            pos += atom_size
            
            if len(elements) > 100:  # 解析制限
                break
        
        return {
            'format': 'MP4',
            'elements': elements,
            'total_size': len(data),
            'metadata': {
                'atom_count': len(elements)
            }
        }
    
    def analyze_text_structure(self, data: bytes) -> Dict:
        """テキストファイルの高度解析"""
        elements = []
        
        try:
            text = data.decode('utf-8')
        except UnicodeDecodeError:
            try:
                text = data.decode('shift_jis')
            except UnicodeDecodeError:
                text = data.decode('utf-8', errors='ignore')
        
        lines = text.split('\n')
        pos = 0
        
        for i, line in enumerate(lines):
            line_bytes = line.encode('utf-8')
            
            # 行の特性に基づく圧縮可能性判定
            if not line.strip():
                compression_potential = 0.95  # 空行は高圧縮
            elif len(set(line)) < 10:
                compression_potential = 0.9   # 繰り返しが多い
            elif line.isdigit() or ',' in line:
                compression_potential = 0.85  # 数値データ
            else:
                compression_potential = 0.7   # 通常テキスト
            
            elements.append({
                'type': f'TEXT_LINE_{i}',
                'position': pos,
                'size': len(line_bytes) + 1,  # 改行文字込み
                'compression_potential': compression_potential,
                'category': 'text_data',
                'metadata': {
                    'line_number': i,
                    'char_count': len(line),
                    'unique_chars': len(set(line))
                }
            })
            
            pos += len(line_bytes) + 1
        
        return {
            'format': 'TEXT',
            'elements': elements,
            'total_size': len(data),
            'metadata': {
                'line_count': len(lines),
                'encoding': 'utf-8'
            }
        }

# エクスポート用
__all__ = ['AdvancedStructureAnalyzer', 'AudioFormat']
