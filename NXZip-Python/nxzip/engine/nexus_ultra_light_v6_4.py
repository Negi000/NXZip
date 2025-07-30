"""
NEXUS Ultra Lightweight Engine v6.4 - 緊急性能最適化版
v6.3の深刻な速度問題を解決する超軽量・高速実装

パフォーマンス危機対応:
- v6.3: 0.8MB/s平均 → 目標: 30MB/s+
- 戦略選択の完全見直し
- 軽量化優先の実装
"""

import numpy as np
import time
import threading
from enum import Enum
from typing import Tuple, Dict, Any, Optional
from collections import defaultdict
import struct


class UltraStrategy(Enum):
    """軽量戦略エンジン"""
    ULTRA_FAST = "ultra_fast"          # 超高速モード: 3MB/s以下用
    SPEED_FIRST = "speed_first"        # 速度優先: 10MB/s以下用  
    VISUAL_QUICK = "visual_quick"      # 視覚高速: 画像用
    PATTERN_LITE = "pattern_lite"      # パターン軽量: 一般用
    AUDIO_BOOST = "audio_boost"        # 音声特化: 音声用


class NEXUSLightweightAnalyzer:
    """超軽量分析エンジン - 512バイト以下サンプリング"""
    
    def __init__(self):
        self.cache = {}
        
    def ultra_fast_analysis(self, data: bytes, file_type: str) -> Dict[str, Any]:
        """200ms以下の超高速分析"""
        start_time = time.perf_counter()
        
        # データサイズベースの即時判定
        size = len(data)
        
        # ファイルタイプ別即座戦略
        if file_type in ['wav', 'mp3']:
            strategy = UltraStrategy.AUDIO_BOOST
            compression_hint = 2.5  # 音声は高圧縮期待
        elif file_type in ['jpg', 'png']:
            if size > 10 * 1024 * 1024:  # 10MB超
                strategy = UltraStrategy.ULTRA_FAST
                compression_hint = 1.2
            else:
                strategy = UltraStrategy.VISUAL_QUICK  
                compression_hint = 1.5
        elif file_type in ['mp4', 'avi']:
            strategy = UltraStrategy.SPEED_FIRST
            compression_hint = 1.8
        else:
            strategy = UltraStrategy.PATTERN_LITE
            compression_hint = 1.4
        
        # 最小限サンプリング (最大256バイト)
        sample_size = min(256, size // 1000 + 64)
        sample = data[:sample_size]
        
        # 高速エントロピー推定
        if len(sample) > 0:
            unique_ratio = len(set(sample)) / len(sample)
            entropy_hint = unique_ratio * 8
        else:
            entropy_hint = 4.0
        
        analysis_time = time.perf_counter() - start_time
        
        return {
            'strategy': strategy,
            'compression_hint': compression_hint,
            'entropy': entropy_hint,
            'sample_size': sample_size,
            'analysis_time': analysis_time,
            'processing_mode': 'ultra_lightweight'
        }


class NEXUSUltraCompressor:
    """軽量圧縮エンジン - 速度最優先"""
    
    def __init__(self):
        self.stats = defaultdict(int)
        
    def compress_ultra_fast(self, data: bytes, strategy: UltraStrategy, hints: Dict) -> Tuple[bytes, Dict]:
        """超高速圧縮実行"""
        start_time = time.perf_counter()
        
        if strategy == UltraStrategy.ULTRA_FAST:
            compressed = self._ultra_fast_compress(data)
            
        elif strategy == UltraStrategy.SPEED_FIRST:
            compressed = self._speed_first_compress(data)
            
        elif strategy == UltraStrategy.VISUAL_QUICK:
            compressed = self._visual_quick_compress(data)
            
        elif strategy == UltraStrategy.PATTERN_LITE:
            compressed = self._pattern_lite_compress(data)
            
        elif strategy == UltraStrategy.AUDIO_BOOST:
            compressed = self._audio_boost_compress(data)
            
        else:
            # フォールバック
            compressed = self._basic_compress(data)
        
        compress_time = time.perf_counter() - start_time
        
        # 統計更新
        self.stats['compressions'] += 1
        self.stats['total_time'] += compress_time
        
        return compressed, {
            'compression_time': compress_time,
            'input_size': len(data),
            'output_size': len(compressed),
            'compression_ratio': (1 - len(compressed) / len(data)) * 100,
            'throughput_mb_s': (len(data) / 1024 / 1024) / compress_time if compress_time > 0 else 0,
            'strategy_used': strategy.value
        }
        
    def _ultra_fast_compress(self, data: bytes) -> bytes:
        """最速モード - RLE + 簡単辞書"""
        if len(data) < 1024:
            return data  # 小ファイルはそのまま
            
        # 超簡単RLE
        compressed = bytearray()
        i = 0
        while i < len(data):
            current = data[i]
            count = 1
            
            # 連続チェック (最大8バイト)
            while i + count < len(data) and data[i + count] == current and count < 8:
                count += 1
            
            if count > 2:
                compressed.extend([255, count, current])  # RLE符号
                i += count
            else:
                compressed.append(current)
                i += 1
                
        return bytes(compressed)
        
    def _speed_first_compress(self, data: bytes) -> bytes:
        """速度優先 - 軽量LZ77"""
        if len(data) < 512:
            return data
            
        compressed = bytearray()
        window_size = 1024  # 小さな窓
        i = 0
        
        while i < len(data):
            # 短い一致検索 (最大32バイト)
            best_length = 0
            best_distance = 0
            max_length = min(32, len(data) - i)
            
            # 過去1KBの範囲で検索
            start = max(0, i - window_size)
            
            for j in range(start, i):
                length = 0
                while (length < max_length and 
                       i + length < len(data) and 
                       data[j + length] == data[i + length]):
                    length += 1
                    
                if length > best_length and length >= 3:
                    best_length = length
                    best_distance = i - j
                    
            if best_length >= 3:
                # LZ77符号化 (距離, 長さ)
                compressed.extend([254, best_distance & 255, (best_distance >> 8) & 15, best_length])
                i += best_length
            else:
                compressed.append(data[i])
                i += 1
                
        return bytes(compressed)
        
    def _visual_quick_compress(self, data: bytes) -> bytes:
        """視覚用高速圧縮"""
        if len(data) < 256:
            return data
            
        # 簡単差分符号化
        compressed = bytearray([data[0]])  # 最初の値
        
        for i in range(1, len(data)):
            diff = (data[i] - data[i-1]) % 256
            
            # 小さな差分は直接記録
            if diff < 128:
                compressed.append(diff)
            else:
                # 大きな差分は特別符号
                compressed.extend([253, data[i]])
                
        return bytes(compressed)
        
    def _pattern_lite_compress(self, data: bytes) -> bytes:
        """軽量パターン圧縮"""
        if len(data) < 128:
            return data
            
        # 頻度解析 (上位16バイトのみ)
        freq = defaultdict(int)
        for b in data:
            freq[b] += 1
            
        # 最頻出16バイト
        top_bytes = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:16]
        
        if not top_bytes:
            return data
            
        # 辞書作成
        dictionary = {byte_val: idx for idx, (byte_val, _) in enumerate(top_bytes)}
        
        # 圧縮
        compressed = bytearray()
        
        # 辞書情報
        compressed.append(252)  # 辞書符号
        compressed.append(len(dictionary))
        for byte_val, _ in top_bytes:
            compressed.append(byte_val)
            
        # データ圧縮
        for b in data:
            if b in dictionary:
                compressed.extend([251, dictionary[b]])  # 辞書参照
            else:
                compressed.append(b)
                
        return bytes(compressed)
        
    def _audio_boost_compress(self, data: bytes) -> bytes:
        """音声特化圧縮"""
        if len(data) < 64:
            return data
            
        # 音声は通常16bit整数
        if len(data) % 2 == 0:
            # 16bitサンプルとして処理
            samples = struct.unpack(f'<{len(data)//2}h', data)
            
            # 差分符号化
            compressed = bytearray()
            compressed.extend(struct.pack('<h', samples[0]))  # 最初のサンプル
            
            for i in range(1, len(samples)):
                diff = samples[i] - samples[i-1]
                
                # 小さな差分は8bitで
                if -128 <= diff <= 127:
                    compressed.extend([250, diff & 255])
                else:
                    # 大きな差分は16bit
                    compressed.extend([249])
                    compressed.extend(struct.pack('<h', samples[i]))
                    
            return bytes(compressed)
        else:
            # 奇数バイトの場合は基本圧縮
            return self._basic_compress(data)
            
    def _basic_compress(self, data: bytes) -> bytes:
        """基本圧縮 - 最小限の処理"""
        # zlib相当の簡単圧縮
        import zlib
        return zlib.compress(data, level=1)  # 最速レベル


class NEXUSUltraLightEngine:
    """NEXUS Ultra Lightweight Engine v6.4 - 緊急性能最適化"""
    
    def __init__(self):
        self.analyzer = NEXUSLightweightAnalyzer()
        self.compressor = NEXUSUltraCompressor()
        self.total_stats = {
            'compressions': 0,
            'total_input': 0,
            'total_output': 0,
            'total_time': 0,
            'strategy_usage': defaultdict(int)
        }
        
    def compress_nexus_ultra_light(self, data: bytes, file_type: str = 'bin') -> Tuple[bytes, Dict[str, Any]]:
        """超軽量NEXUS圧縮 - 性能最優先"""
        overall_start = time.perf_counter()
        
        # Phase 1: 超高速分析 (200ms以下)
        analysis = self.analyzer.ultra_fast_analysis(data, file_type)
        strategy = analysis['strategy']
        
        # Phase 2: 戦略別圧縮
        compressed, compress_info = self.compressor.compress_ultra_fast(data, strategy, analysis)
        
        # Phase 3: 結果統合
        total_time = time.perf_counter() - overall_start
        
        # 統計更新
        self.total_stats['compressions'] += 1
        self.total_stats['total_input'] += len(data)
        self.total_stats['total_output'] += len(compressed)
        self.total_stats['total_time'] += total_time
        self.total_stats['strategy_usage'][strategy.value] += 1
        
        # 総合情報
        info = {
            'compression_ratio': compress_info['compression_ratio'],
            'throughput_mb_s': (len(data) / 1024 / 1024) / total_time if total_time > 0 else 0,
            'total_time': total_time,
            'strategy': strategy.value,
            'nexus_analysis': {
                'compression_multiplier': analysis['compression_hint'],
                'entropy_estimate': analysis['entropy'],
                'analysis_time': analysis['analysis_time'],
                'processing_mode': analysis['processing_mode']
            },
            'compression_details': compress_info
        }
        
        return compressed, info
        
    def get_nexus_ultra_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        if self.total_stats['compressions'] == 0:
            return {'status': 'no_data'}
            
        avg_compression = (1 - self.total_stats['total_output'] / self.total_stats['total_input']) * 100
        avg_throughput = (self.total_stats['total_input'] / 1024 / 1024) / self.total_stats['total_time']
        
        return {
            'total_compressions': self.total_stats['compressions'],
            'total_compression_ratio': avg_compression,
            'average_throughput_mb_s': avg_throughput,
            'strategy_distribution': dict(self.total_stats['strategy_usage']),
            'total_data_processed_mb': self.total_stats['total_input'] / 1024 / 1024,
            'performance_grade': self._calculate_performance_grade(avg_throughput, avg_compression)
        }
        
    def _calculate_performance_grade(self, throughput: float, compression: float) -> str:
        """性能グレード計算"""
        if throughput >= 30 and compression >= 30:
            return "EXCELLENT"
        elif throughput >= 20 and compression >= 20:
            return "VERY_GOOD"
        elif throughput >= 10 and compression >= 15:
            return "GOOD"
        elif throughput >= 5:
            return "ACCEPTABLE"
        else:
            return "NEEDS_IMPROVEMENT"
            
    def reset_stats(self):
        """統計リセット"""
        self.total_stats = {
            'compressions': 0,
            'total_input': 0,
            'total_output': 0,
            'total_time': 0,
            'strategy_usage': defaultdict(int)
        }


# v6.4の使用例
if __name__ == "__main__":
    # テストケース
    engine = NEXUSUltraLightEngine()
    
    # サンプルデータ
    test_data = b"This is a test data for NEXUS Ultra Lightweight compression engine." * 1000
    
    print("🚀 NEXUS Ultra Lightweight Engine v6.4 テスト")
    print(f"📊 入力サイズ: {len(test_data):,} bytes")
    
    # 圧縮実行
    start = time.perf_counter()
    compressed, info = engine.compress_nexus_ultra_light(test_data, 'txt')
    elapsed = time.perf_counter() - start
    
    print(f"✅ 圧縮完了: {elapsed:.3f}秒")
    print(f"📈 圧縮率: {info['compression_ratio']:.2f}%")
    print(f"⚡ スループット: {info['throughput_mb_s']:.1f}MB/s")
    print(f"🧠 戦略: {info['strategy']}")
    
    # 統計表示
    stats = engine.get_nexus_ultra_stats()
    print(f"🏆 性能グレード: {stats['performance_grade']}")
