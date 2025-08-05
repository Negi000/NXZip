#!/usr/bin/env python3
"""
NXZip Core v2.0 実用的総合テストスイート
実際のユースケースを意識した多種多様なテスト

Test Categories:
1. ドキュメント・テキストファイル
2. プログラムコード・設定ファイル  
3. 画像・メディアファイル
4. データベース・ログファイル
5. 科学・数値データ
6. バックアップ・アーカイブ
7. 暗号化・セキュリティ
8. パフォーマンス・ストレステスト
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np

# NXZip Core インポート
try:
    from nxzip_core import NXZipCore, NXZipContainer, CompressionMode
    print("✅ NXZip Core v2.0 実用テストスイート開始")
except ImportError as e:
    print(f"❌ NXZip Core インポート失敗: {e}")
    sys.exit(1)

def progress_callback(info):
    """進捗表示コールバック"""
    progress = info['progress']
    message = info['message']
    speed = info.get('speed', 0)
    
    # 速度を適切な単位で表示
    if speed > 1024 * 1024:
        speed_str = f"{speed / (1024 * 1024):.1f} MB/s"
    elif speed > 1024:
        speed_str = f"{speed / 1024:.1f} KB/s"
    else:
        speed_str = f"{speed:.0f} B/s"
    
    print(f"\r🔄 {progress:5.1f}% | {message[:40]:40} | {speed_str:>10}", end="", flush=True)

class TestResults:
    """テスト結果集計クラス"""
    
    def __init__(self):
        self.results = []
        self.summary = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'total_original_size': 0,
            'total_compressed_size': 0,
            'total_compression_time': 0.0,
            'total_decompression_time': 0.0
        }
    
    def add_result(self, test_name: str, result: dict):
        """テスト結果を追加"""
        self.results.append({
            'test_name': test_name,
            'timestamp': time.time(),
            **result
        })
        
        self.summary['total_tests'] += 1
        if result.get('success', False):
            self.summary['passed'] += 1
        else:
            self.summary['failed'] += 1
        
        self.summary['total_original_size'] += result.get('original_size', 0)
        self.summary['total_compressed_size'] += result.get('compressed_size', 0)
        self.summary['total_compression_time'] += result.get('compression_time', 0.0)
        self.summary['total_decompression_time'] += result.get('decompression_time', 0.0)
    
    def print_summary(self):
        """結果サマリー表示"""
        print("\n" + "="*80)
        print("🎯 NXZip Core v2.0 実用テスト総合結果")
        print("="*80)
        
        total_tests = self.summary['total_tests']
        passed = self.summary['passed']
        failed = self.summary['failed']
        success_rate = (passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"📊 テスト実行数: {total_tests}")
        print(f"✅ 成功: {passed}")
        print(f"❌ 失敗: {failed}")
        print(f"📈 成功率: {success_rate:.1f}%")
        
        # 圧縮効果
        original_size = self.summary['total_original_size']
        compressed_size = self.summary['total_compressed_size']
        if original_size > 0:
            overall_ratio = (1 - compressed_size / original_size) * 100
            print(f"📦 全体圧縮率: {overall_ratio:.2f}%")
            print(f"📏 合計データサイズ: {original_size:,} bytes ({original_size/1024/1024:.1f} MB)")
            print(f"🗜️ 合計圧縮サイズ: {compressed_size:,} bytes ({compressed_size/1024/1024:.1f} MB)")
        
        # 速度
        total_comp_time = self.summary['total_compression_time']
        total_decomp_time = self.summary['total_decompression_time']
        if total_comp_time > 0:
            comp_speed = (original_size / 1024 / 1024) / total_comp_time
            print(f"⚡ 平均圧縮速度: {comp_speed:.1f} MB/s")
        if total_decomp_time > 0:
            decomp_speed = (original_size / 1024 / 1024) / total_decomp_time
            print(f"🚀 平均展開速度: {decomp_speed:.1f} MB/s")

def run_test_case(test_name: str, data: bytes, mode: str, core: NXZipCore, 
                  results: TestResults, encryption_key: bytes = None) -> bool:
    """個別テストケース実行"""
    print(f"\n🧪 {test_name}")
    print(f"   データサイズ: {len(data):,} bytes ({len(data)/1024:.1f} KB)")
    
    try:
        # 圧縮テスト
        start_time = time.time()
        comp_result = core.compress(data, mode=mode, filename=test_name, encryption_key=encryption_key)
        
        if not comp_result.success:
            print(f"   ❌ 圧縮失敗: {comp_result.error_message}")
            results.add_result(test_name, {
                'success': False,
                'error': comp_result.error_message,
                'original_size': len(data),
                'compressed_size': 0,
                'compression_time': 0.0,
                'decompression_time': 0.0
            })
            return False
        
        print(f"\n   ✅ 圧縮成功!")
        print(f"   📦 圧縮率: {comp_result.compression_ratio:.2f}%")
        print(f"   ⏱️ 圧縮時間: {comp_result.compression_time:.3f}秒")
        
        if comp_result.compression_time > 0:
            speed = (len(data) / 1024 / 1024) / comp_result.compression_time
            print(f"   ⚡ 圧縮速度: {speed:.1f} MB/s")
        
        # 目標達成度
        target_eval = comp_result.metadata.get('target_evaluation', {})
        if target_eval:
            achieved = target_eval.get('target_achieved', False)
            concept = target_eval.get('concept', 'N/A')
            print(f"   🎯 目標達成: {'✅' if achieved else '❌'} ({concept})")
        
        # 展開テスト
        print("   🔓 展開テスト中...")
        decomp_result = core.decompress(comp_result.compressed_data, comp_result.metadata)
        
        if not decomp_result.success:
            print(f"   ❌ 展開失敗: {decomp_result.error_message}")
            results.add_result(test_name, {
                'success': False,
                'error': f"Decompression failed: {decomp_result.error_message}",
                'original_size': len(data),
                'compressed_size': len(comp_result.compressed_data),
                'compression_time': comp_result.compression_time,
                'decompression_time': 0.0
            })
            return False
        
        # 整合性確認
        integrity = core.validate_integrity(data, decomp_result.decompressed_data)
        integrity_ok = integrity['integrity_ok']
        
        print(f"   🔍 整合性: {'✅' if integrity_ok else '❌'}")
        print(f"   ⏱️ 展開時間: {decomp_result.decompression_time:.3f}秒")
        
        if decomp_result.decompression_time > 0:
            decomp_speed = (len(data) / 1024 / 1024) / decomp_result.decompression_time
            print(f"   🚀 展開速度: {decomp_speed:.1f} MB/s")
        
        # パイプライン詳細
        stages = comp_result.metadata.get('stages', [])
        transforms_applied = []
        for stage_name, stage_info in stages:
            if stage_name == 'tmc_transform':
                transforms = stage_info.get('transforms_applied', [])
                transforms_applied.extend(transforms)
            elif stage_name == 'spe_integration' and stage_info.get('spe_applied'):
                transforms_applied.append('spe')
        
        if transforms_applied:
            print(f"   🔧 適用変換: {', '.join(transforms_applied)}")
        
        results.add_result(test_name, {
            'success': integrity_ok,
            'compression_ratio': comp_result.compression_ratio,
            'original_size': len(data),
            'compressed_size': len(comp_result.compressed_data),
            'compression_time': comp_result.compression_time,
            'decompression_time': decomp_result.decompression_time,
            'target_achieved': target_eval.get('target_achieved', False),
            'transforms_applied': transforms_applied,
            'mode': mode,
            'encrypted': encryption_key is not None
        })
        
        return integrity_ok
        
    except Exception as e:
        print(f"   ❌ テスト例外: {e}")
        results.add_result(test_name, {
            'success': False,
            'error': str(e),
            'original_size': len(data),
            'compressed_size': 0,
            'compression_time': 0.0,
            'decompression_time': 0.0
        })
        return False

def test_documents_and_text(core: NXZipCore, results: TestResults):
    """ドキュメント・テキストファイルテスト"""
    print("\n" + "="*60)
    print("📄 ドキュメント・テキストファイルテスト")
    print("="*60)
    
    # 1. 日本語テキスト（小説）
    japanese_novel = """
    吾輩は猫である。名前はまだ無い。
    どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。
    吾輩はここで始めて人間というものを見た。しかもあとで聞くとそれは書生という人間中で一番獰猛な種族であったそうだ。
    この書生というのは時々我々を捕えて煮て食うという話である。しかしその当時は何という考もなかったから別段恐しいとも思わなかった。
    ただ彼の掌に載せられてスーと持ち上げられた時何だかフワフワした感じがあったばかりである。
    掌の上で少し落ちついて書生の顔を見たのがいわゆる人間というものの見始めであろう。
    この時妙なものだと思った感じが今でも残っている。第一毛をもって装飾されべきはずの顔がつるつるしてまるで薬缶だ。
    その後猫にもだいぶ逢ったがこんな片輪には一度も出会わした事がない。のみならず顔の真中があまりに突起している。
    そうしてその穴の中から時々ぷうぷうと煙を吹く。どうも咽せぽくて実に弱った。これが人間の飲む煙草というものである事はずっと後になって知った。
    """ * 50  # 繰り返しで大きなファイルを模擬
    
    run_test_case("日本語小説テキスト", japanese_novel.encode('utf-8'), "balanced", core, results)
    
    # 2. 英語テキスト（技術文書）
    english_tech = """
    # Advanced Data Compression Algorithms
    
    ## Introduction
    Data compression is a fundamental technique in computer science that reduces the amount of space needed to store data.
    Modern compression algorithms employ sophisticated mathematical models to achieve high compression ratios while maintaining fast processing speeds.
    
    ## Theoretical Foundation
    The theoretical foundation of compression is based on information theory, developed by Claude Shannon.
    The key insight is that data contains redundancy, and this redundancy can be exploited to reduce storage requirements.
    
    ### Entropy Measurement
    Shannon entropy provides a lower bound for lossless compression:
    H(X) = -∑ P(x) log₂ P(x)
    
    Where P(x) is the probability of symbol x appearing in the data.
    
    ## Practical Algorithms
    
    ### Lempel-Ziv Variants
    - LZ77: Uses a sliding window to find repetitions
    - LZ78: Builds a dictionary of previously seen patterns
    - LZW: Extends LZ78 with adaptive dictionary updates
    
    ### Block-Sorting Algorithms
    - Burrows-Wheeler Transform (BWT): Reversible permutation that improves compressibility
    - Move-to-Front (MTF): Reduces entropy after BWT application
    - Run-Length Encoding (RLE): Efficiently handles repeated symbols
    
    ## Modern Developments
    Recent advances include neural network-based approaches and quantum compression algorithms.
    These methods show promise for achieving compression ratios beyond classical theoretical limits.
    """ * 20
    
    run_test_case("英語技術文書", english_tech.encode('utf-8'), "balanced", core, results)
    
    # 3. 混合言語文書（多言語対応テスト）
    multilingual_doc = """
    多言語文書テスト / Multilingual Document Test / Document Multilingue
    
    日本語: これは多言語対応のテストです。様々な文字エンコーディングに対応しています。
    English: This is a multilingual support test. It supports various character encodings.
    Français: Ceci est un test de support multilingue. Il prend en charge divers encodages de caractères.
    Deutsch: Dies ist ein mehrsprachiger Support-Test. Es unterstützt verschiedene Zeichenkodierungen.
    中文: 这是多语言支持测试。它支持各种字符编码。
    한국어: 이것은 다국어 지원 테스트입니다. 다양한 문자 인코딩을 지원합니다.
    Русский: Это тест многоязыковой поддержки. Он поддерживает различные кодировки символов.
    العربية: هذا اختبار دعم متعدد اللغات. وهو يدعم ترميزات الأحرف المختلفة.
    """ * 30
    
    run_test_case("多言語混合文書", multilingual_doc.encode('utf-8'), "balanced", core, results)

def test_source_code_and_config(core: NXZipCore, results: TestResults):
    """プログラムコード・設定ファイルテスト"""
    print("\n" + "="*60)
    print("💻 プログラムコード・設定ファイルテスト")
    print("="*60)
    
    # 1. Python ソースコード
    python_code = '''
import os
import sys
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import asyncio
import aiohttp
import pandas as pd

class DataProcessor:
    """データ処理クラス"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.cache = {}
        
    def _load_config(self) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._default_config()
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")
    
    def _default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            "batch_size": 1000,
            "max_workers": 4,
            "timeout": 30,
            "retry_attempts": 3,
            "output_format": "json",
            "compression": {
                "enabled": True,
                "algorithm": "lzma",
                "level": 6
            }
        }
    
    async def process_data_batch(self, data_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """データバッチ処理"""
        results = []
        semaphore = asyncio.Semaphore(self.config['max_workers'])
        
        async def process_item(item):
            async with semaphore:
                return await self._process_single_item(item)
        
        tasks = [process_item(item) for item in data_batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # エラーハンドリング
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Error processing item {i}: {result}")
            else:
                valid_results.append(result)
        
        return valid_results
    
    async def _process_single_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """単一アイテム処理"""
        # データ変換
        processed = {
            'id': item.get('id'),
            'timestamp': item.get('timestamp', time.time()),
            'data': self._transform_data(item.get('data', {})),
            'metadata': {
                'processed_at': time.time(),
                'version': '1.0',
                'processor': 'DataProcessor'
            }
        }
        
        # バリデーション
        if not self._validate_item(processed):
            raise ValueError(f"Validation failed for item: {processed['id']}")
        
        return processed
    
    def _transform_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """データ変換処理"""
        transformed = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                transformed[key] = value.strip().lower()
            elif isinstance(value, (int, float)):
                transformed[key] = float(value)
            elif isinstance(value, list):
                transformed[key] = [self._transform_data(item) if isinstance(item, dict) else item for item in value]
            else:
                transformed[key] = value
        
        return transformed
    
    def _validate_item(self, item: Dict[str, Any]) -> bool:
        """アイテムバリデーション"""
        required_fields = ['id', 'timestamp', 'data']
        return all(field in item for field in required_fields)

# 使用例
async def main():
    processor = DataProcessor('config.json')
    
    # テストデータ
    test_data = [
        {'id': f'item_{i}', 'data': {'value': i * 2, 'name': f'test_{i}'}}
        for i in range(1000)
    ]
    
    # バッチ処理実行
    batch_size = processor.config['batch_size']
    results = []
    
    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i + batch_size]
        batch_results = await processor.process_data_batch(batch)
        results.extend(batch_results)
        print(f"Processed batch {i//batch_size + 1}, items: {len(batch_results)}")
    
    print(f"Total processed items: {len(results)}")

if __name__ == "__main__":
    asyncio.run(main())
''' * 10
    
    run_test_case("Pythonソースコード", python_code.encode('utf-8'), "fast", core, results)
    
    # 2. JSON設定ファイル
    json_config = {
        "application": {
            "name": "NXZip Professional",
            "version": "2.0.0",
            "description": "Next-generation compression platform",
            "author": "NXZip Development Team"
        },
        "compression": {
            "default_mode": "balanced",
            "modes": {
                "fast": {
                    "target_speed": 50,
                    "target_ratio": 40,
                    "algorithm": "zlib",
                    "level": 3
                },
                "balanced": {
                    "target_speed": 10,
                    "target_ratio": 60,
                    "algorithm": "lzma",
                    "level": 6
                },
                "maximum": {
                    "target_ratio": 70,
                    "algorithm": "lzma",
                    "level": 9
                }
            }
        },
        "security": {
            "encryption": {
                "enabled": True,
                "algorithm": "AES-256-GCM",
                "key_derivation": "PBKDF2",
                "iterations": 100000
            },
            "integrity": {
                "checksum": "SHA256",
                "signature": "Ed25519"
            }
        },
        "performance": {
            "memory_limit": "1GB",
            "max_threads": 8,
            "chunk_size": "2MB",
            "cache_size": "100MB"
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": "nxzip.log",
            "max_size": "10MB",
            "backup_count": 5
        }
    }
    
    json_data = json.dumps(json_config, indent=2, ensure_ascii=False) * 20
    run_test_case("JSON設定ファイル", json_data.encode('utf-8'), "balanced", core, results)
    
    # 3. XML データ
    xml_data = '''<?xml version="1.0" encoding="UTF-8"?>
<configuration>
  <database>
    <host>localhost</host>
    <port>5432</port>
    <name>nxzip_db</name>
    <user>nxzip_user</user>
    <pool>
      <min_connections>5</min_connections>
      <max_connections>20</max_connections>
      <timeout>30</timeout>
    </pool>
  </database>
  <cache>
    <provider>redis</provider>
    <host>localhost</host>
    <port>6379</port>
    <ttl>3600</ttl>
  </cache>
  <logging>
    <appenders>
      <appender name="console" type="ConsoleAppender">
        <pattern>%d{yyyy-MM-dd HH:mm:ss} [%t] %-5level %logger{36} - %msg%n</pattern>
      </appender>
      <appender name="file" type="FileAppender">
        <file>logs/application.log</file>
        <pattern>%d{yyyy-MM-dd HH:mm:ss} [%t] %-5level %logger{36} - %msg%n</pattern>
      </appender>
    </appenders>
    <loggers>
      <logger name="com.nxzip" level="DEBUG"/>
      <logger name="org.springframework" level="INFO"/>
      <root level="INFO">
        <appender-ref ref="console"/>
        <appender-ref ref="file"/>
      </root>
    </loggers>
  </logging>
</configuration>''' * 30
    
    run_test_case("XML設定ファイル", xml_data.encode('utf-8'), "balanced", core, results)

def test_media_and_binary(core: NXZipCore, results: TestResults):
    """画像・メディア・バイナリファイルテスト"""
    print("\n" + "="*60)
    print("🖼️ 画像・メディア・バイナリファイルテスト")
    print("="*60)
    
    # 1. 模擬画像データ（BMP-like構造）
    width, height = 800, 600
    # 24bit RGB bitmap header simulation
    bmp_header = bytearray(54)
    bmp_header[0:2] = b'BM'  # Signature
    bmp_header[2:6] = (54 + width * height * 3).to_bytes(4, 'little')  # File size
    bmp_header[10:14] = (54).to_bytes(4, 'little')  # Offset to pixel data
    bmp_header[14:18] = (40).to_bytes(4, 'little')  # DIB header size
    bmp_header[18:22] = width.to_bytes(4, 'little')  # Width
    bmp_header[22:26] = height.to_bytes(4, 'little')  # Height
    bmp_header[26:28] = (1).to_bytes(2, 'little')  # Planes
    bmp_header[28:30] = (24).to_bytes(2, 'little')  # Bits per pixel
    
    # Generate gradient image data
    image_data = bytearray()
    for y in range(height):
        for x in range(width):
            r = int((x / width) * 255)
            g = int((y / height) * 255)  
            b = int(((x + y) / (width + height)) * 255)
            image_data.extend([b, g, r])  # BGR format
    
    bmp_data = bytes(bmp_header) + bytes(image_data)
    run_test_case("模擬BMP画像ファイル", bmp_data, "balanced", core, results)
    
    # 2. 音声データ模擬（WAV-like構造）
    sample_rate = 44100
    duration = 5  # seconds
    samples = sample_rate * duration
    
    # WAV header
    wav_header = bytearray(44)
    wav_header[0:4] = b'RIFF'
    wav_header[4:8] = (36 + samples * 2).to_bytes(4, 'little')
    wav_header[8:12] = b'WAVE'
    wav_header[12:16] = b'fmt '
    wav_header[16:20] = (16).to_bytes(4, 'little')  # PCM
    wav_header[20:22] = (1).to_bytes(2, 'little')   # Audio format
    wav_header[22:24] = (1).to_bytes(2, 'little')   # Mono
    wav_header[24:28] = sample_rate.to_bytes(4, 'little')
    wav_header[28:32] = (sample_rate * 2).to_bytes(4, 'little')  # Byte rate
    wav_header[32:34] = (2).to_bytes(2, 'little')   # Block align
    wav_header[34:36] = (16).to_bytes(2, 'little')  # Bits per sample
    wav_header[36:40] = b'data'
    wav_header[40:44] = (samples * 2).to_bytes(4, 'little')
    
    # Generate sine wave
    audio_data = bytearray()
    for i in range(samples):
        # Mix of frequencies to simulate music
        t = i / sample_rate
        freq1, freq2, freq3 = 440, 880, 1320  # A4, A5, E6
        sample = (np.sin(2 * np.pi * freq1 * t) * 0.3 + 
                 np.sin(2 * np.pi * freq2 * t) * 0.2 + 
                 np.sin(2 * np.pi * freq3 * t) * 0.1)
        sample_int = int(sample * 32767)
        audio_data.extend(sample_int.to_bytes(2, 'little', signed=True))
    
    wav_data = bytes(wav_header) + bytes(audio_data)
    run_test_case("模擬WAV音声ファイル", wav_data, "balanced", core, results)
    
    # 3. 実行ファイル模擬（PE-like構造）
    pe_data = bytearray()
    # DOS header
    pe_data.extend(b'MZ')  # DOS signature
    pe_data.extend(b'\x00' * 58)  # DOS header padding
    pe_data.extend((64).to_bytes(4, 'little'))  # PE header offset
    
    # PE header
    pe_data.extend(b'PE\x00\x00')  # PE signature
    pe_data.extend(b'\x4c\x01')    # Machine (i386)
    pe_data.extend(b'\x03\x00')    # Number of sections
    pe_data.extend(b'\x00' * 16)   # Timestamp, etc.
    
    # Add some "code" sections with patterns
    code_section = bytearray()
    for i in range(10000):
        # Simulate x86 instructions patterns
        if i % 100 == 0:
            code_section.extend(b'\x55\x8b\xec')  # push ebp; mov ebp, esp
        elif i % 50 == 0:
            code_section.extend(b'\xff\x15')      # call dword ptr
            code_section.extend(i.to_bytes(4, 'little'))
        else:
            code_section.extend(b'\x90')          # nop
    
    pe_data.extend(code_section)
    run_test_case("模擬実行ファイル", bytes(pe_data), "fast", core, results)

def test_database_and_logs(core: NXZipCore, results: TestResults):
    """データベース・ログファイルテスト"""
    print("\n" + "="*60)
    print("🗄️ データベース・ログファイルテスト")
    print("="*60)
    
    # 1. CSV データベースダンプ
    csv_data = "id,name,email,age,country,registration_date,last_login,status\n"
    countries = ["Japan", "USA", "Germany", "France", "China", "Korea", "Brazil", "India"]
    statuses = ["active", "inactive", "pending", "suspended"]
    
    for i in range(10000):
        csv_data += f"{i+1},User{i+1:04d},user{i+1}@example.com,{25 + (i % 50)},"
        csv_data += f"{countries[i % len(countries)]},2023-{1 + (i % 12):02d}-{1 + (i % 28):02d},"
        csv_data += f"2024-01-{1 + (i % 31):02d},{statuses[i % len(statuses)]}\n"
    
    run_test_case("CSV データベースダンプ", csv_data.encode('utf-8'), "balanced", core, results)
    
    # 2. アプリケーションログ
    log_data = ""
    log_levels = ["DEBUG", "INFO", "WARN", "ERROR", "FATAL"]
    components = ["AuthService", "DatabaseManager", "CacheService", "FileProcessor", "APIController"]
    
    for i in range(5000):
        timestamp = f"2024-01-15 {(i // 200) % 24:02d}:{(i % 60):02d}:{(i * 7) % 60:02d}.{(i * 123) % 1000:03d}"
        level = log_levels[i % len(log_levels)]
        component = components[i % len(components)]
        thread_id = f"Thread-{(i % 20) + 1}"
        
        messages = [
            f"Processing request ID: {i + 1000}",
            f"Database query executed in {(i % 1000) + 1}ms",
            f"Cache hit rate: {85 + (i % 15)}%",
            f"File processed: document_{i}.pdf ({(i % 10000) + 1000} bytes)",
            f"API response sent to client {((i * 7) % 1000) + 1}",
            f"Connection pool size: {(i % 50) + 10}",
            f"Memory usage: {(i % 80) + 20}%",
            "User authentication successful",
            "Session expired, redirecting to login",
            "Backup completed successfully"
        ]
        
        message = messages[i % len(messages)]
        log_data += f"{timestamp} [{thread_id}] {level:5} {component:15} - {message}\n"
        
        # Add occasional stack traces for ERROR/FATAL
        if level in ["ERROR", "FATAL"] and i % 100 == 0:
            log_data += f"    at com.nxzip.{component.lower()}.process(line {100 + (i % 500)})\n"
            log_data += f"    at com.nxzip.core.execute(line {50 + (i % 100)})\n"
            log_data += f"    at java.lang.Thread.run(line {800 + (i % 50)})\n"
    
    run_test_case("アプリケーションログ", log_data.encode('utf-8'), "maximum", core, results)
    
    # 3. SQL ダンプ
    sql_dump = """
-- NXZip Database Dump
-- Generated: 2024-01-15 12:00:00
-- Server: PostgreSQL 15.0

SET statement_timeout = 0;
SET lock_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE files (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    original_size BIGINT NOT NULL,
    compressed_size BIGINT NOT NULL,
    compression_ratio DECIMAL(5,2) NOT NULL,
    algorithm VARCHAR(50) NOT NULL,
    created_by INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_files_created_by ON files(created_by);
CREATE INDEX idx_files_created_at ON files(created_at);

"""
    
    # Generate INSERT statements
    for i in range(1000):
        sql_dump += f"INSERT INTO users (username, email, password_hash) VALUES "
        sql_dump += f"('user{i:04d}', 'user{i}@example.com', '$2b$12$hash{i:04d}');\n"
        
        if i % 10 == 0:  # Add some file records
            for j in range(5):
                file_id = i * 5 + j
                sql_dump += f"INSERT INTO files (filename, original_size, compressed_size, compression_ratio, algorithm, created_by) VALUES "
                sql_dump += f"('document_{file_id}.pdf', {(file_id % 10000) + 5000}, {(file_id % 5000) + 1000}, "
                sql_dump += f"{75.0 + (file_id % 20)}, 'nxzip_balanced', {i + 1});\n"
    
    sql_dump += "\nCOMMIT;\n"
    
    run_test_case("SQL データベースダンプ", sql_dump.encode('utf-8'), "maximum", core, results)

def test_scientific_and_numerical(core: NXZipCore, results: TestResults):
    """科学・数値データテスト"""
    print("\n" + "="*60)
    print("🔬 科学・数値データテスト")
    print("="*60)
    
    # 1. 時系列データ（株価・センサーデータ模擬）
    import math
    
    time_series_data = "timestamp,temperature,humidity,pressure,co2,voltage\n"
    base_time = 1704067200  # 2024-01-01 00:00:00
    
    for i in range(50000):  # 50,000 data points
        timestamp = base_time + i * 60  # Every minute
        
        # Simulate realistic sensor data with trends and noise
        hour_of_day = (i // 60) % 24
        temp = 20 + 10 * math.sin(2 * math.pi * hour_of_day / 24) + np.random.normal(0, 1)
        humidity = 50 + 20 * math.sin(2 * math.pi * hour_of_day / 24 + math.pi/4) + np.random.normal(0, 2)
        pressure = 1013.25 + 10 * math.sin(2 * math.pi * i / (24 * 60 * 7)) + np.random.normal(0, 0.5)  # Weekly cycle
        co2 = 400 + 50 * math.sin(2 * math.pi * hour_of_day / 24 + math.pi) + np.random.normal(0, 5)
        voltage = 12.0 + 0.5 * math.sin(2 * math.pi * i / 1000) + np.random.normal(0, 0.1)
        
        time_series_data += f"{timestamp},{temp:.2f},{humidity:.1f},{pressure:.2f},{co2:.0f},{voltage:.3f}\n"
    
    run_test_case("時系列センサーデータ", time_series_data.encode('utf-8'), "balanced", core, results)
    
    # 2. 科学計算結果（行列データ）
    matrix_size = 500
    matrix_data = f"# Matrix Data {matrix_size}x{matrix_size}\n"
    matrix_data += f"# Generated by NXZip Scientific Test Suite\n"
    matrix_data += f"rows: {matrix_size}\n"
    matrix_data += f"cols: {matrix_size}\n"
    matrix_data += f"format: csv\n\n"
    
    # Generate correlation matrix (symmetric, values between -1 and 1)
    for i in range(matrix_size):
        row_values = []
        for j in range(matrix_size):
            if i == j:
                value = 1.0
            elif i < j:
                # Generate correlation value
                value = math.sin(i * j * 0.001) * math.exp(-abs(i-j) * 0.01)
            else:
                # Use symmetry
                value = math.sin(j * i * 0.001) * math.exp(-abs(j-i) * 0.01)
            
            row_values.append(f"{value:.6f}")
        
        matrix_data += ",".join(row_values) + "\n"
    
    run_test_case("科学計算行列データ", matrix_data.encode('utf-8'), "maximum", core, results)
    
    # 3. バイナリ数値配列（NumPy-like）
    # Float64 array header simulation
    dtype_info = np.dtype(np.float64)
    array_shape = (1000, 100)  # 2D array
    total_elements = array_shape[0] * array_shape[1]
    
    # Create header information
    header = {
        'descr': dtype_info.descr,
        'fortran_order': False,
        'shape': array_shape,
    }
    header_str = str(header).replace("'", '"')
    
    # Simulate .npy format
    magic = b'\x93NUMPY'
    version = b'\x01\x00'
    header_bytes = header_str.encode('latin1')
    header_len = len(header_bytes)
    
    # Pad header to 64-byte boundary
    padding = (64 - (10 + header_len) % 64) % 64
    header_bytes += b' ' * padding
    
    npy_header = magic + version + header_len.to_bytes(2, 'little') + header_bytes
    
    # Generate scientific data (wave interference pattern)
    binary_data = bytearray()
    for i in range(array_shape[0]):
        for j in range(array_shape[1]):
            # Complex wave interference
            x, y = i / array_shape[0], j / array_shape[1]
            value = (math.sin(20 * math.pi * x) * math.cos(15 * math.pi * y) + 
                    math.sin(10 * math.pi * (x + y)) * 0.5 +
                    np.random.normal(0, 0.1))
            
            binary_data.extend(np.array([value], dtype=np.float64).tobytes())
    
    numpy_data = npy_header + bytes(binary_data)
    run_test_case("NumPy科学計算配列", numpy_data, "balanced", core, results)

def test_encryption_and_security(core: NXZipCore, results: TestResults):
    """暗号化・セキュリティテスト"""
    print("\n" + "="*60)
    print("🔐 暗号化・セキュリティテスト")
    print("="*60)
    
    # 1. 機密文書（暗号化圧縮）
    confidential_doc = """
    CONFIDENTIAL - FOR AUTHORIZED PERSONNEL ONLY
    
    Security Report #2024-001
    Classification: TOP SECRET
    Distribution: EYES ONLY
    
    Executive Summary:
    This document contains sensitive information regarding the implementation
    of advanced compression algorithms in the NXZip platform. The information
    herein is proprietary and must not be disclosed to unauthorized parties.
    
    Technical Details:
    - Algorithm: NEXUS TMC v9.1 with SPE integration
    - Compression Ratio: Up to 99.26% for text data
    - Security Features: AES-256-GCM encryption with structure preservation
    - Performance: Exceeds 7-Zip compression with 2x speed improvement
    
    Implementation Notes:
    The Transform-Model-Code (TMC) approach utilizes:
    1. Burrows-Wheeler Transform for text reorganization
    2. Move-to-Front encoding for entropy reduction
    3. Structure-Preserving Encryption (SPE) for security
    4. LZMA/Zstd hybrid compression for final stage
    
    Threat Assessment:
    Current security measures are adequate for protecting against:
    - Brute force attacks (estimated 2^256 operations required)
    - Side-channel analysis (constant-time implementations used)
    - Reverse engineering (obfuscated algorithm parameters)
    
    Recommendations:
    1. Implement key rotation every 90 days
    2. Enable audit logging for all compression operations  
    3. Deploy hardware security modules for key management
    4. Conduct quarterly penetration testing
    
    Contact Information:
    Security Team: security@nxzip.com
    Emergency Contact: +1-555-SECURE
    
    Document Control:
    Created: 2024-01-15
    Version: 1.0
    Next Review: 2024-04-15
    """ * 20
    
    # Generate encryption key
    encryption_key = os.urandom(32)  # 256-bit key
    
    run_test_case("暗号化機密文書", confidential_doc.encode('utf-8'), "maximum", core, results, encryption_key)
    
    # 2. 証明書・キーファイル模擬
    cert_data = """
-----BEGIN CERTIFICATE-----
MIIFfTCCA2WgAwIBAgIJALZlJiPVzxCwMA0GCSqGSIb3DQEBCwUAMFYxCzAJBgNV
BAYTAlVTMQswCQYDVQQIDAJDQTEWMBQGA1UEBwwNU2FuIEZyYW5jaXNjbzEMMAoG
A1UECgwDTlhaSTEUMBIGA1UEAwwLTlhaaXAgQ29ycC4wHhcNMjQwMTE1MTIwMDAw
WhcNMjUwMTE1MTIwMDAwWjBWMQswCQYDVQQGEwJVUzELMAkGA1UECAwCQ0ExFjAU
BgNVBAcMDVNhbiBGcmFuY2lzY28xDDAKBgNVBAoMA05YWjEUMBIGA1UEAwwLTlha
aXAgQ29ycC4wggIiMA0GCSqGSIb3DQEBAQUAA4ICDwAwggIKAoICAQDYuJ5kT8mI
xRv+qJH8cEbDQaI9JbPVkJ5aJwZvN8pqQwYKzJgFH3GJg6lWmm2aBv5VQzJ7uI4X
aPu8K3NjEzBvU2PqL6rD8ZmWaVnXjN9OqG7TpAe5I4YJQ2fN7VsZB8rCqS6Hk3nM
bCZrV4JtPvWFgAw9EcKzLnQP3rUJwNv2B5gXd4SqTjBfEq7nOcGk1TxVlJmYp2Zs
QHgFkOqYuEz4M8KpJ1tFv7XQoVnCqZaP3J9oEb2MkZjYvG4R2wIDAQABo1MwUTAd
BgNVHQ4EFgQUJlYlNxVxWLgJ+sS7OmJ5mNbzUi0wHwYDVR0jBBgwFoAUJlYlNxNx
WLgJ+sS7OmJ5mNbzUi0wDwYDVR0TAQH/BAUwAwEB/zANBgkqhkiG9w0BAQsFAAOC
AgEAKJvUlCJaFnWaVKmBk2OqJ8X4pCzN6YoVgFq8mMhTgN4PqFzrZcMO5cTJ1yZB
9LGFhGlJmKjPwcZvB4L8xVqSvX3JbVvNyZw6QjJzL5kVzO2QgF7rDmE8YbZ4Z3zN
5pZKjP7oEwG1nXrYvNBcJzJfGvZaL4XyQnZ8pV2oGjYzEbN6OqPcX4rVnL9OkJgZ
H4RqY7zCwU4hM4oXjP2vB3nL7GfEwR4Y1tJsZoEoM6KvYxRn3pXy8wRvZnPcOlKj
-----END CERTIFICATE-----

-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDYuJ5kT8mIxRv+
qJH8cEbDQaI9JbPVkJ5aJwZvN8pqQwYKzJgFH3GJg6lWmm2aBv5VQzJ7uI4XaPu8
K3NjEzBvU2PqL6rD8ZmWaVnXjN9OqG7TpAe5I4YJQ2fN7VsZB8rCqS6Hk3nMbCZr
V4JtPvWFgAw9EcKzLnQP3rUJwNv2B5gXd4SqTjBfEq7nOcGk1TxVlJmYp2ZsQHgF
kOqYuEz4M8KpJ1tFv7XQoVnCqZaP3J9oEb2MkZjYvG4R2wIDAQABAoIBADT6oLzN
4xvNKcYjVn8XzGqGH7z3K5lJpZ6vYgPqBzfQoN1rE8KsJgA4OqYpHzNfZKg5rXvB
Nw9mGv6YrJbKz4Q8lMnC3TgJvZpYoLqKg7RzY2fNmO6VpQjNxEtJa5L8XqPvB2zG
WuJ3Q7kM9ZnYoVKjL6rF5TsOqGt7Bp4XyNr8CwZ1QaE5rYvNLgJ6pKjBnZ3fVoTu
Kz4q8O6YgJpL2NrX5vMzKgNqP7z8OfJbLqT6nV9RoYvKgJpO2NzB4qLvMrKgE8Yp
QjTnZ6vOqGt9LpX5rYvNK2z8CwJ6pKjBnO3fVoTuKz4qBO6YgJpL2NrX5vMzKgNq
P7z8OfJbLqT6nV9RoYvKgJpO2ECgYEA7z8K5JbQvYaG3KLqNpOg5J4VmN7zPx1qO
p6vYgFqBzfQoN1rE8KsJgA4OqYpHzNfZKg5rXvBNw9mGv6YrJbKz4Q8lMnC3TgJ
vZpYoLqKg7RzY2fNmO6VpQjNxEtJa5L8XqPvB2zGWuJ3Q7kM9ZnYoVKjL6rF5TsO
qGt7Bp4XyNr8CwZ1QaE5rYvNLgJ6pKjBnZ3fVoTuKz4q8O6YgJpL2NrX5vMzKgNq
-----END PRIVATE KEY-----
""" * 50
    
    run_test_case("証明書・キーファイル", cert_data.encode('utf-8'), "maximum", core, results, encryption_key)
    
    # 3. パスワード・ハッシュデータベース
    password_db = "user_id,username,password_hash,salt,created_at,last_modified\n"
    
    for i in range(5000):
        user_id = i + 1
        username = f"user{i:04d}"
        # Simulate bcrypt hashes
        hash_part = f"$2b$12$" + ''.join([chr(65 + (i * j) % 26) for j in range(22)])
        salt = ''.join([chr(97 + (i * 7 + j) % 26) for j in range(16)])
        password_hash = hash_part + salt + ''.join([chr(48 + (i * j) % 10) for j in range(31)])
        created_at = f"2023-{1 + (i % 12):02d}-{1 + (i % 28):02d}"
        last_modified = f"2024-01-{1 + (i % 31):02d}"
        
        password_db += f"{user_id},{username},{password_hash},{salt},{created_at},{last_modified}\n"
    
    run_test_case("パスワードハッシュDB", password_db.encode('utf-8'), "maximum", core, results, encryption_key)

def test_performance_and_stress(core: NXZipCore, results: TestResults):
    """パフォーマンス・ストレステスト"""
    print("\n" + "="*60)
    print("🚀 パフォーマンス・ストレステスト")
    print("="*60)
    
    # 1. 大容量テキストファイル（10MB）
    large_text = """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor 
    incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis 
    nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
    Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore 
    eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, 
    sunt in culpa qui officia deserunt mollit anim id est laborum.
    
    日本語のテキストも含まれています。これは多言語対応のテストです。
    漢字、ひらがな、カタカナ、そして英数字が混在しています。
    圧縮アルゴリズムは、これらの文字の頻度分析を行い、効率的に圧縮します。
    """ * 5000  # Approximately 10MB
    
    print(f"   大容量テキスト生成完了: {len(large_text):,} bytes")
    run_test_case("大容量テキスト(10MB)", large_text.encode('utf-8'), "balanced", core, results)
    
    # 2. 高エントロピーデータ（疑似ランダム）
    print("   高エントロピーデータ生成中...")
    random_data = os.urandom(1024 * 1024)  # 1MB of random data
    run_test_case("高エントロピーデータ(1MB)", random_data, "fast", core, results)
    
    # 3. 極低エントロピーデータ（高反復）
    repetitive_data = b'A' * (512 * 1024) + b'B' * (256 * 1024) + b'C' * (256 * 1024)  # 1MB
    run_test_case("極低エントロピーデータ(1MB)", repetitive_data, "fast", core, results)
    
    # 4. 全モード比較テスト（中サイズファイル）
    mixed_data = (large_text[:100000] + str(list(range(10000))) * 10).encode('utf-8')
    print(f"\n🔄 全モード比較テスト - データサイズ: {len(mixed_data):,} bytes")
    
    modes = ["fast", "balanced", "maximum"]
    for mode in modes:
        run_test_case(f"全モード比較-{mode.upper()}", mixed_data, mode, core, results)
    
    # 5. メモリ効率テスト（複数の小ファイル）
    print(f"\n📁 複数小ファイルテスト")
    for i in range(10):
        small_file_data = f"Small file #{i+1}\n" + "Data line " * 100 + f"\nEnd of file {i+1}\n"
        run_test_case(f"小ファイル#{i+1:02d}", small_file_data.encode('utf-8'), "fast", core, results)

def main():
    """メインテスト実行"""
    print("🚀 NXZip Core v2.0 実用的総合テストスイート")
    print("="*80)
    
    # テスト結果集計
    results = TestResults()
    
    # NXZip Core初期化
    core = NXZipCore()
    core.set_progress_callback(progress_callback)
    
    try:
        # 各テストカテゴリー実行
        test_documents_and_text(core, results)
        test_source_code_and_config(core, results)
        test_media_and_binary(core, results)
        test_database_and_logs(core, results)
        test_scientific_and_numerical(core, results)
        test_encryption_and_security(core, results)
        test_performance_and_stress(core, results)
        
        # 結果サマリー表示
        results.print_summary()
        
        # 詳細結果をJSONで保存
        detailed_results = {
            'test_suite': 'NXZip Core v2.0 Comprehensive Test',
            'execution_time': time.time(),
            'summary': results.summary,
            'detailed_results': results.results
        }
        
        with open('nxzip_test_results.json', 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 詳細結果を nxzip_test_results.json に保存しました")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ テストが中断されました")
        results.print_summary()
    except Exception as e:
        print(f"\n❌ テスト実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        results.print_summary()

if __name__ == "__main__":
    main()
