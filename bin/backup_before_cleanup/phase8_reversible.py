#!/usr/bin/env python3
"""
Phase 8 完全可逆版 - 100%可逆性保証エンジン
AI強化構造破壊型圧縮の可逆性完全実装
"""

import os
import sys
import time
import json
import struct
import lzma
import zlib
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Phase 8 Turbo エンジンを拡張
sys.path.append('bin')
from nexus_phase8_turbo import Phase8TurboEngine, CompressionResult, DecompressionResult

class ReversibleCompressionResult(CompressionResult):
    """可逆性保証結果クラス"""
    def __init__(self, original_size, compressed_size, compression_ratio, algorithm, 
                 processing_time, structure_map, compressed_data, performance_metrics,
                 original_hash, structure_integrity_hash):
        super().__init__(original_size, compressed_size, compression_ratio, algorithm,
                        processing_time, structure_map, compressed_data, performance_metrics)
        self.original_hash = original_hash
        self.structure_integrity_hash = structure_integrity_hash

class Phase8ReversibleEngine(Phase8TurboEngine):
    """Phase 8 完全可逆版エンジン - 100%可逆性保証"""
    
    def __init__(self):
        super().__init__()
        self.version = "8.0-Reversible"
        self.magic_header = b'NXZ8R'  # Reversible版マジックナンバー
        self.integrity_check = True
    
    def reversible_compress(self, data: bytes, filename: str = "data") -> ReversibleCompressionResult:
        """完全可逆圧縮 - 100%可逆性保証"""
        start_time = time.time()
        original_size = len(data)
        
        print(f"🔒 Phase 8 完全可逆圧縮開始: {filename}")
        print(f"📊 元サイズ: {original_size:,} bytes ({original_size/1024:.1f} KB)")
        
        # Step 1: 元データのハッシュ計算
        original_hash = hashlib.sha256(data).hexdigest()
        print(f"🔐 原本ハッシュ: {original_hash[:16]}...")
        
        # Step 2: AI強化構造解析（詳細情報保存）
        elements = self.analyze_file_structure(data)
        print(f"📈 構造解析完了: {len(elements)}要素")
        
        # Step 3: 完全構造マップ生成（復元に必要な全情報）
        structure_map = self._create_reversible_structure_map(elements, data)
        structure_integrity_hash = hashlib.sha256(structure_map).hexdigest()
        
        # Step 4: 可逆性保証並列圧縮
        compressed_chunks = []
        chunk_metadata = []
        total_chunks = len(elements)
        
        progress_points = [total_chunks//4, total_chunks//2, total_chunks*3//4, total_chunks]
        
        for i, element in enumerate(elements):
            # 可逆性保証圧縮
            compressed_chunk, metadata = self._reversible_compress_chunk(element)
            compressed_chunks.append(compressed_chunk)
            chunk_metadata.append(metadata)
            
            # 進捗表示
            if i + 1 in progress_points:
                percent = ((i + 1) / total_chunks) * 100
                print(f"🔒 可逆圧縮進捗: {percent:.0f}%")
        
        # Step 5: 可逆性統合（メタデータ含む）
        final_compressed = self._integrate_reversible_data(
            compressed_chunks, structure_map, chunk_metadata, original_hash
        )
        
        # Step 6: 結果計算
        compressed_size = len(final_compressed)
        compression_ratio = ((original_size - compressed_size) / original_size) * 100
        processing_time = time.time() - start_time
        
        # AI解析サマリー
        if elements:
            avg_entropy = sum(e.entropy for e in elements) / len(elements)
            ai_recommendations = [e.compression_hint for e in elements]
            most_common_hint = max(set(ai_recommendations), key=ai_recommendations.count)
            
            print(f"🤖 AI解析結果:")
            print(f"   平均エントロピー: {avg_entropy:.2f}")
            print(f"   主要推薦手法: {most_common_hint}")
        
        print(f"✅ 可逆圧縮完了: {compression_ratio:.1f}% ({original_size:,} → {compressed_size:,})")
        print(f"⏱️ 処理時間: {processing_time:.2f}秒")
        print(f"🔐 構造整合性: {structure_integrity_hash[:16]}...")
        
        # 性能指標
        speed_mbps = original_size / processing_time / (1024 * 1024)
        performance_metrics = {
            'analysis_elements': len(elements),
            'avg_entropy': avg_entropy if elements else 0.0,
            'processing_speed_mbps': speed_mbps,
            'ai_recommendation': most_common_hint if elements else 'none',
            'reversible_mode': True,
            'integrity_verification': True
        }
        
        return ReversibleCompressionResult(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            algorithm="Phase8_Reversible",
            processing_time=processing_time,
            structure_map=structure_map,
            compressed_data=final_compressed,
            performance_metrics=performance_metrics,
            original_hash=original_hash,
            structure_integrity_hash=structure_integrity_hash
        )
    
    def reversible_decompress(self, compressed_data: bytes) -> DecompressionResult:
        """完全可逆展開 - 100%復元保証"""
        start_time = time.time()
        
        print("🔓 Phase 8 完全可逆展開開始")
        
        # ヘッダー検証
        if not compressed_data.startswith(self.magic_header):
            raise ValueError("❌ Phase 8 可逆形式ではありません")
        
        offset = len(self.magic_header)
        
        # 元データハッシュ
        original_hash = compressed_data[offset:offset+64].decode('ascii')
        offset += 64
        print(f"🔐 原本ハッシュ確認: {original_hash[:16]}...")
        
        # 構造マップサイズ
        structure_map_size = struct.unpack('<I', compressed_data[offset:offset+4])[0]
        offset += 4
        
        # 構造マップ復元
        structure_map_data = compressed_data[offset:offset+structure_map_size]
        offset += structure_map_size
        
        # 構造整合性検証
        structure_hash = hashlib.sha256(structure_map_data).hexdigest()
        print(f"🔍 構造整合性: {structure_hash[:16]}...")
        
        structure_info = self._parse_reversible_structure_map(structure_map_data)
        print(f"📊 構造復元: {structure_info['total_elements']}要素")
        
        # チャンクメタデータサイズ
        metadata_size = struct.unpack('<I', compressed_data[offset:offset+4])[0]
        offset += 4
        
        # チャンクメタデータ復元
        metadata_data = compressed_data[offset:offset+metadata_size]
        offset += metadata_size
        chunk_metadata = json.loads(lzma.decompress(metadata_data).decode('utf-8'))
        
        # 可逆チャンク復元
        decompressed_chunks = []
        elements_info = structure_info['elements']
        
        for i, element_info in enumerate(elements_info):
            chunk_size = struct.unpack('<I', compressed_data[offset:offset+4])[0]
            offset += 4
            
            if chunk_size > 0:
                chunk_data = compressed_data[offset:offset+chunk_size]
                offset += chunk_size
                
                # 可逆性保証展開
                metadata = chunk_metadata[i] if i < len(chunk_metadata) else {}
                decompressed_chunk = self._reversible_decompress_chunk(
                    chunk_data, element_info, metadata
                )
                decompressed_chunks.append(decompressed_chunk)
            else:
                decompressed_chunks.append(b'')
            
            # 進捗表示
            if (i + 1) % max(1, len(elements_info) // 4) == 0:
                percent = ((i + 1) / len(elements_info)) * 100
                print(f"🔓 展開進捗: {percent:.0f}%")
        
        # 完全復元（構造情報完全使用）
        original_data = self._reconstruct_reversible_original(
            decompressed_chunks, structure_info
        )
        
        # 可逆性検証
        restored_hash = hashlib.sha256(original_data).hexdigest()
        is_identical = (restored_hash == original_hash)
        
        processing_time = time.time() - start_time
        print(f"✅ 展開完了: {len(original_data):,} bytes ({processing_time:.2f}秒)")
        print(f"🔍 可逆性検証: {'✅ 完全一致' if is_identical else '❌ 不一致'}")
        
        if not is_identical:
            print(f"⚠️ 原本: {original_hash[:16]}...")
            print(f"⚠️ 復元: {restored_hash[:16]}...")
            raise ValueError("❌ 可逆性検証失敗")
        
        return DecompressionResult(
            original_data=original_data,
            decompressed_size=len(original_data),
            processing_time=processing_time,
            algorithm="Phase8_Reversible"
        )
    
    def _create_reversible_structure_map(self, elements, original_data: bytes) -> bytes:
        """可逆性保証構造マップ生成"""
        structure_info = {
            'version': self.version,
            'total_elements': len(elements),
            'original_size': len(original_data),
            'ai_enhanced': True,
            'reversible_mode': True,
            'elements': [],
            'global_structure': {
                'file_signature': original_data[:16].hex() if len(original_data) >= 16 else '',
                'file_end': original_data[-16:].hex() if len(original_data) >= 16 else '',
                'total_chunks': len(elements)
            }
        }
        
        # 完全構造情報保存
        cumulative_offset = 0
        for i, element in enumerate(elements):
            element_info = {
                'index': i,
                'type': element.type,
                'absolute_offset': element.offset,
                'cumulative_offset': cumulative_offset,
                'original_size': element.size,
                'entropy': element.entropy,
                'pattern_score': element.pattern_score,
                'compression_hint': element.compression_hint,
                'data_signature': element.data[:8].hex() if len(element.data) >= 8 else '',
                'data_end': element.data[-8:].hex() if len(element.data) >= 8 else ''
            }
            
            # AI解析結果完全保存
            if element.ai_analysis:
                element_info['ai_analysis'] = element.ai_analysis
            
            structure_info['elements'].append(element_info)
            cumulative_offset += element.size
        
        # JSON→バイナリ圧縮
        json_data = json.dumps(structure_info, separators=(',', ':')).encode('utf-8')
        return lzma.compress(json_data, preset=9)
    
    def _reversible_compress_chunk(self, element) -> Tuple[bytes, Dict]:
        """可逆性保証チャンク圧縮"""
        data = element.data
        hint = element.compression_hint
        
        # 元データの完全メタデータ
        metadata = {
            'original_size': len(data),
            'original_hash': hashlib.md5(data).hexdigest(),
            'compression_method': hint,
            'data_characteristics': {
                'entropy': element.entropy,
                'pattern_score': element.pattern_score,
                'first_bytes': data[:16].hex() if len(data) >= 16 else '',
                'last_bytes': data[-16:].hex() if len(data) >= 16 else ''
            }
        }
        
        # 可逆性保証圧縮
        if hint == "minimal_processing" or len(data) < 32:
            # 小さなデータや特殊データは無圧縮保存
            compressed_data = data
            metadata['actual_method'] = 'uncompressed'
        else:
            try:
                if hint == "rle_enhanced":
                    compressed_data = self._safe_rle_compress(data)
                    metadata['actual_method'] = 'rle_enhanced'
                elif hint == "lzma":
                    compressed_data = lzma.compress(data, preset=6, check=lzma.CHECK_CRC64)
                    metadata['actual_method'] = 'lzma'
                elif hint == "zstd":
                    compressed_data = zlib.compress(data, level=6)
                    metadata['actual_method'] = 'zlib'
                else:
                    # 適応的圧縮（最良結果を可逆的に選択）
                    compressed_data = self._safe_adaptive_compress(data)
                    metadata['actual_method'] = 'adaptive'
                
                # 圧縮効果検証
                if len(compressed_data) >= len(data):
                    compressed_data = data
                    metadata['actual_method'] = 'uncompressed'
                    
            except Exception:
                # 圧縮失敗時は無圧縮保存
                compressed_data = data
                metadata['actual_method'] = 'uncompressed'
        
        return compressed_data, metadata
    
    def _safe_rle_compress(self, data: bytes) -> bytes:
        """安全なRLE圧縮（完全可逆保証）"""
        if not data:
            return b''
        
        compressed = bytearray()
        i = 0
        while i < len(data):
            current_byte = data[i]
            count = 1
            
            # 同じバイトの連続をカウント（最大254まで）
            while (i + count < len(data) and 
                   data[i + count] == current_byte and 
                   count < 254):
                count += 1
            
            if count >= 3:  # 3回以上で圧縮効果
                compressed.extend([0xFF, count, current_byte])
                i += count
            else:
                # 0xFFエスケープ処理
                if current_byte == 0xFF:
                    compressed.extend([0xFF, 0, 0xFF])  # エスケープシーケンス
                else:
                    compressed.append(current_byte)
                i += 1
        
        return bytes(compressed)
    
    def _safe_adaptive_compress(self, data: bytes) -> bytes:
        """安全な適応的圧縮（可逆性最優先）"""
        if len(data) < 64:
            return data
        
        # 複数手法を試行し、最良かつ安全なものを選択
        candidates = []
        
        try:
            lzma_result = lzma.compress(data, preset=3, check=lzma.CHECK_CRC64)
            candidates.append(('lzma', lzma_result))
        except:
            pass
        
        try:
            zlib_result = zlib.compress(data, level=3)
            candidates.append(('zlib', zlib_result))
        except:
            pass
        
        try:
            rle_result = self._safe_rle_compress(data)
            candidates.append(('rle', rle_result))
        except:
            pass
        
        # 最小サイズを選択
        if candidates:
            best_method, best_result = min(candidates, key=lambda x: len(x[1]))
            if len(best_result) < len(data):
                return best_result
        
        return data  # フォールバック
    
    def _integrate_reversible_data(self, compressed_chunks, structure_map: bytes, 
                                 chunk_metadata: List[Dict], original_hash: str) -> bytes:
        """可逆性保証データ統合"""
        result = bytearray()
        
        # ヘッダー
        result.extend(self.magic_header)
        
        # 元データハッシュ（64文字固定）
        result.extend(original_hash.encode('ascii'))
        
        # 構造マップ
        result.extend(struct.pack('<I', len(structure_map)))
        result.extend(structure_map)
        
        # チャンクメタデータ
        metadata_json = json.dumps(chunk_metadata, separators=(',', ':')).encode('utf-8')
        metadata_compressed = lzma.compress(metadata_json, preset=9)
        result.extend(struct.pack('<I', len(metadata_compressed)))
        result.extend(metadata_compressed)
        
        # 圧縮チャンク
        for chunk in compressed_chunks:
            result.extend(struct.pack('<I', len(chunk)))
            result.extend(chunk)
        
        return bytes(result)
    
    def _parse_reversible_structure_map(self, structure_map_data: bytes) -> dict:
        """可逆性保証構造マップ解析"""
        try:
            decompressed_json = lzma.decompress(structure_map_data)
            structure_info = json.loads(decompressed_json.decode('utf-8'))
            
            # 可逆性モード検証
            if not structure_info.get('reversible_mode', False):
                raise ValueError("非可逆モードのファイルです")
            
            return structure_info
        except Exception as e:
            raise ValueError(f"構造マップ解析エラー: {e}")
    
    def _reversible_decompress_chunk(self, chunk_data: bytes, element_info: dict, 
                                   metadata: dict) -> bytes:
        """可逆性保証チャンク展開"""
        actual_method = metadata.get('actual_method', 'uncompressed')
        original_size = metadata.get('original_size', 0)
        original_hash = metadata.get('original_hash', '')
        
        try:
            if actual_method == 'uncompressed':
                result = chunk_data
            elif actual_method == 'rle_enhanced':
                result = self._safe_rle_decompress(chunk_data)
            elif actual_method == 'lzma':
                result = lzma.decompress(chunk_data)
            elif actual_method == 'zlib':
                result = zlib.decompress(chunk_data)
            elif actual_method == 'adaptive':
                result = self._safe_adaptive_decompress(chunk_data)
            else:
                result = chunk_data
            
            # 可逆性検証
            if original_hash and len(result) > 0:
                restored_hash = hashlib.md5(result).hexdigest()
                if restored_hash != original_hash:
                    print(f"⚠️ チャンク可逆性警告: 期待{original_hash[:8]} vs 実際{restored_hash[:8]}")
            
            return result
            
        except Exception as e:
            print(f"⚠️ チャンク展開エラー: {e}, フォールバック使用")
            return chunk_data
    
    def _safe_rle_decompress(self, data: bytes) -> bytes:
        """安全なRLE展開（エスケープ処理対応）"""
        if not data:
            return b''
        
        result = bytearray()
        i = 0
        while i < len(data):
            if i + 2 < len(data) and data[i] == 0xFF:
                count = data[i + 1]
                if count == 0:
                    # エスケープシーケンス: 0xFF 0 0xFF → 0xFF
                    result.append(0xFF)
                    i += 3
                else:
                    # RLE圧縮データ: 0xFF count value
                    byte_value = data[i + 2]
                    result.extend([byte_value] * count)
                    i += 3
            else:
                # 通常データ
                result.append(data[i])
                i += 1
        
        return bytes(result)
    
    def _safe_adaptive_decompress(self, data: bytes) -> bytes:
        """安全な適応的展開"""
        # 複数の展開方法を試行
        methods = [
            lzma.decompress,
            zlib.decompress,
            self._safe_rle_decompress
        ]
        
        for method in methods:
            try:
                result = method(data)
                if result:  # 成功した場合
                    return result
            except:
                continue
        
        return data  # フォールバック
    
    def _reconstruct_reversible_original(self, chunks: List[bytes], 
                                       structure_info: dict) -> bytes:
        """完全可逆復元（構造情報完全使用）"""
        elements_info = structure_info['elements']
        original_size = structure_info.get('original_size', 0)
        
        # 元の順序とサイズで完全復元
        result = bytearray()
        
        for i, chunk in enumerate(chunks):
            if i < len(elements_info):
                element_info = elements_info[i]
                expected_size = element_info.get('original_size', len(chunk))
                
                # サイズ検証
                if len(chunk) != expected_size:
                    print(f"⚠️ 要素{i}: サイズ不一致 期待{expected_size} vs 実際{len(chunk)}")
                
                # データ署名検証（可能な場合）
                if len(chunk) >= 8:
                    expected_sig = element_info.get('data_signature', '')
                    actual_sig = chunk[:8].hex()
                    if expected_sig and expected_sig != actual_sig:
                        print(f"⚠️ 要素{i}: 署名不一致 期待{expected_sig[:8]} vs 実際{actual_sig[:8]}")
                
                result.extend(chunk)
            else:
                result.extend(chunk)
        
        # 最終サイズ検証
        if original_size > 0 and len(result) != original_size:
            print(f"⚠️ 全体サイズ不一致: 期待{original_size} vs 実際{len(result)}")
        
        return bytes(result)
    
    def compress_file(self, input_path: str, output_path: str = None) -> bool:
        """ファイル圧縮（可逆性保証）"""
        if not os.path.exists(input_path):
            print(f"❌ ファイルが見つかりません: {input_path}")
            return False
        
        if output_path is None:
            output_path = input_path + '.p8r'  # Phase 8 Reversible
        
        try:
            with open(input_path, 'rb') as f:
                data = f.read()
            
            filename = os.path.basename(input_path)
            result = self.reversible_compress(data, filename)
            
            with open(output_path, 'wb') as f:
                f.write(result.compressed_data)
            
            print(f"💾 可逆圧縮ファイル保存: {output_path}")
            return True
        
        except Exception as e:
            print(f"❌ 圧縮エラー: {e}")
            return False
    
    def decompress_file(self, input_path: str, output_path: str = None) -> bool:
        """ファイル展開（可逆性検証）"""
        if not os.path.exists(input_path):
            print(f"❌ ファイルが見つかりません: {input_path}")
            return False
        
        if output_path is None:
            if input_path.endswith('.p8r'):
                output_path = input_path[:-4]
            else:
                output_path = input_path + '.restored'
        
        try:
            with open(input_path, 'rb') as f:
                compressed_data = f.read()
            
            result = self.reversible_decompress(compressed_data)
            
            with open(output_path, 'wb') as f:
                f.write(result.original_data)
            
            print(f"📁 可逆復元ファイル保存: {output_path}")
            return True
        
        except Exception as e:
            print(f"❌ 展開エラー: {e}")
            return False

def run_reversible_test():
    """完全可逆性テスト"""
    print("🔒 Phase 8 完全可逆性テスト")
    print("=" * 60)
    
    engine = Phase8ReversibleEngine()
    sample_dir = Path("../NXZip-Python/sample")
    
    # 重要: 可逆性重視のテストファイル選択
    test_files = [
        # 基本テスト（小〜中サイズ）
        "陰謀論.mp3",                    # MP3音声 (2MB)
        "COT-001.jpg",                   # JPEG画像 (2.8MB)
        "COT-012.png",                   # PNG画像 (35MB) - 制限版
        
        # 大容量テスト（段階的）
        "出庫実績明細_202412.txt",      # テキスト (97MB) - 最終テスト
    ]
    
    results = []
    failed_files = []
    
    for filename in test_files:
        filepath = sample_dir / filename
        if not filepath.exists():
            print(f"⚠️ ファイルなし: {filename}")
            continue
        
        print(f"\n🔒 可逆性テスト: {filename}")
        print("-" * 40)
        
        try:
            # 段階的テスト（大容量ファイルは一部のみ）
            if filename == "COT-012.png":
                # PNG: 最初の1MBのみテスト
                with open(filepath, 'rb') as f:
                    test_data = f.read(1024*1024)  # 1MB制限
                print(f"📏 部分テスト: {len(test_data):,} bytes (1MB制限)")
            elif filename == "出庫実績明細_202412.txt":
                # テキスト: 最初の5MBのみテスト
                with open(filepath, 'rb') as f:
                    test_data = f.read(5*1024*1024)  # 5MB制限
                print(f"📏 部分テスト: {len(test_data):,} bytes (5MB制限)")
            else:
                # 全体テスト
                with open(filepath, 'rb') as f:
                    test_data = f.read()
                print(f"📏 全体テスト: {len(test_data):,} bytes")
            
            # 可逆圧縮
            result = engine.reversible_compress(test_data, filename)
            
            # 可逆展開
            decompressed_result = engine.reversible_decompress(result.compressed_data)
            
            # 可逆性検証
            is_identical = (test_data == decompressed_result.original_data)
            
            if is_identical:
                print(f"✅ 可逆性成功: 完全一致")
                results.append({
                    'filename': filename,
                    'original_size': len(test_data),
                    'compressed_size': result.compressed_size,
                    'compression_ratio': result.compression_ratio,
                    'reversible': True,
                    'processing_time': result.processing_time
                })
            else:
                print(f"❌ 可逆性失敗: データ不一致")
                failed_files.append(filename)
                
        except Exception as e:
            print(f"❌ テストエラー: {str(e)[:80]}...")
            failed_files.append(filename)
    
    # 総合結果
    print("\n" + "=" * 60)
    print("🏆 Phase 8 完全可逆性テスト結果")
    print("=" * 60)
    
    if results:
        total_original = sum(r['original_size'] for r in results)
        total_compressed = sum(r['compressed_size'] for r in results)
        overall_ratio = (1 - total_compressed / total_original) * 100
        reversible_count = sum(1 for r in results if r['reversible'])
        
        print(f"🔒 可逆性成功率: {reversible_count}/{len(results)} ({reversible_count/len(results)*100:.1f}%)")
        print(f"📊 平均圧縮率: {overall_ratio:.1f}%")
        print(f"📈 テストファイル数: {len(results)}")
        print(f"💾 総データ量: {total_original/1024/1024:.1f} MB")
        print(f"⚡ 平均処理速度: {sum(r['original_size'] for r in results) / sum(r['processing_time'] for r in results) / 1024 / 1024:.1f} MB/s")
        
        # 個別結果
        print(f"\n📋 個別可逆性テスト結果:")
        for result in results:
            filename_short = result['filename'][:25] + ('...' if len(result['filename']) > 25 else '')
            size_mb = result['original_size'] / 1024 / 1024
            speed = result['original_size'] / result['processing_time'] / 1024 / 1024
            print(f"   ✅ {filename_short}: {result['compression_ratio']:.1f}% ({size_mb:.1f}MB, {speed:.1f}MB/s)")
        
        if reversible_count == len(results):
            print("🎉 全ファイル完全可逆性達成！")
        else:
            print(f"⚠️ {len(results) - reversible_count}ファイルで可逆性問題")
    
    if failed_files:
        print(f"\n❌ 失敗ファイル ({len(failed_files)}個):")
        for filename in failed_files:
            print(f"   • {filename}")

def main():
    """メイン処理"""
    if len(sys.argv) < 2:
        print("🔒 Phase 8 完全可逆版")
        print("使用方法:")
        print("  python phase8_reversible.py test                    # 可逆性テスト")
        print("  python phase8_reversible.py compress <file>         # 可逆圧縮")
        print("  python phase8_reversible.py decompress <file.p8r>   # 可逆展開")
        return
    
    command = sys.argv[1].lower()
    engine = Phase8ReversibleEngine()
    
    if command == "test":
        run_reversible_test()
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
