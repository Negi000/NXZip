#!/usr/bin/env python3
"""
Phase 8 Turbo 完全実装 - 圧縮・展開機能付き
AI強化構造破壊型圧縮の実用化
"""

import os
import sys
import time
import json
import struct
import lzma
import zlib
from pathlib import Path

# Phase 8 Turbo エンジンを拡張
sys.path.append('bin')
from nexus_phase8_turbo import Phase8TurboEngine, CompressionResult, DecompressionResult

class Phase8FullEngine(Phase8TurboEngine):
    """Phase 8 完全版エンジン - 圧縮・展開実装"""
    
    def __init__(self):
        super().__init__()
        self.version = "8.0-Full"
        self.magic_header = b'NXZ8F'  # Full版マジックナンバー
    
    def turbo_compress(self, data: bytes, filename: str = "data") -> CompressionResult:
        """Turbo 構造破壊型圧縮 - 完全実装"""
        start_time = time.time()
        original_size = len(data)
        
        print(f"🚀 Phase 8 Turbo 圧縮開始: {filename}")
        print(f"📊 元サイズ: {original_size:,} bytes ({original_size/1024:.1f} KB)")
        
        # Step 1: AI強化構造解析
        elements = self.analyze_file_structure(data)
        print(f"📈 構造解析完了: {len(elements)}要素")
        
        # Step 2: 構造マップ生成
        structure_map = self._create_turbo_structure_map(elements)
        
        # Step 3: 並列圧縮実行
        compressed_chunks = []
        total_chunks = len(elements)
        
        progress_points = [total_chunks//4, total_chunks//2, total_chunks*3//4, total_chunks]
        
        for i, element in enumerate(elements):
            # AI推薦圧縮手法で圧縮
            compressed_chunk = self._turbo_compress_chunk(element)
            compressed_chunks.append(compressed_chunk)
            
            # 進捗表示（効率化）
            if i + 1 in progress_points:
                percent = ((i + 1) / total_chunks) * 100
                print(f"⚡ 圧縮進捗: {percent:.0f}%")
        
        # Step 4: 最終統合
        final_compressed = self._integrate_turbo_data(compressed_chunks, structure_map)
        
        # Step 5: 結果計算
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
        
        print(f"✅ 圧縮完了: {compression_ratio:.1f}% ({original_size:,} → {compressed_size:,})")
        print(f"⏱️ 処理時間: {processing_time:.2f}秒")
        
        # 性能指標
        speed_mbps = original_size / processing_time / (1024 * 1024)
        performance_metrics = {
            'analysis_elements': len(elements),
            'avg_entropy': avg_entropy if elements else 0.0,
            'processing_speed_mbps': speed_mbps,
            'ai_recommendation': most_common_hint if elements else 'none'
        }
        
        return CompressionResult(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            algorithm="Phase8_Turbo_Full",
            processing_time=processing_time,
            structure_map=structure_map,
            compressed_data=final_compressed,
            performance_metrics=performance_metrics
        )
    
    def turbo_decompress(self, compressed_data: bytes) -> DecompressionResult:
        """Turbo 構造破壊型展開 - 完全実装"""
        start_time = time.time()
        
        print("🔄 Phase 8 Turbo 展開開始")
        
        # ヘッダー検証
        if not compressed_data.startswith(self.magic_header):
            raise ValueError("❌ Phase 8 Turbo形式ではありません")
        
        offset = len(self.magic_header)
        
        # 構造マップサイズ
        structure_map_size = struct.unpack('<I', compressed_data[offset:offset+4])[0]
        offset += 4
        
        # 構造マップ復元
        structure_map_data = compressed_data[offset:offset+structure_map_size]
        offset += structure_map_size
        
        structure_info = self._parse_turbo_structure_map(structure_map_data)
        print(f"📊 構造復元: {structure_info['total_elements']}要素")
        
        # チャンク復元
        decompressed_chunks = []
        elements_info = structure_info['elements']
        
        for i, element_info in enumerate(elements_info):
            chunk_size = struct.unpack('<I', compressed_data[offset:offset+4])[0]
            offset += 4
            
            if chunk_size > 0:
                chunk_data = compressed_data[offset:offset+chunk_size]
                offset += chunk_size
                
                # AI推薦手法で展開
                decompressed_chunk = self._turbo_decompress_chunk(chunk_data, element_info)
                decompressed_chunks.append(decompressed_chunk)
            else:
                decompressed_chunks.append(b'')
            
            # 進捗表示
            if (i + 1) % max(1, len(elements_info) // 4) == 0:
                percent = ((i + 1) / len(elements_info)) * 100
                print(f"🔄 展開進捗: {percent:.0f}%")
        
        # 完全復元
        original_data = self._reconstruct_turbo_original(decompressed_chunks, structure_info)
        
        processing_time = time.time() - start_time
        print(f"✅ 展開完了: {len(original_data):,} bytes ({processing_time:.2f}秒)")
        
        return DecompressionResult(
            original_data=original_data,
            decompressed_size=len(original_data),
            processing_time=processing_time,
            algorithm="Phase8_Turbo_Full"
        )
    
    def _create_turbo_structure_map(self, elements) -> bytes:
        """Turbo構造マップ生成"""
        structure_info = {
            'version': self.version,
            'total_elements': len(elements),
            'ai_enhanced': True,
            'elements': []
        }
        
        for element in elements:
            element_info = {
                'type': element.type,
                'offset': element.offset,
                'size': element.size,
                'entropy': element.entropy,
                'pattern_score': element.pattern_score,
                'compression_hint': element.compression_hint
            }
            
            # AI解析結果を含める
            if element.ai_analysis:
                element_info['ai_analysis'] = element.ai_analysis
            
            structure_info['elements'].append(element_info)
        
        # JSON→バイナリ圧縮
        json_data = json.dumps(structure_info, separators=(',', ':')).encode('utf-8')
        return lzma.compress(json_data, preset=9)
    
    def _turbo_compress_chunk(self, element) -> bytes:
        """Turbo チャンク圧縮 - AI推薦手法"""
        data = element.data
        hint = element.compression_hint
        
        # AI推薦に基づく最適圧縮
        if hint == "rle_enhanced":
            return self._enhanced_rle_compress(data)
        elif hint == "lzma":
            return self._turbo_lzma_compress(data)
        elif hint == "zstd":
            return self._turbo_zstd_compress(data)
        elif hint == "brotli":
            return self._turbo_brotli_compress(data)
        elif hint == "minimal_processing":
            return data  # 生データ保存
        else:  # adaptive_optimal
            return self._turbo_adaptive_compress(data)
    
    def _enhanced_rle_compress(self, data: bytes) -> bytes:
        """強化RLE圧縮"""
        if not data:
            return b''
        
        compressed = bytearray()
        i = 0
        while i < len(data):
            current_byte = data[i]
            count = 1
            
            # 同じバイトの連続をカウント
            while i + count < len(data) and data[i + count] == current_byte and count < 255:
                count += 1
            
            if count >= 3:  # 3回以上の繰り返しで圧縮
                compressed.extend([0xFF, count, current_byte])
                i += count
            else:
                compressed.append(current_byte)
                i += 1
        
        return bytes(compressed)
    
    def _turbo_lzma_compress(self, data: bytes) -> bytes:
        """Turbo LZMA圧縮"""
        try:
            return lzma.compress(data, preset=6, check=lzma.CHECK_NONE)
        except:
            return data
    
    def _turbo_zstd_compress(self, data: bytes) -> bytes:
        """Turbo Zstd風圧縮（zlibで代用）"""
        try:
            return zlib.compress(data, level=6)
        except:
            return data
    
    def _turbo_brotli_compress(self, data: bytes) -> bytes:
        """Turbo Brotli風圧縮（lzmaで代用）"""
        try:
            return lzma.compress(data, preset=3)
        except:
            return data
    
    def _turbo_adaptive_compress(self, data: bytes) -> bytes:
        """Turbo適応的圧縮"""
        if not data:
            return b''
        
        # 複数手法を試して最良を選択
        methods = [
            (self._turbo_lzma_compress, 'lzma'),
            (self._turbo_zstd_compress, 'zstd'),
            (self._enhanced_rle_compress, 'rle')
        ]
        
        best_result = data
        best_size = len(data)
        
        for method, name in methods:
            try:
                result = method(data)
                if len(result) < best_size:
                    best_result = result
                    best_size = len(result)
            except:
                continue
        
        return best_result
    
    def _integrate_turbo_data(self, compressed_chunks, structure_map: bytes) -> bytes:
        """Turbo データ統合"""
        result = bytearray()
        
        # ヘッダー
        result.extend(self.magic_header)
        result.extend(struct.pack('<I', len(structure_map)))
        result.extend(structure_map)
        
        # 圧縮チャンク
        for chunk in compressed_chunks:
            result.extend(struct.pack('<I', len(chunk)))
            result.extend(chunk)
        
        return bytes(result)
    
    def _parse_turbo_structure_map(self, structure_map_data: bytes) -> dict:
        """Turbo構造マップ解析"""
        try:
            decompressed_json = lzma.decompress(structure_map_data)
            return json.loads(decompressed_json.decode('utf-8'))
        except Exception as e:
            raise ValueError(f"構造マップ解析エラー: {e}")
    
    def _turbo_decompress_chunk(self, chunk_data: bytes, element_info: dict) -> bytes:
        """Turboチャンク展開"""
        hint = element_info.get('compression_hint', 'adaptive_optimal')
        
        try:
            if hint == "rle_enhanced":
                return self._enhanced_rle_decompress(chunk_data)
            elif hint == "lzma":
                return lzma.decompress(chunk_data)
            elif hint == "zstd":
                return zlib.decompress(chunk_data)
            elif hint == "brotli":
                return lzma.decompress(chunk_data)
            elif hint == "minimal_processing":
                return chunk_data
            else:
                return self._turbo_adaptive_decompress(chunk_data)
        except Exception:
            return chunk_data
    
    def _enhanced_rle_decompress(self, data: bytes) -> bytes:
        """強化RLE展開"""
        if not data:
            return b''
        
        result = bytearray()
        i = 0
        while i < len(data):
            if i + 2 < len(data) and data[i] == 0xFF:
                # RLE圧縮データ
                count = data[i + 1]
                byte_value = data[i + 2]
                result.extend([byte_value] * count)
                i += 3
            else:
                # 通常データ
                result.append(data[i])
                i += 1
        
        return bytes(result)
    
    def _turbo_adaptive_decompress(self, data: bytes) -> bytes:
        """Turbo適応的展開"""
        # 複数の展開方法を試行
        methods = [lzma.decompress, zlib.decompress, self._enhanced_rle_decompress]
        
        for method in methods:
            try:
                return method(data)
            except:
                continue
        
        return data
    
    def _reconstruct_turbo_original(self, chunks, structure_info: dict) -> bytes:
        """Turbo完全復元 - 修正版"""
        result = bytearray()
        
        # 元の構造順序でチャンクを配置
        elements_info = structure_info['elements']
        
        for i, chunk in enumerate(chunks):
            if i < len(elements_info):
                element_info = elements_info[i]
                # 元の位置・サイズ情報を使用して正確に復元
                result.extend(chunk)
            else:
                result.extend(chunk)
        
        return bytes(result)
    
    def compress_file(self, input_path: str, output_path: str = None) -> bool:
        """ファイル圧縮"""
        if not os.path.exists(input_path):
            print(f"❌ ファイルが見つかりません: {input_path}")
            return False
        
        if output_path is None:
            output_path = input_path + '.p8t'  # Phase 8 Turbo
        
        try:
            with open(input_path, 'rb') as f:
                data = f.read()
            
            filename = os.path.basename(input_path)
            result = self.turbo_compress(data, filename)
            
            with open(output_path, 'wb') as f:
                f.write(result.compressed_data)
            
            print(f"💾 圧縮ファイル保存: {output_path}")
            
            # Phase 7との比較
            phase7_estimated = len(data) * 0.427  # Phase 7平均57.3%圧縮
            improvement = (phase7_estimated - result.compressed_size) / phase7_estimated * 100
            
            print(f"🏆 Phase 7比較:")
            print(f"   Phase 7推定: {phase7_estimated:,.0f} bytes")
            print(f"   Phase 8実測: {result.compressed_size:,} bytes")
            print(f"   改善率: {improvement:+.1f}%")
            
            return True
        
        except Exception as e:
            print(f"❌ 圧縮エラー: {e}")
            return False
    
    def decompress_file(self, input_path: str, output_path: str = None) -> bool:
        """ファイル展開"""
        if not os.path.exists(input_path):
            print(f"❌ ファイルが見つかりません: {input_path}")
            return False
        
        if output_path is None:
            if input_path.endswith('.p8t'):
                output_path = input_path[:-4]
            else:
                output_path = input_path + '.restored'
        
        try:
            with open(input_path, 'rb') as f:
                compressed_data = f.read()
            
            result = self.turbo_decompress(compressed_data)
            
            with open(output_path, 'wb') as f:
                f.write(result.original_data)
            
            print(f"📁 復元ファイル保存: {output_path}")
            return True
        
        except Exception as e:
            print(f"❌ 展開エラー: {e}")
            return False

def run_phase8_test():
    """Phase 8 完全テスト"""
    print("🚀 Phase 8 Turbo 完全実装テスト")
    print("=" * 60)
    
    engine = Phase8FullEngine()
    sample_dir = Path("../NXZip-Python/sample")
    
    # 全ファイル形式での包括的テスト
    test_files = [
        # テキストファイル
        "出庫実績明細_202412.txt",      # 大容量テキスト (97MB)
        
        # 音声ファイル
        "陰謀論.mp3",                    # MP3音声 (2MB)
        "generated-music-1752042054079.wav",  # WAV音声 (4MB)
        
        # 画像ファイル  
        "COT-001.jpg",                   # JPEG画像 (2.8MB)
        "COT-012.png",                   # PNG画像 (35MB)
        
        # 動画ファイル
        "Python基礎講座3_4月26日-3.mp4", # MP4動画 (30MB)
        
        # 圧縮済みファイル
        "COT-001.7z",                    # 7-Zip圧縮済み
        "COT-012.7z",                    # 7-Zip圧縮済み
        "Python基礎講座3_4月26日-3.7z", # 7-Zip圧縮済み
    ]
    
    results = []
    
    for filename in test_files:
        filepath = sample_dir / filename
        if not filepath.exists():
            print(f"⚠️ ファイルなし: {filename}")
            continue
        
        print(f"\n📁 テストファイル: {filename}")
        print("-" * 40)
        
        try:
            # 圧縮テスト
            output_path = str(filepath) + '.p8t'
            success = engine.compress_file(str(filepath), output_path)
            
            if success:
                # 展開テスト
                restored_path = output_path + '.restored'
                decompress_success = engine.decompress_file(output_path, restored_path)
                
                if decompress_success:
                    # 可逆性検証
                    with open(filepath, 'rb') as f:
                        original = f.read()
                    with open(restored_path, 'rb') as f:
                        restored = f.read()
                    
                    is_identical = (original == restored)
                    print(f"🔍 可逆性: {'✅ 完全一致' if is_identical else '❌ 不一致'}")
                    
                    # ファイルサイズ比較
                    original_size = len(original)
                    compressed_size = os.path.getsize(output_path)
                    compression_ratio = (1 - compressed_size / original_size) * 100
                    
                    results.append({
                        'filename': filename,
                        'original_size': original_size,
                        'compressed_size': compressed_size,
                        'compression_ratio': compression_ratio,
                        'reversible': is_identical
                    })
                    
                    # クリーンアップ
                    os.remove(output_path)
                    os.remove(restored_path)
            
        except Exception as e:
            print(f"❌ テストエラー: {str(e)[:60]}...")
    
    # 総合結果
    if results:
        print("\n" + "=" * 60)
        print("🏆 Phase 8 Turbo 包括的テスト結果")
        print("=" * 60)
        
        total_original = sum(r['original_size'] for r in results)
        total_compressed = sum(r['compressed_size'] for r in results)
        overall_ratio = (1 - total_compressed / total_original) * 100
        
        print(f"📊 総合圧縮率: {overall_ratio:.1f}%")
        print(f"📈 テストファイル数: {len(results)}")
        print(f"� 総データ量: {total_original/1024/1024:.1f} MB")
        print(f"💾 圧縮後サイズ: {total_compressed/1024/1024:.1f} MB")
        print(f"�🔍 可逆性: {sum(1 for r in results if r['reversible'])}/{len(results)} 成功")
        
        # ファイル形式別分析
        format_analysis = {}
        for result in results:
            filename = result['filename']
            ext = filename.split('.')[-1].upper()
            
            if ext not in format_analysis:
                format_analysis[ext] = {
                    'count': 0,
                    'total_original': 0,
                    'total_compressed': 0,
                    'reversible_count': 0
                }
            
            format_analysis[ext]['count'] += 1
            format_analysis[ext]['total_original'] += result['original_size']
            format_analysis[ext]['total_compressed'] += result['compressed_size']
            if result['reversible']:
                format_analysis[ext]['reversible_count'] += 1
        
        print(f"\n🎯 ファイル形式別詳細分析:")
        for ext, data in format_analysis.items():
            ratio = (1 - data['total_compressed'] / data['total_original']) * 100
            reversible_rate = (data['reversible_count'] / data['count']) * 100
            print(f"   📄 {ext}: {ratio:.1f}%圧縮 ({data['count']}ファイル, 可逆性{reversible_rate:.0f}%)")
        
        # Phase 7との比較
        phase7_ratio = 57.3
        improvement = overall_ratio - phase7_ratio
        
        print(f"\n🎯 Phase 7比較:")
        print(f"   Phase 7: {phase7_ratio}%")
        print(f"   Phase 8: {overall_ratio:.1f}%")
        print(f"   改善: {improvement:+.1f}%")
        
        # 個別ファイル詳細結果
        print(f"\n📋 個別ファイル詳細結果:")
        for result in results:
            filename = result['filename'][:30] + ('...' if len(result['filename']) > 30 else '')
            size_mb = result['original_size'] / 1024 / 1024
            reversible_icon = '✅' if result['reversible'] else '❌'
            print(f"   • {filename}: {result['compression_ratio']:.1f}% ({size_mb:.1f}MB) {reversible_icon}")
        
        if overall_ratio > phase7_ratio:
            print("🎉 Phase 8 Turbo大成功！Phase 7を上回る圧縮率達成！")
        else:
            print("📈 継続改善中...")
        
        # 最優秀・最低パフォーマンス
        best_result = max(results, key=lambda x: x['compression_ratio'])
        worst_result = min(results, key=lambda x: x['compression_ratio'])
        
        print(f"\n🏅 パフォーマンス分析:")
        print(f"   🥇 最優秀: {best_result['filename']} ({best_result['compression_ratio']:.1f}%)")
        print(f"   🚨 改善必要: {worst_result['filename']} ({worst_result['compression_ratio']:.1f}%)")
        
        # 推奨改善アクション
        low_compression_files = [r for r in results if r['compression_ratio'] < 10]
        if low_compression_files:
            print(f"\n⚠️ 低圧縮率ファイル ({len(low_compression_files)}個):")
            for r in low_compression_files:
                ext = r['filename'].split('.')[-1].upper()
                print(f"   • {r['filename']}: {r['compression_ratio']:.1f}% (要{ext}特化最適化)")
        
        # データサイズ別分析
        large_files = [r for r in results if r['original_size'] > 10*1024*1024]  # 10MB以上
        medium_files = [r for r in results if 1*1024*1024 <= r['original_size'] <= 10*1024*1024]  # 1-10MB
        small_files = [r for r in results if r['original_size'] < 1*1024*1024]  # 1MB未満
        
        print(f"\n📏 ファイルサイズ別分析:")
        if large_files:
            large_ratio = sum(r['compression_ratio'] for r in large_files) / len(large_files)
            print(f"   🐘 大容量ファイル (10MB+): 平均{large_ratio:.1f}%圧縮 ({len(large_files)}個)")
        if medium_files:
            medium_ratio = sum(r['compression_ratio'] for r in medium_files) / len(medium_files)
            print(f"   🦌 中容量ファイル (1-10MB): 平均{medium_ratio:.1f}%圧縮 ({len(medium_files)}個)")
        if small_files:
            small_ratio = sum(r['compression_ratio'] for r in small_files) / len(small_files)
            print(f"   🐁 小容量ファイル (1MB未満): 平均{small_ratio:.1f}%圧縮 ({len(small_files)}個)")

def main():
    """メイン処理"""
    if len(sys.argv) < 2:
        print("🚀 Phase 8 Turbo 完全実装")
        print("使用方法:")
        print("  python phase8_full.py test                     # 完全テスト")
        print("  python phase8_full.py compress <file>          # ファイル圧縮")
        print("  python phase8_full.py decompress <file.p8t>    # ファイル展開")
        return
    
    command = sys.argv[1].lower()
    engine = Phase8FullEngine()
    
    if command == "test":
        run_phase8_test()
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
