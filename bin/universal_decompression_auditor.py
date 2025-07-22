#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 Universal Decompression Auditor - 汎用解凍監査システム
あらゆる圧縮形式の可逆性を検証

🎯 検証範囲:
- 全形式圧縮ファイル (.nxz)
- 複数解凍アルゴリズム試行
- 完全可逆性検証
- 偽装圧縮検出
"""

import os
import sys
import time
import zlib
import bz2
import lzma
from pathlib import Path
import hashlib
import struct

class UniversalDecompressionAuditor:
    """汎用解凍監査システム"""
    
    def __init__(self):
        pass
    
    def audit_all_compressed_files(self) -> dict:
        """全圧縮ファイルの可逆性監査"""
        print("🔍 Universal Decompression Audit")
        print("🎯 全圧縮ファイル可逆性検証")
        print("=" * 70)
        
        # 圧縮ファイル検索
        compressed_files = self._find_compressed_files()
        print(f"📦 発見圧縮ファイル: {len(compressed_files)}")
        
        if not compressed_files:
            print("⚠️ 圧縮ファイルが見つかりません")
            return {'success': False, 'error': 'No compressed files found'}
        
        audit_results = []
        
        for compressed_file in compressed_files:
            print(f"\n🧪 監査: {Path(compressed_file).name}")
            print("-" * 50)
            
            result = self._audit_single_file(compressed_file)
            audit_results.append(result)
            
            # 結果表示
            status = result.get('reversibility_status', 'UNKNOWN')
            if status == 'PERFECT':
                print(f"✅ 完全可逆: {result.get('compression_ratio', 0):.1f}%")
            elif status == 'PARTIAL':
                print(f"⚠️ 部分可逆: {result.get('compression_ratio', 0):.1f}%")
            elif status == 'FAILED':
                print(f"❌ 可逆失敗: {result.get('compression_ratio', 0):.1f}%")
            else:
                print(f"🔧 監査エラー: {result.get('error', 'Unknown')}")
        
        # 総合評価
        print("\n" + "=" * 70)
        print("🏆 総合可逆性監査結果")
        print("=" * 70)
        
        perfect_count = sum(1 for r in audit_results if r.get('reversibility_status') == 'PERFECT')
        partial_count = sum(1 for r in audit_results if r.get('reversibility_status') == 'PARTIAL')
        failed_count = sum(1 for r in audit_results if r.get('reversibility_status') == 'FAILED')
        error_count = len(audit_results) - perfect_count - partial_count - failed_count
        
        print(f"📊 監査統計:")
        print(f"   ✅ 完全可逆: {perfect_count}/{len(audit_results)}")
        print(f"   ⚠️ 部分可逆: {partial_count}/{len(audit_results)}")
        print(f"   ❌ 可逆失敗: {failed_count}/{len(audit_results)}")
        print(f"   🔧 エラー: {error_count}/{len(audit_results)}")
        
        if perfect_count == len(audit_results):
            print("\n🎉🎉🎉🎉 全ファイル完全可逆!")
            print("🏆 すべての圧縮が可逆性を保証!")
        elif perfect_count > 0:
            print(f"\n🎉 部分成功: {perfect_count}ファイルが完全可逆")
            print("🔧 一部のエンジンに可逆性問題あり")
        else:
            print("\n🚨 重大な問題: 完全可逆ファイルなし")
            print("❌ すべてのエンジンに可逆性問題あり")
        
        return {
            'success': True,
            'audit_results': audit_results,
            'statistics': {
                'total': len(audit_results),
                'perfect': perfect_count,
                'partial': partial_count,
                'failed': failed_count,
                'error': error_count
            }
        }
    
    def _find_compressed_files(self) -> list:
        """圧縮ファイル検索"""
        compressed_files = []
        
        # 検索ディレクトリ
        search_dirs = [
            r"c:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\sample",
            r"c:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\test-data",
            "."  # 現在のディレクトリ
        ]
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for root, dirs, files in os.walk(search_dir):
                    for file in files:
                        if file.endswith('.nxz'):
                            full_path = os.path.join(root, file)
                            compressed_files.append(full_path)
        
        return compressed_files
    
    def _audit_single_file(self, compressed_file: str) -> dict:
        """単一ファイル監査"""
        try:
            # 元ファイル推定
            original_file = self._find_original_file(compressed_file)
            
            if not original_file:
                return {
                    'compressed_file': compressed_file,
                    'reversibility_status': 'ERROR',
                    'error': 'Original file not found'
                }
            
            print(f"📄 元ファイル: {Path(original_file).name}")
            
            # 元ファイル読み込み
            with open(original_file, 'rb') as f:
                original_data = f.read()
            
            # 圧縮ファイル読み込み
            with open(compressed_file, 'rb') as f:
                compressed_data = f.read()
            
            # 解凍試行
            decompressed_data = self._attempt_universal_decompression(compressed_data)
            
            if decompressed_data is None:
                return {
                    'compressed_file': compressed_file,
                    'original_file': original_file,
                    'reversibility_status': 'FAILED',
                    'error': 'Decompression failed',
                    'compression_ratio': (1 - len(compressed_data) / len(original_data)) * 100
                }
            
            # 可逆性検証
            size_match = len(original_data) == len(decompressed_data)
            byte_match = original_data == decompressed_data
            
            original_hash = hashlib.sha256(original_data).hexdigest()
            decompressed_hash = hashlib.sha256(decompressed_data).hexdigest()
            hash_match = original_hash == decompressed_hash
            
            # 判定
            if size_match and byte_match and hash_match:
                status = 'PERFECT'
            elif size_match:
                status = 'PARTIAL'
            else:
                status = 'FAILED'
            
            compression_ratio = (1 - len(compressed_data) / len(original_data)) * 100
            
            print(f"📊 圧縮率: {compression_ratio:.1f}%")
            print(f"🔍 サイズ一致: {'✅' if size_match else '❌'}")
            print(f"🔍 バイト一致: {'✅' if byte_match else '❌'}")
            print(f"🔍 ハッシュ一致: {'✅' if hash_match else '❌'}")
            
            return {
                'compressed_file': compressed_file,
                'original_file': original_file,
                'reversibility_status': status,
                'compression_ratio': compression_ratio,
                'original_size': len(original_data),
                'compressed_size': len(compressed_data),
                'decompressed_size': len(decompressed_data),
                'size_match': size_match,
                'byte_match': byte_match,
                'hash_match': hash_match
            }
            
        except Exception as e:
            return {
                'compressed_file': compressed_file,
                'reversibility_status': 'ERROR',
                'error': str(e)
            }
    
    def _find_original_file(self, compressed_file: str) -> str:
        """元ファイル検索"""
        base_name = Path(compressed_file).stem
        dir_path = Path(compressed_file).parent
        
        # 可能な拡張子
        possible_extensions = ['.mp4', '.mp3', '.png', '.txt', '.wav', '.jpg', '.pdf']
        
        # 同じディレクトリで検索
        for ext in possible_extensions:
            possible_path = dir_path / f"{base_name}{ext}"
            if possible_path.exists():
                return str(possible_path)
        
        # 他のディレクトリでも検索
        search_dirs = [
            r"c:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\sample",
            r"c:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\test-data"
        ]
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for ext in possible_extensions:
                    possible_path = Path(search_dir) / f"{base_name}{ext}"
                    if possible_path.exists():
                        return str(possible_path)
        
        return None
    
    def _attempt_universal_decompression(self, compressed_data: bytes) -> bytes:
        """汎用解凍試行"""
        try:
            print("🔄 汎用解凍試行中...")
            
            # ヘッダーベース解凍
            if compressed_data.startswith(b'NXMP4_OPTIMAL_BALANCE_V1'):
                return self._decompress_optimal_balance(compressed_data)
            elif compressed_data.startswith(b'NXMP4_PERFECT_REVERSIBLE'):
                return self._decompress_perfect_reversible(compressed_data)
            elif compressed_data.startswith(b'NXMP4_VIDEO_BREAKTHROUGH'):
                return self._decompress_video_breakthrough(compressed_data)
            elif compressed_data.startswith(b'NEXUS_LIGHTNING_ULTRA'):
                return self._decompress_lightning_ultra(compressed_data)
            
            # 直接解凍試行
            algorithms = [
                ('LZMA', lzma.decompress),
                ('BZ2', bz2.decompress),
                ('ZLIB', zlib.decompress),
            ]
            
            for name, decompress_func in algorithms:
                try:
                    result = decompress_func(compressed_data)
                    print(f"✅ {name}直接解凍成功")
                    return result
                except:
                    continue
            
            # ヘッダーを除去して試行
            for header_size in [16, 20, 24, 32]:
                payload = compressed_data[header_size:]
                for name, decompress_func in algorithms:
                    try:
                        result = decompress_func(payload)
                        print(f"✅ {name}ヘッダー除去解凍成功 (ヘッダー: {header_size}bytes)")
                        return result
                    except:
                        continue
            
            print("❌ 全解凍方法失敗")
            return None
            
        except Exception as e:
            print(f"❌ 解凍エラー: {e}")
            return None
    
    def _decompress_optimal_balance(self, data: bytes) -> bytes:
        """最適バランス解凍"""
        try:
            import json
            
            # メタデータサイズ取得
            metadata_size = struct.unpack('<I', data[24:28])[0]
            metadata_compressed = data[28:28 + metadata_size]
            
            # メタデータ解凍
            metadata_json = zlib.decompress(metadata_compressed).decode('utf-8')
            metadata = json.loads(metadata_json)
            
            # コア部分解凍
            core_start = 28 + metadata_size
            compressed_core = data[core_start:]
            
            # 複数アルゴリズム試行
            for decompress_func in [lzma.decompress, bz2.decompress, zlib.decompress]:
                try:
                    return decompress_func(compressed_core)
                except:
                    continue
            
            return None
            
        except:
            return None
    
    def _decompress_perfect_reversible(self, data: bytes) -> bytes:
        """完全可逆解凍"""
        try:
            # 簡易実装 - ヘッダー除去後解凍試行
            payload = data[32:]  # ヘッダー除去
            
            for decompress_func in [lzma.decompress, bz2.decompress, zlib.decompress]:
                try:
                    return decompress_func(payload)
                except:
                    continue
            
            return None
            
        except:
            return None
    
    def _decompress_video_breakthrough(self, data: bytes) -> bytes:
        """動画突破解凍"""
        try:
            # ヘッダー除去
            if data.startswith(b'NXMP4_VIDEO_BREAKTHROUGH_SUCCESS'):
                payload = data[32:]
            else:
                payload = data[29:]  # その他の変種
            
            for decompress_func in [lzma.decompress, bz2.decompress, zlib.decompress]:
                try:
                    return decompress_func(payload)
                except:
                    continue
            
            return None
            
        except:
            return None
    
    def _decompress_lightning_ultra(self, data: bytes) -> bytes:
        """Lightning Ultra解凍"""
        try:
            # ヘッダー除去
            payload = data[20:]  # NEXUS_LIGHTNING_ULTRA
            
            for decompress_func in [lzma.decompress, bz2.decompress, zlib.decompress]:
                try:
                    return decompress_func(payload)
                except:
                    continue
            
            return None
            
        except:
            return None

def run_universal_audit():
    """汎用監査実行"""
    auditor = UniversalDecompressionAuditor()
    result = auditor.audit_all_compressed_files()
    
    if result['success']:
        stats = result['statistics']
        
        if stats['perfect'] == stats['total'] and stats['total'] > 0:
            print("\n🎉🎉🎉🎉 全エンジン可逆性確認!")
            print("🏆 すべての圧縮技術が信頼できます!")
        elif stats['perfect'] > 0:
            print(f"\n⚠️ 混在状況: {stats['perfect']}/{stats['total']}エンジンが可逆")
            print("🔧 一部エンジンに問題があります")
        else:
            print("\n🚨 深刻な問題: 可逆エンジンなし")
            print("❌ 全圧縮技術の見直しが必要")

def main():
    """メイン関数"""
    if len(sys.argv) < 2:
        print("🔍 Universal Decompression Auditor")
        print("使用方法:")
        print("  python universal_decompression_auditor.py audit    # 汎用可逆性監査")
        return
    
    command = sys.argv[1].lower()
    
    if command == "audit":
        run_universal_audit()
    else:
        print("❌ 無効なコマンドです")

if __name__ == "__main__":
    main()
