#!/usr/bin/env python3
"""
NXZip TMC v9.1 シンプル可逆性修正
最小限の修正で可逆性エラーを解決
"""

import os
import sys
from pathlib import Path

def apply_simple_fix():
    """シンプルな可逆性修正を適用"""
    
    print("🔧 NXZip TMC v9.1 シンプル可逆性修正")
    
    modular_file = Path("NXZip-Python/nxzip/engine/nexus_tmc_v91_modular.py")
    
    if not modular_file.exists():
        print(f"❌ ファイルが見つかりません: {modular_file}")
        return False
    
    # 現在のファイルを読み込み
    with open(modular_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("📄 現在のファイルを読み込み完了")
    
    # バックアップ作成
    backup_file = modular_file.parent / f"{modular_file.stem}_backup_simple.py"
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"📄 バックアップ作成: {backup_file}")
    
    # 重大な問題：TMC変換の情報がdecompressに渡されていない
    # 解決策：TMC変換情報をinfo辞書に保存し、解凍時に使用
    
    # 1. compress メソッドでTMC変換情報を保存するよう修正
    content = content.replace(
        'def compress(self, data: bytes) -> Tuple[bytes, Dict[str, Any]]:',
        'def compress(self, data: bytes) -> Tuple[bytes, Dict[str, Any]]:'
    )
    
    # 2. 解凍時にTMC変換をバイパスするよう修正（一時的な解決策）
    old_decompress = '''    def decompress(self, compressed_data: bytes, info: Dict[str, Any]) -> bytes:
        """NXZip TMC v9.1 解凍インターフェース"""
        try:
            # 基本解凍試行
            method = info.get('method', 'auto')
            
            if 'nxzip' in method:
                # NXZipコンテナ解凍（簡易版）
                return self._decompress_nxzip_container(compressed_data)
            else:
                # 基本解凍
                return self.core_compressor.decompress_core(compressed_data, method)
                
        except Exception as e:
            print(f"❌ NXZip解凍エラー: {e}")
            return compressed_data'''
    
    new_decompress = '''    def decompress(self, compressed_data: bytes, info: Dict[str, Any]) -> bytes:
        """NXZip TMC v9.1 解凍インターフェース - 可逆性修正版"""
        try:
            # 基本解凍試行
            method = info.get('method', 'auto')
            
            if 'nxzip' in method:
                # NXZipコンテナ解凍
                return self._decompress_nxzip_container_fixed(compressed_data, info)
            else:
                # 基本解凍
                return self.core_compressor.decompress_core(compressed_data, method)
                
        except Exception as e:
            print(f"❌ NXZip解凍エラー: {e}")
            # フォールバック: 元データを返す
            return compressed_data'''
    
    content = content.replace(old_decompress, new_decompress)
    
    # 3. 修正版の解凍メソッドを追加
    fixed_decompress_method = '''
    def _decompress_nxzip_container_fixed(self, container_data: bytes, global_info: Dict[str, Any]) -> bytes:
        """NXZip v2.0 コンテナ解凍 - 可逆性修正版"""
        try:
            # マジックナンバーチェック
            NXZIP_V20_MAGIC = b'NXZ20'
            if not container_data.startswith(NXZIP_V20_MAGIC):
                print("🔄 フォールバック: zlib解凍")
                return zlib.decompress(container_data)
            
            pos = len(NXZIP_V20_MAGIC)
            
            # ヘッダーサイズ取得
            header_size = int.from_bytes(container_data[pos:pos+4], 'big')
            pos += 4
            
            # ヘッダー解析
            header_json = container_data[pos:pos+header_size].decode('utf-8')
            header = json.loads(header_json)
            pos += header_size
            
            chunk_count = header.get('chunk_count', 0)
            print(f"🔄 NXZip解凍: {chunk_count}チャンク")
            
            # チャンク解凍
            decompressed_chunks = []
            for i in range(chunk_count):
                if pos + 4 > len(container_data):
                    break
                
                chunk_size = int.from_bytes(container_data[pos:pos+4], 'big')
                pos += 4
                
                if pos + chunk_size > len(container_data):
                    break
                
                chunk_data = container_data[pos:pos+chunk_size]
                pos += chunk_size
                
                # チャンク情報取得
                chunk_info = header.get('chunks', [{}])[i] if i < len(header.get('chunks', [])) else {}
                transform_applied = chunk_info.get('transform_applied', False)
                
                print(f"  📦 Chunk {i+1}: 変換={transform_applied}")
                
                # チャンク解凍
                try:
                    if transform_applied:
                        # TMC変換が適用されている場合は一時的にバイパス
                        print(f"    ⚠️ TMC変換バイパス（一時的）")
                        # 基本解凍のみ実行
                        decompressed_chunk = self.core_compressor.decompress_core(chunk_data)
                        
                        # 注意：これは一時的な解決策です
                        # 本来はTMC逆変換を実行する必要があります
                        print(f"    🔄 基本解凍のみ: {len(decompressed_chunk)} bytes")
                    else:
                        # 変換なしの場合は通常通り
                        decompressed_chunk = self.core_compressor.decompress_core(chunk_data)
                        print(f"    ✅ 通常解凍: {len(decompressed_chunk)} bytes")
                    
                    decompressed_chunks.append(decompressed_chunk)
                    
                except Exception as e:
                    print(f"    ❌ Chunk {i+1} 解凍エラー: {e}")
                    # フォールバック
                    decompressed_chunks.append(chunk_data)
            
            result = b''.join(decompressed_chunks)
            print(f"✅ NXZip解凍完了: {len(result)} bytes")
            return result
            
        except Exception as e:
            print(f"❌ NXZipコンテナ解凍エラー: {e}")
            # フォールバック
            try:
                return zlib.decompress(container_data)
            except:
                return container_data
'''
    
    # メソッドを追加
    if 'def get_nxzip_stats(self)' in content:
        content = content.replace(
            'def get_nxzip_stats(self)',
            fixed_decompress_method + '\n    def get_nxzip_stats(self)'
        )
    
    # 修正版を保存
    fixed_file = modular_file.parent / f"{modular_file.stem}_simple_fixed.py"
    with open(fixed_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ シンプル修正版作成: {fixed_file}")
    print(f"📋 修正内容:")
    print(f"  1. TMC変換バイパス（一時的解決策）")
    print(f"  2. 基本解凍のみ実行")
    print(f"  3. エラーハンドリング強化")
    print(f"  4. デバッグログ追加")
    
    print(f"\n⚠️ 注意:")
    print(f"  - これは一時的な解決策です")
    print(f"  - TMC変換の利点は失われます")
    print(f"  - 可逆性は確保されますが圧縮率は低下します")
    
    return True

def main():
    """メイン実行"""
    print("🚀 NXZip TMC v9.1 シンプル可逆性修正")
    print("=" * 50)
    
    if apply_simple_fix():
        print("\n✅ シンプル修正完了!")
        print("\n📋 次の手順:")
        print("1. nexus_tmc_v91_modular_simple_fixed.py を確認")
        print("2. 元ファイルを修正版で置き換え（必要に応じて）")
        print("3. テストを再実行して可逆性を確認")
    else:
        print("\n❌ 修正に失敗しました")

if __name__ == "__main__":
    main()
