#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 量子圧縮データ構造解析
圧縮されたファイルの内部構造を詳細分析
"""

import os
import struct

def analyze_quantum_file(filepath):
    """量子圧縮ファイルの構造解析"""
    
    print(f"🔍 量子圧縮ファイル解析: {filepath}")
    print("=" * 60)
    
    if not os.path.exists(filepath):
        print("❌ ファイルが見つかりません")
        return
    
    with open(filepath, 'rb') as f:
        data = f.read()
    
    print(f"📊 ファイルサイズ: {len(data):,} bytes")
    
    # ヘッダー解析
    print(f"\n🏷️ ヘッダー解析:")
    if data.startswith(b'NXQNT_PNG_V1'):
        print("   形式: NXQNT_PNG_V1")
        header_size = 12
    elif data.startswith(b'NXQNT_JPEG_V1'):
        print("   形式: NXQNT_JPEG_V1") 
        header_size = 13
    else:
        print(f"   不明な形式: {data[:20]}")
        return
    
    # 先頭バイトの詳細表示
    print(f"\n🔍 先頭100バイト:")
    for i in range(0, min(100, len(data)), 16):
        hex_part = ' '.join(f'{b:02x}' for b in data[i:i+16])
        ascii_part = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in data[i:i+16])
        print(f"   {i:04x}: {hex_part:<48} {ascii_part}")
    
    # メタデータ位置の解析
    if len(data) > header_size + 16:
        print(f"\n📋 メタデータ解析:")
        metadata_start = header_size
        
        # ハッシュ部分（16バイト）
        hash_data = data[metadata_start:metadata_start + 16]
        print(f"   ハッシュ (16bytes): {hash_data.hex()}")
        
        # 量子ヘッダー部分
        quantum_start = metadata_start + 16
        if len(data) > quantum_start + 6:
            quantum_phase_data = data[quantum_start:quantum_start + 4]
            pairs_count_data = data[quantum_start + 4:quantum_start + 6]
            
            try:
                quantum_phase = struct.unpack('>f', quantum_phase_data)[0]
                pairs_count = struct.unpack('>H', pairs_count_data)[0]
                print(f"   量子位相: {quantum_phase}")
                print(f"   ペア数: {pairs_count}")
            except:
                print(f"   量子ヘッダー解析失敗")
            
            # LZMA部分の開始
            lzma_start = quantum_start + 6
            lzma_data = data[lzma_start:]
            print(f"   LZMA開始位置: {lzma_start}")
            print(f"   LZMAデータサイズ: {len(lzma_data):,} bytes")
            
            if len(lzma_data) > 0:
                lzma_header = lzma_data[:10]
                print(f"   LZMAヘッダー: {lzma_header.hex()}")
                
                # LZMA形式確認
                if lzma_data.startswith(b'\xfd7zXZ'):
                    print("   ✅ 正常なLZMAヘッダー")
                else:
                    print("   ❌ 不正なLZMAヘッダー")
                    
                    # LZMA解凍テスト
                    try:
                        import lzma
                        decompressed = lzma.decompress(lzma_data)
                        print(f"   ✅ LZMA解凍成功: {len(decompressed):,} bytes")
                    except Exception as e:
                        print(f"   ❌ LZMA解凍失敗: {str(e)}")

def main():
    quantum_file = "NXZip-Python/sample/COT-001.nxz"
    analyze_quantum_file(quantum_file)

if __name__ == "__main__":
    main()
