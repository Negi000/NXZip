#!/usr/bin/env python3
"""画像ファイル検証スクリプト"""

import sys
import struct
from pathlib import Path

def check_png_header(filepath):
    """PNG基本情報を確認"""
    try:
        with open(filepath, 'rb') as f:
            data = f.read(50)
        
        # PNG署名確認
        if data[:8] != b'\x89PNG\r\n\x1a\n':
            print(f"❌ {filepath}: PNG署名が正しくありません")
            return False
        
        # IHDR情報抽出
        if len(data) >= 25:
            width = struct.unpack('>I', data[16:20])[0]
            height = struct.unpack('>I', data[20:24])[0]
            bit_depth = data[24]
            color_type = data[25]
            
            print(f"✅ {filepath}: PNG形式確認")
            print(f"   サイズ: {width}x{height}")
            print(f"   ビット深度: {bit_depth}")
            print(f"   カラータイプ: {color_type}")
            return True
        else:
            print(f"❌ {filepath}: IHDRデータが不足")
            return False
            
    except Exception as e:
        print(f"❌ {filepath}: エラー {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("使用法: python check_image.py <image_file>")
        return
    
    filepath = sys.argv[1]
    
    if not Path(filepath).exists():
        print(f"❌ ファイルが見つかりません: {filepath}")
        return
    
    # ファイルサイズ確認
    file_size = Path(filepath).stat().st_size
    print(f"📏 ファイルサイズ: {file_size:,} bytes ({file_size/1024/1024:.1f}MB)")
    
    # PNG確認
    check_png_header(filepath)

if __name__ == "__main__":
    main()
