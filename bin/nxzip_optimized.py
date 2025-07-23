#!/usr/bin/env python3
"""
NXZip Optimized Engine Selector
最適化エンジン選択器 - 自動最適エンジン選択
"""

import os
import sys
from pathlib import Path

def select_optimal_engine(file_path):
    """ファイル形式に基づく最適エンジン選択"""
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        return 'nxzip_smart_image_compressor.py'
    elif file_ext in ['.wav', '.mp3', '.flac']:
        return 'nexus_lightning_fast.py'  # 音声超高速エンジン
    elif file_ext in ['.mp4', '.avi', '.mkv']:
        return 'nexus_phase8_turbo.py'  # 動画AI強化エンジン
    else:
        return 'nxzip_ultra_fast_binary_collapse.py'  # 汎用

def main():
    if len(sys.argv) != 2:
        print("使用法: python nxzip_optimized.py <ファイルパス>")
        print("\n🎯 NXZip 最適化エンジン選択器")
        print("📋 対応:")
        print("  🖼️ 画像: Smart Image Compressor")
        print("  🎵 音声: Lightning Fast Engine (79.1%/100%)")
        print("  🎬 動画: Phase8 Turbo Engine (40.2%)")
        print("  📄 その他: Ultra Fast Binary Collapse")
        sys.exit(1)
    
    file_path = sys.argv[1]
    engine = select_optimal_engine(file_path)
    
    print(f"🎯 最適エンジン選択: {engine}")
    print(f"📁 対象ファイル: {Path(file_path).name}")
    
    os.system(f'python {engine} "{file_path}"')

if __name__ == "__main__":
    main()
