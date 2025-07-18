#!/usr/bin/env python3
"""
NXZip CLI Adaptive - フォーマット特化型CLI
全フォーマットでトップを目指す統合ツール
"""

import argparse
import os
import sys
import time
from pathlib import Path

# プロジェクトパス追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nxzip.engine.nexus_adaptive import NEXUSAdaptive

def format_size(size_bytes):
    """サイズをMBで表示"""
    if size_bytes == 0:
        return "0 B"
    size_mb = size_bytes / (1024 * 1024)
    if size_mb < 1:
        return f"{size_bytes} B"
    else:
        return f"{size_mb:.2f} MB"

def compress_file(input_file, output_file=None):
    """フォーマット特化型圧縮"""
    print(f"🎯 NXZip NEXUS Adaptive - フォーマット特化型")
    print(f"📄 圧縮: {input_file}")
    
    # 入力ファイル確認
    if not os.path.exists(input_file):
        print(f"❌ エラー: ファイル '{input_file}' が見つかりません")
        return False
    
    # 出力ファイル名生成
    if output_file is None:
        output_file = str(input_file) + ".nxz"
    
    # データ読み込み
    print(f"📖 データ読み込み中...")
    with open(input_file, 'rb') as f:
        data = f.read()
    
    original_size = len(data)
    print(f"📊 元サイズ: {format_size(original_size)}")
    
    # NEXUS Adaptive初期化
    nexus = NEXUSAdaptive()
    
    # フォーマット検出
    format_type = nexus.detect_format(data)
    print(f"🔍 検出フォーマット: {format_type}")
    
    # 圧縮実行
    print(f"🎯 NEXUS Adaptive 圧縮中...")
    start_time = time.perf_counter()
    compressed = nexus.compress(data)
    compress_time = time.perf_counter() - start_time
    
    # 圧縮結果
    compressed_size = len(compressed)
    compression_ratio = (1 - compressed_size / original_size) * 100
    speed = (original_size / 1024 / 1024) / compress_time
    
    # 圧縮データ保存
    with open(output_file, 'wb') as f:
        f.write(compressed)
    
    print(f"✅ 圧縮完了!")
    print(f"   📈 圧縮率: {compression_ratio:.2f}%")
    print(f"   ⚡ 圧縮速度: {speed:.2f} MB/s")
    print(f"   ⏱️ 時間: {compress_time:.2f}秒")
    print(f"   💾 出力: {output_file}")
    print(f"   📊 サイズ: {format_size(compressed_size)}")
    
    # 段階的目標判定
    if compression_ratio >= 50 and speed >= 100:
        print(f"🎯 第1段階目標達成! 圧縮率50%+圧縮速度100MB/s")
    elif compression_ratio >= 50:
        print(f"⚠️ 圧縮率目標達成、速度改善が必要")
    else:
        print(f"⚠️ さらなる改善が必要")
    
    return True

def decompress_file(input_file, output_file=None):
    """フォーマット特化型展開"""
    print(f"🔄 NXZip NEXUS Adaptive - 展開")
    print(f"📄 展開: {input_file}")
    
    # 入力ファイル確認
    if not os.path.exists(input_file):
        print(f"❌ エラー: ファイル '{input_file}' が見つかりません")
        return False
    
    # 出力ファイル名生成
    if output_file is None:
        if input_file.endswith('.nxz'):
            output_file = input_file[:-4] + '_restored'
        else:
            output_file = input_file + "_restored"
    
    # データ読み込み
    print(f"📖 データ読み込み中...")
    with open(input_file, 'rb') as f:
        compressed_data = f.read()
    
    print(f"📊 圧縮サイズ: {format_size(len(compressed_data))}")
    
    # NEXUS Adaptive初期化
    nexus = NEXUSAdaptive()
    
    # 展開実行
    print(f"🔄 NEXUS Adaptive 展開中...")
    start_time = time.perf_counter()
    try:
        decompressed = nexus.decompress(compressed_data)
        decomp_time = time.perf_counter() - start_time
        
        # 展開結果
        decompressed_size = len(decompressed)
        speed = (decompressed_size / 1024 / 1024) / decomp_time
        
        # 展開データ保存
        with open(output_file, 'wb') as f:
            f.write(decompressed)
        
        print(f"✅ 展開完了!")
        print(f"   ⚡ 展開速度: {speed:.2f} MB/s")
        print(f"   ⏱️ 時間: {decomp_time:.2f}秒")
        print(f"   💾 出力: {output_file}")
        print(f"   📊 サイズ: {format_size(decompressed_size)}")
        
        # 速度目標判定
        if speed >= 200:
            print(f"🎯 展開速度目標達成! 200MB/s以上")
        
        return True
    except Exception as e:
        print(f"❌ 展開エラー: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="NXZip NEXUS Adaptive - フォーマット特化型")
    parser.add_argument('command', choices=['compress', 'decompress'],
                        help='実行するコマンド')
    parser.add_argument('input_file', 
                        help='入力ファイル')
    parser.add_argument('output_file', nargs='?',
                        help='出力ファイル（省略可）')
    
    args = parser.parse_args()
    
    if args.command == 'compress':
        compress_file(args.input_file, args.output_file)
    elif args.command == 'decompress':
        decompress_file(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
