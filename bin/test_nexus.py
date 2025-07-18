#!/usr/bin/env python3
"""
NXZip NEXUS 統合テストツール
AV1/SRLA/AVIF制約除去技術による次世代圧縮のテスト
"""

import os
import sys
import time
from pathlib import Path

# プロジェクトパス追加
current_dir = Path(__file__).parent
project_root = current_dir.parent / "NXZip-Python"
sys.path.insert(0, str(project_root))

from nxzip.engine.nexus_unified import NEXUSUnified

def format_size(size_bytes):
    """サイズをMBで表示"""
    if size_bytes == 0:
        return "0 B"
    size_mb = size_bytes / (1024 * 1024)
    if size_mb < 1:
        return f"{size_bytes} B"
    else:
        return f"{size_mb:.2f} MB"

def test_nexus_unified():
    """NEXUS統合テスト"""
    print("🔥 NXZip NEXUS 統合テスト")
    print("=" * 70)
    print("🎯 目標: 80%圧縮率(テキスト95%), 100MB/s圧縮, 200MB/s展開")
    print("=" * 70)
    
    # テストファイル
    test_files = [
        # テキストファイル
        r"C:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\sample\出庫実績明細_202412.txt",
        # 動画ファイル
        r"C:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\sample\Python基礎講座3_4月26日-3.mp4",
        # 画像ファイル
        r"C:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\sample\COT-001.jpg",
        # 音声ファイル
        r"C:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\sample\陰謀論.mp3",
    ]
    
    nexus = NEXUSUnified()
    
    total_original = 0
    total_compressed = 0
    total_comp_time = 0
    total_decomp_time = 0
    
    for file_path in test_files:
        path = Path(file_path)
        if not path.exists():
            print(f"⚠️  スキップ: {path.name}")
            continue
            
        print(f"\n📄 テスト: {path.name}")
        print(f"   パス: {path}")
        
        with open(path, 'rb') as f:
            data = f.read()
        
        original_size = len(data)
        print(f"   元サイズ: {format_size(original_size)}")
        
        # 圧縮
        print(f"   🚀 圧縮中...")
        start = time.perf_counter()
        compressed = nexus.compress(data)
        compress_time = time.perf_counter() - start
        
        # 展開
        print(f"   💨 展開中...")
        start = time.perf_counter()
        decompressed = nexus.decompress(compressed)
        decomp_time = time.perf_counter() - start
        
        # 結果
        compressed_size = len(compressed)
        ratio = (1 - compressed_size / original_size) * 100
        comp_speed = (original_size / 1024 / 1024) / compress_time
        decomp_speed = (original_size / 1024 / 1024) / decomp_time
        correct = data == decompressed
        
        print(f"   📊 圧縮率: {ratio:.1f}%")
        print(f"   ⚡ 圧縮速度: {comp_speed:.1f} MB/s")
        print(f"   💨 展開速度: {decomp_speed:.1f} MB/s")
        print(f"   ✅ 正確性: {'OK' if correct else 'NG'}")
        print(f"   💾 圧縮後: {format_size(compressed_size)}")
        
        # 統計更新
        total_original += original_size
        total_compressed += compressed_size
        total_comp_time += compress_time
        total_decomp_time += decomp_time
        
        # 目標達成チェック
        target_ratio = 95 if path.suffix.lower() == '.txt' else 80
        comp_ok = ratio >= target_ratio
        speed_ok = comp_speed >= 100 and decomp_speed >= 200
        
        print(f"   🎯 目標達成: 圧縮率{'✅' if comp_ok else '❌'} 速度{'✅' if speed_ok else '❌'}")
    
    # 全体統計
    print(f"\n{'='*70}")
    print(f"🏆 統合結果")
    print(f"{'='*70}")
    
    overall_ratio = (1 - total_compressed / total_original) * 100
    overall_comp_speed = (total_original / 1024 / 1024) / total_comp_time
    overall_decomp_speed = (total_original / 1024 / 1024) / total_decomp_time
    
    print(f"📊 全体圧縮率: {overall_ratio:.1f}%")
    print(f"⚡ 全体圧縮速度: {overall_comp_speed:.1f} MB/s")
    print(f"💨 全体展開速度: {overall_decomp_speed:.1f} MB/s")
    print(f"💾 元サイズ: {format_size(total_original)}")
    print(f"💾 圧縮後: {format_size(total_compressed)}")
    
    print(f"\n🎯 目標達成状況:")
    print(f"   圧縮率: {'✅' if overall_ratio >= 80 else '❌'} (目標: 80%)")
    print(f"   圧縮速度: {'✅' if overall_comp_speed >= 100 else '❌'} (目標: 100MB/s)")
    print(f"   展開速度: {'✅' if overall_decomp_speed >= 200 else '❌'} (目標: 200MB/s)")

def main():
    """メイン処理"""
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_nexus_unified()
    else:
        print("使用法: python test_nexus.py test")

if __name__ == "__main__":
    main()
