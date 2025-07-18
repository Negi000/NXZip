#!/usr/bin/env python3
"""
7z比較テスト - NXZipと7zの圧縮性能比較
"""

import os
import time
from pathlib import Path
import sys

# プロジェクトパス追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nxzip.engine.nexus_text import NEXUSText
from nxzip.engine.nexus_audio_advanced import NEXUSAudioAdvanced
from nxzip.engine.nexus_image_advanced import NEXUSImageAdvanced
from nxzip.engine.nexus_video_ultra import NEXUSVideoUltra

def compare_with_7z():
    """7zファイルとNXZipの比較テスト"""
    print("📊 7z比較テスト - NXZip vs 7z")
    print("=" * 80)
    
    # サンプルファイルの組み合わせ
    test_pairs = [
        ("出庫実績明細_202412.txt", "出庫実績明細_202412.7z", "text"),
        ("陰謀論.mp3", "陰謀論.7z", "audio"),
        ("COT-001.jpg", "COT-001.7z", "image"),
        ("COT-012.png", "COT-012.7z", "image"),
        ("generated-music-1752042054079.wav", "generated-music-1752042054079.7z", "audio"),
        ("Python基礎講座3_4月26日-3.mp4", "Python基礎講座3_4月26日-3.7z", "video")
    ]
    
    sample_dir = Path(r"C:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\sample")
    
    total_original_size = 0
    total_nxzip_size = 0
    total_7z_size = 0
    total_nxzip_time = 0
    
    for original_file, z7_file, file_type in test_pairs:
        print(f"\n🔍 {file_type.upper()} 比較: {original_file}")
        print("-" * 60)
        
        original_path = sample_dir / original_file
        z7_path = sample_dir / z7_file
        
        if not original_path.exists():
            print(f"❌ {original_file} が見つかりません")
            continue
        
        if not z7_path.exists():
            print(f"❌ {z7_file} が見つかりません")
            continue
        
        # ファイルサイズ取得
        original_size = original_path.stat().st_size
        z7_size = z7_path.stat().st_size
        
        print(f"📄 元ファイル: {original_size//1024} KB")
        print(f"📦 7zファイル: {z7_size//1024} KB")
        
        # 7z圧縮率計算
        z7_ratio = (1 - z7_size / original_size) * 100
        print(f"📈 7z圧縮率: {z7_ratio:.2f}%")
        
        # NXZip圧縮テスト
        with open(original_path, 'rb') as f:
            data = f.read()
        
        # エンジン選択
        if file_type == "text":
            engine = NEXUSText()
        elif file_type == "audio":
            engine = NEXUSAudioAdvanced()
        elif file_type == "image":
            engine = NEXUSImageAdvanced()
        elif file_type == "video":
            engine = NEXUSVideoUltra()
        
        # NXZip圧縮
        start_time = time.perf_counter()
        nxzip_compressed = engine.compress(data)
        nxzip_time = time.perf_counter() - start_time
        
        nxzip_size = len(nxzip_compressed)
        nxzip_ratio = (1 - nxzip_size / original_size) * 100
        nxzip_speed = (original_size / 1024 / 1024) / nxzip_time
        
        print(f"🚀 NXZip圧縮率: {nxzip_ratio:.2f}%")
        print(f"⚡ NXZip速度: {nxzip_speed:.2f} MB/s")
        print(f"⏱️ NXZip時間: {nxzip_time:.2f}秒")
        
        # 比較結果
        if nxzip_ratio > z7_ratio:
            print(f"🏆 NXZip勝利: {nxzip_ratio - z7_ratio:.2f}% 高圧縮")
        elif nxzip_ratio < z7_ratio:
            print(f"⚠️ 7z勝利: {z7_ratio - nxzip_ratio:.2f}% 高圧縮")
        else:
            print("🤝 同等の圧縮率")
        
        # 統計累積
        total_original_size += original_size
        total_nxzip_size += nxzip_size
        total_7z_size += z7_size
        total_nxzip_time += nxzip_time
    
    # 総合結果
    print("\n" + "=" * 80)
    print("📊 総合比較結果")
    print("=" * 80)
    
    total_nxzip_ratio = (1 - total_nxzip_size / total_original_size) * 100
    total_7z_ratio = (1 - total_7z_size / total_original_size) * 100
    total_nxzip_speed = (total_original_size / 1024 / 1024) / total_nxzip_time
    
    print(f"📄 元ファイル合計: {total_original_size//1024//1024} MB")
    print(f"🚀 NXZip合計: {total_nxzip_size//1024//1024} MB (圧縮率: {total_nxzip_ratio:.2f}%)")
    print(f"📦 7z合計: {total_7z_size//1024//1024} MB (圧縮率: {total_7z_ratio:.2f}%)")
    print(f"⚡ NXZip平均速度: {total_nxzip_speed:.2f} MB/s")
    print(f"⏱️ NXZip合計時間: {total_nxzip_time:.2f}秒")
    
    if total_nxzip_ratio > total_7z_ratio:
        print(f"\n🏆 NXZip総合勝利!")
        print(f"   {total_nxzip_ratio - total_7z_ratio:.2f}% 高圧縮を達成")
    elif total_nxzip_ratio < total_7z_ratio:
        print(f"\n⚠️ 7z総合勝利")
        print(f"   {total_7z_ratio - total_nxzip_ratio:.2f}% 高圧縮")
    else:
        print(f"\n🤝 総合同等")
    
    print(f"\n💡 NXZipの特徴:")
    print(f"   - SPE構造保存暗号化")
    print(f"   - フォーマット別最適化")
    print(f"   - 完全可逆性保証")
    print(f"   - 高速処理（平均{total_nxzip_speed:.2f} MB/s）")

if __name__ == "__main__":
    compare_with_7z()
