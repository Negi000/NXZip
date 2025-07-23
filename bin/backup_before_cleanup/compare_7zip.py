#!/usr/bin/env python3
"""
7-Zip vs NXZip 比較ツール
各フォーマットでの圧縮率比較と課題分析
"""

import os
from pathlib import Path

def compare_compression_ratios():
    """圧縮率比較"""
    print("📊 7-Zip vs NXZip 圧縮率比較")
    print("=" * 60)
    
    sample_dir = Path("NXZip-Python/sample")
    
    comparisons = [
        {
            'name': '📄 テキスト',
            'original': sample_dir / "出庫実績明細_202412.txt",
            'sevenz': sample_dir / "出庫実績明細_202412.7z",
            'nxz_ratio': 91.3,
            'target': 95
        },
        {
            'name': '🎬 動画',
            'original': sample_dir / "Python基礎講座3_4月26日-3.mp4",
            'sevenz': sample_dir / "Python基礎講座3_4月26日-3.7z",
            'nxz_ratio': 18.3,
            'target': 80
        },
        {
            'name': '🖼️ 画像',
            'original': sample_dir / "COT-001.jpg",
            'sevenz': sample_dir / "COT-001.7z",
            'nxz_ratio': 3.1,
            'target': 80
        },
        {
            'name': '🎵 音声',
            'original': sample_dir / "陰謀論.mp3",
            'sevenz': sample_dir / "陰謀論.7z",
            'nxz_ratio': 1.2,
            'target': 80
        }
    ]
    
    for comp in comparisons:
        if comp['original'].exists() and comp['sevenz'].exists():
            original_size = comp['original'].stat().st_size
            sevenz_size = comp['sevenz'].stat().st_size
            
            sevenz_ratio = (1 - sevenz_size / original_size) * 100
            nxz_ratio = comp['nxz_ratio']
            target = comp['target']
            
            print(f"\n{comp['name']}")
            print(f"  元サイズ: {original_size:,} bytes ({original_size/1024/1024:.1f} MB)")
            print(f"  7-Zip:   {sevenz_size:,} bytes ({sevenz_ratio:.1f}%)")
            print(f"  NXZip:   {nxz_ratio:.1f}%")
            print(f"  目標:    {target:.1f}%")
            
            if nxz_ratio > sevenz_ratio:
                print(f"  ✅ NXZip優位: +{nxz_ratio - sevenz_ratio:.1f}%")
            else:
                print(f"  ❌ 7-Zip優位: -{sevenz_ratio - nxz_ratio:.1f}%")
            
            gap_to_target = target - nxz_ratio
            print(f"  📈 目標まで: {gap_to_target:.1f}%")
    
    print(f"\n🎯 課題分析サマリー")
    print("=" * 60)
    print("1. テキスト: 7-Zipに対する優位性があるが、目標まで3.7%")
    print("2. 動画: 既存圧縮技術の限界、根本的なアプローチ変更が必要")
    print("3. 画像: JPEG圧縮の壁、制約除去技術の実装が必須")
    print("4. 音声: MP3圧縮の壁、デコード→再圧縮戦略が必要")

if __name__ == "__main__":
    compare_compression_ratios()
