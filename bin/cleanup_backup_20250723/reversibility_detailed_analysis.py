#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 量子圧縮可逆性詳細分析
サイズは一致しているがハッシュが不一致の原因を特定
"""

import os
import hashlib
import struct

def analyze_compression_reversibility():
    """圧縮可逆性の詳細分析"""
    
    print("🔍 量子圧縮可逆性詳細分析")
    print("=" * 60)
    
    # ファイルサイズ比較
    original_file = "NXZip-Python/sample/COT-001.png"
    restored_file = "test-data/COT-001_restored_reversible.png"
    
    original_size = os.path.getsize(original_file)
    restored_size = os.path.getsize(restored_file)
    
    print(f"📊 ファイルサイズ比較:")
    print(f"   元ファイル: {original_size:,} bytes")
    print(f"   復元ファイル: {restored_size:,} bytes")
    print(f"   サイズ一致: {'はい' if original_size == restored_size else 'いいえ'}")
    
    # ハッシュ比較
    with open(original_file, 'rb') as f:
        original_data = f.read()
    original_hash = hashlib.sha256(original_data).hexdigest()
    
    with open(restored_file, 'rb') as f:
        restored_data = f.read()
    restored_hash = hashlib.sha256(restored_data).hexdigest()
    
    print(f"\\n🔑 ハッシュ比較:")
    print(f"   元ハッシュ: {original_hash}")
    print(f"   復元ハッシュ: {restored_hash}")
    print(f"   ハッシュ一致: {'はい' if original_hash == restored_hash else 'いいえ'}")
    
    # バイト単位の差分分析
    differences = []
    for i in range(min(len(original_data), len(restored_data))):
        if original_data[i] != restored_data[i]:
            differences.append((i, original_data[i], restored_data[i]))
    
    print(f"\\n🔍 バイト差分分析:")
    print(f"   総バイト数: {len(original_data)}")
    print(f"   差分箇所数: {len(differences)}")
    print(f"   一致率: {((len(original_data) - len(differences)) / len(original_data) * 100):.6f}%")
    
    if differences:
        print(f"\\n🎯 最初の10個の差分:")
        for i, (pos, orig, rest) in enumerate(differences[:10]):
            print(f"   位置 {pos}: {orig:02x} → {rest:02x} (差分: {abs(orig - rest)})")
    
    # 差分パターン分析
    if differences:
        diff_values = [abs(orig - rest) for _, orig, rest in differences]
        
        print(f"\\n📊 差分パターン:")
        print(f"   最小差分: {min(diff_values)}")
        print(f"   最大差分: {max(diff_values)}")
        print(f"   平均差分: {sum(diff_values) / len(diff_values):.2f}")
        
        # 差分分布
        diff_counts = {}
        for diff in diff_values:
            diff_counts[diff] = diff_counts.get(diff, 0) + 1
        
        print(f"   差分分布 (上位5個):")
        for diff, count in sorted(diff_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"      差分{diff}: {count}回 ({count/len(differences)*100:.1f}%)")
    
    # 特定領域の分析
    chunk_size = 1024
    chunk_differences = []
    
    for i in range(0, len(original_data), chunk_size):
        chunk_orig = original_data[i:i+chunk_size]
        chunk_rest = restored_data[i:i+chunk_size]
        
        chunk_diffs = sum(1 for j in range(min(len(chunk_orig), len(chunk_rest))) 
                         if chunk_orig[j] != chunk_rest[j])
        
        if chunk_diffs > 0:
            chunk_differences.append((i, chunk_diffs))
    
    print(f"\\n🗂️ チャンク分析 ({chunk_size}バイト単位):")
    print(f"   総チャンク数: {(len(original_data) + chunk_size - 1) // chunk_size}")
    print(f"   差分有りチャンク: {len(chunk_differences)}")
    
    if chunk_differences:
        print(f"   差分上位5チャンク:")
        for i, (pos, diffs) in enumerate(sorted(chunk_differences, 
                                              key=lambda x: x[1], reverse=True)[:5]):
            print(f"      位置 {pos}: {diffs}個の差分")

def main():
    analyze_compression_reversibility()

if __name__ == "__main__":
    main()
