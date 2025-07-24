#!/usr/bin/env python3
"""ファイル比較スクリプト"""

import hashlib
import sys

def compare_files(file1, file2):
    # ファイル1読み込み
    try:
        with open(file1, 'rb') as f:
            data1 = f.read()
        hash1 = hashlib.sha256(data1).hexdigest()
        print(f"元ファイル       : {file1}")
        print(f"サイズ          : {len(data1):,} bytes")
        print(f"SHA256          : {hash1}")
        print()
    except Exception as e:
        print(f"❌ ファイル1読み込みエラー: {e}")
        return
    
    # ファイル2読み込み
    try:
        with open(file2, 'rb') as f:
            data2 = f.read()
        hash2 = hashlib.sha256(data2).hexdigest()
        print(f"復元ファイル      : {file2}")
        print(f"サイズ          : {len(data2):,} bytes")
        print(f"SHA256          : {hash2}")
        print()
    except Exception as e:
        print(f"❌ ファイル2読み込みエラー: {e}")
        return
    
    # 比較結果
    if hash1 == hash2:
        print("🎉✅ 完全一致確認！真の可逆性実現！")
        print("🎯 バイトレベルで100%同一のファイルです")
    else:
        print("❌ ハッシュ値が異なります")
        print(f"   サイズ差: {len(data2) - len(data1)} bytes")

if __name__ == "__main__":
    file1 = r"c:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\sample\COT-001.png"
    file2 = "COT-001_perfect_restored.png"
    compare_files(file1, file2)
