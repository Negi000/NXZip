#!/usr/bin/env python3
"""
超シンプル個別ファイルテスト - ログなし版
"""

import os
import hashlib
import sys
from pathlib import Path

# パスを追加
sys.path.append('NXZip-Python/sample')

def test_file(filename):
    """ファイルテスト - 最小ログ版"""
    filepath = Path("sample") / filename
    
    if not filepath.exists():
        print(f"{filename}: ❌ NOT FOUND")
        return False
    
    print(f"{filename}: ", end="")
    
    try:
        # 読み込み
        with open(filepath, 'rb') as f:
            data = f.read()
        
        original_size = len(data)
        original_hash = hashlib.sha256(data).hexdigest()
        
        # 圧縮
        from nexus_advanced_engine import NexusAdvancedCompressor
        compressor = NexusAdvancedCompressor(use_ai=False, max_recursion_level=0)
        compressed = compressor.compress(data)
        
        # 解凍
        decompressed = compressor.decompress(compressed)
        
        # 検証
        if len(decompressed) == original_size:
            restored_hash = hashlib.sha256(decompressed).hexdigest()
            if restored_hash == original_hash:
                ratio = len(compressed) / original_size
                print(f"✅ OK (ratio: {ratio:.3f})")
                
                # .nxz ファイル保存
                nxz_path = filepath.with_suffix('.nxz')
                with open(nxz_path, 'wb') as f:
                    f.write(compressed)
                
                # 復元ファイル保存
                restored_path = filepath.with_name(f"{filepath.stem}_restored{filepath.suffix}")
                with open(restored_path, 'wb') as f:
                    f.write(decompressed)
                
                return True
            else:
                print("❌ HASH MISMATCH")
                return False
        else:
            print("❌ SIZE MISMATCH")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

def main():
    """メイン実行"""
    print("=== クリーン個別ファイルテスト ===")
    
    test_files = [
        "test_small.txt",
        "element_test_small.bin", 
        "element_test_medium.bin",
        "COT-001.png"
    ]
    
    success = 0
    for filename in test_files:
        if test_file(filename):
            success += 1
    
    print(f"\n結果: {success}/{len(test_files)} 成功")

if __name__ == "__main__":
    main()
