#!/usr/bin/env python3
"""
個別ファイルテスト - クリーン版
各ファイルを個別に .nxz 圧縮・解凍テスト
"""

import os
import hashlib
import time
from pathlib import Path

# サンプルフォルダのパス
SAMPLE_FOLDER = Path("sample")

def get_file_hash(filepath):
    """ファイルのSHA256ハッシュを計算"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def test_single_file(filepath):
    """単一ファイルのNXZ圧縮・解凍テスト"""
    print(f"\n📁 Testing: {filepath.name}")
    
    try:
        # 元ファイル情報
        original_size = filepath.stat().st_size
        original_hash = get_file_hash(filepath)
        print(f"   Original: {original_size:,} bytes")
        
        # 圧縮実行
        start_time = time.time()
        
        # ここでNEXUS圧縮を実行
        import sys
        sys.path.append('NXZip-Python/sample')
        from nexus_advanced_engine import NexusAdvancedCompressor
        compressor = NexusAdvancedCompressor(use_ai=True, max_recursion_level=0)
        
        with open(filepath, 'rb') as f:
            data = f.read()
        
        compressed = compressor.compress(data)
        compress_time = time.time() - start_time
        
        # .nxz ファイル保存
        nxz_path = filepath.with_suffix(filepath.suffix + '.nxz')
        with open(nxz_path, 'wb') as f:
            f.write(compressed)
        
        compressed_size = len(compressed)
        ratio = compressed_size / original_size
        
        print(f"   Compressed: {compressed_size:,} bytes (ratio: {ratio:.4f})")
        print(f"   💾 Saved: {nxz_path.name}")
        
        # 解凍テスト
        start_time = time.time()
        decompressed = compressor.decompress(compressed)
        decompress_time = time.time() - start_time
        
        # 復元ファイル保存
        restored_path = filepath.with_name(f"{filepath.stem}_restored{filepath.suffix}")
        with open(restored_path, 'wb') as f:
            f.write(decompressed)
        
        # 検証
        if len(decompressed) == original_size:
            restored_hash = hashlib.sha256(decompressed).hexdigest()
            if restored_hash == original_hash:
                print(f"   🎉 REVERSIBLE: ✅ YES")
                print(f"   📤 Restored: {restored_path.name}")
                return True
            else:
                print(f"   ❌ REVERSIBLE: NO (hash mismatch)")
                print(f"      Expected: {original_hash}")
                print(f"      Got:      {restored_hash}")
                return False
        else:
            print(f"   ❌ REVERSIBLE: NO (size mismatch)")
            print(f"      Expected: {original_size:,} bytes")
            print(f"      Got:      {len(decompressed):,} bytes")
            return False
            
    except Exception as e:
        print(f"   ❌ ERROR: {str(e)}")
        return False

def main():
    """メイン実行"""
    print("=== 個別ファイルテスト (クリーン版) ===")
    
    # サンプルファイル取得
    if not SAMPLE_FOLDER.exists():
        print(f"❌ Sample folder not found: {SAMPLE_FOLDER}")
        return
    
    # テスト対象ファイル
    test_files = [
        "test_small.txt",
        "element_test_small.bin", 
        "element_test_medium.bin",
        "COT-001.png"
    ]
    
    results = []
    
    for filename in test_files:
        filepath = SAMPLE_FOLDER / filename
        if filepath.exists():
            success = test_single_file(filepath)
            results.append((filename, success))
        else:
            print(f"\n📁 {filename}: ❌ File not found")
            results.append((filename, False))
    
    # 結果サマリー
    print(f"\n{'='*50}")
    print("📊 テスト結果サマリー:")
    
    success_count = 0
    for filename, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {filename}: {status}")
        if success:
            success_count += 1
    
    print(f"\n🎯 成功率: {success_count}/{len(results)} ({100*success_count/len(results):.1f}%)")

if __name__ == "__main__":
    main()
