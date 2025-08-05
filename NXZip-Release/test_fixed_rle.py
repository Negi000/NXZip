#!/usr/bin/env python3
"""
修正版RLE可逆性テスト
"""

import os
import sys
import hashlib

# NXZip Core インポート
try:
    from nxzip_core import TMCEngine, CompressionMode, NXZipCore
    print("✅ 修正版NXZip Core インポート成功")
except ImportError as e:
    print(f"❌ NXZip Core インポート失敗: {e}")
    sys.exit(1)

def test_rle_reversibility():
    """RLE可逆性の詳細テスト"""
    print("🧪 修正版RLE可逆性テスト")
    
    # テストケース1: 簡単な繰り返し
    test_cases = [
        # (テスト名, データ)
        ("単純繰り返し", b'AAAABBBBCCCCDDDD'),
        ("混合データ", b'AAA\xFE\xFE\xFEBBBBCCCC'),
        ("0xFEエスケープ", b'ABC\xFEDEF\xFE\xFE\xFE\xFEGHI'),
        ("境界ケース", b'AAA'),  # 4未満
        ("長い繰り返し", b'X' * 100),
        ("模擬実行ファイル", create_mock_pe_data()),
    ]
    
    core = NXZipCore()
    tmc_engine = TMCEngine(CompressionMode.FAST)
    
    for test_name, original_data in test_cases:
        print(f"\n🔍 {test_name}")
        print(f"   元データ: {len(original_data)} bytes")
        
        # RLE圧縮
        compressed = tmc_engine._reduce_redundancy(original_data)
        print(f"   圧縮後: {len(compressed)} bytes")
        
        # RLE復元
        restored = core._restore_redundancy(compressed)
        print(f"   復元後: {len(restored)} bytes")
        
        # 可逆性確認
        original_hash = hashlib.sha256(original_data).hexdigest()
        restored_hash = hashlib.sha256(restored).hexdigest()
        
        reversible = original_hash == restored_hash
        print(f"   可逆性: {'✅' if reversible else '❌'}")
        
        if not reversible:
            print(f"   ❌ 失敗詳細:")
            print(f"      元ハッシュ: {original_hash[:16]}...")
            print(f"      復元ハッシュ: {restored_hash[:16]}...")
            
            # バイト比較
            min_len = min(len(original_data), len(restored))
            diffs = 0
            for i in range(min_len):
                if original_data[i] != restored[i]:
                    diffs += 1
                    if diffs <= 3:
                        print(f"      位置{i}: 元={original_data[i]:02x} 復元={restored[i]:02x}")
            
            if diffs > 3:
                print(f"      ... 他 {diffs-3} 箇所の違い")

def create_mock_pe_data():
    """模擬PEデータ作成"""
    pe_data = bytearray()
    # DOS header
    pe_data.extend(b'MZ')  # DOS signature
    pe_data.extend(b'\x00' * 58)  # DOS header padding
    pe_data.extend((64).to_bytes(4, 'little'))  # PE header offset
    
    # PE header
    pe_data.extend(b'PE\x00\x00')  # PE signature
    pe_data.extend(b'\x4c\x01')    # Machine (i386)
    pe_data.extend(b'\x03\x00')    # Number of sections
    pe_data.extend(b'\x00' * 16)   # Timestamp, etc.
    
    # Add code patterns with repetitions
    code_section = bytearray()
    for i in range(200):
        if i % 20 == 0:
            code_section.extend(b'\x90' * 10)  # 10個のNOP (should be RLE compressed)
        elif i % 15 == 0:
            code_section.extend(b'\xFE\xFE\xFE\xFE\xFE')  # 0xFE repetition (escape test)
        elif i % 10 == 0:
            code_section.extend(b'\x55\x8b\xec')  # push ebp; mov ebp, esp
        else:
            code_section.append(i % 256)
    
    pe_data.extend(code_section)
    return bytes(pe_data)

def test_full_pipeline():
    """修正後のフルパイプラインテスト"""
    print(f"\n{'='*60}")
    print("🔧 修正後フルパイプラインテスト")
    
    core = NXZipCore()
    
    # 問題があったテストケース再実行
    test_data = create_mock_pe_data()
    print(f"テストデータ: {len(test_data)} bytes")
    
    # 圧縮
    comp_result = core.compress(test_data, mode="fast", filename="fixed_test")
    
    if not comp_result.success:
        print(f"❌ 圧縮失敗: {comp_result.error_message}")
        return
    
    print(f"✅ 圧縮成功: {comp_result.compression_ratio:.2f}%")
    
    # 展開
    decomp_result = core.decompress(comp_result.compressed_data, comp_result.metadata)
    
    if not decomp_result.success:
        print(f"❌ 展開失敗: {decomp_result.error_message}")
        return
    
    print(f"✅ 展開成功")
    
    # 整合性確認
    integrity = core.validate_integrity(test_data, decomp_result.decompressed_data)
    print(f"🔍 整合性: {'✅ 完全' if integrity['integrity_ok'] else '❌ 失敗'}")
    
    if not integrity['integrity_ok']:
        print(f"   サイズ: {integrity['original_size']} → {integrity['decompressed_size']}")
        print(f"   ハッシュ一致: {'✅' if integrity['hash_match'] else '❌'}")

def main():
    print("🔧 修正版RLE可逆性検証")
    print("="*60)
    
    test_rle_reversibility()
    test_full_pipeline()

if __name__ == "__main__":
    main()
