#!/usr/bin/env python3
"""
実際の失敗ケース再現
"""

import hashlib
from nxzip_core import NXZipCore

def create_exact_failing_pe():
    """実際に失敗したPEデータを正確に再現"""
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
    
    # Add code patterns that will trigger redundancy_reduction
    code_section = bytearray()
    for i in range(1000):  # 確実に冗長性削減が発生するように
        # Simulate x86 instructions patterns
        if i % 100 == 0:
            code_section.extend(b'\x55\x8b\xec')  # push ebp; mov ebp, esp
        elif i % 50 == 0:
            code_section.extend(b'\xff\x15')      # call dword ptr
            code_section.extend(i.to_bytes(4, 'little'))
        elif i % 20 == 0:
            code_section.extend(b'\x90' * 10)     # 10個のNOP（確実にRLE対象）
        else:
            code_section.extend(b'\x90')          # nop
    
    pe_data.extend(code_section)
    return bytes(pe_data)

def test_exact_failing_case():
    print("🔍 実際の失敗ケース再現テスト")
    
    test_data = create_exact_failing_pe()
    print(f"テストデータ: {len(test_data)} bytes")
    
    core = NXZipCore()
    
    # 圧縮実行
    comp_result = core.compress(test_data, mode="fast", filename="failing_test")
    
    if not comp_result.success:
        print(f"❌ 圧縮失敗: {comp_result.error_message}")
        return
    
    print(f"✅ 圧縮成功: {comp_result.compression_ratio:.2f}%")
    
    # パイプライン確認
    stages = comp_result.metadata.get('stages', [])
    redundancy_applied = False
    
    for stage_name, stage_info in stages:
        if stage_name == 'tmc_transform':
            transforms = stage_info.get('transforms_applied', [])
            if 'redundancy_reduction' in transforms:
                redundancy_applied = True
                print(f"🔧 冗長性削減が適用されました")
                print(f"   元サイズ: {stage_info.get('original_size', 0)} bytes")
                print(f"   変換後サイズ: {stage_info.get('transformed_size', 0)} bytes")
    
    if not redundancy_applied:
        print("⚠️ 冗長性削減が適用されていません")
        return
    
    # 展開実行
    print(f"🔍 comp_result.metadata keys: {list(comp_result.metadata.keys())}")
    print(f"🔍 metadata engine: {comp_result.metadata.get('engine', 'NOT FOUND')}")
    decomp_result = core.decompress(comp_result.compressed_data, comp_result.metadata)
    
    if not decomp_result.success:
        print(f"❌ 展開失敗: {decomp_result.error_message}")
        return
    
    print(f"✅ 展開成功")
    
    # 整合性確認
    integrity = core.validate_integrity(test_data, decomp_result.decompressed_data)
    print(f"🔍 整合性: {'✅ 完全' if integrity['integrity_ok'] else '❌ 失敗'}")
    
    if not integrity['integrity_ok']:
        print(f"   元サイズ: {integrity['original_size']} bytes")
        print(f"   復元サイズ: {integrity['decompressed_size']} bytes")
        print(f"   サイズ差: {abs(integrity['original_size'] - integrity['decompressed_size'])} bytes")
        
        # 手動でパイプライン逆変換を実行してデバッグ
        print(f"\n🔍 手動逆変換デバッグ:")
        
        current_data = comp_result.compressed_data
        
        # Stage 3: final_compression 逆変換
        final_comp_info = None
        for stage_name, stage_info in stages:
            if stage_name == 'final_compression':
                final_comp_info = stage_info
                break
        
        if final_comp_info:
            method = final_comp_info.get('method', 'zlib_fast')
            print(f"   最終圧縮逆変換: {method}")
            
            if method.startswith('zlib'):
                import zlib
                current_data = zlib.decompress(current_data)
            
            print(f"   zlib展開後: {len(current_data)} bytes")
        
        # Stage 2: SPE逆変換（パススルー）
        print(f"   SPE逆変換（パススルー）")
        
        # Stage 1: TMC逆変換
        print(f"   TMC逆変換実行中...")
        restored_data = core._restore_redundancy(current_data)
        print(f"   冗長性復元後: {len(restored_data)} bytes")
        
        # 最終確認
        final_hash = hashlib.sha256(restored_data).hexdigest()
        original_hash = hashlib.sha256(test_data).hexdigest()
        
        print(f"   手動逆変換結果:")
        print(f"     元ハッシュ: {original_hash[:16]}...")
        print(f"     復元ハッシュ: {final_hash[:16]}...")
        print(f"     手動可逆性: {'✅' if original_hash == final_hash else '❌'}")

def main():
    test_exact_failing_case()

if __name__ == "__main__":
    main()
