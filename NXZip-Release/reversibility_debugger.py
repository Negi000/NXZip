#!/usr/bin/env python3
"""
NXZip 可逆性デバッガー
どのステージで可逆性が失われているかを詳細調査

失敗したテストケースを特定し、各ステージでの変換を追跡
"""

import os
import sys
import hashlib
import json
from pathlib import Path

# NXZip Core インポート
try:
    from nxzip_core import NXZipCore, NXZipContainer, CompressionMode
    print("✅ NXZip Core インポート成功")
except ImportError as e:
    print(f"❌ NXZip Core インポート失敗: {e}")
    sys.exit(1)

def calculate_hash(data: bytes) -> str:
    """データのSHA256ハッシュ計算"""
    return hashlib.sha256(data).hexdigest()

def analyze_pipeline_stages(data: bytes, mode: str, core: NXZipCore):
    """パイプライン各ステージの可逆性を詳細分析"""
    print(f"\n🔍 パイプライン可逆性分析")
    print(f"   モード: {mode}")
    print(f"   データサイズ: {len(data):,} bytes")
    
    original_hash = calculate_hash(data)
    print(f"   元データハッシュ: {original_hash[:16]}...")
    
    # 圧縮実行
    print("\n📦 圧縮実行中...")
    comp_result = core.compress(data, mode=mode, filename="debug_test")
    
    if not comp_result.success:
        print(f"❌ 圧縮失敗: {comp_result.error_message}")
        return False
    
    print(f"✅ 圧縮成功 - 圧縮率: {comp_result.compression_ratio:.2f}%")
    
    # 各ステージの詳細分析
    stages = comp_result.metadata.get('stages', [])
    print(f"\n🔧 パイプラインステージ数: {len(stages)}")
    
    for i, (stage_name, stage_info) in enumerate(stages):
        print(f"\n   Stage {i+1}: {stage_name}")
        
        if stage_name == 'tmc_transform':
            original_size = stage_info.get('original_size', 0)
            transformed_size = stage_info.get('transformed_size', 0)
            transforms = stage_info.get('transforms_applied', [])
            
            print(f"      元サイズ: {original_size:,} bytes")
            print(f"      変換後サイズ: {transformed_size:,} bytes")
            print(f"      適用変換: {transforms}")
            
            # 可逆性の危険信号
            if transforms:
                print(f"      ⚠️ データ変換が適用されています - 逆変換必須")
                
        elif stage_name == 'spe_integration':
            spe_applied = stage_info.get('spe_applied', False)
            original_size = stage_info.get('original_size', 0)
            spe_size = stage_info.get('spe_size', 0)
            encrypted = stage_info.get('encrypted', False)
            
            print(f"      SPE適用: {spe_applied}")
            print(f"      暗号化: {encrypted}")
            if spe_applied:
                print(f"      元サイズ: {original_size:,} bytes")
                print(f"      SPE後サイズ: {spe_size:,} bytes")
                
        elif stage_name == 'final_compression':
            method = stage_info.get('method', 'unknown')
            input_size = stage_info.get('input_size', 0)
            output_size = stage_info.get('output_size', 0)
            stage_ratio = stage_info.get('stage_ratio', 0)
            
            print(f"      圧縮方式: {method}")
            print(f"      入力サイズ: {input_size:,} bytes")
            print(f"      出力サイズ: {output_size:,} bytes")
            print(f"      ステージ圧縮率: {stage_ratio:.2f}%")
    
    # 展開実行
    print(f"\n🔓 展開実行中...")
    decomp_result = core.decompress(comp_result.compressed_data, comp_result.metadata)
    
    if not decomp_result.success:
        print(f"❌ 展開失敗: {decomp_result.error_message}")
        return False
    
    # 可逆性検証
    print(f"\n🔍 可逆性検証")
    decompressed_hash = calculate_hash(decomp_result.decompressed_data)
    print(f"   展開後ハッシュ: {decompressed_hash[:16]}...")
    
    integrity = core.validate_integrity(data, decomp_result.decompressed_data)
    
    print(f"   サイズ一致: {'✅' if integrity['size_match'] else '❌'}")
    print(f"   ハッシュ一致: {'✅' if integrity['hash_match'] else '❌'}")
    print(f"   全体整合性: {'✅' if integrity['integrity_ok'] else '❌'}")
    
    if not integrity['integrity_ok']:
        print(f"\n❌ 可逆性失敗の詳細:")
        print(f"   元サイズ: {integrity['original_size']:,} bytes")
        print(f"   展開後サイズ: {integrity['decompressed_size']:,} bytes")
        print(f"   サイズ差: {abs(integrity['original_size'] - integrity['decompressed_size']):,} bytes")
        
        # バイト単位比較（最初の1000バイト）
        if len(data) > 0 and len(decomp_result.decompressed_data) > 0:
            print(f"\n🔍 バイト単位比較（最初の100バイト）:")
            min_len = min(100, len(data), len(decomp_result.decompressed_data))
            
            differences = 0
            for i in range(min_len):
                if data[i] != decomp_result.decompressed_data[i]:
                    differences += 1
                    if differences <= 5:  # 最初の5つの違いを表示
                        print(f"   位置{i}: 元={data[i]:02x} 展開後={decomp_result.decompressed_data[i]:02x}")
            
            if differences > 5:
                print(f"   ... 他 {differences-5} 箇所の違い")
            
            print(f"   最初の100バイトでの違い: {differences} 箇所")
    
    return integrity['integrity_ok']

def test_problematic_cases():
    """問題のあるテストケースを再現"""
    print("🧪 問題のあるテストケースを調査")
    
    core = NXZipCore()
    
    # 失敗が予想されるケース1: 模擬実行ファイル（PE-like構造）
    print("\n" + "="*60)
    print("🔍 Case 1: 模擬実行ファイル")
    
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
    
    # Add some "code" sections with patterns
    code_section = bytearray()
    for i in range(1000):  # 小さくして問題を特定しやすく
        # Simulate x86 instructions patterns
        if i % 100 == 0:
            code_section.extend(b'\x55\x8b\xec')  # push ebp; mov ebp, esp
        elif i % 50 == 0:
            code_section.extend(b'\xff\x15')      # call dword ptr
            code_section.extend(i.to_bytes(4, 'little'))
        else:
            code_section.extend(b'\x90')          # nop
    
    pe_data.extend(code_section)
    test_data = bytes(pe_data)
    
    result = analyze_pipeline_stages(test_data, "fast", core)
    
    if not result:
        print("\n🔍 冗長性削減の個別テスト")
        # 冗長性削減の単体テスト
        from nxzip_core import TMCEngine, CompressionMode
        
        tmc_engine = TMCEngine(CompressionMode.FAST)
        reduced_data = tmc_engine._reduce_redundancy(test_data)
        
        print(f"   元データサイズ: {len(test_data):,} bytes")
        print(f"   冗長性削減後: {len(reduced_data):,} bytes")
        
        # 逆変換テスト
        core_instance = NXZipCore()
        restored_data = core_instance._restore_redundancy(reduced_data)
        
        print(f"   復元後サイズ: {len(restored_data):,} bytes")
        
        # 冗長性削減の可逆性確認
        original_hash = calculate_hash(test_data)
        restored_hash = calculate_hash(restored_data)
        
        print(f"   元データハッシュ: {original_hash[:16]}...")
        print(f"   復元後ハッシュ: {restored_hash[:16]}...")
        print(f"   冗長性削減可逆性: {'✅' if original_hash == restored_hash else '❌'}")
        
        if original_hash != restored_hash:
            print("\n❌ 冗長性削減で可逆性が失われています！")
            
            # 詳細比較
            min_len = min(len(test_data), len(restored_data))
            for i in range(min(min_len, 50)):
                if test_data[i] != restored_data[i]:
                    print(f"   位置{i}: 元={test_data[i]:02x} 復元後={restored_data[i]:02x}")

def main():
    """メイン調査実行"""
    print("🔍 NXZip 可逆性デバッガー")
    print("="*60)
    
    test_problematic_cases()

if __name__ == "__main__":
    main()
