#!/usr/bin/env python3
"""
パイプライン展開デバッグ
"""

import hashlib
from nxzip_core import NXZipCore

def debug_pipeline_decompression():
    """パイプライン展開の詳細デバッグ"""
    print("🔍 パイプライン展開デバッグ")
    
    # 問題のあるデータを作成
    test_data = bytearray()
    test_data.extend(b'MZ')  
    test_data.extend(b'\x00' * 60)  # 60個の0x00の繰り返し
    test_data.extend(b'PE')
    test_data.extend(b'\x90' * 20)   # 20個のNOPの繰り返し  
    test_data = bytes(test_data)
    
    print(f"元データ: {len(test_data)} bytes")
    print(f"元ハッシュ: {hashlib.sha256(test_data).hexdigest()[:16]}...")
    
    core = NXZipCore()
    
    # 圧縮
    comp_result = core.compress(test_data, mode="fast", filename="debug_test")
    
    if not comp_result.success:
        print(f"❌ 圧縮失敗: {comp_result.error_message}")
        return
    
    print(f"✅ 圧縮成功: {comp_result.compression_ratio:.2f}%")
    print(f"圧縮データサイズ: {len(comp_result.compressed_data)} bytes")
    
    # パイプライン情報表示
    stages = comp_result.metadata.get('stages', [])
    print(f"\n🔧 パイプライン情報:")
    for i, (stage_name, stage_info) in enumerate(stages):
        print(f"  Stage {i+1}: {stage_name}")
        if stage_name == 'tmc_transform':
            transforms = stage_info.get('transforms_applied', [])
            print(f"    適用変換: {transforms}")
            print(f"    元サイズ: {stage_info.get('original_size', 0)} bytes")
            print(f"    変換後サイズ: {stage_info.get('transformed_size', 0)} bytes")
    
    # 手動でパイプライン逆変換をデバッグ実行
    print(f"\n🔓 手動パイプライン逆変換デバッグ:")
    
    current_data = comp_result.compressed_data
    print(f"開始データ: {len(current_data)} bytes")
    
    # 逆順で各ステージを処理
    for i, (stage_name, stage_info) in enumerate(reversed(stages)):
        print(f"\n  逆変換ステップ {i+1}: {stage_name}")
        print(f"    入力サイズ: {len(current_data)} bytes")
        
        if stage_name == 'final_compression':
            # 最終圧縮の逆変換
            method = stage_info.get('method', 'zlib_balanced')
            print(f"    圧縮方式: {method}")
            
            if method.startswith('lzma'):
                import lzma
                current_data = lzma.decompress(current_data)
            elif method.startswith('zlib'):
                import zlib
                current_data = zlib.decompress(current_data)
            
            print(f"    展開後サイズ: {len(current_data)} bytes")
                
        elif stage_name == 'spe_integration':
            # SPE逆変換（パススルー）
            print(f"    SPE逆変換（パススルー）")
            pass
                
        elif stage_name == 'tmc_transform':
            # TMC逆変換
            transforms = stage_info.get('transforms_applied', [])
            print(f"    逆変換対象: {transforms}")
            
            for transform in reversed(transforms):
                print(f"      逆変換実行: {transform}")
                
                if transform == 'redundancy_reduction':
                    before_size = len(current_data)
                    current_data = core._restore_redundancy(current_data)
                    after_size = len(current_data)
                    print(f"        冗長性復元: {before_size} → {after_size} bytes")
                    
                    # 詳細チェック
                    if before_size != after_size:
                        print(f"        ✅ 冗長性復元が実行されました")
                    else:
                        print(f"        ⚠️ サイズが変わっていません")
        
        print(f"    出力サイズ: {len(current_data)} bytes")
    
    # 最終結果確認
    print(f"\n🔍 最終結果:")
    print(f"復元データサイズ: {len(current_data)} bytes")
    
    final_hash = hashlib.sha256(current_data).hexdigest()
    original_hash = hashlib.sha256(test_data).hexdigest()
    
    print(f"元ハッシュ:   {original_hash[:16]}...")
    print(f"復元ハッシュ: {final_hash[:16]}...")
    print(f"可逆性: {'✅' if original_hash == final_hash else '❌'}")
    
    if original_hash != final_hash and len(current_data) == len(test_data):
        # サイズは一致するがハッシュが違う場合の詳細確認
        print(f"\n❌ ハッシュ不一致の詳細:")
        for i in range(min(50, len(test_data))):
            if test_data[i] != current_data[i]:
                print(f"  位置{i}: 元=0x{test_data[i]:02x} 復元=0x{current_data[i]:02x}")

def main():
    debug_pipeline_decompression()

if __name__ == "__main__":
    main()
