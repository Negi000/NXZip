#!/usr/bin/env python3
"""
NXZip軽量モード vs 通常モード性能テスト
"""

import time
from nxzip.formats.enhanced_nxz import SuperNXZipFile

def main():
    print("=== NXZip 軽量・通常モード性能比較テスト ===")
    
    # テストデータ準備
    test_data = b'Hello compression benchmark test data ' * 50
    print(f"📊 テストデータサイズ: {len(test_data)} bytes")
    
    # 軽量モードテスト
    print("\n🚀 軽量モード圧縮テスト")
    try:
        start_time = time.time()
        nxz_light = SuperNXZipFile(lightweight_mode=True)
        compressed_light = nxz_light.create_archive(test_data, show_progress=True)
        light_time = time.time() - start_time
        light_ratio = (1 - len(compressed_light) / len(test_data)) * 100
        
        print(f"⚡ 軽量モード結果: {len(compressed_light)} bytes ({light_ratio:.1f}% 圧縮) {light_time:.3f}秒")
        
    except Exception as e:
        print(f"❌ 軽量モードエラー: {e}")
        compressed_light = b''
        light_time = 0
        light_ratio = 0
    
    # 通常モードテスト
    print("\n🎯 通常モード圧縮テスト")
    try:
        start_time = time.time()
        nxz_normal = SuperNXZipFile(lightweight_mode=False)
        compressed_normal = nxz_normal.create_archive(test_data, show_progress=True)
        normal_time = time.time() - start_time
        normal_ratio = (1 - len(compressed_normal) / len(test_data)) * 100
        
        print(f"🎯 通常モード結果: {len(compressed_normal)} bytes ({normal_ratio:.1f}% 圧縮) {normal_time:.3f}秒")
        
    except Exception as e:
        print(f"❌ 通常モードエラー: {e}")
        compressed_normal = b''
        normal_time = 0
        normal_ratio = 0
    
    # 結果比較
    print(f"\n📈 性能比較結果:")
    if light_time > 0 and normal_time > 0:
        speed_factor = light_time / normal_time
        speed_desc = "高速" if light_time < normal_time else "低速"
        ratio_diff = normal_ratio - light_ratio
        
        print(f"   ⚡ 速度比較: 軽量 {light_time:.3f}秒 vs 通常 {normal_time:.3f}秒")
        print(f"   📊 軽量モードが通常モードより {speed_factor:.1f}x {speed_desc}")
        print(f"   🗜️  圧縮率: 軽量 {light_ratio:.1f}% vs 通常 {normal_ratio:.1f}%")
        print(f"   📈 通常モードが {ratio_diff:+.1f}% 高圧縮")
        
        # 可逆性テスト
        print(f"\n🔄 可逆性テスト:")
        try:
            # 軽量モード展開
            if compressed_light:
                decompressed_light = nxz_light.extract_archive(compressed_light)
                light_integrity = decompressed_light == test_data
                print(f"   ⚡ 軽量モード: {'✅ 可逆' if light_integrity else '❌ 不可逆'}")
            
            # 通常モード展開
            if compressed_normal:
                decompressed_normal = nxz_normal.extract_archive(compressed_normal)
                normal_integrity = decompressed_normal == test_data
                print(f"   🎯 通常モード: {'✅ 可逆' if normal_integrity else '❌ 不可逆'}")
                
        except Exception as e:
            print(f"   ❌ 可逆性テストエラー: {e}")
    else:
        print("   ❌ テスト失敗により比較不可")

if __name__ == "__main__":
    main()
