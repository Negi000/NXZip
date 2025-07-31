#!/usr/bin/env python3
"""
NEXUS TMC v9.0 実ファイルテスト
"""

def test_with_larger_data():
    """より大きなデータでテスト"""
    import sys
    sys.path.insert(0, '.')
    
    from nxzip.engine.nexus_tmc import NEXUSTMCEngineV9
    
    print("🧪 NEXUS TMC v9.0 大容量データテスト")
    print("=" * 60)
    
    engine = NEXUSTMCEngineV9(max_workers=2)
    
    # 様々なサイズのテストデータ
    test_cases = [
        ("小サイズ", b"Hello TMC" * 10),  # 90 bytes
        ("中サイズ", b"NEXUS TMC Engine Test Data " * 100),  # 2,600 bytes
        ("大サイズ", b"Large data compression test with NEXUS TMC v9.0 " * 1000),  # 51,000 bytes
        ("繰り返し", b"A" * 5000),  # 5,000 bytes
        ("バイナリ", bytes(range(256)) * 20),  # 5,120 bytes
    ]
    
    for name, test_data in test_cases:
        print(f"\n📄 {name}テスト: {len(test_data):,} bytes")
        
        try:
            # 圧縮
            compressed, meta = engine.compress_tmc(test_data)
            
            # 展開
            decompressed, decomp_meta = engine.decompress_tmc(compressed)
            
            # 結果分析
            is_identical = test_data == decompressed
            compression_ratio = len(compressed) / len(test_data) * 100
            space_saved = (1 - len(compressed) / len(test_data)) * 100
            
            print(f"  📦 圧縮後: {len(compressed):,} bytes")
            print(f"  📊 圧縮率: {compression_ratio:.1f}%")
            print(f"  💾 節約: {space_saved:.1f}%")
            print(f"  🔄 可逆性: {'✅ OK' if is_identical else '❌ NG'}")
            
            if 'data_type' in meta:
                print(f"  🔍 データ型: {meta.get('data_type', 'N/A')}")
            
            if not is_identical:
                print(f"    ⚠️ サイズ不一致: 元={len(test_data)}, 復元={len(decompressed)}")
                
        except Exception as e:
            print(f"  ❌ エラー: {e}")
    
    print(f"\n🎯 NEXUS TMC v9.0 大容量テスト完了")

if __name__ == "__main__":
    test_with_larger_data()
