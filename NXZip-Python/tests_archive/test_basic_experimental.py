#!/usr/bin/env python3
# 基本テスト - インポート確認

print("=== 基本インポートテスト ===")

try:
    print("1. 実験版エンジンインポート中...")
    from nxzip.engine.nexus_experimental import NEXUSExperimentalEngine
    print("✅ インポート成功")
    
    print("2. エンジン初期化中...")
    engine = NEXUSExperimentalEngine()
    print(f"✅ 初期化成功: {engine.version}")
    
    print("3. 小規模テストデータ準備...")
    test_data = b"Hello NEXUS Experimental!" * 1000
    print(f"✅ テストデータ: {len(test_data)} bytes")
    
    print("4. 圧縮テスト...")
    compressed, comp_stats = engine.compress(test_data)
    print(f"✅ 圧縮完了: {comp_stats['compression_ratio']:.2f}%")
    
    print("5. 展開テスト...")
    decompressed, decomp_stats = engine.decompress(compressed)
    print(f"✅ 展開完了: {decomp_stats['speed_mbps']:.2f} MB/s")
    
    print("6. データ検証...")
    is_valid = decompressed == test_data
    print(f"✅ 検証結果: {'OK' if is_valid else 'NG'}")
    
    print("\n📊 基本テスト完了")
    
except Exception as e:
    print(f"❌ エラー発生: {e}")
    import traceback
    traceback.print_exc()
