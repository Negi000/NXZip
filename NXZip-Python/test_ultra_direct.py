#!/usr/bin/env python3
# 🔍 zlib_ultra_compress 直接テスト

from nxzip.engine.nexus_experimental import NEXUSExperimentalEngine

print("🔍 zlib_ultra_compress 直接テスト")
print("=" * 60)

engine = NEXUSExperimentalEngine()

# 1MBテストデータ
test_data = """これはTSVファイルのテストデータです。
カラム1	カラム2	カラム3	カラム4
データ1	値1	100	テスト
データ2	値2	200	テスト
""" * 5000  # 約1MB

data = test_data.encode('utf-8')
print(f"📊 テストデータ: {len(data):,} bytes ({len(data)/(1024*1024):.1f} MB)")

# zlib_ultra_compress を直接実行
print("🗜️  zlib_ultra_compress 直接実行...")
try:
    compressed = engine._zlib_ultra_compress(data)
    
    print(f"✅ 圧縮完了!")
    print(f"📦 圧縮サイズ: {len(compressed):,} bytes")
    print(f"📊 圧縮率: {(1 - len(compressed)/len(data)) * 100:.4f}%")
    if len(compressed) >= 4:
        header = compressed[:4]
        print(f"🔍 ヘッダー: {header}")
    
    # 展開実行
    print("\n⚡ 展開実行...")
    decompressed = engine._zlib_ultra_decompress_optimized(compressed)
    
    print(f"✅ 展開完了!")
    print(f"📤 展開サイズ: {len(decompressed):,} bytes")
    
    # データ検証
    if data == decompressed:
        print(f"🔍 データ検証: ✅ 完全一致!")
    else:
        print(f"🔍 データ検証: ❌ データ不一致!")
        print(f"   元データ: {len(data)} bytes")
        print(f"   展開データ: {len(decompressed)} bytes")
        
except Exception as e:
    print(f"❌ エラー: {e}")
    import traceback
    traceback.print_exc()

print(f"\n🔚 テスト完了")
