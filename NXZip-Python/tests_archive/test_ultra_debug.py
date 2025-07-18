#!/usr/bin/env python3
# 🔍 zlib_ultra_compress 詳細デバッグテスト

from nxzip.engine.nexus_experimental import NEXUSExperimentalEngine

print("🔍 zlib_ultra_compress 詳細デバッグテスト")
print("=" * 60)

engine = NEXUSExperimentalEngine()

# 中サイズテストデータ（zlib_ultra_compressを誘発）
test_data = """これはTSVファイルのテストデータです。
カラム1	カラム2	カラム3	カラム4
データ1	値1	100	テスト
データ2	値2	200	テスト
""" * 2000  # 中サイズファイル

data = test_data.encode('utf-8')
print(f"📊 テストデータ: {len(data):,} bytes ({len(data)/(1024*1024):.1f} MB)")

# 圧縮実行
print("🗜️  圧縮実行...")
compressed, stats = engine.compress(data)

print(f"✅ 圧縮完了!")
print(f"📦 圧縮サイズ: {len(compressed):,} bytes")
print(f"📊 圧縮率: {stats['compression_ratio']:.4f}%")
print(f"🏷️  手法: {stats['method']}")

# パッケージ解析
print("\n🔍 パッケージ解析...")
try:
    unpacked_data, method, original_size = engine._lightning_unpackage_data(compressed)
    print(f"📦 解析結果:")
    print(f"   🏷️  手法: {method}")
    print(f"   📏 元サイズ: {original_size:,} bytes")
    print(f"   📦 圧縮データ: {len(unpacked_data):,} bytes")
    if len(unpacked_data) >= 4:
        header = unpacked_data[:4]
        print(f"   🔍 内部ヘッダー: {header}")
    
    # 展開実行
    print("\n⚡ 展開実行...")
    if method == 'zlib_ultra_compress':
        print("🔍 zlib_ultra_compress 専用展開を使用")
        decompressed = engine._zlib_ultra_decompress_optimized(unpacked_data)
    else:
        print(f"🔍 標準展開を使用 ({method})")
        decompressed = engine._execute_optimized_decompression(unpacked_data, method)
    
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
