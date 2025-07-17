#!/usr/bin/env python3
# 🔍 展開デバッグテスト

from nxzip.engine.nexus_experimental import NEXUSExperimentalEngine

print("🔍 展開デバッグテスト")
print("=" * 50)

engine = NEXUSExperimentalEngine()

# 小容量テストデータ
test_data = "これはテストデータです。" * 1000
data = test_data.encode('utf-8')

print(f"📊 テストデータ: {len(data):,} bytes")

# 圧縮実行
print("🗜️  圧縮実行...")
compressed, stats = engine.compress(data)

print(f"✅ 圧縮完了!")
print(f"📦 圧縮サイズ: {len(compressed):,} bytes")
print(f"📊 圧縮率: {stats['compression_ratio']:.4f}%")
print(f"🏷️  手法: {stats['method']}")

# ヘッダー確認
if len(compressed) >= 4:
    header = compressed[:4]
    print(f"🔍 ヘッダー: {header}")

# 展開実行
print("\n⚡ 展開実行...")
try:
    decompressed, decomp_stats = engine.decompress(compressed)
    print(f"✅ 展開完了!")
    print(f"📤 展開サイズ: {len(decompressed):,} bytes")
    
    # データ検証
    if data == decompressed:
        print(f"🔍 データ検証: ✅ 完全一致!")
    else:
        print(f"🔍 データ検証: ❌ データ不一致!")
        
except Exception as e:
    print(f"❌ 展開エラー: {e}")

print(f"\n🔚 テスト完了")
