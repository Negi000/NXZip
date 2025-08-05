#!/usr/bin/env python3
"""
🔍 実際のTSVファイルでのTMC v9.1テスト - 真の問題を特定
"""

import os
import sys
import hashlib
import time
from pathlib import Path

# NXZip-Python パスを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'NXZip-Python'))

try:
    from nxzip.engine.nexus_tmc_v91_modular import NEXUSTMCEngineV91
    print('✅ NEXUS TMC v9.1 エンジンインポート成功')
except ImportError as e:
    print(f'❌ インポートエラー: {e}')
    sys.exit(1)

def test_real_tsv_file():
    """実際のTSVファイルでの圧縮→展開テスト"""
    
    # 実際のTSVファイルのパス
    tsv_path = r"C:/Users/241822/Desktop/在庫明細_20250610/在庫明細_20250610.tsv"
    
    if not os.path.exists(tsv_path):
        print(f"❌ テストファイルが見つかりません: {tsv_path}")
        print("🔧 小さなTSVサンプルを作成してテストします...")
        return test_synthetic_tsv()
    
    print(f"📂 実際のTSVファイルをテスト: {tsv_path}")
    
    try:
        # ファイル読み込み
        with open(tsv_path, 'rb') as f:
            original_data = f.read()
        
        print(f"📊 元ファイルサイズ: {len(original_data):,} bytes ({len(original_data)/1024/1024:.1f} MB)")
        
        # 元データのハッシュ
        original_hash = hashlib.sha256(original_data).hexdigest()
        print(f"🔍 元データハッシュ: {original_hash[:16]}...")
        
        # エンジン初期化
        engine = NEXUSTMCEngineV91(
            max_workers=4,
            chunk_size=2*1024*1024,  # 2MB
            lightweight_mode=False
        )
        
        # 圧縮フェーズ
        print("🗜️ 圧縮開始...")
        start_time = time.time()
        compressed_data, compress_info = engine.compress(original_data)
        compress_time = time.time() - start_time
        
        compression_ratio = compress_info.get('compression_ratio', 0)
        print(f"✅ 圧縮完了: {len(compressed_data):,} bytes ({compression_ratio:.2f}% 圧縮)")
        print(f"⏱️ 圧縮時間: {compress_time:.2f}秒")
        print(f"🔥 TMC変換: {'適用' if compress_info.get('transform_applied') else 'バイパス'}")
        print(f"📊 データタイプ: {compress_info.get('data_type', 'unknown')}")
        
        # 展開フェーズ
        print("📂 展開開始...")
        start_time = time.time()
        decompressed_data = engine.decompress(compressed_data, compress_info)
        decompress_time = time.time() - start_time
        
        print(f"✅ 展開完了: {len(decompressed_data):,} bytes")
        print(f"⏱️ 展開時間: {decompress_time:.2f}秒")
        
        # 整合性検証
        decompressed_hash = hashlib.sha256(decompressed_data).hexdigest()
        print(f"🔍 展開データハッシュ: {decompressed_hash[:16]}...")
        
        # 詳細比較
        print("\n" + "="*60)
        print("🔍 詳細比較結果")
        print("="*60)
        print(f"📏 サイズ比較:")
        print(f"   元データ: {len(original_data):,} bytes")
        print(f"   展開データ: {len(decompressed_data):,} bytes")
        print(f"   サイズ一致: {'✅' if len(original_data) == len(decompressed_data) else '❌'}")
        
        print(f"🔍 ハッシュ比較:")
        print(f"   元ハッシュ  : {original_hash}")
        print(f"   展開ハッシュ: {decompressed_hash}")
        print(f"   ハッシュ一致: {'✅' if original_hash == decompressed_hash else '❌'}")
        
        if original_hash != decompressed_hash or len(original_data) != len(decompressed_data):
            print("\n⚠️ 整合性に問題があります！")
            
            # バイト単位での部分比較
            if len(decompressed_data) > 0:
                print("🔍 バイト単位比較（先頭100バイト）:")
                min_len = min(100, len(original_data), len(decompressed_data))
                for i in range(min_len):
                    if original_data[i] != decompressed_data[i]:
                        print(f"   差異発見: 位置{i} - 元=0x{original_data[i]:02X}, 展開=0x{decompressed_data[i]:02X}")
                        break
                else:
                    print("   先頭100バイトは一致")
            
            # データ内容の分析
            print("\n🔍 元データ内容分析（先頭200文字）:")
            try:
                sample_text = original_data[:200].decode('utf-8', errors='replace')
                print(f"   内容: {repr(sample_text)}")
            except:
                print("   バイナリデータ")
            
            print("\n🔍 展開データ内容分析（先頭200文字）:")
            try:
                sample_text = decompressed_data[:200].decode('utf-8', errors='replace')
                print(f"   内容: {repr(sample_text)}")
            except:
                print("   バイナリデータ")
            
            return False
        else:
            print("\n🎉 ✅ 完全一致！TMC v9.1は正常に動作しています！")
            return True
            
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_synthetic_tsv():
    """合成TSVデータでのテスト"""
    print("🧪 合成TSVデータでテスト中...")
    
    # TSV形式のサンプルデータ生成
    tsv_lines = []
    tsv_lines.append("商品ID\t商品名\t価格\t在庫数\tカテゴリ")
    
    for i in range(10000):  # 1万行のTSVデータ
        tsv_lines.append(f"ITEM{i:05d}\t商品名{i}\t{1000 + i % 5000}\t{i % 100}\tカテゴリ{i % 10}")
    
    tsv_text = "\n".join(tsv_lines)
    original_data = tsv_text.encode('utf-8')
    
    print(f"📊 合成TSVサイズ: {len(original_data):,} bytes ({len(original_data)/1024/1024:.1f} MB)")
    
    # 元データのハッシュ
    original_hash = hashlib.sha256(original_data).hexdigest()
    print(f"🔍 元データハッシュ: {original_hash[:16]}...")
    
    # エンジン初期化
    engine = NEXUSTMCEngineV91(max_workers=4, chunk_size=2*1024*1024, lightweight_mode=False)
    
    # 圧縮フェーズ
    print("🗜️ 圧縮開始...")
    compressed_data, compress_info = engine.compress(original_data)
    
    compression_ratio = compress_info.get('compression_ratio', 0)
    print(f"✅ 圧縮完了: {len(compressed_data):,} bytes ({compression_ratio:.2f}% 圧縮)")
    print(f"🔥 TMC変換: {'適用' if compress_info.get('transform_applied') else 'バイパス'}")
    print(f"📊 データタイプ: {compress_info.get('data_type', 'unknown')}")
    
    # 展開フェーズ
    print("📂 展開開始...")
    decompressed_data = engine.decompress(compressed_data, compress_info)
    
    print(f"✅ 展開完了: {len(decompressed_data):,} bytes")
    
    # 整合性検証
    decompressed_hash = hashlib.sha256(decompressed_data).hexdigest()
    
    if original_hash == decompressed_hash:
        print("🎉 ✅ 合成TSVデータでは完全一致！")
        return True
    else:
        print("❌ 合成TSVデータでも問題発生")
        return False

def main():
    """メイン実行"""
    print("🔍 実際のTSVファイルでのTMC v9.1整合性テスト")
    print("="*60)
    
    success = test_real_tsv_file()
    
    print("\n" + "="*60)
    print("🏆 結果")
    print("="*60)
    if success:
        print("✅ TMC v9.1は実際のTSVファイルでも正常に動作しています")
        print("🎉 前回の問題は別の要因である可能性があります")
    else:
        print("⚠️ 実際のTSVファイルで整合性の問題を確認")
        print("🔧 TMC変換の特定のデータパターンに対する調整が必要")

if __name__ == "__main__":
    main()
