#!/usr/bin/env python3
# 🔬 実ファイル段階的テスト - 小サイズから順番に

import time
import os
from nxzip.engine.nexus_experimental import NEXUSExperimentalEngine

print("🔬 実ファイル段階的テスト - 小サイズから順番に")
print("=" * 60)

# ファイル情報
test_files = [
    {
        'name': '需要引当予測リスト クエリ.txt',
        'path': r'C:\Users\241822\Desktop\新しいフォルダー (2)\需要引当予測リスト クエリ.txt',
        'expected_size_mb': 161.6,
        'description': '最小ファイル (161.6 MB)'
    },
    {
        'name': '出庫実績明細_202408.tsv',
        'path': r'C:\Users\241822\Desktop\新しいフォルダー (2)\出庫実績明細_202408.tsv',
        'expected_size_mb': 617.21,
        'description': '中サイズファイル (617.21 MB)'
    },
    {
        'name': '出庫実績明細_202410.tsv',
        'path': r'C:\Users\241822\Desktop\新しいフォルダー (2)\出庫実績明細_202410.tsv',
        'expected_size_mb': 1606.75,
        'description': '大サイズファイル (1606.75 MB) - 既知'
    }
]

engine = NEXUSExperimentalEngine()

for i, file_info in enumerate(test_files, 1):
    print(f"\n📋 テスト {i}/3: {file_info['name']}")
    print(f"📁 {file_info['description']}")
    print("=" * 50)
    
    try:
        # ファイル読み込み
        print(f"📖 ファイル読み込み中...")
        if not os.path.exists(file_info['path']):
            print(f"❌ ファイルが見つかりません: {file_info['path']}")
            continue
            
        with open(file_info['path'], 'rb') as f:
            data = f.read()
        
        actual_size_mb = len(data) / (1024 * 1024)
        print(f"✅ 読み込み完了: {len(data):,} bytes ({actual_size_mb:.2f} MB)")
        
        # 圧縮テスト
        print(f"\n🗜️  圧縮テスト実行...")
        start_time = time.time()
        
        compressed, stats = engine.compress(data, file_info['name'])
        
        compression_time = time.time() - start_time
        
        print(f"✅ 圧縮完了!")
        print(f"📊 圧縮率: {stats['compression_ratio']:.4f}%")
        print(f"🚀 圧縮速度: {stats['speed_mbps']:.2f} MB/s")
        print(f"⏱️  圧縮時間: {compression_time:.2f}秒")
        print(f"🏷️  圧縮手法: {stats['method']}")
        print(f"📦 圧縮サイズ: {len(compressed):,} bytes")
        
        # 展開テスト
        print(f"\n⚡ 展開テスト実行...")
        start_time = time.time()
        
        try:
            decompressed, decomp_stats = engine.decompress(compressed)
            decompression_time = time.time() - start_time
            
            print(f"✅ 展開完了!")
            print(f"📤 展開サイズ: {len(decompressed):,} bytes")
            print(f"⚡ 展開速度: {decomp_stats['speed_mbps']:.2f} MB/s")
            print(f"⏱️  展開時間: {decompression_time:.2f}秒")
            print(f"🎯 性能等級: {decomp_stats.get('performance_grade', 'N/A')}")
            
            # データ検証
            if data == decompressed:
                print(f"🔍 データ検証: ✅ 完全一致!")
                
                # 性能評価
                ratio = stats['compression_ratio']
                comp_speed = stats['speed_mbps']
                decomp_speed = decomp_stats['speed_mbps']
                
                print(f"\n🏆 性能評価:")
                print(f"   📈 圧縮率: {ratio:.2f}% {'🎉' if ratio >= 90 else '🔶' if ratio >= 80 else '📊'}")
                print(f"   🚀 圧縮速度: {comp_speed:.1f} MB/s {'🎉' if comp_speed >= 100 else '🔶' if comp_speed >= 50 else '📊'}")
                print(f"   ⚡ 展開速度: {decomp_speed:.1f} MB/s {'🎉' if decomp_speed >= 200 else '🔶' if decomp_speed >= 100 else '📊'}")
                
                if ratio >= 90 and comp_speed >= 50 and decomp_speed >= 100:
                    print(f"🎉🏆 優秀な性能達成!")
                elif ratio >= 80 and comp_speed >= 30:
                    print(f"🎯 良好な性能!")
                else:
                    print(f"📊 標準的な性能")
            else:
                print(f"🔍 データ検証: ❌ データ不一致!")
                print(f"   元サイズ: {len(data)} bytes")
                print(f"   展開サイズ: {len(decompressed)} bytes")
                
        except Exception as e:
            print(f"❌ 展開エラー: {e}")
            # 展開できなくても圧縮性能は評価
            ratio = stats['compression_ratio']
            comp_speed = stats['speed_mbps']
            print(f"\n📊 圧縮のみ評価:")
            print(f"   📈 圧縮率: {ratio:.2f}%")
            print(f"   🚀 圧縮速度: {comp_speed:.1f} MB/s")
        
        # サイズが大きい場合は継続確認
        if actual_size_mb > 500 and i < len(test_files):
            print(f"\n⚠️  次のファイルはさらに大きいです ({test_files[i]['expected_size_mb']:.1f} MB)")
            print(f"💡 現在の結果で十分な場合は手動で停止してください")
            
    except Exception as e:
        print(f"❌ ファイル処理エラー: {e}")
        continue

print(f"\n🔚 全テスト完了")
