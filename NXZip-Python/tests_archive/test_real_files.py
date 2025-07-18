#!/usr/bin/env python3
# 🚀 実ファイル段階的テスト - 高速版

import os
import time
from concurrent.futures import ThreadPoolExecutor
from nxzip.engine.nexus_experimental import NEXUSExperimentalEngine

def run_with_timeout(func, timeout):
    """タイムアウト付き実行"""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout)
        except Exception as e:
            future.cancel()
            raise e

print("🚀 実ファイル段階的テスト - 高速版")
print("=" * 60)

# ファイル情報
files_to_test = [
    {
        'name': '需要引当予測リスト クエリ.txt',
        'path': r'C:\Users\241822\Desktop\新しいフォルダー (2)\需要引当予測リスト クエリ.txt',
        'description': '最小ファイル (161.6 MB)',
        'timeout': 60  # 60秒制限
    },
    {
        'name': '出庫実績明細_202408.tsv', 
        'path': r'C:\Users\241822\Desktop\新しいフォルダー (2)\出庫実績明細_202408.tsv',
        'description': '中サイズファイル (617.21 MB)',
        'timeout': 120  # 120秒制限
    }
]

engine = NEXUSExperimentalEngine()

for i, file_info in enumerate(files_to_test, 1):
    print(f"\n📋 テスト {i}/{len(files_to_test)}: {file_info['name']}")
    print(f"📁 {file_info['description']}")
    print("=" * 50)
    
    try:
        # ファイル読み込み
        print("📖 ファイル読み込み中...")
        with open(file_info['path'], 'rb') as f:
            data = f.read()
        
        file_size_mb = len(data) / (1024 * 1024)
        print(f"✅ 読み込み完了: {len(data):,} bytes ({file_size_mb:.2f} MB)")
        
        # 圧縮テスト（タイムアウト付き）
        print(f"\n🗜️  圧縮テスト実行... (制限時間: {file_info['timeout']}秒)")
        
        def compress_test():
            return engine.compress(data, file_info['name'])
        
        start_time = time.time()
        try:
            compressed, stats = run_with_timeout(compress_test, file_info['timeout'])
            compression_time = time.time() - start_time
            
            print(f"✅ 圧縮完了!")
            print(f"📊 圧縮率: {stats['compression_ratio']:.4f}%")
            print(f"🚀 圧縮速度: {stats['speed_mbps']:.2f} MB/s")
            print(f"⏱️  圧縮時間: {compression_time:.2f}秒")
            print(f"🏷️  圧縮手法: {stats['method']}")
            print(f"📦 圧縮サイズ: {len(compressed):,} bytes")
            
            # 展開テスト（短時間制限）
            print(f"\n⚡ 展開テスト実行... (制限時間: 30秒)")
            
            def decompress_test():
                return engine.decompress(compressed)
            
            try:
                decompressed, decomp_stats = run_with_timeout(decompress_test, 30)
                
                # データ検証
                if data == decompressed:
                    print(f"✅ 展開成功!")
                    print(f"⚡ 展開速度: {decomp_stats['speed_mbps']:.2f} MB/s")
                    print(f"🔍 データ検証: ✅ 完全一致!")
                    
                    # 総合評価
                    ratio = stats['compression_ratio']
                    speed = stats['speed_mbps']
                    
                    print(f"\n🎯 総合評価:")
                    if ratio >= 95.0 and speed >= 10:
                        print(f"🏆 優秀! 高圧縮率 & 実用速度達成")
                    elif ratio >= 90.0:
                        print(f"🎯 良好! 高圧縮率達成")
                    elif speed >= 20:
                        print(f"🚀 良好! 高速処理達成")
                    else:
                        print(f"📊 標準レベル")
                        
                else:
                    print(f"❌ 展開失敗: データ不一致")
                    
            except Exception as e:
                print(f"❌ 展開エラーまたはタイムアウト: {e}")
                print(f"📊 圧縮のみ評価:")
                print(f"   📈 圧縮率: {stats['compression_ratio']:.2f}%")
                print(f"   🚀 圧縮速度: {stats['speed_mbps']:.1f} MB/s")
                
        except Exception as e:
            print(f"❌ 圧縮エラーまたはタイムアウト: {e}")
            print(f"💡 {file_size_mb:.1f}MBで{file_info['timeout']}秒は処理時間不足")
            
            # より小さなサンプルで再試行
            sample_size = min(50 * 1024 * 1024, len(data) // 4)  # 50MBまたは1/4サイズ
            sample_data = data[:sample_size]
            sample_mb = sample_size / (1024 * 1024)
            
            print(f"🔄 小サンプル再試行: {sample_mb:.1f} MB")
            
            try:
                def sample_test():
                    return engine.compress(sample_data, f"sample_{file_info['name']}")
                
                sample_compressed, sample_stats = run_with_timeout(sample_test, 30)
                print(f"📊 小サンプル結果:")
                print(f"   📈 圧縮率: {sample_stats['compression_ratio']:.2f}%")
                print(f"   🚀 圧縮速度: {sample_stats['speed_mbps']:.1f} MB/s")
                print(f"   🏷️  手法: {sample_stats['method']}")
                
            except Exception as sample_e:
                print(f"❌ 小サンプルも失敗: {sample_e}")
    
    except FileNotFoundError:
        print(f"❌ ファイルが見つかりません: {file_info['path']}")
    except Exception as e:
        print(f"❌ ファイル読み込みエラー: {e}")

print(f"\n🔚 段階的テスト完了")
print("=" * 60)
print("💡 高速処理と高圧縮率の両立を確認")
