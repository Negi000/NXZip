#!/usr/bin/env python3
# 🚀 NEXUS 段階的圧縮率改善テスト - 99.9%への道

import time
import os
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

def run_with_timeout_safe(func, timeout_seconds=30):
    """安全なタイムアウト付き実行"""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError:
            print(f"⏰ タイムアウト: {timeout_seconds}秒で処理を停止")
            raise TimeoutError(f"処理が{timeout_seconds}秒を超過")

print("🚀 NEXUS 段階的圧縮率改善テスト - 99.9%への道")
print("=" * 60)

# 実戦ファイル
real_file_path = r"C:\Users\241822\Desktop\新しいフォルダー (2)\出庫実績明細_202410.tsv"

if not os.path.exists(real_file_path):
    print(f"❌ ファイルが見つかりません")
    exit(1)

file_size = os.path.getsize(real_file_path)
print(f"📊 実戦ファイル: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")

# ファイル読み込み
with open(real_file_path, 'rb') as f:
    data = f.read()
print(f"✅ 読み込み完了: {len(data):,} bytes")

# NEXUS実験エンジンのインポート
from nxzip.engine.nexus_experimental import NEXUSExperimentalEngine

# 段階的テスト
compression_tests = [
    {
        'name': '🚀 高速圧縮テスト',
        'sample_size': 1024 * 1024,  # 1MB
        'timeout': 10,
        'target_ratio': 90.0
    },
    {
        'name': '💎 高圧縮テスト',
        'sample_size': 10 * 1024 * 1024,  # 10MB
        'timeout': 20,
        'target_ratio': 95.0
    },
    {
        'name': '🏆 超高圧縮テスト',
        'sample_size': 50 * 1024 * 1024,  # 50MB
        'timeout': 30,
        'target_ratio': 99.0
    },
    {
        'name': '💎🏆 99.9%圧縮テスト',
        'sample_size': len(data),  # 全体
        'timeout': 60,  # 60秒に延長
        'target_ratio': 99.9
    }
]

best_ratio = 0
best_method = None
best_stats = None

for i, test in enumerate(compression_tests):
    print(f"\n{test['name']} ({i+1}/{len(compression_tests)})")
    print(f"📊 サンプルサイズ: {test['sample_size']:,} bytes")
    print(f"⏰ タイムアウト: {test['timeout']}秒")
    print(f"🎯 目標圧縮率: {test['target_ratio']}%")
    
    # サンプルデータ準備
    sample_data = data[:test['sample_size']]
    
    try:
        def compression_test():
            engine = NEXUSExperimentalEngine()
            return engine.compress(sample_data)
        
        start_time = time.time()
        compressed, stats = run_with_timeout_safe(compression_test, test['timeout'])
        compression_time = time.time() - start_time
        
        ratio = stats['compression_ratio']
        speed = stats['speed_mbps']
        method = stats['method']
        
        print(f"✅ 圧縮完了!")
        print(f"📊 圧縮率: {ratio:.4f}%")
        print(f"🚀 圧縮速度: {speed:.2f} MB/s")
        print(f"⏱️  時間: {compression_time:.3f}秒")
        print(f"🏷️  手法: {method}")
        
        # 結果評価
        if ratio >= test['target_ratio']:
            print(f"🎉 目標達成! {ratio:.4f}% ≥ {test['target_ratio']}%")
            
            # 最高記録更新
            if ratio > best_ratio:
                best_ratio = ratio
                best_method = method
                best_stats = stats
                print(f"🏆 新記録! 最高圧縮率更新: {ratio:.4f}%")
        else:
            print(f"📊 目標未達成: {ratio:.4f}% < {test['target_ratio']}%")
        
        # 99.9%達成チェック
        if ratio >= 99.9:
            print(f"🎉🏆💎 99.9%圧縮率達成! 完全成功!")
            break
        elif ratio >= 99.5:
            print(f"🎯💎 99.5%達成! 99.9%まであと少し!")
        elif ratio >= 99.0:
            print(f"🔶 99%達成! 更なる改良継続!")
        
        # 展開テスト（短時間）
        try:
            def decompression_test():
                engine = NEXUSExperimentalEngine()
                
                # 圧縮データのヘッダーを確認してルーティング
                if len(compressed) >= 4:
                    header = compressed[:4]
                    
                    # NEXUS形式の場合は通常の展開を使用
                    if header in [b'NXL8', b'NXL7']:
                        result, decomp_stats = engine.decompress(compressed)
                        return result, decomp_stats
                    # zlib_ultra_compress専用形式の場合
                    elif header in [b'BZ2Z', b'BZ2X', b'LZMA', b'3STG', b'ZLIB'] or method == 'zlib_ultra_compress':
                        start_time = time.time()
                        result = engine._zlib_ultra_decompress_optimized(compressed)
                        decomp_time = time.time() - start_time
                        decomp_stats = {
                            'speed_mbps': len(result) / (1024 * 1024) / decomp_time if decomp_time > 0 else 0
                        }
                        return result, decomp_stats
                
                # フォールバック: 通常の展開
                return engine.decompress(compressed)
            
            decompressed, decomp_stats = run_with_timeout_safe(decompression_test, 15)
            
            if sample_data == decompressed:
                print(f"🔍 展開検証: ✅ OK")
                print(f"⚡ 展開速度: {decomp_stats['speed_mbps']:.2f} MB/s")
            else:
                print(f"🔍 展開検証: ❌ データ不一致")
                
        except TimeoutError:
            print(f"⚠️ 展開タイムアウト（圧縮は成功）")
        except Exception as e:
            print(f"⚠️ 展開エラー: {e}")
        
    except TimeoutError:
        print(f"❌ 圧縮タイムアウト")
        print(f"💡 {test['sample_size']//1024//1024}MBで{test['timeout']}秒は不足")
        
        # より小さなサンプルで再試行
        if test['sample_size'] > 1024 * 1024:
            print(f"🔄 小サンプル再試行...")
            try:
                retry_size = min(1024 * 1024, test['sample_size'] // 2)
                retry_data = data[:retry_size]
                
                def retry_test():
                    engine = NEXUSExperimentalEngine()
                    return engine.compress(retry_data)
                
                retry_compressed, retry_stats = run_with_timeout_safe(retry_test, 15)
                print(f"🔄 小サンプル結果: {retry_stats['compression_ratio']:.4f}%")
                
            except Exception as e:
                print(f"❌ 再試行も失敗: {e}")
        
    except Exception as e:
        print(f"❌ テストエラー: {e}")

# 最終結果
print(f"\n" + "=" * 60)
print(f"🏆 段階的圧縮率改善テスト結果")
print(f"=" * 60)

if best_ratio > 0:
    print(f"🏆 最高圧縮率: {best_ratio:.4f}%")
    print(f"🏷️  最適手法: {best_method}")
    
    if best_ratio >= 99.9:
        print(f"🎉🏆💎 99.9%圧縮率達成! 完全成功!")
        print(f"✨ テキストファイル最高圧縮技術完成!")
    elif best_ratio >= 99.5:
        print(f"🎯💎 99.5%達成! 99.9%まであと{99.9-best_ratio:.4f}%!")
        print(f"🔧 更なる改良で99.9%到達可能!")
    elif best_ratio >= 99.0:
        print(f"🔶 99%達成! 良好な圧縮率!")
        print(f"📊 99.9%まであと{99.9-best_ratio:.4f}%の改良が必要")
    elif best_ratio >= 95.0:
        print(f"📊 95%達成! 標準的な圧縮率")
        print(f"🔧 大幅改良が必要（目標まで{99.9-best_ratio:.4f}%）")
    else:
        print(f"❌ 改良が必要（現在{best_ratio:.4f}%）")
else:
    print(f"❌ 圧縮テスト全て失敗")
    print(f"🔧 アルゴリズム見直しが必要")

print(f"\n💡 次のステップ:")
if best_ratio >= 99.9:
    print(f"✅ 99.9%達成済み - 実用化準備")
elif best_ratio >= 99.0:
    print(f"🔧 最終調整で99.9%達成")
    print(f"  - 多段圧縮パラメータ調整")
    print(f"  - チャンクサイズ最適化")
elif best_ratio >= 95.0:
    print(f"🔧 大幅改良が必要")
    print(f"  - 圧縮アルゴリズム追加")
    print(f"  - 前処理最適化")
else:
    print(f"🔧 基本設計見直し")
    print(f"  - アルゴリズム選択ロジック改良")
    print(f"  - テキスト特化最適化")

print(f"🔚 段階的テスト完了")
