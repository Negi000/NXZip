#!/usr/bin/env python3
# 🚀 NEXUS Experimental v8.1 - 実戦対応改良版テスト（30秒強制タイムアウト付き）

import time
import threading
import os
import multiprocessing
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

def run_with_timeout(func, timeout_seconds=30):
    """強制タイムアウト付き関数実行 - multiprocessing版"""
    
    def target(queue):
        try:
            result = func()
            queue.put(('success', result))
        except Exception as e:
            queue.put(('error', str(e)))
    
    # マルチプロセス実行
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=target, args=(queue,))
    process.start()
    
    try:
        process.join(timeout=timeout_seconds)
        
        if process.is_alive():
            print(f"⏰ 強制終了: {timeout_seconds}秒でプロセスを終了します")
            process.terminate()
            process.join(timeout=5)
            
            if process.is_alive():
                print(f"💀 強制kill: プロセスを強制終了します")
                process.kill()
                process.join()
                
            raise TimeoutError(f"処理が{timeout_seconds}秒を超過しました（強制終了実行）")
        
        if not queue.empty():
            status, result = queue.get()
            if status == 'success':
                return result
            else:
                raise Exception(result)
        else:
            raise Exception("プロセスが結果を返しませんでした")
            
    finally:
        if process.is_alive():
            process.terminate()

def run_with_timeout_thread(func, timeout_seconds=30):
    """ThreadPoolExecutor版タイムアウト（フォールバック用）"""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError:
            print(f"⏰ タイムアウト警告: {timeout_seconds}秒で処理を中断しました")
            raise TimeoutError(f"処理が{timeout_seconds}秒を超過しました（Thread版）")

print("🚀 NEXUS Experimental v8.1 - 実戦対応改良版テスト（強制タイムアウト対応）")
print("=" * 70)
print("⏰ 30秒強制タイムアウト機能: multiprocessing + 強制終了対応")

# 実戦ファイル
real_file_path = r"C:\Users\241822\Desktop\新しいフォルダー (2)\出庫実績明細_202410.tsv"

if not os.path.exists(real_file_path):
    print(f"❌ ファイルが見つかりません")
    exit(1)

file_size = os.path.getsize(real_file_path)
print(f"📊 実戦ファイル: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")

print("📖 ファイル読み込み中...")

try:
    with open(real_file_path, 'rb') as f:
        data = f.read()
    print(f"✅ 読み込み完了: {len(data):,} bytes")
except FileNotFoundError:
    print(f"❌ ファイルが見つかりません: {real_file_path}")
    exit(1)

print("\n🧪 改良版エンジンテスト開始（強制タイムアウト対応）")

# NEXUS実験エンジンのインポート
from nxzip.engine.nexus_experimental import NEXUSExperimentalEngine

def create_experimental_engine():
    """エンジン作成（プロセス分離対応）"""
    return NEXUSExperimentalEngine()

def test_method_selection():
    """手法選択テスト（プロセス分離対応）"""
    engine = NEXUSExperimentalEngine()
    return engine._instant_method_selection(data)

print("🎯 手法選択テスト...")
try:
    method = run_with_timeout(test_method_selection, 10)
    print(f"🎯 選択手法: {method} (実戦大容量対応版)")
except TimeoutError as e:
    print(f"❌ 手法選択タイムアウト: {e}")
    # フォールバック手法
    method = 'zlib_ultra_compress'
    print(f"🔄 フォールバック手法: {method}")
except Exception as e:
    print(f"❌ 手法選択エラー: {e}")
    method = 'zlib_ultra_compress'
    print(f"🔄 フォールバック手法: {method}")

print(f"🗜️  改良版圧縮実行（30秒タイムアウト）...")
start_time = time.time()

try:
    # 30秒タイムアウト付き圧縮（プロセス分離版）
    def compress_task():
        engine = NEXUSExperimentalEngine()
        return engine.compress(data)
    
    compressed_exp, stats_exp = run_with_timeout(compress_task, 30)
    
    compression_time = time.time() - start_time
    compression_ratio = stats_exp['compression_ratio']
    compression_speed = stats_exp['speed_mbps']
    
    print(f"✅ 圧縮完了!")
    print(f"📊 圧縮率: {compression_ratio:.2f}%")
    print(f"🚀 圧縮速度: {compression_speed:.2f} MB/s")
    print(f"⏱️  圧縮時間: {compression_time:.3f}秒")
    print(f"🏷️  使用手法: {stats_exp['method']}")
    
    # 圧縮率評価
    if compression_ratio >= 99.9:
        print(f"🎉🏆 圧縮率目標達成! 99.9%超え!")
    elif compression_ratio >= 99.0:
        print(f"🎯 圧縮率99%超え（99.9%まであと少し）")
    elif compression_ratio >= 95.0:
        print(f"🔶 圧縮率良好（95%超え）")
    else:
        print(f"📊 圧縮率要改善（{compression_ratio:.2f}% < 95%）")
    
    print(f"⚡ 改良版展開実行（30秒タイムアウト）...")
    start_time = time.time()
    
    try:
        # 30秒タイムアウト付き展開（プロセス分離版）
        def decompress_task():
            engine = NEXUSExperimentalEngine()
            return engine.decompress(compressed_exp)
        
        decompressed_exp, decomp_stats_exp = run_with_timeout(decompress_task, 30)
        
        decompression_time = time.time() - start_time
        decompression_speed = decomp_stats_exp['speed_mbps']
        
        print(f"✅ 展開完了!")
        print(f"⚡ 展開速度: {decompression_speed:.2f} MB/s")
        print(f"⏱️  展開時間: {decompression_time:.3f}秒")
        
        # データ検証
        if data == decompressed_exp:
            print(f"🔍 データ検証: ✅ OK")
            print(f"🎯 性能等級: {decomp_stats_exp.get('performance_grade', 'N/A')}")
            
            print(f"\n🏆 実戦対応改良版結果（強制タイムアウト対応）:")
            print(f"🎯 手法選択: {method} ← 実戦大容量モード")
            print(f"🚀 圧縮性能: {compression_speed:.1f} MB/s")
            print(f"⚡ 展開性能: {decompression_speed:.1f} MB/s")
            print(f"📊 圧縮率: {compression_ratio:.4f}%")
            
            # 99.9%圧縮率評価
            if compression_ratio >= 99.9:
                print(f"🎉🏆💎 99.9%圧縮率達成! テキスト最高圧縮成功!")
            elif compression_ratio >= 99.5:
                print(f"🎯💎 99.5%圧縮率達成! 99.9%まであと少し!")
            elif compression_ratio >= 99.0:
                print(f"🔶 99%圧縮率達成! 更なる改良継続!")
            else:
                print(f"📊 圧縮率要改善: {compression_ratio:.2f}% → 99.9%目標")
            
            if decompression_speed >= 200:
                print(f"🎉 展開速度目標達成! {decompression_speed:.0f} MB/s")
            elif decompression_speed >= 150:
                print(f"🔶 展開速度良好! {decompression_speed:.0f} MB/s")
            else:
                print(f"📊 展開速度要改善: {decompression_speed:.0f} MB/s")
                
            if compression_speed >= 100:
                print(f"🎉 圧縮速度目標達成! {compression_speed:.0f} MB/s")
            elif compression_speed >= 75:
                print(f"🔶 圧縮速度良好! {compression_speed:.0f} MB/s")
            else:
                print(f"📊 圧縮速度要改善: {compression_speed:.0f} MB/s")
                
        else:
            print(f"🔍 データ検証: ❌ NG - データ不一致")
            
    except TimeoutError as e:
        print(f"❌ 展開タイムアウト（強制終了実行）: {e}")
        print(f"💡 圧縮は成功しましたが、展開に時間がかかりすぎています")
    except Exception as e:
        print(f"❌ 展開エラー: {e}")
        
except TimeoutError as e:
    print(f"❌ 圧縮タイムアウト（強制終了実行）: {e}")
    print(f"💡 99.9%圧縮率モードは処理時間が長いため、タイムアウト対策を検討します")
    
    # フォールバック圧縮テスト
    print(f"\n🔄 フォールバック圧縮テスト（ThreadPoolExecutor版）")
    try:
        def fallback_compress_task():
            engine = NEXUSExperimentalEngine()
            # より軽量な手法で圧縮
            temp_data = data[:1024*1024] if len(data) > 1024*1024 else data  # 1MB制限
            return engine.compress(temp_data)
        
        fb_compressed, fb_stats = run_with_timeout_thread(fallback_compress_task, 15)
        print(f"🔄 フォールバック圧縮率: {fb_stats['compression_ratio']:.2f}%")
        print(f"🔄 フォールバック速度: {fb_stats['speed_mbps']:.2f} MB/s")
    except Exception as fb_e:
        print(f"❌ フォールバックも失敗: {fb_e}")
        
except Exception as e:
    print(f"❌ 圧縮エラー: {e}")

print(f"🔚 実戦対応改良版テスト完了（強制タイムアウト対応）")
