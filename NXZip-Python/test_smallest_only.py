#!/usr/bin/env python3
# 🚀 最小ファイル専用テスト - 基準クリア確認

import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

def run_with_timeout(func, timeout, *args, **kwargs):
    """タイムアウト付き実行"""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            print(f"⏰ タイムアウト: {timeout}秒で処理が完了しませんでした")
            future.cancel()
            return None
        except Exception as e:
            future.cancel()
            raise e

def test_smallest_file_with_criteria(filepath, nxzip):
    """最小ファイルで基準テスト"""
    filename = os.path.basename(filepath)
    
    try:
        # ファイル読み込み
        print("📖 ファイル読み込み中...")
        with open(filepath, 'rb') as f:
            data = f.read()
        
        file_size_mb = len(data) / (1024 * 1024)
        print(f"✅ 読み込み完了: {len(data):,} bytes ({file_size_mb:.2f} MB)")
        
        # 空ファイルチェック
        if len(data) == 0:
            print(f"❌ 空ファイルのためテストスキップ")
            return False
        
        # 圧縮テスト
        print(f"\n🗜️  圧縮実行中...")
        start_time = time.time()
        compressed, stats = nxzip.compress(data, filename)
        compression_time = time.time() - start_time
        
        # 圧縮結果
        ratio = stats['compression_ratio']
        speed = stats['speed_mbps']
        method = stats.get('method', 'unknown')
        
        print(f"✅ 圧縮完了: {compression_time:.2f}秒")
        print(f"📈 圧縮率: {ratio:.4f}%")
        print(f"🚀 圧縮速度: {speed:.2f} MB/s")
        print(f"🔧 使用手法: {method}")
        
        # 基準チェック1: 圧縮率（99%目標）
        if file_size_mb < 1.0:  # 1MB未満は圧縮率50%以上
            min_ratio = 50.0
        elif file_size_mb < 10.0:  # 10MB未満は圧縮率70%以上  
            min_ratio = 70.0
        elif file_size_mb >= 100.0:  # 100MB以上は99%目標
            min_ratio = 99.0
        else:  # 10MB〜100MBは85%以上
            min_ratio = 85.0
            
        if ratio < min_ratio:
            print(f"❌ 基準未達成: 圧縮率 {ratio:.2f}% < {min_ratio}%")
            return False
            
        # 展開テスト
        print(f"\n🔓 展開実行中...")
        start_time = time.time()
        decompressed, decomp_stats = nxzip.decompress(compressed)
        decompression_time = time.time() - start_time
        
        # 展開結果
        decomp_speed = decomp_stats['speed_mbps']
        print(f"✅ 展開完了: {decompression_time:.2f}秒")
        print(f"⚡ 展開速度: {decomp_speed:.2f} MB/s")
        
        # 基準チェック2: データ完全性
        if data != decompressed:
            print(f"❌ 基準未達成: データ不一致")
            return False
        
        print(f"🔍 データ検証: ✅ 完全一致!")
        
        # 基準チェック3: 処理時間（圧縮+展開で30秒以内、目標は2-3秒）
        total_time = compression_time + decompression_time
        if total_time > 30:
            print(f"❌ 基準未達成: 処理時間 {total_time:.2f}秒 > 30秒")
            return False
        
        # パフォーマンス評価
        if total_time <= 3:
            time_performance = "🏆 優秀 (≤3秒)"
        elif total_time <= 5:
            time_performance = "🎯 良好 (≤5秒)"  
        elif total_time <= 10:
            time_performance = "✅ 実用 (≤10秒)"
        else:
            time_performance = "⚠️ 要改善 (>10秒)"
        
        # 全基準クリア
        print(f"\n🎯 基準達成度:")
        print(f"  📈 圧縮率: ✅ {ratio:.2f}% (≥{min_ratio}%)")
        print(f"  🔍 完全性: ✅ 100%一致")
        print(f"  ⏱️ 処理時間: ✅ {total_time:.1f}秒 (≤30秒)")
        print(f"  🚀 総合速度: {file_size_mb/(total_time):.1f} MB/s")
        print(f"  ⭐ パフォーマンス: {time_performance}")
        
        return True
        
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        return False

def main():
    """指定された実際のファイルでの基準テスト"""
    # 指定されたファイルを直接テスト
    target_file = r"C:\Users\241822\Desktop\新しいフォルダー (2)\需要引当予測リスト クエリ.txt"
    
    print("🚀 指定ファイル基準テスト")
    print(f"📁 対象: 需要引当予測リスト クエリ.txt")
    print("🎯 基準: 圧縮成功、展開成功、処理時間30秒以内（目標2-3秒）")
    print("=" * 60)
    
    # ファイル存在確認
    if not os.path.exists(target_file):
        print(f"❌ ファイルが見つかりません: {target_file}")
        return False
    
    # ファイルサイズ確認
    try:
        file_size = os.path.getsize(target_file)
        size_mb = file_size / (1024 * 1024)
        print(f"� ファイルサイズ: {size_mb:.2f} MB ({file_size:,} bytes)")
    except Exception as e:
        print(f"❌ ファイルサイズ取得エラー: {e}")
        return False
    
    # NXZip初期化
    from nxzip.engine.nexus import NEXUSExperimentalEngine
    nxzip = NEXUSExperimentalEngine()
    
    # テスト実行（30秒タイムアウト）
    print(f"\n🚀 基準テスト開始 (タイムアウト: 30秒, 目標: 2-3秒)")
    print("=" * 40)
    
    try:
        success = run_with_timeout(test_smallest_file_with_criteria, 30, target_file, nxzip)
        
        if success:
            print(f"\n🏆 指定ファイル基準クリア!")
            print("=" * 40)
            print(f"✅ 需要引当予測リスト クエリ.txt ({size_mb:.2f} MB)")
            print(f"🎉 基準達成済み - より大きなファイルもテスト可能")
            print(f"💡 次回は中サイズファイルもテストしてください")
            return True
        else:
            print(f"\n❌ 指定ファイル基準未達成")
            print("=" * 40)
            print(f"🛠️ アルゴリズム改善が必要です")
            return False
            
    except Exception as e:
        print(f"\n❌ テスト実行エラー: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
