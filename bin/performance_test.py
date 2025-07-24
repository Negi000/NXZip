#!/usr/bin/env python3
"""
NEXUS圧縮パフォーマンステスト
最適化前後の処理速度を比較
"""

import time
import sys
import os
from pathlib import Path

# パスの設定
sys.path.append(str(Path(__file__).parent))

from nexus_compression_engine import NEXUSCompressor

def test_compression_speed(file_path: str) -> dict:
    """圧縮速度テスト"""
    try:
        # ファイル読み込み
        with open(file_path, 'rb') as f:
            data = f.read()
        
        file_size = len(data)
        print(f"📁 テストファイル: {os.path.basename(file_path)} ({file_size:,} bytes)")
        
        # 圧縮器初期化
        start_init = time.time()
        compressor = NEXUSCompressor()
        init_time = time.time() - start_init
        
        # 圧縮テスト
        start_compress = time.time()
        try:
            compressed, state = compressor.nexus_compress_with_ml(data)
            compress_time = time.time() - start_compress
            
            # 結果計算
            compression_ratio = (1 - len(compressed) / file_size) * 100
            speed_mbps = (file_size / (1024 * 1024)) / compress_time
            
            return {
                'file_size': file_size,
                'compressed_size': len(compressed),
                'init_time': init_time,
                'compress_time': compress_time,
                'compression_ratio': compression_ratio,
                'speed_mbps': speed_mbps,
                'success': True
            }
            
        except Exception as e:
            return {
                'file_size': file_size,
                'init_time': init_time,
                'compress_time': 0,
                'error': str(e),
                'success': False
            }
            
    except FileNotFoundError:
        return {'error': 'ファイルが見つかりません', 'success': False}

def main():
    """メインテスト実行"""
    print("🚀 NEXUS圧縮パフォーマンステスト")
    print("=" * 50)
    
    # テストファイル一覧
    test_files = [
        'red_simple.png',
        'green_simple.png', 
        'small_test.png',
        'medium_test.png'
    ]
    
    results = []
    
    for file_name in test_files:
        if os.path.exists(file_name):
            print(f"\n🔍 テスト中: {file_name}")
            result = test_compression_speed(file_name)
            results.append((file_name, result))
            
            if result['success']:
                print(f"  ✅ 初期化: {result['init_time']:.3f}秒")
                print(f"  ⚡ 圧縮時間: {result['compress_time']:.3f}秒")
                print(f"  📊 圧縮率: {result['compression_ratio']:.1f}%")
                print(f"  🚄 処理速度: {result['speed_mbps']:.2f} MB/s")
                
                # 速度判定
                if result['compress_time'] < 1.0:
                    print("  🟢 高速処理")
                elif result['compress_time'] < 5.0:
                    print("  🟡 標準処理")
                else:
                    print("  🔴 低速処理")
            else:
                print(f"  ❌ エラー: {result.get('error', '不明')}")
        else:
            print(f"⚠️ ファイルが見つかりません: {file_name}")
    
    # サマリー表示
    print("\n" + "=" * 50)
    print("📋 処理速度サマリー")
    print("=" * 50)
    
    successful_results = [r for _, r in results if r['success']]
    
    if successful_results:
        avg_speed = sum(r['speed_mbps'] for r in successful_results) / len(successful_results)
        max_speed = max(r['speed_mbps'] for r in successful_results)
        min_speed = min(r['speed_mbps'] for r in successful_results)
        
        print(f"平均処理速度: {avg_speed:.2f} MB/s")
        print(f"最高処理速度: {max_speed:.2f} MB/s") 
        print(f"最低処理速度: {min_speed:.2f} MB/s")
        
        # 総合評価
        if avg_speed > 1.0:
            print("🎉 最適化により高速処理を実現！")
        elif avg_speed > 0.5:
            print("✅ 実用的な処理速度")
        else:
            print("⚠️ さらなる最適化が必要")
    else:
        print("❌ 有効な結果がありません")

if __name__ == "__main__":
    main()
