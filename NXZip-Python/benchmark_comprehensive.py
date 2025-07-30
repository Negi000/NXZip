#!/usr/bin/env python3
"""
NEXUS TMC v9.0 - 包括的ベンチマークテスト
100%可逆性保証版の性能評価
"""

import sys
import os
import time
import json
import numpy as np
sys.path.append(os.path.dirname(__file__))

from nxzip.engine.nexus_tmc_v4_unified import NEXUSTMCEngineV9

def format_size(size_bytes):
    """バイトサイズを読みやすい形式に変換"""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f}{size_names[i]}"

def calculate_compression_ratio(original_size, compressed_size):
    """圧縮率を%で計算（数値が小さいほど高圧縮）"""
    if original_size == 0:
        return 0.0
    return (compressed_size / original_size) * 100

def main():
    print("=" * 80)
    print("🚀 NEXUS TMC v9.0 - 包括的ベンチマークテスト")
    print("=" * 80)
    print("🎯 100%可逆性保証版の性能評価")
    print("📊 圧縮率: %表記（数値が小さいほど高圧縮）")
    print()

    # TMCエンジン初期化
    engine = NEXUSTMCEngineV9()

    # 包括的テストケース
    test_cases = {
        # 小データ（高速パス）
        "JSON小": {
            "data": json.dumps({"name": "test", "value": 123}, ensure_ascii=False).encode('utf-8'),
            "category": "小データ（高速パス）"
        },
        "XML小": {
            "data": '<?xml version="1.0"?><root><item>test</item></root>'.encode('utf-8'),
            "category": "小データ（高速パス）"
        },
        "Binary小": {
            "data": bytes(range(256)),
            "category": "小データ（高速パス）"
        },

        # 構造化データ
        "JSON中規模": {
            "data": json.dumps({
                "users": [{"id": i, "name": f"user{i}", "email": f"user{i}@example.com"} for i in range(100)],
                "settings": {"theme": "dark", "language": "ja", "version": "1.0"}
            }, ensure_ascii=False).encode('utf-8'),
            "category": "構造化データ"
        },
        "CSV形式": {
            "data": "\n".join([f"{i},user{i},user{i}@example.com,{i*1000}" for i in range(500)]).encode('utf-8'),
            "category": "構造化データ"
        },

        # テキストデータ
        "英語テキスト": {
            "data": ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 200).encode('utf-8'),
            "category": "テキストデータ"
        },
        "日本語テキスト": {
            "data": ("これは日本語のテストデータです。圧縮率を測定しています。" * 100).encode('utf-8'),
            "category": "テキストデータ"
        },
        "プログラムコード": {
            "data": '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

for i in range(20):
    print(f"fib({i}) = {fibonacci(i)}")
'''.encode('utf-8') * 50,
            "category": "テキストデータ"
        },

        # 数値データ
        "浮動小数点配列": {
            "data": np.linspace(0, 1000, 5000, dtype=np.float32).tobytes(),
            "category": "数値データ"
        },
        "整数シーケンス": {
            "data": np.arange(0, 10000, dtype=np.int32).tobytes(),
            "category": "数値データ"
        },
        "時系列データ": {
            "data": (np.sin(np.linspace(0, 100*np.pi, 8000)) * 1000).astype(np.int16).tobytes(),
            "category": "数値データ"
        },

        # パターンデータ
        "反復パターン": {
            "data": b"ABCDEFGH" * 500,
            "category": "パターンデータ"
        },
        "バイナリパターン": {
            "data": bytes([i % 256 for i in range(2000)]),
            "category": "パターンデータ"
        },

        # 混合データ
        "ゼロバイト混合": {
            "data": b'\x00' * 100 + b'DATA' * 50 + b'\x00' * 100,
            "category": "混合データ"
        },
        "Unicode混合": {
            "data": ("Hello世界🌍Test™®©αβγδε" * 100).encode('utf-8'),
            "category": "混合データ"
        },

        # 大容量データ
        "大容量テキスト": {
            "data": ("This is a large text data for compression benchmark testing. " * 1000).encode('utf-8'),
            "category": "大容量データ"
        },
        "大容量数値": {
            "data": np.random.randint(0, 1000000, 20000, dtype=np.int32).tobytes(),
            "category": "大容量データ"
        },

        # ランダムデータ（圧縮困難）
        "ランダムバイト": {
            "data": np.random.bytes(2000),
            "category": "ランダムデータ"
        }
    }

    # 結果収集
    results_by_category = {}
    total_original_size = 0
    total_compressed_size = 0
    total_compression_time = 0
    total_decompression_time = 0
    reversibility_success = 0
    total_tests = len(test_cases)

    print("🧪 ベンチマークテスト実行中...")
    print("-" * 80)

    for test_name, test_info in test_cases.items():
        test_data = test_info["data"]
        category = test_info["category"]
        
        print(f"📋 テスト: {test_name}")
        print(f"   カテゴリ: {category}")
        print(f"   データサイズ: {format_size(len(test_data))}")
        
        try:
            # 圧縮テスト
            compress_start = time.time()
            compressed_result = engine.compress_tmc(test_data)
            compress_time = time.time() - compress_start
            
            if isinstance(compressed_result, tuple):
                compressed_data, compression_info = compressed_result
            else:
                compressed_data = compressed_result
                compression_info = {}
            
            # 解凍テスト
            decompress_start = time.time()
            decompress_result = engine.decompress_tmc(compressed_data)
            decompress_time = time.time() - decompress_start
            
            if isinstance(decompress_result, tuple):
                decompressed_data, decompress_info = decompress_result
            else:
                decompressed_data = decompress_result
            
            # 可逆性検証
            is_reversible = test_data == decompressed_data
            if is_reversible:
                reversibility_success += 1
            
            # 圧縮率計算
            compression_ratio = calculate_compression_ratio(len(test_data), len(compressed_data))
            
            # 結果表示
            print(f"   圧縮率: {compression_ratio:.2f}% ({format_size(len(test_data))} → {format_size(len(compressed_data))})")
            print(f"   圧縮時間: {compress_time*1000:.2f}ms")
            print(f"   解凍時間: {decompress_time*1000:.2f}ms")
            print(f"   可逆性: {'✅ 成功' if is_reversible else '❌ 失敗'}")
            print(f"   メソッド: {compression_info.get('method', 'unknown')}")
            
            # カテゴリ別結果集計
            if category not in results_by_category:
                results_by_category[category] = {
                    'tests': [],
                    'total_original': 0,
                    'total_compressed': 0,
                    'total_compress_time': 0,
                    'total_decompress_time': 0,
                    'reversible_count': 0
                }
            
            cat_results = results_by_category[category]
            cat_results['tests'].append({
                'name': test_name,
                'original_size': len(test_data),
                'compressed_size': len(compressed_data),
                'compression_ratio': compression_ratio,
                'compress_time': compress_time,
                'decompress_time': decompress_time,
                'reversible': is_reversible,
                'method': compression_info.get('method', 'unknown')
            })
            cat_results['total_original'] += len(test_data)
            cat_results['total_compressed'] += len(compressed_data)
            cat_results['total_compress_time'] += compress_time
            cat_results['total_decompress_time'] += decompress_time
            if is_reversible:
                cat_results['reversible_count'] += 1
            
            # 全体統計
            total_original_size += len(test_data)
            total_compressed_size += len(compressed_data)
            total_compression_time += compress_time
            total_decompression_time += decompress_time
            
        except Exception as e:
            print(f"   ❌ エラー: {e}")
        
        print("-" * 80)

    # 総合結果表示
    print("\n" + "=" * 80)
    print("📊 ベンチマーク結果サマリ")
    print("=" * 80)
    
    overall_compression_ratio = calculate_compression_ratio(total_original_size, total_compressed_size)
    reversibility_rate = (reversibility_success / total_tests) * 100
    
    print(f"🎯 総合成績:")
    print(f"   全体圧縮率: {overall_compression_ratio:.2f}% ({format_size(total_original_size)} → {format_size(total_compressed_size)})")
    print(f"   可逆性成功率: {reversibility_rate:.1f}% ({reversibility_success}/{total_tests})")
    print(f"   合計圧縮時間: {total_compression_time*1000:.1f}ms")
    print(f"   合計解凍時間: {total_decompression_time*1000:.1f}ms")
    print(f"   平均圧縮速度: {(total_original_size/1024/1024)/total_compression_time:.1f} MB/s")
    print(f"   平均解凍速度: {(total_original_size/1024/1024)/total_decompression_time:.1f} MB/s")
    print()

    # カテゴリ別結果
    print("📋 カテゴリ別詳細結果:")
    print("-" * 80)
    
    for category, results in results_by_category.items():
        cat_compression_ratio = calculate_compression_ratio(results['total_original'], results['total_compressed'])
        cat_reversibility_rate = (results['reversible_count'] / len(results['tests'])) * 100
        
        print(f"🏷️  {category}:")
        print(f"   テスト数: {len(results['tests'])}")
        print(f"   カテゴリ圧縮率: {cat_compression_ratio:.2f}%")
        print(f"   可逆性: {cat_reversibility_rate:.1f}%")
        print(f"   合計サイズ: {format_size(results['total_original'])} → {format_size(results['total_compressed'])}")
        
        # 最高・最低圧縮率
        if results['tests']:
            best_test = min(results['tests'], key=lambda x: x['compression_ratio'])
            worst_test = max(results['tests'], key=lambda x: x['compression_ratio'])
            print(f"   最高圧縮: {best_test['name']} ({best_test['compression_ratio']:.2f}%)")
            print(f"   最低圧縮: {worst_test['name']} ({worst_test['compression_ratio']:.2f}%)")
        print()

    # パフォーマンス評価
    print("⚡ パフォーマンス評価:")
    print("-" * 80)
    
    if overall_compression_ratio < 30:
        compression_grade = "🏆 優秀"
    elif overall_compression_ratio < 50:
        compression_grade = "🥈 良好"
    elif overall_compression_ratio < 70:
        compression_grade = "🥉 普通"
    else:
        compression_grade = "⚠️  要改善"
    
    if reversibility_rate == 100:
        reversibility_grade = "🏆 完璧"
    elif reversibility_rate >= 95:
        reversibility_grade = "🥈 優秀"
    elif reversibility_rate >= 90:
        reversibility_grade = "🥉 良好"
    else:
        reversibility_grade = "⚠️  要改善"
    
    avg_throughput = (total_original_size/1024/1024)/(total_compression_time + total_decompression_time)
    if avg_throughput > 50:
        speed_grade = "🏆 高速"
    elif avg_throughput > 20:
        speed_grade = "🥈 良好"
    elif avg_throughput > 10:
        speed_grade = "🥉 普通"
    else:
        speed_grade = "⚠️  低速"
    
    print(f"圧縮性能: {compression_grade} (圧縮率 {overall_compression_ratio:.2f}%)")
    print(f"可逆性: {reversibility_grade} (成功率 {reversibility_rate:.1f}%)")
    print(f"処理速度: {speed_grade} (平均 {avg_throughput:.1f} MB/s)")
    print()

    # 推奨事項
    print("💡 推奨事項:")
    print("-" * 80)
    
    recommendations = []
    
    if overall_compression_ratio > 50:
        recommendations.append("• より高い圧縮率が期待される場合は、データタイプ別の専用アルゴリズムの調整を検討")
    
    if reversibility_rate < 100:
        recommendations.append(f"• 可逆性が{reversibility_rate:.1f}%のため、失敗ケースの詳細調査が必要")
    
    if avg_throughput < 20:
        recommendations.append("• 処理速度向上のため、並列処理の最適化やアルゴリズムチューニングを検討")
    
    # データタイプ別推奨
    text_categories = ["テキストデータ", "構造化データ"]
    numeric_categories = ["数値データ"]
    
    for category in text_categories:
        if category in results_by_category:
            cat_ratio = calculate_compression_ratio(
                results_by_category[category]['total_original'],
                results_by_category[category]['total_compressed']
            )
            if cat_ratio > 40:
                recommendations.append(f"• {category}の圧縮率({cat_ratio:.1f}%)改善のため、BWT/MTF変換の調整を検討")
    
    for category in numeric_categories:
        if category in results_by_category:
            cat_ratio = calculate_compression_ratio(
                results_by_category[category]['total_original'],
                results_by_category[category]['total_compressed']
            )
            if cat_ratio > 30:
                recommendations.append(f"• {category}の圧縮率({cat_ratio:.1f}%)改善のため、予測符号化の精度向上を検討")
    
    if not recommendations:
        recommendations.append("• 現在の性能は良好です。継続的な監視とテストケース拡充を推奨")
    
    for rec in recommendations:
        print(rec)
    
    print("\n" + "=" * 80)
    print(f"✅ ベンチマークテスト完了 - TMC v9.0エンジン評価結果")
    print("=" * 80)

if __name__ == '__main__':
    main()
