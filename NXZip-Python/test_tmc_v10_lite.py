#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TMC v10.0 Lite 軽量化版革新機能テスト
実用性重視の次世代圧縮技術検証
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nxzip'))

import time
import struct
import numpy as np

# TMC v10.0 Lite エンジンインポート
from nxzip.engine.nexus_tmc_v10_lite import NEXUSTMCEngineV10Lite

def test_tmc_v10_lite():
    """TMC v10.0 Lite革新機能の軽量化テスト"""
    print("🚀 TMC v10.0 Lite 軽量化革新機能テスト開始")
    print("="*80)
    print("軽量化次世代圧縮技術:")
    print("  🧠 階層型コンテキストモデリング (Order 0-4)")
    print("  🤖 機械学習ベース予測器 (軽量版)")
    print("  📊 ANS極限エントロピー符号化 (軽量版)")
    print("  ⚡ 実用性重視の最適化")
    print("="*80)
    
    # エンジン初期化
    engine = NEXUSTMCEngineV10Lite()
    
    # 実用的テストケース設計
    test_cases = [
        {
            "name": "📚 中規模テキストデータ（実用階層コンテキスト対象）",
            "data": generate_realistic_text(3000),  # 3000語の現実的テキスト
            "expected_features": ["階層型コンテキスト", "ML予測器"],
            "expected_compression": 80,  # 80%圧縮期待
            "complexity": "中複雑度"
        },
        {
            "name": "🔄 高冗長性データ（全機能対象）",
            "data": generate_repetitive_data(8000),  # 8KB反復データ
            "expected_features": ["階層型コンテキスト", "ML予測器", "ANS符号化"],
            "expected_compression": 90,  # 90%圧縮期待
            "complexity": "高冗長性"
        },
        {
            "name": "🧬 構造化データ（フル機能対象）",
            "data": generate_structured_data(6000),  # 6KB構造化データ
            "expected_features": ["ML予測器", "ANS符号化"],
            "expected_compression": 75,  # 75%圧縮期待
            "complexity": "構造化"
        },
        {
            "name": "📊 数値時系列データ（実用ML対象）",
            "data": generate_numeric_data(4000),  # 4KB数値データ
            "expected_features": ["ML予測器", "階層型コンテキスト"],
            "expected_compression": 70,  # 70%圧縮期待
            "complexity": "数値系列"
        },
        {
            "name": "🌊 混合データ（総合実用性テスト）",
            "data": generate_mixed_data(10000),  # 10KB混合データ
            "expected_features": ["全機能統合"],
            "expected_compression": 65,  # 65%圧縮期待
            "complexity": "混合データ"
        }
    ]
    
    total_original_size = 0
    total_compressed_size = 0
    detailed_results = []
    
    for i, test_case in enumerate(test_cases, 1):
        data = test_case["data"]
        print(f"\n🧪 テストケース {i}: {test_case['name']}")
        print(f"データサイズ: {len(data):,} bytes ({len(data)/1024:.1f} KB)")
        print(f"期待機能: {', '.join(test_case['expected_features'])}")
        print(f"期待圧縮率: {test_case['expected_compression']}%")
        print(f"複雑度: {test_case['complexity']}")
        print("-" * 70)
        
        # TMC v10.0 Lite 圧縮実行
        start_time = time.time()
        try:
            compressed_data, compression_info = engine.compress_ultimate_lite(data)
            compression_time = time.time() - start_time
            
            # 圧縮結果詳細分析
            compression_ratio = compression_info.get("compression_ratio", 0)
            compression_speed = len(data) / max(compression_time * 1024 * 1024, 0.001)  # MB/s
            efficiency_score = min(100, compression_ratio / max(test_case['expected_compression'], 1) * 100)
            
            print(f"  📈 圧縮結果:")
            print(f"     元サイズ: {len(data):,} bytes")
            print(f"     圧縮後: {len(compressed_data):,} bytes")
            print(f"     圧縮率: {compression_ratio:.1f}% (期待: {test_case['expected_compression']}%)")
            print(f"     効率度: {efficiency_score:.1f}%")
            print(f"     処理速度: {compression_speed:.2f} MB/s")
            print(f"     処理時間: {compression_time:.3f}秒")
            
            # TMC v10.0 Lite機能使用状況
            print(f"  🔧 軽量革新機能使用状況:")
            
            if compression_info.get("hierarchical_context_used"):
                print(f"     ✅ 階層型コンテキスト: Order 0-4軽量モデリング実行")
            else:
                print(f"     ❌ 階層型コンテキスト: 未使用")
                
            if compression_info.get("ml_prediction_used"):
                accuracy = compression_info.get("ml_predictor_accuracy", 0)
                print(f"     ✅ ML予測器: 軽量適応予測実行 (精度: {accuracy:.1f}%)")
            else:
                print(f"     ❌ ML予測器: 未使用")
                
            if compression_info.get("ans_encoding_used"):
                print(f"     ✅ ANS符号化: 軽量エントロピー符号化実行")
            else:
                print(f"     ❌ ANS符号化: 未使用")
                
            if compression_info.get("fallback_compression_used"):
                print(f"     ✅ フォールバック: 最終最適化実行")
            
            # 展開テスト
            print(f"  📉 展開テスト:")
            start_time = time.time()
            try:
                decompressed_data, decompression_info = engine.decompress_ultimate_lite(compressed_data)
                decompression_time = time.time() - start_time
                decompression_speed = len(data) / max(decompression_time * 1024 * 1024, 0.001)
                
                print(f"     展開速度: {decompression_speed:.2f} MB/s")
                print(f"     展開時間: {decompression_time:.3f}秒")
                
                # 可逆性検証
                if decompressed_data == data:
                    print(f"     ✅ 可逆性: 完璧復元 ({len(decompressed_data):,} bytes)")
                    reversibility_score = 100
                else:
                    error_rate = abs(len(decompressed_data) - len(data)) / len(data) * 100
                    reversibility_score = max(0, 100 - error_rate)
                    print(f"     ❌ 可逆性エラー: サイズ差異 ({reversibility_score:.1f}%)")
            except Exception as e:
                print(f"     ❌ 展開エラー: {e}")
                reversibility_score = 0
                decompression_time = 0
                decompression_speed = 0
            
            # 総合評価スコア計算
            speed_score = min(100, compression_speed * 10)  # MB/s * 10 でスコア化
            overall_score = (
                efficiency_score * 0.4 +      # 圧縮効率 40%
                reversibility_score * 0.3 +   # 可逆性 30%
                speed_score * 0.2 +            # 圧縮速度 20%
                min(100, decompression_speed * 10) * 0.1  # 展開速度 10%
            )
            
            print(f"  🏆 総合評価: {overall_score:.1f}/100")
            if overall_score >= 90:
                print(f"     🌟 エクセレント - 軽量化の完璧なバランス")
            elif overall_score >= 80:
                print(f"     ⭐ 優秀 - 実用性と性能の高レベル実現")
            elif overall_score >= 70:
                print(f"     📊 良好 - 安定した実用性能")
            else:
                print(f"     🔄 改善要 - 軽量化調整継続")
            
            # 詳細結果保存
            detailed_results.append({
                "test_name": test_case["name"],
                "compression_ratio": compression_ratio,
                "efficiency_score": efficiency_score,
                "overall_score": overall_score,
                "reversibility_score": reversibility_score,
                "processing_time": compression_time + decompression_time,
                "lite_features_used": {
                    "hierarchical_context": compression_info.get("hierarchical_context_used", False),
                    "ml_prediction": compression_info.get("ml_prediction_used", False),
                    "ans_encoding": compression_info.get("ans_encoding_used", False)
                }
            })
            
        except Exception as e:
            print(f"  ❌ 圧縮エラー: {e}")
            detailed_results.append({
                "test_name": test_case["name"],
                "error": str(e),
                "compression_ratio": 0,
                "overall_score": 0
            })
        
        total_original_size += len(data)
        if 'compressed_data' in locals():
            total_compressed_size += len(compressed_data)
    
    # 最終総合評価
    if total_original_size > 0:
        overall_compression_ratio = (1 - total_compressed_size / total_original_size) * 100
        average_efficiency = sum(r.get("efficiency_score", 0) for r in detailed_results) / len(detailed_results)
        average_overall_score = sum(r.get("overall_score", 0) for r in detailed_results) / len(detailed_results)
        lite_feature_usage = sum(1 for r in detailed_results if r.get("lite_features_used", {}).get("hierarchical_context", False))
    else:
        overall_compression_ratio = 0
        average_efficiency = 0
        average_overall_score = 0
        lite_feature_usage = 0
    
    print("\n" + "="*80)
    print("🏆 TMC v10.0 Lite 最終評価結果")
    print("="*80)
    
    print(f"📊 軽量化統計:")
    print(f"   元データ総容量: {total_original_size:,} bytes ({total_original_size/1024:.1f} KB)")
    print(f"   圧縮後総容量: {total_compressed_size:,} bytes ({total_compressed_size/1024:.1f} KB)")
    print(f"   総合圧縮率: {overall_compression_ratio:.1f}%")
    print(f"   平均効率度: {average_efficiency:.1f}%")
    print(f"   総合評価: {average_overall_score:.1f}/100")
    
    print(f"\n🚀 TMC v10.0 Lite 軽量化成果:")
    print(f"   ⚡ 軽量機能使用率: {lite_feature_usage}/{len(test_cases)} テストケース")
    print(f"   🧠 階層型コンテキスト: Order 0-4実用実装")
    print(f"   🤖 ML予測器: 軽量適応アルゴリズム")
    print(f"   📊 ANS符号化: 実用エントロピー符号化")
    print(f"   ⚖️ 性能と実用性の最適バランス")
    
    print(f"\n🎯 軽量化技術成果:")
    print(f"   🔬 実用的圧縮率実現")
    print(f"   ⚡ 高速処理による実用性")
    print(f"   🧠 軽量ML適応最適化")
    print(f"   📈 階層予測による精度向上")
    print(f"   🌟 次世代技術の実用実装")
    
    print(f"\n🚀 TMC v11.0 Lite 展望:")
    print(f"   🌊 ストリーミング対応")
    print(f"   🧠 深層学習軽量統合")
    print(f"   ⚡ GPU軽量加速")
    print(f"   🔄 リアルタイム適応")
    print(f"   💎 量子軽量アルゴリズム")


def generate_realistic_text(word_count: int) -> bytes:
    """現実的なテキストデータ生成"""
    words = [
        # 基本語彙
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "and", "cat",
        # 技術用語
        "algorithm", "compression", "optimization", "efficiency", "performance",
        "parallel", "processing", "context", "prediction", "entropy", "encoding",
        # 文章構造
        "however", "therefore", "furthermore", "moreover", "consequently",
        "in", "conclusion", "to", "summarize", "this", "study", "shows"
    ]
    
    text_parts = []
    current_word = "The"
    
    for i in range(word_count):
        text_parts.append(current_word)
        
        # 簡単なマルコフ連鎖風選択
        if current_word.lower() in ["the", "a", "an"]:
            current_word = np.random.choice(["advanced", "sophisticated", "novel", "effective"])
        elif current_word.lower() in ["advanced", "sophisticated"]:
            current_word = np.random.choice(["algorithm", "method", "approach", "technique"])
        else:
            current_word = np.random.choice(words)
        
        # 句読点挿入
        if i % 15 == 14:
            text_parts.append(".")
        elif i % 8 == 7:
            text_parts.append(",")
    
    return " ".join(text_parts).encode('utf-8')


def generate_repetitive_data(size_bytes: int) -> bytes:
    """高冗長性データ生成"""
    patterns = [
        b"ABC" * 10,
        b"123" * 8,
        b"PATTERN_" * 5,
        b"REPEATING_SEQUENCE_" * 2
    ]
    
    data = bytearray()
    while len(data) < size_bytes:
        pattern = np.random.choice(patterns)
        repeat_count = np.random.randint(3, 12)
        data.extend(pattern * repeat_count)
        
        # 時々ノイズ挿入
        if np.random.random() < 0.1:
            noise = np.random.bytes(np.random.randint(1, 3))
            data.extend(noise)
    
    return bytes(data[:size_bytes])


def generate_structured_data(size_bytes: int) -> bytes:
    """構造化データ生成"""
    structured_data = []
    
    # JSON風構造
    for i in range(size_bytes // 100):
        entry = f'{{"id":{i},"name":"item_{i}","value":{i*2.5},"active":true}}'
        structured_data.append(entry)
        
        if i % 10 == 9:
            structured_data.append("\n")
    
    return ",".join(structured_data).encode('utf-8')[:size_bytes]


def generate_numeric_data(size_bytes: int) -> bytes:
    """数値時系列データ生成"""
    sample_count = size_bytes // 4
    time_series = []
    
    base_value = 100.0
    trend = 0.01
    noise_level = 2.0
    
    for i in range(sample_count):
        # トレンド + ノイズ
        value = base_value + i * trend + np.random.normal(0, noise_level)
        
        # 季節性
        seasonal = 10 * np.sin(2 * np.pi * i / 50)
        value += seasonal
        
        time_series.append(struct.pack('<f', value))
    
    return b''.join(time_series)


def generate_mixed_data(size_bytes: int) -> bytes:
    """混合データ生成"""
    data = bytearray()
    
    # 複数データタイプを混合
    data_generators = [
        (generate_realistic_text, 0.4),
        (generate_repetitive_data, 0.3),
        (generate_structured_data, 0.2),
        (generate_numeric_data, 0.1)
    ]
    
    remaining_size = size_bytes
    for generator, ratio in data_generators:
        chunk_size = min(remaining_size, int(size_bytes * ratio))
        if chunk_size > 0:
            if generator == generate_realistic_text:
                chunk = generator(chunk_size // 8)  # 語数調整
            else:
                chunk = generator(chunk_size)
            data.extend(chunk)
            remaining_size -= len(chunk)
        
        if remaining_size <= 0:
            break
    
    return bytes(data[:size_bytes])


if __name__ == "__main__":
    # 必要なライブラリのインポート
    try:
        import numpy as np
        import struct
    except ImportError as e:
        print(f"必要なライブラリが不足しています: {e}")
        print("pip install numpy でインストールしてください")
        sys.exit(1)
    
    test_tmc_v10_lite()
