#!/usr/bin/env python3
"""
NEXUS TMC モジュール分離テスト

分離したモジュールの正常動作を確認します。
"""

import sys
import os

# パスの設定
sys.path.insert(0, os.path.dirname(__file__))

def test_core_modules():
    """Core モジュールのテスト"""
    print("🧪 Core モジュール テスト開始")
    
    try:
        # Data types のインポートテスト
        from nxzip.engine.core.data_types import DataType, ChunkInfo, PipelineStage
        print("  ✅ data_types モジュール: インポート成功")
        
        # DataType enum のテスト
        data_type = DataType.TEXT_REPETITIVE
        print(f"  ✅ DataType enum: {data_type.value}")
        
        # ChunkInfo dataclass のテスト
        chunk = ChunkInfo(
            chunk_id=1,
            original_size=1024,
            compressed_size=512,
            data_type="test",
            compression_ratio=0.5,
            processing_time=0.1
        )
        print(f"  ✅ ChunkInfo dataclass: {chunk.chunk_id}")
        
        # Memory manager のインポートテスト
        from nxzip.engine.core.memory_manager import MemoryManager, MEMORY_MANAGER
        print("  ✅ memory_manager モジュール: インポート成功")
        
        # Memory manager のテスト
        stats = MEMORY_MANAGER.get_memory_stats()
        print(f"  ✅ MemoryManager: {stats.get('optimization_status', 'OK')}")
        
        print("✅ Core モジュール: 全テスト成功\n")
        return True
        
    except Exception as e:
        print(f"❌ Core モジュール テストエラー: {e}\n")
        return False


def test_analyzer_modules():
    """Analyzers モジュールのテスト"""
    print("🧪 Analyzers モジュール テスト開始")
    
    try:
        # Entropy calculator のインポートテスト
        from nxzip.engine.analyzers.entropy_calculator import calculate_entropy
        print("  ✅ entropy_calculator モジュール: インポート成功")
        
        # Entropy計算のテスト
        test_data = b"Hello, World! This is a test string for entropy calculation."
        entropy = calculate_entropy(test_data)
        print(f"  ✅ Entropy calculation: {entropy:.3f} bits")
        
        # Meta analyzer のインポートテスト  
        from nxzip.engine.analyzers.meta_analyzer import MetaAnalyzer
        print("  ✅ meta_analyzer モジュール: インポート成功")
        
        print("✅ Analyzers モジュール: 全テスト成功\n")
        return True
        
    except Exception as e:
        print(f"❌ Analyzers モジュール テストエラー: {e}\n")
        return False


def test_module_integration():
    """統合インポートテスト"""
    print("🧪 統合インポート テスト開始")
    
    try:
        # 統合インポートテスト
        from nxzip.engine.core import MemoryManager, DataType, ChunkInfo
        from nxzip.engine.analyzers import calculate_entropy, MetaAnalyzer
        
        print("  ✅ 統合インポート: 成功")
        print("  ✅ 分離されたモジュール間の依存関係: 正常")
        
        print("✅ 統合テスト: 全テスト成功\n")
        return True
        
    except Exception as e:
        print(f"❌ 統合テスト エラー: {e}\n")
        return False


def main():
    """メインテスト実行"""
    print("🚀 NEXUS TMC モジュール分離テスト 開始")
    print("=" * 50)
    
    results = []
    
    # Core モジュールテスト
    results.append(test_core_modules())
    
    # Analyzers モジュールテスト  
    results.append(test_analyzer_modules())
    
    # 統合テスト
    results.append(test_module_integration())
    
    # 結果集計
    print("=" * 50)
    print("📊 テスト結果:")
    success_count = sum(results)
    total_count = len(results)
    
    if success_count == total_count:
        print(f"🎉 全テスト成功! ({success_count}/{total_count})")
        print("✅ モジュール分離は正常に完了しました")
        print("🚀 次のフェーズ（Transforms分離）に進めます")
    else:
        print(f"⚠️ 一部テスト失敗 ({success_count}/{total_count})")
        print("🔧 問題を修正してから次のフェーズに進んでください")
    
    return success_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
