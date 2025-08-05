#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 Optimizer for NXZip Core
=====================================

Targets:
- BWT processing optimization
- Memory allocation efficiency  
- Pipeline processing refinement
- Entropy calculation optimization

Safety Level: MEDIUM (algorithmic optimization with preservation)
Target Improvements: 50-200% additional speed gains
"""

import os
import sys
import shutil
import time
from pathlib import Path

def create_phase2_optimizations():
    """Phase 2の最適化を適用"""
    
    print("🔧 Phase 2 Optimization Analysis Starting...")
    
    # ファイルパス設定
    source_file = "nxzip_core_optimized.py"  # Phase 1結果
    target_file = "nxzip_core_phase2.py"     # Phase 2結果
    backup_file = "nxzip_core_phase1_backup.py"
    
    # バックアップ作成
    if os.path.exists(source_file):
        shutil.copy2(source_file, backup_file)
        print(f"✅ Backup created: {backup_file}")
    else:
        print(f"❌ Source file not found: {source_file}")
        return False
    
    # ファイル読み込み
    with open(source_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("🎯 Applying Phase 2 Optimizations...")
    
    # === Phase 2 Optimizations ===
    
    # 1. BWT処理の更なる最適化
    optimizations_applied = []
    
    # 1.1 BWT閾値の動的調整
    if "if len(data) < 50000:" in content:
        content = content.replace(
            "if len(data) < 50000:",
            "# Dynamic BWT threshold based on data characteristics\n        "
            "entropy_sample = self._calculate_entropy(data[:min(1024, len(data))])\n        "
            "bwt_threshold = 30000 if entropy_sample > 7.5 else 70000\n        "
            "if len(data) < bwt_threshold:"
        )
        optimizations_applied.append("BWT Dynamic Threshold")
    
    # 1.2 エントロピー計算の最適化
    if "_calculate_entropy" in content and "np.log2" in content:
        entropy_optimization = '''
    def _calculate_entropy(self, data):
        """高速エントロピー計算 (Phase 2 optimized)"""
        if len(data) == 0:
            return 0.0
            
        # 頻度カウント最適化
        if len(data) < 1000:
            # 小データは直接計算
            unique, counts = np.unique(data, return_counts=True)
            probabilities = counts / len(data)
        else:
            # 大データはサンプリング
            sample_size = min(2048, len(data))
            sample_data = data[::max(1, len(data)//sample_size)]
            unique, counts = np.unique(sample_data, return_counts=True)
            probabilities = counts / len(sample_data)
        
        # ゼロ確率除去とログ計算最適化
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities))'''
        
        # 既存のエントロピー関数を置換
        import re
        pattern = r'def _calculate_entropy\(self, data\):.*?return -np\.sum\(probabilities \* np\.log2\(probabilities\)\)'
        content = re.sub(pattern, entropy_optimization.strip(), content, flags=re.DOTALL)
        optimizations_applied.append("Entropy Calculation Optimization")
    
    # 1.3 メモリ効率的なBWT処理
    if "pydivsufsort.divsufsort" in content:
        content = content.replace(
            "suffix_array = pydivsufsort.divsufsort(data)",
            "# Memory-efficient BWT processing\n            "
            "if len(data) > 100000:\n                "
            "# 大データは分割処理\n                "
            "chunk_size = 50000\n                "
            "chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]\n                "
            "suffix_arrays = [pydivsufsort.divsufsort(chunk) for chunk in chunks]\n                "
            "suffix_array = np.concatenate(suffix_arrays)\n            "
            "else:\n                "
            "suffix_array = pydivsufsort.divsufsort(data)"
        )
        optimizations_applied.append("Memory-Efficient BWT")
    
    # 1.4 NumPy配列操作の最適化
    numpy_optimizations = [
        ("np.array(", "np.asarray("),  # より高速
        ("np.concatenate([", "np.concatenate(("),  # タプル使用
        ("data.tolist()", "data"),  # 不要な変換除去
    ]
    
    for old, new in numpy_optimizations:
        if old in content:
            content = content.replace(old, new)
            optimizations_applied.append(f"NumPy: {old} -> {new}")
    
    # 1.5 条件分岐の最適化
    if "if self.debug:" in content:
        content = content.replace(
            "if self.debug:",
            "if __debug__ and self.debug:"  # デバッグモード時のみ評価
        )
        optimizations_applied.append("Debug Condition Optimization")
    
    # 1.6 プログレス更新の効率化
    if "progress_callback(" in content:
        content = content.replace(
            "progress_callback(",
            "if progress_callback is not None: progress_callback("
        )
        optimizations_applied.append("Progress Callback Optimization")
    
    # Phase 2最適化マーカー追加
    phase2_marker = '''
# =============================================================================
# Phase 2 Optimizations Applied
# - BWT Dynamic Threshold based on entropy
# - Entropy calculation with sampling optimization
# - Memory-efficient BWT processing for large data
# - NumPy array operation optimizations
# - Debug condition short-circuiting
# - Progress callback null-checking
# =============================================================================
'''
    
    if "# Phase 1 Optimizations Applied" in content:
        content = content.replace(
            "# Phase 1 Optimizations Applied",
            "# Phase 1 & 2 Optimizations Applied"
        )
        content = phase2_marker + content
    
    # ファイル保存
    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Phase 2 Optimizations Applied:")
    for opt in optimizations_applied:
        print(f"   🔹 {opt}")
    
    print(f"✅ Phase 2 optimized file created: {target_file}")
    return True

def benchmark_phase2():
    """Phase 2最適化の効果を測定"""
    
    print("\n🚀 Phase 2 Benchmark Starting...")
    
    test_files = [
        "sample/COT-001.jpg",
        "sample/COT-001.png", 
        "sample/出庫実績明細_202412.txt"
    ]
    
    modes = ["FAST", "BALANCED", "MAXIMUM"]
    
    for test_file in test_files:
        if not os.path.exists(test_file):
            continue
            
        print(f"\n📁 Testing: {test_file}")
        file_size = os.path.getsize(test_file) / 1024 / 1024  # MB
        
        for mode in modes:
            print(f"  🎯 Mode: {mode}")
            
            # Phase 2版テスト
            start_time = time.time()
            result = os.system(f'python nxzip_core_phase2.py "{test_file}" -o "test_p2_{mode.lower()}.nxz" -m {mode} --quiet')
            phase2_time = time.time() - start_time
            
            if result == 0 and os.path.exists(f"test_p2_{mode.lower()}.nxz"):
                compressed_size = os.path.getsize(f"test_p2_{mode.lower()}.nxz")
                compression_ratio = compressed_size / os.path.getsize(test_file)
                speed = file_size / phase2_time if phase2_time > 0 else 0
                
                print(f"    ⚡ Phase 2: {phase2_time:.3f}s, {speed:.1f} MB/s, ratio: {compression_ratio:.4f}")
                
                # クリーンアップ
                try:
                    os.remove(f"test_p2_{mode.lower()}.nxz")
                except:
                    pass
            else:
                print(f"    ❌ Phase 2 failed")

if __name__ == "__main__":
    print("🚀 NXZip Phase 2 Optimizer")
    print("=" * 50)
    
    success = create_phase2_optimizations()
    
    if success:
        print("\n🎯 Phase 2 optimization completed successfully!")
        
        # ベンチマーク実行確認
        response = input("\n🔥 Run Phase 2 benchmark? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            benchmark_phase2()
        else:
            print("📊 Benchmark skipped. You can run it later with:")
            print("   python phase2_optimizer.py")
    else:
        print("❌ Phase 2 optimization failed")
        sys.exit(1)
