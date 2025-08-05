#!/usr/bin/env python3
"""
NXZip SPE + TMC 効果検証
標準ライブラリ単体 vs SPE + TMC + 標準ライブラリ の圧縮率比較
"""

import os
import sys
import time
import warnings
import zlib
import lzma
from pathlib import Path

# 警告を抑制
warnings.filterwarnings("ignore")

# NXZipを直接インポート
sys.path.insert(0, '.')

# NXZip Core v2.0のパス追加
nxzip_core_path = Path("c:\\Users\\241822\\Desktop\\新しいフォルダー (2)\\NXZip\\NXZip-Release")
sys.path.insert(0, str(nxzip_core_path))

from nxzip_core import NXZipCore, CompressionPipeline, CompressionMode, DataAnalyzer
from nxzip.engine.spe_core_jit import SPECoreJIT

class CompressionEffectivenessAnalyzer:
    def __init__(self):
        self.test_files = [
            ("sample/COT-001.jpg", "画像(JPEG)"),
            ("sample/COT-001.png", "画像(PNG)"), 
            ("sample/出庫実績明細_202412.txt", "テキスト"),
            ("sample/generated-music-1752042054079.wav", "音声")
        ]
        
    def test_standard_library_only(self, data: bytes, data_type: str):
        """標準ライブラリ単体での圧縮テスト"""
        results = {}
        
        # zlib レベル別
        for level in [1, 6, 9]:
            compressed = zlib.compress(data, level=level)
            ratio = (1 - len(compressed) / len(data)) * 100
            results[f'zlib_level_{level}'] = {
                'size': len(compressed),
                'ratio': ratio,
                'method': f'zlib(level={level})'
            }
        
        # lzma プリセット別
        for preset in [1, 6, 9]:
            compressed = lzma.compress(data, preset=preset)
            ratio = (1 - len(compressed) / len(data)) * 100
            results[f'lzma_preset_{preset}'] = {
                'size': len(compressed),
                'ratio': ratio,
                'method': f'lzma(preset={preset})'
            }
        
        return results
    
    def test_spe_tmc_enhanced(self, data: bytes, data_type: str):
        """SPE + TMC前処理 + 標準ライブラリ圧縮"""
        results = {}
        
        # SPE Core JIT でデータ変換
        spe = SPECoreJIT()
        spe_data = spe.apply_transform(data)
        
        print(f"    SPE変換: {len(data)} → {len(spe_data)} bytes")
        
        # TMC Pipeline でデータ変換
        pipeline = CompressionPipeline(CompressionMode.BALANCED)
        data_analyzer = DataAnalyzer()
        detected_type = data_analyzer.analyze_data_type(data)
        
        # TMC変換のみ実行（最終圧縮は除く）
        transformed_data, transform_info = pipeline.tmc_engine.transform_data(data, detected_type)
        
        print(f"    TMC変換: {len(data)} → {len(transformed_data)} bytes")
        print(f"    適用変換: {transform_info.get('transforms_applied', [])}")
        
        # SPE + TMC組み合わせ
        spe_tmc_data = spe.apply_transform(transformed_data)
        
        print(f"    SPE+TMC: {len(data)} → {len(spe_tmc_data)} bytes")
        
        # 変換されたデータで標準ライブラリ圧縮
        test_datasets = [
            ("original", data),
            ("spe_only", spe_data), 
            ("tmc_only", transformed_data),
            ("spe_tmc", spe_tmc_data)
        ]
        
        for dataset_name, test_data in test_datasets:
            # zlib最適レベル
            zlib_compressed = zlib.compress(test_data, level=6)
            zlib_ratio = (1 - len(zlib_compressed) / len(data)) * 100
            
            # lzma最適レベル
            lzma_compressed = lzma.compress(test_data, preset=6)
            lzma_ratio = (1 - len(lzma_compressed) / len(data)) * 100
            
            results[f'{dataset_name}_zlib'] = {
                'size': len(zlib_compressed),
                'ratio': zlib_ratio,
                'method': f'{dataset_name} + zlib',
                'preprocessing_size': len(test_data)
            }
            
            results[f'{dataset_name}_lzma'] = {
                'size': len(lzma_compressed),
                'ratio': lzma_ratio,
                'method': f'{dataset_name} + lzma',
                'preprocessing_size': len(test_data)
            }
        
        return results
    
    def compare_effectiveness(self, filepath: str, file_description: str):
        """圧縮効果比較"""
        print(f"\n📊 {file_description} 圧縮効果分析: {os.path.basename(filepath)}")
        print("=" * 70)
        
        # データ読み込み
        with open(filepath, 'rb') as f:
            data = f.read()
        
        original_size = len(data)
        print(f"元サイズ: {original_size:,} bytes ({original_size/1024/1024:.2f} MB)")
        
        # 標準ライブラリ単体テスト
        print(f"\n🔸 標準ライブラリ単体:")
        standard_results = self.test_standard_library_only(data, file_description)
        
        best_standard_ratio = 0
        best_standard_method = ""
        
        for method, result in standard_results.items():
            print(f"  {result['method']:15} : {result['ratio']:6.2f}% ({result['size']:,} bytes)")
            if result['ratio'] > best_standard_ratio:
                best_standard_ratio = result['ratio']
                best_standard_method = result['method']
        
        print(f"  最良標準手法: {best_standard_method} ({best_standard_ratio:.2f}%)")
        
        # SPE + TMC強化テスト
        print(f"\n🔸 SPE + TMC強化:")
        enhanced_results = self.test_spe_tmc_enhanced(data, file_description)
        
        print(f"\n  圧縮結果比較:")
        improvements = []
        
        for method, result in enhanced_results.items():
            improvement = result['ratio'] - best_standard_ratio
            color = "✅" if improvement > 0 else "❌" if improvement < -1 else "⚠️"
            
            print(f"  {result['method']:20} : {result['ratio']:6.2f}% ({improvement:+5.2f}%) {color}")
            
            if improvement > 0:
                improvements.append((method, improvement, result['ratio']))
        
        # 改善効果評価
        print(f"\n🎯 改善効果評価:")
        if improvements:
            best_improvement = max(improvements, key=lambda x: x[1])
            print(f"  最良改善: {best_improvement[0]} (+{best_improvement[1]:.2f}%)")
            print(f"  改善手法数: {len(improvements)}/{len(enhanced_results)}")
            
            if best_improvement[1] > 5:
                print(f"  🏆 大幅改善達成! SPE + TMC前処理の効果あり")
            elif best_improvement[1] > 1:
                print(f"  🥈 軽微改善: SPE + TMC前処理に効果あり")
            else:
                print(f"  🥉 微改善: SPE + TMC前処理の効果限定的")
        else:
            print(f"  ❌ 改善なし: SPE + TMC前処理が逆効果")
            print(f"  💡 提案: このファイル形式ではSPE + TMCをスキップ")
        
        return {
            'file': filepath,
            'original_size': original_size,
            'best_standard': best_standard_ratio,
            'improvements': improvements,
            'effective': len(improvements) > 0
        }
    
    def run_comprehensive_analysis(self):
        """包括的な圧縮効果分析"""
        print("🔥 NXZip SPE + TMC 圧縮効果検証")
        print("標準ライブラリ単体 vs SPE + TMC + 標準ライブラリ")
        print("=" * 80)
        
        all_results = []
        
        for filepath, file_desc in self.test_files:
            if os.path.exists(filepath):
                result = self.compare_effectiveness(filepath, file_desc)
                all_results.append(result)
            else:
                print(f"⚠️ ファイル未発見: {filepath}")
        
        # 総合評価
        print(f"\n" + "=" * 80)
        print(f"🎯 総合評価")
        print("=" * 80)
        
        effective_count = sum(1 for r in all_results if r['effective'])
        total_count = len(all_results)
        
        print(f"📊 SPE + TMC効果:")
        print(f"  効果的なファイル: {effective_count}/{total_count}")
        print(f"  効果率: {effective_count/total_count*100:.1f}%" if total_count > 0 else "  効果率: N/A")
        
        if effective_count == 0:
            print(f"\n❌ 重大な問題:")
            print(f"  SPE + TMC前処理が全てのファイル形式で無効果")
            print(f"  現在の実装では標準ライブラリ単体より劣る")
            print(f"\n💡 改善提案:")
            print(f"  1. SPE変換アルゴリズムの見直し")
            print(f"  2. TMC変換の最適化")
            print(f"  3. ファイル形式別の前処理選択")
            print(f"  4. 前処理スキップ機能の実装")
        elif effective_count < total_count * 0.5:
            print(f"\n⚠️ 改善必要:")
            print(f"  SPE + TMC前処理の効果が限定的")
            print(f"  ファイル形式によって効果にばらつき")
            print(f"\n💡 改善提案:")
            print(f"  1. ファイル形式別最適化")
            print(f"  2. 効果的でない場合の自動スキップ")
            print(f"  3. アダプティブ前処理選択")
        else:
            print(f"\n✅ 効果確認:")
            print(f"  SPE + TMC前処理が多数のファイル形式で有効")
            print(f"  標準ライブラリ単体を上回る圧縮率を実現")
        
        # 最良改善例表示
        best_improvements = []
        for result in all_results:
            if result['improvements']:
                best = max(result['improvements'], key=lambda x: x[1])
                best_improvements.append((result['file'], best[1], best[2]))
        
        if best_improvements:
            print(f"\n🏆 最良改善例:")
            best_improvements.sort(key=lambda x: x[1], reverse=True)
            for i, (file, improvement, ratio) in enumerate(best_improvements[:3], 1):
                filename = os.path.basename(file)
                print(f"  {i}. {filename}: +{improvement:.2f}% (最終{ratio:.2f}%)")
        
        return all_results

if __name__ == "__main__":
    analyzer = CompressionEffectivenessAnalyzer()
    analyzer.run_comprehensive_analysis()
