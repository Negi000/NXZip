#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧹 NXZip Repository Cleanup Script
リポジトリ整理スクリプト - 不要ファイル削除

🎯 整理方針:
- 最終統合エンジンのみ保持
- 開発中間ファイル削除
- テスト・分析ファイル保持
- 重要なドキュメント保持
"""

import os
import shutil
from pathlib import Path

class RepositoryCleanup:
    """リポジトリ整理クラス"""
    
    def __init__(self, bin_dir: str):
        self.bin_dir = Path(bin_dir)
        
        # 保持するファイル（最終版・重要ファイル）
        self.keep_files = {
            # 最終統合エンジン
            'nxzip_final_engines.py',
            'nxzip_final_decompressor.py',
            
            # 最適バランスエンジン（動画用）
            'nexus_optimal_balance.py',
            'optimal_decompressor.py',
            
            # 分析・テストツール
            'analyze_formats.py',
            'compare_7zip.py',
            'comprehensive_reversibility_audit.py',
            
            # 特化エンジン（参考用）
            'nexus_image_sdc.py',  # 画像専用
            'nexus_lightning_ultra.py',  # 超高速
            
            # テスト・評価
            'phase8_benchmark.py',
            'universal_decompression_auditor.py',
            
            # ドキュメント・レポート
            'reversibility_audit_report_20250722_175853.json',
            'reversibility_audit_report_20250723_100152.json',
            
            # 作業ファイル
            'test_compress.p8t'
        }
        
        # 削除対象ファイル（開発中間ファイル）
        self.delete_files = {
            # 開発中エンジン
            'nexus_absolute_final.py',
            'nexus_absolute_final_video.py',
            'nexus_ai_driven.py',
            'nexus_av1_revolutionary.py',
            'nexus_av_specialized.py',
            'nexus_complete_media_analysis.py',
            'nexus_data_revolution.py',
            'nexus_final_evaluation.py',
            'nexus_final_integrated.py',
            'nexus_improvement_phase7.py',
            'nexus_lightning_fast.py',
            'nexus_media_revolutionary.py',
            'nexus_optimization_phase3.py',
            'nexus_optimization_phase6.py',
            'nexus_perfect_reversible.py',
            'nexus_phase8_revolutionary.py',
            'nexus_phase8_turbo.py',
            'nexus_quantum.py',
            'nexus_quantum_video_revolution.py',
            'nexus_reversibility_test.py',
            'nexus_revolutionary_ai.py',
            'nexus_revolutionary_breakthrough.py',
            'nexus_revolution_final.py',
            'nexus_sdc_engine_backup.py',
            'nexus_sdc_engine_concise.py',
            'nexus_sdc_enhanced.py',
            'nexus_speed_optimized.py',
            'nexus_structure_freedom.py',
            'nexus_ultimate_final.py',
            'nexus_ultimate_lightning.py',
            'nexus_ultimate_media_breakthrough.py',
            'nexus_ultimate_video_breakthrough.py',
            'nexus_ultra_efficient.py',
            'nexus_unified_test.py',
            'nexus_video_breakthrough.py',
            
            # 中間テストファイル
            'nxzip_comprehensive_test.py',
            'nxzip_comprehensive_test_simple.py',
            'nxzip_final.py',
            'nxzip_nexus.py',
            'perfect_decompressor.py',
            'phase8_full.py',
            'phase8_media.py',
            'phase8_media_final.py',
            'phase8_media_optimized.py',
            'phase8_reversible.py',
            'phase8_simple_reversible.py',
            'png_quantum_comparison_test.py',
            'progress_display.py',
            'structure_destructive_analysis.py',
            'structure_destructive_v2.py',
            'test_nexus.py',
            'workflow_test.py',
            'workflow_test_enhanced.py',
            'workflow_test_improved.py',
            
            # 重複・古いエンジン
            'nexus_sdc_analyzer.py',
            'nexus_sdc_engine.py'  # nexus_image_sdc.pyに統合済み
        }
    
    def analyze_directory(self):
        """ディレクトリ分析"""
        if not self.bin_dir.exists():
            print(f"❌ ディレクトリが見つかりません: {self.bin_dir}")
            return
        
        all_files = [f.name for f in self.bin_dir.glob('*.py') if f.is_file()]
        
        # 分類
        keep_files = [f for f in all_files if f in self.keep_files]
        delete_files = [f for f in all_files if f in self.delete_files]
        unknown_files = [f for f in all_files if f not in self.keep_files and f not in self.delete_files]
        
        print("🧹 NXZip Repository Cleanup Analysis")
        print("=" * 60)
        print(f"📁 対象ディレクトリ: {self.bin_dir}")
        print(f"📊 総ファイル数: {len(all_files)} files")
        
        print(f"\n✅ 保持ファイル ({len(keep_files)} files):")
        for file in sorted(keep_files):
            print(f"  • {file}")
        
        print(f"\n🗑️ 削除予定ファイル ({len(delete_files)} files):")
        for file in sorted(delete_files):
            print(f"  • {file}")
        
        if unknown_files:
            print(f"\n❓ 未分類ファイル ({len(unknown_files)} files):")
            for file in sorted(unknown_files):
                print(f"  • {file}")
        
        return keep_files, delete_files, unknown_files
    
    def create_backup(self):
        """バックアップ作成"""
        backup_dir = self.bin_dir.parent / "bin_backup"
        
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        
        shutil.copytree(self.bin_dir, backup_dir)
        print(f"💾 バックアップ作成: {backup_dir}")
        
        return backup_dir
    
    def perform_cleanup(self, create_backup=True):
        """整理実行"""
        if create_backup:
            backup_dir = self.create_backup()
        
        deleted_count = 0
        
        print(f"\n🧹 整理実行中...")
        
        for file_name in self.delete_files:
            file_path = self.bin_dir / file_name
            if file_path.exists():
                try:
                    file_path.unlink()
                    print(f"🗑️ 削除: {file_name}")
                    deleted_count += 1
                except Exception as e:
                    print(f"❌ 削除失敗: {file_name} - {e}")
        
        # __pycache__も削除
        pycache_dir = self.bin_dir / "__pycache__"
        if pycache_dir.exists():
            try:
                shutil.rmtree(pycache_dir)
                print(f"🗑️ 削除: __pycache__/")
            except Exception as e:
                print(f"❌ __pycache__削除失敗: {e}")
        
        print(f"\n✅ 整理完了: {deleted_count} files削除")
        
        # 最終状態確認
        remaining_files = [f.name for f in self.bin_dir.glob('*.py') if f.is_file()]
        print(f"📁 残存ファイル数: {len(remaining_files)} files")
        
        print(f"\n🎯 最終ファイル構成:")
        for file in sorted(remaining_files):
            if file in self.keep_files:
                print(f"  ✅ {file}")
            else:
                print(f"  ❓ {file}")

def main():
    """メイン関数"""
    import sys
    
    bin_dir = r"c:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\bin"
    cleanup = RepositoryCleanup(bin_dir)
    
    if len(sys.argv) < 2:
        print("🧹 NXZip Repository Cleanup Script")
        print("使用方法:")
        print("  python repository_cleanup.py analyze   # 分析のみ")
        print("  python repository_cleanup.py cleanup   # 整理実行")
        return
    
    command = sys.argv[1].lower()
    
    if command == "analyze":
        cleanup.analyze_directory()
    elif command == "cleanup":
        print("⚠️ リポジトリ整理を実行します。")
        response = input("続行しますか? (y/N): ")
        if response.lower() == 'y':
            cleanup.perform_cleanup()
        else:
            print("❌ 整理をキャンセルしました")
    else:
        print("❌ 無効なコマンドです")

if __name__ == "__main__":
    main()
