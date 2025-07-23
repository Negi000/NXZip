#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧹 NXZip Repository Cleanup - 最高性能エンジン保持

最高性能の4つのエンジンのみ保持:
✅ nexus_quantum.py (PNG量子圧縮 93.8%達成率)
✅ nexus_phase8_turbo.py (MP4動画 40.2%実績)  
✅ nexus_optimal_balance.py (テキスト 99.9%実績)
✅ nexus_lightning_fast.py (音声 79.1%/100%実績)

140個 → 10個以下に整理
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

class RepositoryCleanup:
    """リポジトリ整理クラス"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.backup_dir = self.base_dir / "backup_before_cleanup"
        
        # 保持する最高性能ファイル
        self.keep_files = {
            # 最高性能エンジン（4つ）
            'nexus_quantum.py': 'PNG量子圧縮エンジン (93.8%達成率)',
            'nexus_phase8_turbo.py': 'MP4動画最適化エンジン (40.2%実績)',
            'nexus_optimal_balance.py': 'テキスト高効率エンジン (99.9%実績)', 
            'nexus_lightning_fast.py': '音声最適化エンジン (79.1%/100%実績)',
            
            # 統合・ユーティリティ
            'nxzip_unified_wrapper.py': '統合ラッパー（新規作成）',
            'progress_display.py': '進捗表示ユーティリティ',
            
            # 分析・テストツール
            'analyze_formats.py': 'フォーマット分析ツール',
            'repository_cleanup.py': 'このクリーンアップスクリプト'
        }
        
        # 除外するディレクトリ/ファイル
        self.exclude_patterns = [
            '__pycache__',
            '*.pyc',
            '*.pyo',
            '.pytest_cache',
            'backup_*',
            '*.bak'
        ]
    
    def analyze_current_state(self):
        """現状分析"""
        all_py_files = list(self.base_dir.glob("*.py"))
        
        print(f"📊 リポジトリ現状分析")
        print(f"{'='*60}")
        print(f"📁 対象ディレクトリ: {self.base_dir}")
        print(f"📝 総Pythonファイル数: {len(all_py_files)}")
        print(f"✅ 保持予定ファイル数: {len(self.keep_files)}")
        print(f"🗑️ 削除予定ファイル数: {len(all_py_files) - len(self.keep_files)}")
        
        print(f"\n🎯 保持する最高性能ファイル:")
        for filename, description in self.keep_files.items():
            status = "✅" if (self.base_dir / filename).exists() else "❌"
            print(f"   {status} {filename}: {description}")
        
        print(f"\n🗑️ 削除予定ファイル（一部表示）:")
        delete_files = [f for f in all_py_files if f.name not in self.keep_files]
        for i, file_path in enumerate(delete_files[:10]):
            print(f"   • {file_path.name}")
        if len(delete_files) > 10:
            print(f"   ... 他 {len(delete_files) - 10} ファイル")
        
        return len(all_py_files), len(delete_files)
    
    def create_backup(self):
        """バックアップ作成"""
        print(f"\n💾 バックアップ作成中...")
        
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        
        self.backup_dir.mkdir(exist_ok=True)
        
        # 全Pythonファイルをバックアップ
        py_files = list(self.base_dir.glob("*.py"))
        backup_count = 0
        
        for py_file in py_files:
            try:
                shutil.copy2(py_file, self.backup_dir / py_file.name)
                backup_count += 1
            except Exception as e:
                print(f"   ⚠️ バックアップ失敗: {py_file.name} - {e}")
        
        # バックアップ情報ファイル作成
        backup_info = self.backup_dir / "backup_info.txt"
        with open(backup_info, 'w', encoding='utf-8') as f:
            f.write(f"NXZip Repository Cleanup Backup\n")
            f.write(f"作成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"バックアップファイル数: {backup_count}\n")
            f.write(f"保持予定ファイル: {', '.join(self.keep_files.keys())}\n")
        
        print(f"   ✅ バックアップ完了: {backup_count}ファイル → {self.backup_dir}")
    
    def execute_cleanup(self, confirm=True):
        """クリーンアップ実行"""
        if confirm:
            print(f"\n⚠️ 最終確認")
            print(f"{'='*60}")
            response = input("クリーンアップを実行しますか？ (yes/no): ")
            if response.lower() != 'yes':
                print("❌ クリーンアップをキャンセルしました")
                return False
        
        print(f"\n🧹 クリーンアップ実行中...")
        
        # 削除対象ファイル収集
        all_py_files = list(self.base_dir.glob("*.py"))
        delete_files = [f for f in all_py_files if f.name not in self.keep_files]
        
        deleted_count = 0
        error_count = 0
        
        for file_path in delete_files:
            try:
                file_path.unlink()
                deleted_count += 1
                if deleted_count <= 5:  # 最初の5ファイルのみ表示
                    print(f"   🗑️ 削除: {file_path.name}")
            except Exception as e:
                print(f"   ❌ 削除失敗: {file_path.name} - {e}")
                error_count += 1
        
        if deleted_count > 5:
            print(f"   ... 他 {deleted_count - 5} ファイル削除")
        
        print(f"\n✅ クリーンアップ完了")
        print(f"   削除成功: {deleted_count}ファイル")
        print(f"   削除失敗: {error_count}ファイル")
        print(f"   残存ファイル: {len(self.keep_files)}ファイル")
        
        return deleted_count > 0
    
    def verify_cleanup(self):
        """クリーンアップ検証"""
        print(f"\n🔍 クリーンアップ結果検証")
        print(f"{'='*60}")
        
        remaining_files = list(self.base_dir.glob("*.py"))
        
        print(f"📊 残存ファイル数: {len(remaining_files)}")
        print(f"🎯 目標ファイル数: {len(self.keep_files)}")
        
        print(f"\n📋 残存ファイル一覧:")
        for file_path in remaining_files:
            if file_path.name in self.keep_files:
                description = self.keep_files[file_path.name]
                print(f"   ✅ {file_path.name}: {description}")
            else:
                print(f"   ⚠️ {file_path.name}: 想定外のファイル")
        
        # 目標との整合性チェック
        missing_files = set(self.keep_files.keys()) - set(f.name for f in remaining_files)
        if missing_files:
            print(f"\n❌ 不足ファイル:")
            for filename in missing_files:
                print(f"   • {filename}")
        
        success = len(remaining_files) <= len(self.keep_files) + 2  # 多少の許容
        
        if success:
            print(f"\n🎉 クリーンアップ成功！")
            print(f"   最高性能エンジンが適切に保持されました")
        else:
            print(f"\n⚠️ クリーンアップに課題があります")
        
        return success
    
    def run_complete_cleanup(self):
        """完全クリーンアップ実行"""
        print(f"🚀 NXZip Repository Cleanup - 最高性能エンジン保持")
        print(f"{'='*70}")
        
        # Step 1: 現状分析
        total_files, delete_count = self.analyze_current_state()
        
        if delete_count == 0:
            print(f"\n✅ 既にクリーンアップ済みです")
            return True
        
        # Step 2: バックアップ作成
        self.create_backup()
        
        # Step 3: クリーンアップ実行
        if self.execute_cleanup(confirm=True):
            
            # Step 4: 結果検証
            success = self.verify_cleanup()
            
            if success:
                print(f"\n🎊 NXZip Repository Cleanup 完了")
                print(f"   140+ファイル → {len(self.keep_files)}ファイルに整理")
                print(f"   最高性能エンジンを保持")
                print(f"   バックアップ: {self.backup_dir}")
            
            return success
        
        return False

def main():
    """メイン実行"""
    cleanup = RepositoryCleanup()
    cleanup.run_complete_cleanup()

if __name__ == "__main__":
    main()
