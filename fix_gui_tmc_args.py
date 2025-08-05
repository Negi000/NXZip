#!/usr/bin/env python3
"""
GUI TMC引数修正パッチ
chunk_callback引数の問題を修正
"""

import os
import shutil

def fix_gui_tmc_args():
    """GUI TMC引数問題の修正"""
    print("🔧 GUI TMC引数修正開始")
    print("=" * 50)
    
    gui_file = "NXZip-Release/NXZip_Professional_v2.py"
    if not os.path.exists(gui_file):
        print(f"❌ ファイルが見つかりません: {gui_file}")
        return
    
    # バックアップを作成
    backup_file = f"{gui_file}.backup"
    shutil.copy2(gui_file, backup_file)
    print(f"📋 バックアップ作成: {backup_file}")
    
    # ファイル読み込み
    with open(gui_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # TMC引数修正: chunk_callbackの呼び出し方法を安全にする
    old_tmc_call = '''                result = self.tmc_engine.compress(data, chunk_callback=tmc_progress_callback)'''
    
    new_tmc_call = '''                # TMC安全呼び出し（引数チェック付き）
                try:
                    result = self.tmc_engine.compress(data, chunk_callback=tmc_progress_callback)
                except TypeError as te:
                    if 'chunk_callback' in str(te):
                        # chunk_callbackをサポートしていない場合のフォールバック
                        print("[TMC] chunk_callback未サポート - 代替処理")
                        result = self.tmc_engine.compress(data)
                    else:
                        raise'''
    
    # 置換実行
    replacements = 0
    if old_tmc_call in content:
        content = content.replace(old_tmc_call, new_tmc_call)
        replacements += content.count(new_tmc_call) - content.count(old_tmc_call)
        print(f"✅ TMC呼び出し修正完了: {replacements}箇所")
    else:
        print("⚠️ 対象のTMC呼び出しが見つかりません")
    
    # ファイル書き込み
    with open(gui_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"💾 GUI修正完了: {gui_file}")
    print("✅ GUI TMC引数修正完了")
    print("🔍 次回GUI実行時に正常動作するはずです")

if __name__ == "__main__":
    fix_gui_tmc_args()
