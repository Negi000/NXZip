#!/usr/bin/env python3
"""
Real-time Progress Display for NEXUS SDC
リアルタイム進捗表示機能
"""

import sys
import time
import threading
from typing import Optional

class ProgressDisplay:
    """リアルタイム進捗表示クラス"""
    
    def __init__(self):
        self.current_task = ""
        self.current_step = 0
        self.total_steps = 0
        self.start_time = 0
        self.is_running = False
        self.display_thread: Optional[threading.Thread] = None
        self.current_file = ""
        self.bytes_processed = 0
        self.total_bytes = 0
        
    def start_task(self, task_name: str, total_steps: int = 100, file_name: str = "", total_bytes: int = 0):
        """タスク開始"""
        self.current_task = task_name
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.is_running = True
        self.current_file = file_name
        self.bytes_processed = 0
        self.total_bytes = total_bytes
        
        # 進捗表示スレッド開始
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()
        
        print(f"\n🚀 開始: {task_name}")
        if file_name:
            print(f"📁 ファイル: {file_name}")
        if total_bytes > 0:
            print(f"💾 サイズ: {self._format_bytes(total_bytes)}")
    
    def update_progress(self, step: int, message: str = "", bytes_processed: int = 0):
        """進捗更新"""
        self.current_step = min(step, self.total_steps)
        if message:
            print(f"\n📊 {message}")
        if bytes_processed > 0:
            self.bytes_processed = bytes_processed
    
    def set_substep(self, message: str):
        """サブステップ表示"""
        print(f"   ⚡ {message}")
    
    def finish_task(self, success: bool = True, final_message: str = ""):
        """タスク終了"""
        self.is_running = False
        if self.display_thread:
            self.display_thread.join(timeout=0.1)
        
        elapsed_time = time.time() - self.start_time
        
        if success:
            print(f"\n✅ 完了: {self.current_task}")
        else:
            print(f"\n❌ 失敗: {self.current_task}")
        
        if final_message:
            print(f"📋 結果: {final_message}")
        
        print(f"⏱️  所要時間: {elapsed_time:.1f}秒")
        
        if self.total_bytes > 0 and elapsed_time > 0:
            speed_mbps = (self.total_bytes / (1024 * 1024)) / elapsed_time
            print(f"⚡ 処理速度: {speed_mbps:.1f} MB/s")
        
        print("-" * 50)
    
    def _display_loop(self):
        """進捗表示ループ - 改良版リアルタイム更新"""
        spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        spinner_idx = 0
        last_line_length = 0
        
        while self.is_running:
            # 進捗バー作成（より見やすいスタイル）
            if self.total_steps > 0:
                progress_percent = (self.current_step / self.total_steps) * 100
                bar_length = 25
                filled_length = int(bar_length * self.current_step // self.total_steps)
                bar = '▓' * filled_length + '░' * (bar_length - filled_length)
            else:
                progress_percent = 0
                bar = '░' * 25
            
            # 経過時間
            elapsed = time.time() - self.start_time
            
            # バイト進捗（コンパクト表示）
            if self.total_bytes > 0:
                byte_percent = (self.bytes_processed / self.total_bytes) * 100
                processed_str = self._format_bytes(self.bytes_processed)
                total_str = self._format_bytes(self.total_bytes)
                progress_info = f"{processed_str}/{total_str}"
            else:
                progress_info = f"Step {self.current_step}/{self.total_steps}"
            
            # コンパクトな進捗表示
            current_task_name = self.current_task[:30] + "..." if len(self.current_task) > 30 else self.current_task
            display_text = f"{spinner[spinner_idx]} {current_task_name}: [{bar}] {progress_percent:.1f}% | {elapsed:.1f}s | {progress_info}"
            
            # 前の行をクリアして新しい行を表示
            clear_padding = " " * max(last_line_length - len(display_text), 0)
            print(f"\r{display_text}{clear_padding}", end='', flush=True)
            last_line_length = len(display_text)
            
            spinner_idx = (spinner_idx + 1) % len(spinner)
            time.sleep(0.08)  # より高速な更新
        
        # 最終表示をクリア
        print('\r' + ' ' * (last_line_length + 10) + '\r', end='', flush=True)
    
    def _format_bytes(self, bytes_value: int) -> str:
        """バイト数の表示フォーマット"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f}{unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f}TB"

# グローバル進捗表示インスタンス
progress = ProgressDisplay()

def show_step(message: str):
    """ステップ表示のヘルパー関数"""
    print(f"🔧 {message}")

def show_substep(message: str):
    """サブステップ表示のヘルパー関数"""
    print(f"   💫 {message}")

def show_warning(message: str):
    """警告表示のヘルパー関数"""
    print(f"⚠️  {message}")

def show_error(message: str):
    """エラー表示のヘルパー関数"""
    print(f"❌ {message}")

def show_success(message: str):
    """成功表示のヘルパー関数"""
    print(f"✅ {message}")

# エクスポート用
__all__ = ['ProgressDisplay', 'progress', 'show_step', 'show_substep', 'show_warning', 'show_error', 'show_success']
