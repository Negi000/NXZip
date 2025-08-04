#!/usr/bin/env python3
"""
NXZip用の軽量アイコンを作成
"""

import tkinter as tk
from tkinter import Canvas

def create_simple_icon():
    """軽量なアイコンをCanvasで作成"""
    root = tk.Tk()
    root.withdraw()  # メインウィンドウを隠す
    
    # 32x32のキャンバス
    canvas = Canvas(root, width=32, height=32, bg='white')
    
    # 背景円（青）
    canvas.create_oval(2, 2, 30, 30, fill='#3498db', outline='#2980b9', width=2)
    
    # テキスト "NX"
    canvas.create_text(16, 16, text="NX", fill='white', font=('Arial', 8, 'bold'))
    
    # PostScriptで保存してからPNGに変換
    try:
        canvas.postscript(file="temp_icon.ps")
        print("✅ Icon created successfully")
    except Exception as e:
        print(f"⚠️ Icon creation failed: {e}")
    
    root.destroy()

if __name__ == "__main__":
    create_simple_icon()
