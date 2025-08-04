#!/usr/bin/env python3
"""
簡単なアイコン作成スクリプト
大きなPNGファイルを小さなアイコンサイズに変換
"""

import tkinter as tk
from PIL import Image, ImageDraw, ImageFont
import os

def create_simple_icon():
    """シンプルなNXZipアイコンを作成"""
    # 32x32のアイコンを作成
    size = (32, 32)
    icon = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(icon)
    
    # 背景円
    draw.ellipse([2, 2, 30, 30], fill=(52, 152, 219, 255), outline=(41, 128, 185, 255), width=2)
    
    # テキスト "NX"
    try:
        # フォントを試行
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    # テキスト描画
    draw.text((8, 10), "NX", fill=(255, 255, 255, 255), font=font)
    
    return icon

def main():
    """メイン処理"""
    print("🎨 Creating NXZip icon...")
    
    try:
        # アイコン作成
        icon = create_simple_icon()
        
        # 保存
        icon_path = "small_icon.png"
        icon.save(icon_path, "PNG")
        print(f"✅ Icon created: {icon_path}")
        
        # プレビュー表示
        root = tk.Tk()
        root.title("Icon Preview")
        root.geometry("100x100")
        
        # アイコンを表示
        photo = tk.PhotoImage(file=icon_path)
        label = tk.Label(root, image=photo)
        label.pack(expand=True)
        
        root.mainloop()
        
    except ImportError:
        print("⚠️ PIL (Pillow) not available, creating text-based icon")
        create_text_icon()

def create_text_icon():
    """テキストベースのシンプルアイコン"""
    print("📝 Creating text-based fallback icon...")
    
    # 極小PNGアイコンをバイナリで作成
    png_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00 \x00\x00\x00 \x08\x06\x00\x00\x00szz\xf4\x00\x00\x00\x04sBIT\x08\x08\x08\x08|\x08d\x88\x00\x00\x01\x8eIDATX\x85\xed\x97\xc1\n\x830\x10D\x9f\xa5\xe8\xa1\x87\x1e\xfa\x0e\xbd\xf6\xd1G\x0f=\xf4\xd0\xa7\x1ez\xe8!\x87\x16\xfaH!F\x92\x98d\xec\xa1\x85\xee\xcc\x99\xf9\xcd\xcc\x9b\x01\xc0\x7f\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\xf8\x0f\x80\xff\x00\x00\x00\x00IEND\xaeB`\x82'
    
    with open("small_icon.png", "wb") as f:
        f.write(png_data)
    
    print("✅ Fallback icon created")

if __name__ == "__main__":
    main()
