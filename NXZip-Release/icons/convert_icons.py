#!/usr/bin/env python3
"""
NXZip アイコン変換スクリプト
大きなPNGファイルを適切なサイズのアイコンに変換

Requirements:
- Python 3.6+
- Pillow (PIL) library: pip install Pillow

Usage:
    python convert_icons.py
"""

import os
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ Pillow not available. Install with: pip install Pillow")

def convert_large_icon_to_small(input_path: str, output_path: str, size: tuple = (32, 32)):
    """大きなアイコンを小さなサイズに変換"""
    try:
        print(f"🔄 Converting {input_path} to {size[0]}x{size[1]}...")
        
        # 元画像を開く
        with Image.open(input_path) as img:
            # RGBAモードに変換（透明度対応）
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # アスペクト比を保持してリサイズ
            img.thumbnail(size, Image.Resampling.LANCZOS)
            
            # 新しい画像を作成（透明背景）
            new_img = Image.new('RGBA', size, (0, 0, 0, 0))
            
            # 中央に配置
            x = (size[0] - img.width) // 2
            y = (size[1] - img.height) // 2
            new_img.paste(img, (x, y), img)
            
            # 保存
            new_img.save(output_path, 'PNG', optimize=True)
            
            file_size = os.path.getsize(output_path)
            print(f"✅ Converted successfully: {file_size} bytes")
            return True
            
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

def create_simple_nxzip_icon(output_path: str, size: tuple = (32, 32)):
    """シンプルなNXZipアイコンを作成"""
    try:
        print(f"🎨 Creating simple NXZip icon ({size[0]}x{size[1]})...")
        
        # 新しい画像を作成
        img = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # 背景円（グラデーション風）
        margin = 2
        circle_size = min(size) - margin * 2
        x1 = (size[0] - circle_size) // 2
        y1 = (size[1] - circle_size) // 2
        x2 = x1 + circle_size
        y2 = y1 + circle_size
        
        # メインの円
        draw.ellipse([x1, y1, x2, y2], fill=(52, 152, 219, 255), outline=(41, 128, 185, 255), width=2)
        
        # 内側の強調円
        inner_margin = 4
        ix1 = x1 + inner_margin
        iy1 = y1 + inner_margin
        ix2 = x2 - inner_margin
        iy2 = y2 - inner_margin
        draw.ellipse([ix1, iy1, ix2, iy2], outline=(255, 255, 255, 100), width=1)
        
        # テキスト "NX"
        try:
            # フォントサイズを動的に調整
            font_size = max(8, size[0] // 4)
            font = ImageFont.truetype("arial.ttf", font_size)
        except (OSError, IOError):
            # フォールバック
            font = ImageFont.load_default()
        
        # テキストサイズを測定
        text = "NX"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # テキストを中央配置
        text_x = (size[0] - text_width) // 2
        text_y = (size[1] - text_height) // 2 - 1  # 少し上に調整
        
        # 影効果
        shadow_offset = 1
        draw.text((text_x + shadow_offset, text_y + shadow_offset), text, 
                 fill=(0, 0, 0, 128), font=font)
        # メインテキスト
        draw.text((text_x, text_y), text, fill=(255, 255, 255, 255), font=font)
        
        # 保存
        img.save(output_path, 'PNG', optimize=True)
        
        file_size = os.path.getsize(output_path)
        print(f"✅ Simple icon created: {file_size} bytes")
        return True
        
    except Exception as e:
        print(f"❌ Simple icon creation failed: {e}")
        return False

def create_archive_icon(output_path: str, size: tuple = (32, 32)):
    """アーカイブファイル用のアイコンを作成"""
    try:
        print(f"📦 Creating archive icon ({size[0]}x{size[1]})...")
        
        img = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # ファイルの形
        margin = 4
        file_width = size[0] - margin * 2
        file_height = size[1] - margin * 2
        corner_size = 4
        
        # ファイルの背景
        points = [
            (margin, margin + corner_size),
            (margin + file_width - corner_size, margin),
            (margin + file_width, margin + corner_size),
            (margin + file_width, margin + file_height),
            (margin, margin + file_height)
        ]
        draw.polygon(points, fill=(236, 240, 241, 255), outline=(149, 165, 166, 255), width=1)
        
        # 折り目
        fold_points = [
            (margin + file_width - corner_size, margin),
            (margin + file_width - corner_size, margin + corner_size),
            (margin + file_width, margin + corner_size)
        ]
        draw.polygon(fold_points, fill=(189, 195, 199, 255))
        
        # "NZ" テキスト
        try:
            font_size = max(6, size[0] // 6)
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        text = "NZ"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        text_x = (size[0] - text_width) // 2
        text_y = (size[1] - text_height) // 2 + 2
        
        draw.text((text_x, text_y), text, fill=(52, 73, 94, 255), font=font)
        
        # 保存
        img.save(output_path, 'PNG', optimize=True)
        
        file_size = os.path.getsize(output_path)
        print(f"✅ Archive icon created: {file_size} bytes")
        return True
        
    except Exception as e:
        print(f"❌ Archive icon creation failed: {e}")
        return False

def create_fallback_icons():
    """Pillowが利用できない場合のフォールバック"""
    print("🔧 Creating fallback text-based icons...")
    
    # 極小PNGデータ（32x32の透明画像）
    minimal_png = (
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00 \x00\x00\x00 '
        b'\x08\x06\x00\x00\x00szz\xf4\x00\x00\x00\x19tEXtSoftware\x00'
        b'Adobe ImageReadyq\xc9e<\x00\x00\x00\x0eIDATx\xdac\xf8\x0f\x00'
        b'\x00\x01\x00\x01\x00\x00\x00\x00\x00\x00IEND\xaeB`\x82'
    )
    
    try:
        # アプリアイコン
        with open("rogo_small.png", "wb") as f:
            f.write(minimal_png)
        print("✅ Fallback app icon created")
        
        # アーカイブアイコン
        with open("archive_nxz_small.png", "wb") as f:
            f.write(minimal_png)
        print("✅ Fallback archive icon created")
        
        return True
    except Exception as e:
        print(f"❌ Fallback creation failed: {e}")
        return False

def main():
    """メイン処理"""
    global PIL_AVAILABLE  # グローバル宣言を最初に
    
    print("🎨 NXZip Icon Converter v1.0")
    print("=" * 50)
    
    # 現在のディレクトリを確認
    current_dir = Path.cwd()
    print(f"📁 Working directory: {current_dir}")
    
    # Pillowの可用性をチェック
    if not PIL_AVAILABLE:
        print("⚠️ Pillow library not found!")
        print("📥 Installing Pillow...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'Pillow'])
            print("✅ Pillow installed successfully!")
            # 再インポート
            from PIL import Image, ImageDraw, ImageFont
            PIL_AVAILABLE = True
        except Exception as e:
            print(f"❌ Pillow installation failed: {e}")
            print("🔧 Using fallback mode...")
            return create_fallback_icons()
    
    # 元のアイコンファイルを確認
    original_app_icon = Path("rogo.png")
    original_archive_icon = Path("archive_nxz.png")
    
    success_count = 0
    
    # アプリアイコンの変換
    if original_app_icon.exists():
        file_size = original_app_icon.stat().st_size
        print(f"📄 Found app icon: {file_size:,} bytes")
        
        if file_size > 50000:  # 50KB以上の場合は変換
            if convert_large_icon_to_small(str(original_app_icon), "rogo_small.png", (32, 32)):
                success_count += 1
        else:
            print("✅ App icon size is acceptable, copying...")
            try:
                import shutil
                shutil.copy2(original_app_icon, "rogo_small.png")
                success_count += 1
            except Exception as e:
                print(f"❌ Copy failed: {e}")
    else:
        print("⚠️ Original app icon not found, creating new one...")
        if create_simple_nxzip_icon("rogo_small.png", (32, 32)):
            success_count += 1
    
    # アーカイブアイコンの変換
    if original_archive_icon.exists():
        file_size = original_archive_icon.stat().st_size
        print(f"📦 Found archive icon: {file_size:,} bytes")
        
        if file_size > 50000:  # 50KB以上の場合は変換
            if convert_large_icon_to_small(str(original_archive_icon), "archive_nxz_small.png", (32, 32)):
                success_count += 1
        else:
            print("✅ Archive icon size is acceptable, copying...")
            try:
                import shutil
                shutil.copy2(original_archive_icon, "archive_nxz_small.png")
                success_count += 1
            except Exception as e:
                print(f"❌ Copy failed: {e}")
    else:
        print("⚠️ Original archive icon not found, creating new one...")
        if create_archive_icon("archive_nxz_small.png", (32, 32)):
            success_count += 1
    
    # 複数サイズのアイコンも作成
    sizes = [(16, 16), (24, 24), (48, 48)]
    for size in sizes:
        size_suffix = f"_{size[0]}x{size[1]}"
        
        # アプリアイコン
        if create_simple_nxzip_icon(f"rogo{size_suffix}.png", size):
            print(f"✅ Created app icon {size[0]}x{size[1]}")
        
        # アーカイブアイコン
        if create_archive_icon(f"archive_nxz{size_suffix}.png", size):
            print(f"✅ Created archive icon {size[0]}x{size[1]}")
    
    print("=" * 50)
    print(f"🎯 Conversion completed! {success_count}/2 main icons converted")
    
    # 作成されたファイルをリスト表示
    print("\n📋 Created files:")
    for icon_file in Path(".").glob("*_small.png"):
        size = icon_file.stat().st_size
        print(f"   {icon_file.name}: {size:,} bytes")
    
    for icon_file in Path(".").glob("*_16x16.png"):
        size = icon_file.stat().st_size
        print(f"   {icon_file.name}: {size:,} bytes")
    
    for icon_file in Path(".").glob("*_24x24.png"):
        size = icon_file.stat().st_size
        print(f"   {icon_file.name}: {size:,} bytes")
        
    for icon_file in Path(".").glob("*_48x48.png"):
        size = icon_file.stat().st_size
        print(f"   {icon_file.name}: {size:,} bytes")
    
    print("\n🚀 Icons are ready for NXZip Professional!")
    print("💡 Tip: The application will now use the optimized icons automatically.")

if __name__ == "__main__":
    main()
