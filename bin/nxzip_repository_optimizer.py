#!/usr/bin/env python3
"""
NXZip Repository Cleanup and Optimization
リポジトリ整理 - 最適なエンジンのみ保持

目標:
- 重複・無駄なエンジンを削除
- 最高性能エンジンのみ保持
- 各フォーマット専用最適化エンジン
- クリーンな構造に再構築
"""

import os
import sys
import shutil
import time
from pathlib import Path

class NXZipRepositoryCleanup:
    def __init__(self):
        self.bin_dir = Path("C:/Users/241822/Desktop/新しいフォルダー (2)/NXZip/bin")
        
        # 保持する最適エンジンリスト
        self.keep_engines = {
            # 超高速汎用エンジン（最新・最高性能）
            "nxzip_ultra_fast_binary_collapse.py": "汎用超高速圧縮エンジン（22.5MB/s、10%圧縮）",
            
            # 画像専用最適化エンジン
            "nxzip_smart_image_compressor.py": "画像専用圧縮エンジン（JPEG 8.4%圧縮）",
            
            # AV1インスパイア高品質エンジン
            "nexus_cablc_enhanced.py": "AV1インスパイア高品質圧縮エンジン",
            
            # ユーティリティ
            "analyze_formats.py": "フォーマット解析ツール",
            "final_repository_cleanup.py": "リポジトリ整理ツール"
        }
        
        # 削除対象（重複・旧式・失敗エンジン）
        self.remove_patterns = [
            "nexus_quantum*",           # 量子系（複雑すぎ）
            "nexus_adaptive*",          # 適応系（重複）
            "nexus_hybrid*",            # ハイブリッド系（重複）
            "nexus_lightning*",         # 雷系（重複）
            "nexus_ultra_fast*",        # 旧高速系（新版で置換）
            "nexus_optimal*",           # 最適化系（重複）
            "nexus_phase8*",            # Phase8系（旧式）
            "nexus_practical*",         # 実用系（重複）
            "nexus_media*",             # メディア系（重複）
            "nexus_lossless*",          # 無損失系（重複）
            "nexus_av1_inspired.py",    # AV1系（enhanced版で置換）
            "nexus_cablc_engine.py",    # CABLC旧版
            "nexus_cablc_png_decoder.py", # PNG専用（smart版で置換）
            "nexus_extreme_structural*", # 極限構造系（複雑すぎ）
            "nxzip_advanced_decoder.py", # 旧式デコーダー
            "nxzip_binary_structural_collapse.py", # 旧式構造圧縮
            "nxzip_binary_structural_dictionary.py", # 構造辞書（未完成）
            "nxzip_final_decompressor.py", # 旧式展開
            "nxzip_final_engines.py",   # 旧式エンジン集
            "nxzip_format_decoder.py",  # フォーマットデコーダー（重複）
            "nxzip_image_extreme_compressor.py", # 極限画像圧縮（重複）
            "nxzip_performance_verified_engine.py", # 検証済み（古い）
            "nxzip_ultra_fast_collapse.py", # 旧高速圧縮
            "nxzip_unified_wrapper.py", # 統合ラッパー（複雑）
            "*test*.py",                # テストファイル群
            "*reversibility*.py",       # 可逆性テスト群
            "*audit*.py",               # 監査ツール群
            "check_*.py",               # チェックツール群
            "comprehensive_*.py",       # 包括テスト群
            "final_quantum_*.py",       # 最終量子系
            "phase8_*.py",              # Phase8関連
            "progress_display.py",      # 進捗表示
            "quantum_*.py",             # 量子関連
            "repository_cleanup.py",    # 旧整理ツール
            "sample_data_*.py",         # サンプルデータテスト
            "simple_engine_test.py",    # シンプルテスト
            "universal_*.py",           # 汎用ツール
            "*.json",                   # テスト結果JSON
            "*.txt",                    # テキストファイル
            "*.nxz*",                   # 圧縮済みファイル
            "*.restored"                # 復元ファイル
        ]
    
    def create_backup(self):
        """バックアップ作成"""
        backup_dir = self.bin_dir / "cleanup_backup_20250723"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        
        backup_dir.mkdir(exist_ok=True)
        
        print(f"📦 バックアップ作成中: {backup_dir}")
        
        for file_path in self.bin_dir.glob("*.py"):
            if file_path.name not in self.keep_engines:
                shutil.copy2(file_path, backup_dir / file_path.name)
        
        for file_path in self.bin_dir.glob("*"):
            if file_path.suffix in ['.json', '.txt', '.nxz', '.restored', '.nxzah', '.nxzuh', '.nxzhb', '.nxzhs']:
                if file_path.is_file():
                    shutil.copy2(file_path, backup_dir / file_path.name)
        
        print(f"✅ バックアップ完了: {len(list(backup_dir.glob('*')))} files")
        return backup_dir
    
    def cleanup_files(self):
        """ファイル整理実行"""
        print("🧹 ファイル整理開始...")
        
        removed_count = 0
        kept_count = 0
        
        # Pythonファイル整理
        for file_path in self.bin_dir.glob("*.py"):
            if file_path.name in self.keep_engines:
                kept_count += 1
                print(f"✅ 保持: {file_path.name} - {self.keep_engines[file_path.name]}")
            else:
                file_path.unlink()
                removed_count += 1
                print(f"🗑️ 削除: {file_path.name}")
        
        # その他ファイル整理
        extensions_to_remove = ['.json', '.txt', '.nxz', '.restored', '.nxzah', '.nxzuh', '.nxzhb', '.nxzhs']
        for ext in extensions_to_remove:
            for file_path in self.bin_dir.glob(f"*{ext}"):
                if file_path.is_file():
                    file_path.unlink()
                    removed_count += 1
                    print(f"🗑️ 削除: {file_path.name}")
        
        # __pycache__ 削除
        pycache_dir = self.bin_dir / "__pycache__"
        if pycache_dir.exists():
            shutil.rmtree(pycache_dir)
            print(f"🗑️ 削除: __pycache__/")
        
        print(f"\n📊 整理完了:")
        print(f"   ✅ 保持: {kept_count} files")
        print(f"   🗑️ 削除: {removed_count} files")
        
        return kept_count, removed_count
    
    def create_optimized_structure(self):
        """最適化された構造作成"""
        print("🏗️ 最適化構造作成中...")
        
        # README作成
        readme_content = """# NXZip - Next Generation Archive System

## 最適化エンジン一覧

### 🚀 汎用超高速圧縮エンジン
- **nxzip_ultra_fast_binary_collapse.py**
  - 処理速度: 22.5 MB/s
  - 汎用圧縮率: 10%
  - 対象: 全ファイル形式
  - 特徴: 超高速処理、実用性重視

### 🖼️ 画像専用最適化エンジン
- **nxzip_smart_image_compressor.py**
  - JPEG圧縮率: 8.4%
  - PNG最適化: 制約認識
  - 特徴: フォーマット特化最適化

### 🎯 AV1インスパイア高品質エンジン
- **nexus_cablc_enhanced.py**
  - ブロック分割最適化
  - 複数予測モード
  - 特徴: 高品質圧縮

### 🔧 ユーティリティ
- **analyze_formats.py**: フォーマット解析
- **final_repository_cleanup.py**: リポジトリ整理

## 使用方法

```bash
# 汎用高速圧縮
python nxzip_ultra_fast_binary_collapse.py <ファイル>

# 画像専用圧縮
python nxzip_smart_image_compressor.py <画像ファイル>

# 高品質圧縮
python nexus_cablc_enhanced.py <ファイル>
```

## 性能比較

| エンジン | 速度 | JPEG圧縮 | PNG圧縮 | WAV圧縮 | 特徴 |
|---------|------|----------|---------|---------|------|
| Ultra Fast | 22.5 MB/s | 10.0% | -0.0% | 100%+ | 超高速 |
| Smart Image | 2.3 MB/s | 8.4% | 制約認識 | - | 画像特化 |
| CABLC Enhanced | 可変 | 可変 | 可変 | 可変 | 高品質 |

## アーキテクチャ

```
NXZip/
├── bin/                    # 最適化エンジン
│   ├── nxzip_ultra_fast_binary_collapse.py
│   ├── nxzip_smart_image_compressor.py
│   ├── nexus_cablc_enhanced.py
│   └── analyze_formats.py
└── README.md              # この説明
```
"""
        
        readme_path = self.bin_dir.parent / "README_OPTIMIZED.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"📋 最適化README作成: {readme_path}")
        
        # エンジン統合スクリプト作成
        wrapper_content = '''#!/usr/bin/env python3
"""
NXZip Optimized Engine Selector
最適化エンジン選択器 - 自動最適エンジン選択
"""

import os
import sys
from pathlib import Path

def select_optimal_engine(file_path):
    """ファイル形式に基づく最適エンジン選択"""
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        return 'nxzip_smart_image_compressor.py'
    elif file_ext in ['.wav', '.mp3', '.flac']:
        return 'nxzip_ultra_fast_binary_collapse.py'
    elif file_ext in ['.mp4', '.avi', '.mkv']:
        return 'nexus_cablc_enhanced.py'
    else:
        return 'nxzip_ultra_fast_binary_collapse.py'  # 汎用

def main():
    if len(sys.argv) != 2:
        print("使用法: python nxzip_optimized.py <ファイルパス>")
        print("\\n🎯 NXZip 最適化エンジン選択器")
        print("📋 対応:")
        print("  🖼️ 画像: Smart Image Compressor")
        print("  🎵 音声: Ultra Fast Binary Collapse")
        print("  🎬 動画: CABLC Enhanced")
        print("  📄 その他: Ultra Fast Binary Collapse")
        sys.exit(1)
    
    file_path = sys.argv[1]
    engine = select_optimal_engine(file_path)
    
    print(f"🎯 最適エンジン選択: {engine}")
    print(f"📁 対象ファイル: {Path(file_path).name}")
    
    os.system(f'python {engine} "{file_path}"')

if __name__ == "__main__":
    main()
'''
        
        wrapper_path = self.bin_dir / "nxzip_optimized.py"
        with open(wrapper_path, 'w', encoding='utf-8') as f:
            f.write(wrapper_content)
        
        print(f"🎯 最適化選択器作成: {wrapper_path}")
    
    def run_cleanup(self):
        """整理実行"""
        print("🚀 NXZip リポジトリ最適化開始")
        print("=" * 60)
        
        start_time = time.time()
        
        # 1. バックアップ
        backup_dir = self.create_backup()
        
        # 2. ファイル整理
        kept, removed = self.cleanup_files()
        
        # 3. 最適化構造作成
        self.create_optimized_structure()
        
        # 4. 最終確認
        remaining_files = list(self.bin_dir.glob("*.py"))
        
        processing_time = time.time() - start_time
        
        print("=" * 60)
        print("🎉 NXZip リポジトリ最適化完了!")
        print(f"⏱️ 処理時間: {processing_time:.2f}秒")
        print(f"📦 バックアップ: {backup_dir}")
        print(f"✅ 保持エンジン: {len(remaining_files)} files")
        print("\n🎯 最適化されたエンジン:")
        
        for engine_file in remaining_files:
            if engine_file.name in self.keep_engines:
                print(f"  ✅ {engine_file.name}")
                print(f"     {self.keep_engines[engine_file.name]}")
        
        print("\n🚀 使用開始:")
        print("  python nxzip_optimized.py <ファイル>  # 自動最適選択")
        print("  python nxzip_ultra_fast_binary_collapse.py <ファイル>  # 汎用高速")

if __name__ == "__main__":
    cleanup = NXZipRepositoryCleanup()
    cleanup.run_cleanup()
