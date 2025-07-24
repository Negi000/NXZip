"""
NEXUS圧縮エンジン - 実践ファイルテストスイート
.nxz形式での完全可逆性テスト
"""

import os
import hashlib
import time
from pathlib import Path
from nexus_compression_engine import NEXUSCompressor
import shutil

class NXZipFileTester:
    """実践的なファイル圧縮テスター"""
    
    def __init__(self, sample_dir: str):
        from nexus_compression_engine import MLCompressionConfig
        
        self.sample_dir = Path(sample_dir)
        self.output_dir = self.sample_dir / "nxz_output"
        self.output_dir.mkdir(exist_ok=True)
        
        # ログ出力を最小限に制御
        config = MLCompressionConfig(verbose=False)  # ログ無効化
        self.compressor = NEXUSCompressor(config)
        
    def test_all_files(self):
        """サンプルディレクトリの全ファイルをテスト"""
        print("🧪 NEXUS実践ファイルテスト開始")
        print(f"📂 サンプルディレクトリ: {self.sample_dir}")
        print(f"📁 出力ディレクトリ: {self.output_dir}")
        print("=" * 60)
        
        # テスト対象ファイルを収集（サイズ制限を追加）
        test_files = []
        for file_path in self.sample_dir.iterdir():
            if file_path.is_file() and not file_path.name.startswith('.'):
                # 適当なサイズのファイルを選択（効率化のため制限）
                size = file_path.stat().st_size
                if 100 <= size <= 10 * 1024 * 1024:  # 100B ~ 10MB（範囲縮小）
                    test_files.append((file_path, size))
        
        # サイズ順にソート（小さいファイルから処理）
        test_files.sort(key=lambda x: x[1])
        
        print(f"📋 テスト対象ファイル: {len(test_files)}個")
        
        total_original = 0
        total_compressed = 0
        success_count = 0
        
        for file_path, file_size in test_files:
            try:
                # 大きなファイルは警告表示
                if file_size > 5 * 1024 * 1024:
                    print(f"⚠️  大ファイル処理: {self._format_size(file_size)}")
                
                result = self.test_single_file(file_path)
                if result:
                    total_original += result['original_size']
                    total_compressed += result['compressed_size']
                    success_count += 1
            except Exception as e:
                print(f"❌ {file_path.name}: エラー - {e}")
        
        # 総合結果
        print("\n" + "=" * 60)
        print("🎯 総合結果")
        print(f"成功ファイル: {success_count}/{len(test_files)}")
        print(f"総元サイズ: {self._format_size(total_original)}")
        print(f"総圧縮サイズ: {self._format_size(total_compressed)}")
        
        if total_original > 0:
            overall_ratio = (total_compressed / total_original) * 100
            print(f"総合圧縮率: {overall_ratio:.1f}%")
            print(f"総合削減: {100 - overall_ratio:.1f}%")
        
        print("🏁 テスト完了")
        
    def test_single_file(self, file_path: Path) -> dict:
        """単一ファイルの圧縮・展開テスト"""
        print(f"\n📄 テスト: {file_path.name}")
        
        # 元ファイル読み込み
        with open(file_path, 'rb') as f:
            original_data = f.read()
        
        original_size = len(original_data)
        original_hash = hashlib.sha256(original_data).hexdigest()
        
        print(f"   元サイズ: {self._format_size(original_size)}")
        print(f"   元ハッシュ: {original_hash[:16]}...")
        
        # タイムアウト設定（大ファイル用）
        timeout_seconds = 30 if original_size < 1024*1024 else 120
        
        # 圧縮
        compress_start = time.time()
        try:
            compressed_data = self.compressor.compress(original_data)
            compress_time = time.time() - compress_start
            
            # タイムアウトチェック
            if compress_time > timeout_seconds:
                print(f"   ⏰ タイムアウト: {compress_time:.1f}s > {timeout_seconds}s")
                return None
            
            compressed_size = len(compressed_data)
            compression_ratio = (compressed_size / original_size) * 100
            
            print(f"   圧縮サイズ: {self._format_size(compressed_size)}")
            print(f"   圧縮率: {compression_ratio:.1f}%")
            print(f"   圧縮時間: {compress_time:.3f}s")
            
            # .nxz ファイルとして保存
            nxz_file = self.output_dir / f"{file_path.stem}.nxz"
            with open(nxz_file, 'wb') as f:
                f.write(compressed_data)
            print(f"   保存: {nxz_file.name}")
            
            # 展開
            decompress_start = time.time()
            decompressed_data = self.compressor.decompress(compressed_data)
            decompress_time = time.time() - decompress_start
            
            decompressed_hash = hashlib.sha256(decompressed_data).hexdigest()
            
            print(f"   展開時間: {decompress_time:.3f}s")
            print(f"   展開ハッシュ: {decompressed_hash[:16]}...")
            
            # 完全性チェック
            is_identical = (original_hash == decompressed_hash)
            size_match = (len(original_data) == len(decompressed_data))
            
            if is_identical and size_match:
                print("   ✅ 完全可逆性確認")
                
                # 復元ファイルを保存
                restored_file = self.output_dir / f"{file_path.stem}_restored{file_path.suffix}"
                with open(restored_file, 'wb') as f:
                    f.write(decompressed_data)
                print(f"   復元保存: {restored_file.name}")
                
                # 効率性評価
                if compression_ratio < 100:
                    reduction = 100 - compression_ratio
                    print(f"   📈 削減率: {reduction:.1f}%")
                else:
                    expansion = compression_ratio - 100
                    print(f"   📉 膨張率: {expansion:.1f}%")
                
                return {
                    'file_name': file_path.name,
                    'original_size': original_size,
                    'compressed_size': compressed_size,
                    'compression_ratio': compression_ratio,
                    'compress_time': compress_time,
                    'decompress_time': decompress_time,
                    'success': True
                }
            else:
                print("   ❌ データ破損検出")
                print(f"      サイズ一致: {size_match}")
                print(f"      ハッシュ一致: {is_identical}")
                return None
                
        except Exception as e:
            compress_time = time.time() - compress_start
            print(f"   ❌ 圧縮エラー ({compress_time:.1f}s): {e}")
            return None
    
    def compare_with_existing(self, file_path: Path):
        """既存の圧縮形式との比較"""
        print(f"\n🔍 圧縮形式比較: {file_path.name}")
        
        # 7z ファイルが存在する場合の比較
        sevenz_file = file_path.with_suffix('.7z')
        if sevenz_file.exists():
            sevenz_size = sevenz_file.stat().st_size
            print(f"   7z サイズ: {self._format_size(sevenz_size)}")
            
        # NXZ 圧縮
        try:
            with open(file_path, 'rb') as f:
                original_data = f.read()
            
            compressed_data = self.compressor.compress(original_data)
            nxz_size = len(compressed_data)
            
            print(f"   NXZ サイズ: {self._format_size(nxz_size)}")
            
            if sevenz_file.exists():
                ratio = (nxz_size / sevenz_size) * 100
                print(f"   NXZ vs 7z: {ratio:.1f}%")
                
        except Exception as e:
            print(f"   ❌ 比較エラー: {e}")
    
    def _format_size(self, size_bytes: int) -> str:
        """ファイルサイズの人間可読形式変換"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f}TB"

def main():
    """メイン実行"""
    sample_dir = r"C:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\sample"
    
    if not os.path.exists(sample_dir):
        print(f"❌ サンプルディレクトリが見つかりません: {sample_dir}")
        return
    
    tester = NXZipFileTester(sample_dir)
    tester.test_all_files()
    
    # 特定ファイルの詳細比較
    print("\n" + "="*60)
    print("🔍 既存形式との比較分析")
    
    sample_path = Path(sample_dir)
    for txt_file in sample_path.glob("*.txt"):
        tester.compare_with_existing(txt_file)
    
    for png_file in sample_path.glob("*.png"):
        if png_file.stat().st_size < 1024 * 1024:  # 1MB未満
            tester.compare_with_existing(png_file)
            break

if __name__ == "__main__":
    main()
