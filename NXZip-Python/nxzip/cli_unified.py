#!/usr/bin/env python3
"""
NXZip CLI - 統合圧縮システム
AV1/SRLA/AVIF制約除去技術による次世代圧縮
"""

import argparse
import os
import sys
import time
from pathlib import Path

# プロジェクトパス追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# NXZip Core v2.0を使用（SPE + NEXUS TMC統合）
nxzip_core_path = Path("c:\\Users\\241822\\Desktop\\新しいフォルダー (2)\\NXZip\\NXZip-Release")
sys.path.insert(0, str(nxzip_core_path))

try:
    from nxzip_core import NXZipCore, NXZipContainer
    NXZIP_CORE_AVAILABLE = True
    print("✅ NXZip Core v2.0 (SPE + NEXUS TMC) loaded")
except ImportError as e:
    # フォールバック: NEXUS Unified
    from nxzip.engine.nexus_unified import NEXUSUnified
    NXZIP_CORE_AVAILABLE = False
    print(f"⚠️ NXZip Core読み込み失敗、フォールバック: {e}")

def format_size(size_bytes):
    """サイズをMBで表示"""
    if size_bytes == 0:
        return "0 B"
    size_mb = size_bytes / (1024 * 1024)
    if size_mb < 1:
        return f"{size_bytes} B"
    else:
        return f"{size_mb:.2f} MB"

def compress_file(input_file, output_file=None):
    """ファイル圧縮 - NXZip Core v2.0 (SPE + NEXUS TMC) 使用"""
    print(f"🔥 NXZip - 次世代圧縮システム")
    print(f"📄 圧縮: {input_file}")
    
    # 入力ファイル確認
    if not os.path.exists(input_file):
        print(f"❌ エラー: ファイル '{input_file}' が見つかりません")
        return False
    
    # 出力ファイル名生成
    if output_file is None:
        output_file = str(input_file) + ".nxz"
    
    # データ読み込み
    print(f"📖 データ読み込み中...")
    with open(input_file, 'rb') as f:
        data = f.read()
    
    original_size = len(data)
    print(f"📊 元サイズ: {format_size(original_size)}")
    
    # NXZip Core v2.0 または フォールバック選択
    if NXZIP_CORE_AVAILABLE:
        # 🚀 NXZip Core v2.0 使用 (SPE + NEXUS TMC)
        print(f"🚀 NXZip Core v2.0 (SPE + NEXUS TMC) 使用")
        
        nxzip_core = NXZipCore()
        
        # 圧縮実行
        print(f"🚀 圧縮中...")
        start_time = time.perf_counter()
        result = nxzip_core.compress(data, mode="balanced", filename=input_file)
        compress_time = time.perf_counter() - start_time
        
        if result.success:
            # NXZipコンテナとしてパック
            container_data = NXZipContainer.pack(
                result.compressed_data, 
                result.metadata,
                os.path.basename(input_file)
            )
            
            # 結果保存
            with open(output_file, 'wb') as f:
                f.write(container_data)
            
            # 結果表示
            compressed_size = len(container_data)
            ratio = result.compression_ratio
            speed = (original_size / 1024 / 1024) / compress_time
            
            print(f"✅ 完了: {output_file}")
            print(f"📊 圧縮率: {ratio:.1f}%")
            print(f"⚡ 速度: {speed:.1f} MB/s")
            print(f"💾 圧縮後: {format_size(compressed_size)}")
            print(f"🔐 使用技術: SPE + NEXUS TMC v2.0")
            
            # 技術詳細表示
            metadata = result.metadata
            if 'stages' in metadata:
                for stage_name, stage_info in metadata['stages']:
                    if stage_name == 'spe_integration' and stage_info.get('spe_applied'):
                        print(f"   🔐 SPE: 構造保持変換実行")
                    elif stage_name == 'tmc_transform' and stage_info.get('transforms_applied'):
                        transforms = stage_info['transforms_applied']
                        print(f"   🔄 TMC: {', '.join(transforms)} 実行")
            
            return True
        else:
            print(f"❌ NXZip Core圧縮失敗: {result.error_message}")
            return False
    else:
        # フォールバック: NEXUS Unified
        print(f"⚠️ フォールバック: NEXUS Unified使用")
        
        nexus = NEXUSUnified()
        
        # 圧縮実行
        print(f"🚀 圧縮中...")
        start_time = time.perf_counter()
        compressed = nexus.compress(data)
        compress_time = time.perf_counter() - start_time
        
        # 結果保存
        with open(output_file, 'wb') as f:
            f.write(compressed)
        
        # 結果表示
        compressed_size = len(compressed)
        ratio = (1 - compressed_size / original_size) * 100
        speed = (original_size / 1024 / 1024) / compress_time
        
        print(f"✅ 完了: {output_file}")
        print(f"📊 圧縮率: {ratio:.1f}%")
        print(f"⚡ 速度: {speed:.1f} MB/s")
        print(f"💾 圧縮後: {format_size(compressed_size)}")
        print(f"⚠️ 使用技術: NEXUS Unified (フォールバック)")
        
        return True

def decompress_file(input_file, output_file=None):
    """ファイル展開 - NXZip Core v2.0 対応"""
    print(f"💨 NXZip - 統合展開システム")
    print(f"📄 展開: {input_file}")
    
    # 入力ファイル確認
    if not os.path.exists(input_file):
        print(f"❌ エラー: ファイル '{input_file}' が見つかりません")
        return False
    
    # 出力ファイル名生成
    if output_file is None:
        if input_file.endswith('.nxz'):
            output_file = input_file[:-4]
        else:
            output_file = input_file + ".restored"
    
    # データ読み込み
    print(f"📖 データ読み込み中...")
    with open(input_file, 'rb') as f:
        compressed = f.read()
    
    compressed_size = len(compressed)
    print(f"📊 圧縮サイズ: {format_size(compressed_size)}")
    
    # NXZip Core v2.0 または フォールバック選択
    if NXZIP_CORE_AVAILABLE:
        # 🚀 NXZip Core v2.0 使用
        try:
            print(f"🚀 NXZip Core v2.0 (SPE + NEXUS TMC) 展開中...")
            
            # NXZipコンテナ形式の場合
            if compressed.startswith(b'NXZIP200'):
                compressed_data, compression_info = NXZipContainer.unpack(compressed)
                
                nxzip_core = NXZipCore()
                start_time = time.perf_counter()
                result = nxzip_core.decompress(compressed_data, compression_info)
                decompress_time = time.perf_counter() - start_time
                
                if result.success:
                    decompressed = result.decompressed_data
                    print(f"✅ NXZip Core v2.0展開成功")
                else:
                    raise Exception(f"NXZip Core展開失敗: {result.error_message}")
            else:
                # フォールバック: NEXUS Unified形式
                raise Exception("NXZip Core v2.0形式ではありません")
                
        except Exception as e:
            print(f"⚠️ NXZip Core展開失敗: {e}")
            print(f"📂 フォールバック: NEXUS Unified展開")
            
            nexus = NEXUSUnified()
            start_time = time.perf_counter()
            decompressed = nexus.decompress(compressed)
            decompress_time = time.perf_counter() - start_time
    else:
        # フォールバック: NEXUS Unified
        print(f"📂 NEXUS Unified展開中...")
        
        nexus = NEXUSUnified()
        start_time = time.perf_counter()
        decompressed = nexus.decompress(compressed)
        decompress_time = time.perf_counter() - start_time
    
    # 結果保存
    with open(output_file, 'wb') as f:
        f.write(decompressed)
    
    # 結果表示
    decompressed_size = len(decompressed)
    speed = (decompressed_size / 1024 / 1024) / decompress_time
    
    print(f"✅ 完了: {output_file}")
    print(f"📊 展開サイズ: {format_size(decompressed_size)}")
    print(f"⚡ 速度: {speed:.1f} MB/s")
    
    return True
    nexus = NEXUSUnified()
    
    # 展開実行
    print(f"💨 展開中...")
    start_time = time.perf_counter()
    decompressed = nexus.decompress(compressed)
    decomp_time = time.perf_counter() - start_time
    
    # 結果保存
    with open(output_file, 'wb') as f:
        f.write(decompressed)
    
    # 結果表示
    original_size = len(decompressed)
    speed = (original_size / 1024 / 1024) / decomp_time
    
    print(f"✅ 完了: {output_file}")
    print(f"⚡ 速度: {speed:.1f} MB/s")
    print(f"💾 展開サイズ: {format_size(original_size)}")
    
    return True

def test_nexus():
    """NEXUSテスト"""
    print("🧪 NEXUS統合テスト")
    print("=" * 50)
    
    # テストファイル
    test_files = [
        r"C:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\sample\出庫実績明細_202412.txt",
        r"C:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\sample\Python基礎講座3_4月26日-3.mp4"
    ]
    
    nexus = NEXUSUnified()
    
    for file_path in test_files:
        path = Path(file_path)
        if not path.exists():
            print(f"⚠️ スキップ: {path.name}")
            continue
            
        print(f"\n📄 テスト: {path.name}")
        
        with open(path, 'rb') as f:
            data = f.read()
        
        # 圧縮
        start = time.perf_counter()
        compressed = nexus.compress(data)
        compress_time = time.perf_counter() - start
        
        # 展開
        start = time.perf_counter()
        decompressed = nexus.decompress(compressed)
        decomp_time = time.perf_counter() - start
        
        # 結果
        ratio = (1 - len(compressed) / len(data)) * 100
        comp_speed = (len(data) / 1024 / 1024) / compress_time
        decomp_speed = (len(data) / 1024 / 1024) / decomp_time
        correct = data == decompressed
        
        print(f"📊 圧縮率: {ratio:.1f}%")
        print(f"⚡ 圧縮速度: {comp_speed:.1f} MB/s")
        print(f"💨 展開速度: {decomp_speed:.1f} MB/s")
        print(f"✅ 正確性: {'OK' if correct else 'NG'}")

def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="NXZip - 次世代圧縮システム")
    parser.add_argument("action", choices=["compress", "decompress", "test"], 
                       help="実行するアクション")
    parser.add_argument("input_file", nargs="?", help="入力ファイル")
    parser.add_argument("output_file", nargs="?", help="出力ファイル")
    parser.add_argument("--verbose", "-v", action="store_true", help="詳細出力")
    
    args = parser.parse_args()
    
    if args.action == "test":
        test_nexus()
        return
    
    if not args.input_file:
        print("エラー: 入力ファイルが指定されていません")
        return
    
    # 処理実行
    if args.action == "compress":
        compress_file(args.input_file, args.output_file)
    elif args.action == "decompress":
        decompress_file(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
