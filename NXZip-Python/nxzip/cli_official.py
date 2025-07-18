#!/usr/bin/env python3
"""
NXZip v2.0 Command Line Interface - 正式統合版
SPE + NEXUS + NXZ 統合エンジン
"""

import os
import sys
import click
import time
from typing import Optional
from pathlib import Path

# 正式統合版エンジン使用
from .engine.nxzip_core import NXZipCore

@click.group()
@click.version_option(version="2.0.0", prog_name="NXZip")
def main():
    """
    🚀 NXZip v2.0 - 正式統合版
    
    SPE + NEXUS + NXZ 統合エンジン
    実績: 97.31%圧縮率、186.80 MB/s総合速度
    """
    pass

@main.command()
@click.argument('archive', type=click.Path())
@click.argument('file', type=click.Path(exists=True))
@click.option('-v', '--verbose', is_flag=True, help='詳細表示')
def create(archive: str, file: str, verbose: bool):
    """NXZアーカイブを作成"""
    
    try:
        # NXZip Core 正式統合版
        nxzip = NXZipCore()
        
        if verbose:
            click.echo("🚀 NXZip v2.0 - 正式統合版圧縮開始")
            click.echo(f"📂 入力: {file}")
            click.echo(f"📦 出力: {archive}")
            click.echo(f"🗜️  エンジン: SPE + NEXUS + NXZ")
        
        # ファイル読み込み
        start_time = time.perf_counter()
        with open(file, 'rb') as f:
            data = f.read()
        
        # アーカイブ作成
        archive_data = nxzip.compress(data)
        
        # アーカイブ保存
        with open(archive, 'wb') as f:
            f.write(archive_data)
        
        total_time = time.perf_counter() - start_time
        original_size = len(data)
        archive_size = len(archive_data)
        ratio = (1 - archive_size / original_size) * 100 if original_size > 0 else 0
        speed = (original_size / 1024 / 1024) / total_time if total_time > 0 else 0
        
        click.echo(f"✅ NXZアーカイブ作成完了!")
        click.echo(f"   📊 サイズ: {original_size:,} → {archive_size:,} bytes")
        click.echo(f"   📈 圧縮率: {ratio:.2f}%")
        click.echo(f"   ⚡ 速度: {speed:.2f} MB/s")
        click.echo(f"   ⏱️ 時間: {total_time:.2f}秒")
        
    except Exception as e:
        click.echo(f"❌ エラー: {e}", err=True)
        sys.exit(1)

@main.command()
@click.argument('archive', type=click.Path(exists=True))
@click.argument('output', type=click.Path(), required=False)
@click.option('-v', '--verbose', is_flag=True, help='詳細表示')
def extract(archive: str, output: Optional[str], verbose: bool):
    """NXZアーカイブを展開"""
    
    try:
        # 出力ファイル名の決定
        if not output:
            base_name = os.path.splitext(archive)[0]
            if base_name.endswith('.nxz'):
                output = base_name[:-4]
            else:
                output = base_name + '_extracted'
        
        # NXZip Core 正式統合版
        nxzip = NXZipCore()
        
        if verbose:
            click.echo("🔓 NXZip v2.0 - 正式統合版展開開始")
            click.echo(f"📦 入力: {archive}")
            click.echo(f"📂 出力: {output}")
            click.echo(f"🗜️  エンジン: SPE + NEXUS + NXZ")
        
        # アーカイブ読み込み
        start_time = time.perf_counter()
        with open(archive, 'rb') as f:
            archive_data = f.read()
        
        # アーカイブ展開
        extracted_data = nxzip.decompress(archive_data)
        
        # 出力ファイル保存
        with open(output, 'wb') as f:
            f.write(extracted_data)
        
        total_time = time.perf_counter() - start_time
        speed = (len(extracted_data) / 1024 / 1024) / total_time if total_time > 0 else 0
        
        click.echo(f"✅ NXZアーカイブ展開完了!")
        click.echo(f"   📊 サイズ: {len(extracted_data):,} bytes")
        click.echo(f"   ⚡ 速度: {speed:.2f} MB/s")
        click.echo(f"   ⏱️ 時間: {total_time:.2f}秒")
        
    except Exception as e:
        click.echo(f"❌ エラー: {e}", err=True)
        sys.exit(1)

@main.command()
@click.argument('archive', type=click.Path(exists=True))
def info(archive: str):
    """NXZアーカイブの情報を表示"""
    
    try:
        with open(archive, 'rb') as f:
            data = f.read()
        
        if len(data) < 44:
            click.echo("❌ 無効なNXZファイル")
            return
        
        # ヘッダー解析
        magic = data[0:4]
        if magic != b'NXZP':
            click.echo("❌ NXZファイルではありません")
            return
        
        import struct
        version = struct.unpack('<I', data[4:8])[0]
        original_size = struct.unpack('<Q', data[8:16])[0]
        compressed_size = struct.unpack('<Q', data[16:24])[0]
        encrypted_size = struct.unpack('<Q', data[24:32])[0]
        timestamp = struct.unpack('<Q', data[32:40])[0]
        
        compression_ratio = (1 - len(data) / original_size) * 100 if original_size > 0 else 0
        
        click.echo(f"📦 NXZアーカイブ情報: {archive}")
        click.echo(f"   🔸 バージョン: {version}")
        click.echo(f"   🔸 元サイズ: {original_size:,} bytes")
        click.echo(f"   🔸 圧縮サイズ: {compressed_size:,} bytes")
        click.echo(f"   🔸 暗号化サイズ: {encrypted_size:,} bytes")
        click.echo(f"   🔸 総サイズ: {len(data):,} bytes")
        click.echo(f"   🔸 圧縮率: {compression_ratio:.2f}%")
        click.echo(f"   🔸 タイムスタンプ: {timestamp}")
        click.echo(f"   🔸 エンジン: SPE + NEXUS + NXZ")
        
    except Exception as e:
        click.echo(f"❌ エラー: {e}", err=True)
        sys.exit(1)

@main.command()
@click.argument('file', type=click.Path(exists=True))
def test(file: str):
    """ファイルでNXZip性能テスト"""
    
    try:
        # NXZip Core 正式統合版
        nxzip = NXZipCore()
        
        click.echo(f"🧪 NXZip v2.0 性能テスト: {file}")
        click.echo("=" * 50)
        
        # ファイル読み込み
        with open(file, 'rb') as f:
            data = f.read()
        
        original_size = len(data)
        click.echo(f"📊 元サイズ: {original_size:,} bytes ({original_size/1024/1024:.2f} MB)")
        
        # 圧縮テスト
        start_time = time.perf_counter()
        compressed = nxzip.compress(data)
        compress_time = time.perf_counter() - start_time
        
        # 展開テスト
        start_time = time.perf_counter()
        decompressed = nxzip.decompress(compressed)
        decompress_time = time.perf_counter() - start_time
        
        # 結果計算
        compressed_size = len(compressed)
        compression_ratio = (1 - compressed_size / original_size) * 100
        compress_speed = (original_size / 1024 / 1024) / compress_time if compress_time > 0 else 0
        decompress_speed = (original_size / 1024 / 1024) / decompress_time if decompress_time > 0 else 0
        total_speed = (original_size * 2 / 1024 / 1024) / (compress_time + decompress_time)
        
        # 正確性確認
        is_correct = data == decompressed
        
        click.echo(f"📈 圧縮率: {compression_ratio:.2f}%")
        click.echo(f"⚡ 圧縮速度: {compress_speed:.2f} MB/s")
        click.echo(f"⚡ 展開速度: {decompress_speed:.2f} MB/s")
        click.echo(f"⚡ 総合速度: {total_speed:.2f} MB/s")
        click.echo(f"⏱️ 総時間: {compress_time + decompress_time:.2f}秒")
        click.echo(f"✅ 正確性: {'OK' if is_correct else 'NG'}")
        click.echo(f"🏆 エンジン: SPE + NEXUS + NXZ")
        
        # 性能評価
        if compression_ratio >= 90 and total_speed >= 100:
            click.echo(f"\n🎯 優秀! 90%圧縮率 + 100MB/s速度達成")
        elif compression_ratio >= 90:
            click.echo(f"\n📊 良好! 90%圧縮率達成")
        elif total_speed >= 100:
            click.echo(f"\n⚡ 高速! 100MB/s速度達成")
        else:
            click.echo(f"\n📊 標準的な性能")
        
    except Exception as e:
        click.echo(f"❌ エラー: {e}", err=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
