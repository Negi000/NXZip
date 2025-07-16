#!/usr/bin/env python3
"""
NXZip v2.0 Command Line Interface
次世代アーカイブシステムのCLI
"""

import os
import sys
import click
from typing import Optional

from .formats.enhanced_nxz import SuperNXZipFile, NXZipError
from .utils.constants import CompressionAlgorithm, EncryptionAlgorithm, KDFAlgorithm


@click.group()
@click.version_option(version="2.0.0", prog_name="NXZip")
def main():
    """
    🚀 NXZip v2.0 - 次世代超高速・高圧縮・多重暗号化アーカイブシステム
    
    7Zipを超える圧縮率と超高速処理を実現する革新的なアーカイブツール
    """
    pass


@main.command()
@click.argument('archive', type=click.Path())
@click.argument('file', type=click.Path(exists=True))
@click.option('-p', '--password', help='暗号化パスワード')
@click.option('-c', '--compression', 
              type=click.Choice(['auto', 'zlib', 'lzma2', 'zstd']),
              default='auto', help='圧縮アルゴリズム')
@click.option('-e', '--encryption',
              type=click.Choice(['aes-gcm', 'xchacha20-poly1305']),
              default='aes-gcm', help='暗号化アルゴリズム')
@click.option('-k', '--kdf',
              type=click.Choice(['pbkdf2', 'scrypt']),
              default='pbkdf2', help='鍵導出方式')
@click.option('-l', '--level', type=click.IntRange(1, 9), default=6,
              help='圧縮レベル (1-9)')
@click.option('-v', '--verbose', is_flag=True, help='詳細表示')
def create(archive: str, file: str, password: Optional[str], 
          compression: str, encryption: str, kdf: str, 
          level: int, verbose: bool):
    """アーカイブを作成"""
    
    try:
        # NXZipファイルインスタンス作成
        nxzip = SuperNXZipFile(
            compression_algo=compression,
            encryption_algo=encryption,
            kdf_algo=kdf
        )
        
        if verbose:
            click.echo("🚀 NXZip v2.0 - 超高速圧縮開始")
            click.echo(f"📂 入力: {file}")
            click.echo(f"📦 出力: {archive}")
            click.echo(f"🗜️  圧縮: {compression} (レベル {level})")
            if password:
                click.echo(f"🔒 暗号化: {encryption} (KDF: {kdf})")
        
        # ファイル読み込み
        with open(file, 'rb') as f:
            data = f.read()
        
        # アーカイブ作成
        archive_data = nxzip.create_archive(
            data, 
            password, 
            level, 
            show_progress=verbose
        )
        
        # アーカイブ保存
        with open(archive, 'wb') as f:
            f.write(archive_data)
        
        if not verbose:
            original_size = len(data)
            archive_size = len(archive_data)
            ratio = (1 - archive_size / original_size) * 100 if original_size > 0 else 0
            click.echo(f"✅ アーカイブ作成完了: {original_size:,} → {archive_size:,} bytes ({ratio:.1f}% 削減)")
        
    except Exception as e:
        click.echo(f"❌ エラー: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('archive', type=click.Path(exists=True))
@click.argument('output', type=click.Path(), required=False)
@click.option('-p', '--password', help='復号化パスワード')
@click.option('-v', '--verbose', is_flag=True, help='詳細表示')
def extract(archive: str, output: Optional[str], password: Optional[str], verbose: bool):
    """アーカイブを展開"""
    
    try:
        # 出力ファイル名の決定
        if not output:
            base_name = os.path.splitext(archive)[0]
            if base_name.endswith('.nxz'):
                output = base_name[:-4]
            else:
                output = base_name + '_extracted'
        
        # NXZipファイルインスタンス作成
        nxzip = SuperNXZipFile()
        
        if verbose:
            click.echo("🔓 NXZip v2.0 - 超高速展開開始")
            click.echo(f"📦 入力: {archive}")
            click.echo(f"📁 出力: {output}")
        
        # アーカイブ読み込み
        with open(archive, 'rb') as f:
            archive_data = f.read()
        
        # 展開
        extracted_data = nxzip.extract_archive(
            archive_data,
            password,
            show_progress=verbose
        )
        
        # ファイル保存
        with open(output, 'wb') as f:
            f.write(extracted_data)
        
        if not verbose:
            click.echo(f"✅ 展開完了: {len(extracted_data):,} bytes → {output}")
        
    except Exception as e:
        click.echo(f"❌ エラー: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('archive', type=click.Path(exists=True))
def info(archive: str):
    """アーカイブ情報を表示"""
    
    try:
        nxzip = SuperNXZipFile()
        
        with open(archive, 'rb') as f:
            archive_data = f.read()
        
        info_data = nxzip.get_info(archive_data)
        
        click.echo("📊 NXZip アーカイブ情報")
        click.echo("=" * 40)
        click.echo(f"バージョン: {info_data['version']}")
        click.echo(f"元サイズ: {info_data['original_size']:,} bytes")
        click.echo(f"圧縮後サイズ: {info_data['compressed_size']:,} bytes")
        click.echo(f"アーカイブサイズ: {info_data['archive_size']:,} bytes")
        click.echo(f"圧縮アルゴリズム: {info_data['compression_algorithm']}")
        click.echo(f"圧縮率: {info_data['compression_ratio']:.1f}%")
        click.echo(f"総圧縮率: {info_data['total_compression_ratio']:.1f}%")
        click.echo(f"暗号化: {'有効' if info_data['is_encrypted'] else '無効'}")
        if info_data['is_encrypted']:
            click.echo(f"  アルゴリズム: {info_data['encryption_algorithm']}")
            click.echo(f"  KDF: {info_data['kdf_algorithm']}")
        click.echo(f"チェックサム: {info_data['checksum']}")
        
    except Exception as e:
        click.echo(f"❌ エラー: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('archive', type=click.Path(exists=True))
@click.option('-p', '--password', help='復号化パスワード')
def test(archive: str, password: Optional[str]):
    """アーカイブをテスト"""
    
    try:
        nxzip = SuperNXZipFile()
        
        click.echo(f"🧪 アーカイブテスト中: {archive}")
        
        with open(archive, 'rb') as f:
            archive_data = f.read()
        
        # 情報取得テスト
        info_data = nxzip.get_info(archive_data)
        click.echo(f"✅ ヘッダー: 正常 ({info_data['version']})")
        
        # 展開テスト
        extracted_data = nxzip.extract_archive(archive_data, password, show_progress=False)
        click.echo(f"✅ 展開: 正常 ({len(extracted_data):,} bytes)")
        
        click.echo("✅ アーカイブは正常です")
        
    except Exception as e:
        click.echo(f"❌ アーカイブエラー: {e}", err=True)
        sys.exit(1)


@main.command()
def benchmark():
    """ベンチマークテストを実行"""
    
    try:
        import tempfile
        import time
        
        click.echo("🚀 NXZip v2.0 ベンチマーク開始")
        click.echo("=" * 50)
        
        # テストデータ作成
        test_sizes = [1024, 10240, 102400, 1048576]  # 1KB, 10KB, 100KB, 1MB
        
        for size in test_sizes:
            click.echo(f"\n📊 テストサイズ: {size:,} bytes")
            
            # ランダムデータ生成
            import secrets
            test_data = secrets.token_bytes(size)
            
            nxzip = SuperNXZipFile()
            
            # 圧縮テスト
            start_time = time.time()
            archive = nxzip.create_archive(test_data)
            compress_time = time.time() - start_time
            
            # 展開テスト
            start_time = time.time()
            restored = nxzip.extract_archive(archive)
            decompress_time = time.time() - start_time
            
            # 結果表示
            ratio = (1 - len(archive) / size) * 100
            compress_speed = size / compress_time / 1024 / 1024
            decompress_speed = size / decompress_time / 1024 / 1024
            
            click.echo(f"  圧縮率: {ratio:.1f}% ({size:,} → {len(archive):,} bytes)")
            click.echo(f"  圧縮速度: {compress_speed:.1f} MB/s ({compress_time:.3f}s)")
            click.echo(f"  展開速度: {decompress_speed:.1f} MB/s ({decompress_time:.3f}s)")
            click.echo(f"  整合性: {'✅ OK' if restored == test_data else '❌ NG'}")
        
        click.echo("\n✅ ベンチマーク完了")
        
    except Exception as e:
        click.echo(f"❌ ベンチマークエラー: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
