#!/usr/bin/env python3
"""
🖥️ NXZip Command Line Interface

次世代アーカイブシステム - コマンドライン統合ツール
Copyright (c) 2025 NXZip Project
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path
from typing import List, Optional

from ..core.archive import NXZipArchive
from ..core.nexus import NEXUSCompressor
from ..crypto.spe import SPECrypto
from ..formats.nxz_format import NXZFormatValidator
from .. import __version__, __description__


class NXZipCLI:
    """🖥️ NXZip コマンドライン処理"""
    
    def __init__(self):
        self.compressor = NEXUSCompressor()
        self.crypto = SPECrypto()
        self.validator = NXZFormatValidator()
    
    def create_archive(self, archive_path: str, input_paths: List[str], 
                      password: Optional[str] = None, recursive: bool = True,
                      compression: str = 'nexus', verbose: bool = False) -> bool:
        """アーカイブ作成"""
        try:
            print(f"🚀 NXZip アーカイブ作成開始: {archive_path}")
            
            # アーカイブオブジェクト作成
            archive = NXZipArchive(archive_path, password)
            
            total_files = 0
            start_time = time.time()
            
            # 入力パス処理
            for input_path in input_paths:
                if os.path.isfile(input_path):
                    # ファイル追加
                    if archive.add_file(input_path):
                        total_files += 1
                        if verbose:
                            print(f"  ✅ ファイル追加: {input_path}")
                elif os.path.isdir(input_path):
                    # ディレクトリ追加
                    added_count = archive.add_directory(input_path, recursive)
                    total_files += added_count
                    if verbose:
                        print(f"  📁 ディレクトリ追加: {input_path} ({added_count} ファイル)")
                else:
                    print(f"  ⚠️ パスが見つかりません: {input_path}")
            
            # アーカイブ保存
            if archive.save():
                processing_time = time.time() - start_time
                stats = archive.get_stats()
                
                print(f"\n🎉 アーカイブ作成完了!")
                print(f"📄 ファイル数: {total_files}")
                print(f"📊 元サイズ: {self._format_size(stats['total_original_size'])}")
                print(f"📦 圧縮サイズ: {self._format_size(stats['total_compressed_size'])}")
                print(f"⚡ 圧縮率: {stats['overall_compression_ratio']:.2f}%")
                print(f"🔒 暗号化: {'有効' if password else '無効'}")
                print(f"⏱️ 処理時間: {processing_time:.2f}秒")
                
                return True
            else:
                print("❌ アーカイブ保存に失敗しました")
                return False
                
        except Exception as e:
            print(f"❌ アーカイブ作成エラー: {e}")
            return False
    
    def extract_archive(self, archive_path: str, output_dir: str,
                       password: Optional[str] = None, files: Optional[List[str]] = None,
                       verbose: bool = False) -> bool:
        """アーカイブ展開"""
        try:
            print(f"📂 NXZip アーカイブ展開開始: {archive_path}")
            
            if not os.path.exists(archive_path):
                print(f"❌ アーカイブファイルが見つかりません: {archive_path}")
                return False
            
            # アーカイブ読み込み
            archive = NXZipArchive(archive_path, password)
            if not archive.load():
                print("❌ アーカイブの読み込みに失敗しました")
                return False
            
            # 出力ディレクトリ作成
            os.makedirs(output_dir, exist_ok=True)
            
            start_time = time.time()
            
            if files:
                # 指定ファイルのみ展開
                extracted_count = 0
                for filename in files:
                    output_path = os.path.join(output_dir, filename)
                    if archive.extract_file(filename, output_path):
                        extracted_count += 1
                        if verbose:
                            print(f"  ✅ 展開: {filename}")
                    else:
                        print(f"  ❌ 展開失敗: {filename}")
            else:
                # 全ファイル展開
                extracted_count = archive.extract_all(output_dir)
                if verbose:
                    entries = archive.list_entries()
                    for entry in entries:
                        print(f"  ✅ 展開: {entry['filepath']}")
            
            processing_time = time.time() - start_time
            
            print(f"\n🎉 展開完了!")
            print(f"📄 展開ファイル数: {extracted_count}")
            print(f"📁 出力先: {output_dir}")
            print(f"⏱️ 処理時間: {processing_time:.2f}秒")
            
            return True
            
        except Exception as e:
            print(f"❌ 展開エラー: {e}")
            return False
    
    def list_archive(self, archive_path: str, password: Optional[str] = None,
                    detailed: bool = False) -> bool:
        """アーカイブ内容一覧表示"""
        try:
            if not os.path.exists(archive_path):
                print(f"❌ アーカイブファイルが見つかりません: {archive_path}")
                return False
            
            # アーカイブ読み込み
            archive = NXZipArchive(archive_path, password)
            if not archive.load():
                print("❌ アーカイブの読み込みに失敗しました")
                return False
            
            # 統計情報
            stats = archive.get_stats()
            print(f"📦 NXZip アーカイブ: {archive_path}")
            print(f"📄 ファイル数: {stats['total_files']}")
            print(f"📊 合計サイズ: {self._format_size(stats['total_original_size'])}")
            print(f"📦 圧縮サイズ: {self._format_size(stats['total_compressed_size'])}")
            print(f"⚡ 圧縮率: {stats['overall_compression_ratio']:.2f}%")
            print(f"🔒 暗号化: {'有' if stats['has_encryption'] else '無'}")
            print()
            
            # エントリ一覧
            entries = archive.list_entries()
            
            if detailed:
                # 詳細表示
                print(f"{'ファイル名':<40} {'元サイズ':<12} {'圧縮サイズ':<12} {'圧縮率':<8} {'暗号化':<6}")
                print("-" * 85)
                
                for entry in entries:
                    encrypted = "有" if entry['is_encrypted'] else "無"
                    print(f"{entry['filepath']:<40} "
                          f"{self._format_size(entry['original_size']):<12} "
                          f"{self._format_size(entry['compressed_size']):<12} "
                          f"{entry['compression_ratio']:.1f}%{'':<3} "
                          f"{encrypted:<6}")
            else:
                # 簡易表示
                for entry in entries:
                    encrypted_mark = "🔒" if entry['is_encrypted'] else ""
                    print(f"  {entry['filepath']} {encrypted_mark}")
            
            return True
            
        except Exception as e:
            print(f"❌ 一覧表示エラー: {e}")
            return False
    
    def test_archive(self, archive_path: str, password: Optional[str] = None) -> bool:
        """アーカイブ整合性テスト"""
        try:
            print(f"🔍 NXZip アーカイブテスト: {archive_path}")
            
            if not os.path.exists(archive_path):
                print(f"❌ アーカイブファイルが見つかりません: {archive_path}")
                return False
            
            # フォーマット検証
            validation_result = self.validator.validate_archive(archive_path)
            
            if not validation_result['valid']:
                print(f"❌ フォーマット検証失敗: {validation_result['error']}")
                return False
            
            print(f"✅ フォーマット検証: OK")
            print(f"📋 バージョン: {validation_result['version']}")
            print(f"📄 エントリ数: {validation_result['entry_count']}")
            
            # アーカイブ読み込みテスト
            archive = NXZipArchive(archive_path, password)
            if not archive.load():
                print("❌ アーカイブ読み込み失敗")
                return False
            
            print("✅ アーカイブ読み込み: OK")
            
            # TODO: より詳細な整合性チェック
            # - チェックサム検証
            # - 展開テスト
            # - 暗号化検証
            
            print("🎉 アーカイブテスト完了: 正常")
            return True
            
        except Exception as e:
            print(f"❌ テストエラー: {e}")
            return False
    
    def benchmark(self, test_files: List[str], output_file: Optional[str] = None) -> bool:
        """圧縮性能ベンチマーク"""
        try:
            print("⚡ NXZip ベンチマーク実行")
            
            results = []
            total_start = time.time()
            
            for test_file in test_files:
                if not os.path.exists(test_file):
                    print(f"⚠️ テストファイルが見つかりません: {test_file}")
                    continue
                
                print(f"🔄 テスト中: {os.path.basename(test_file)}")
                
                # ファイル読み込み
                with open(test_file, 'rb') as f:
                    data = f.read()
                
                # 圧縮テスト
                start_time = time.time()
                compressed_data, metadata = self.compressor.compress(data, test_file)
                compression_time = time.time() - start_time
                
                # 結果記録
                result = {
                    'filename': os.path.basename(test_file),
                    'original_size': len(data),
                    'compressed_size': len(compressed_data),
                    'compression_ratio': metadata['ratio'],
                    'compression_time': compression_time,
                    'throughput_mbps': (len(data) / 1024 / 1024) / compression_time if compression_time > 0 else 0,
                    'format_detected': metadata['format']
                }
                results.append(result)
                
                print(f"  📊 圧縮率: {result['compression_ratio']:.2f}%")
                print(f"  ⚡ スループット: {result['throughput_mbps']:.2f} MB/s")
            
            total_time = time.time() - total_start
            
            # 統計計算
            if results:
                avg_ratio = sum(r['compression_ratio'] for r in results) / len(results)
                avg_throughput = sum(r['throughput_mbps'] for r in results) / len(results)
                total_original = sum(r['original_size'] for r in results)
                total_compressed = sum(r['compressed_size'] for r in results)
                overall_ratio = (1 - total_compressed / total_original) * 100 if total_original > 0 else 0
                
                print(f"\n📊 ベンチマーク結果:")
                print(f"  ファイル数: {len(results)}")
                print(f"  平均圧縮率: {avg_ratio:.2f}%")
                print(f"  総合圧縮率: {overall_ratio:.2f}%")
                print(f"  平均スループット: {avg_throughput:.2f} MB/s")
                print(f"  総処理時間: {total_time:.2f}秒")
                
                # 結果保存
                if output_file:
                    benchmark_data = {
                        'timestamp': int(time.time()),
                        'nxzip_version': __version__,
                        'total_files': len(results),
                        'average_compression_ratio': avg_ratio,
                        'overall_compression_ratio': overall_ratio,
                        'average_throughput_mbps': avg_throughput,
                        'total_processing_time': total_time,
                        'individual_results': results
                    }
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(benchmark_data, f, ensure_ascii=False, indent=2)
                    
                    print(f"📄 結果保存: {output_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ ベンチマークエラー: {e}")
            return False
    
    def _format_size(self, size_bytes: int) -> str:
        """サイズ整形"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"


def create_parser() -> argparse.ArgumentParser:
    """コマンドライン引数パーサー作成"""
    parser = argparse.ArgumentParser(
        prog='nxzip',
        description=f'{__description__} (v{__version__})',
        epilog='Examples:\n'
               '  nxzip create archive.nxz file1.txt file2.txt\n'
               '  nxzip extract archive.nxz -o output_dir\n'
               '  nxzip list archive.nxz\n'
               '  nxzip test archive.nxz',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('-v', '--version', action='version', version=f'NXZip {__version__}')
    
    subparsers = parser.add_subparsers(dest='command', help='利用可能なコマンド')
    
    # create サブコマンド
    create_parser = subparsers.add_parser('create', help='アーカイブ作成')
    create_parser.add_argument('archive', help='作成するアーカイブファイル')
    create_parser.add_argument('inputs', nargs='+', help='圧縮するファイル/ディレクトリ')
    create_parser.add_argument('-p', '--password', help='暗号化パスワード')
    create_parser.add_argument('-r', '--recursive', action='store_true', default=True, help='ディレクトリ再帰処理')
    create_parser.add_argument('-c', '--compression', default='nexus', choices=['nexus', 'lzma', 'zlib'], help='圧縮アルゴリズム')
    create_parser.add_argument('--verbose', action='store_true', help='詳細出力')
    
    # extract サブコマンド
    extract_parser = subparsers.add_parser('extract', help='アーカイブ展開')
    extract_parser.add_argument('archive', help='展開するアーカイブファイル')
    extract_parser.add_argument('-o', '--output', default='.', help='出力ディレクトリ')
    extract_parser.add_argument('-p', '--password', help='復号化パスワード')
    extract_parser.add_argument('-f', '--files', nargs='*', help='展開する特定ファイル')
    extract_parser.add_argument('--verbose', action='store_true', help='詳細出力')
    
    # list サブコマンド
    list_parser = subparsers.add_parser('list', help='アーカイブ内容一覧')
    list_parser.add_argument('archive', help='一覧表示するアーカイブファイル')
    list_parser.add_argument('-p', '--password', help='復号化パスワード')
    list_parser.add_argument('-d', '--detailed', action='store_true', help='詳細表示')
    
    # test サブコマンド
    test_parser = subparsers.add_parser('test', help='アーカイブ整合性テスト')
    test_parser.add_argument('archive', help='テストするアーカイブファイル')
    test_parser.add_argument('-p', '--password', help='復号化パスワード')
    
    # benchmark サブコマンド
    benchmark_parser = subparsers.add_parser('benchmark', help='圧縮性能ベンチマーク')
    benchmark_parser.add_argument('files', nargs='+', help='テストファイル')
    benchmark_parser.add_argument('-o', '--output', help='結果出力ファイル')
    
    return parser


def main():
    """メイン関数"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    cli = NXZipCLI()
    
    try:
        if args.command == 'create':
            success = cli.create_archive(
                args.archive, args.inputs, args.password,
                args.recursive, args.compression, args.verbose
            )
        elif args.command == 'extract':
            success = cli.extract_archive(
                args.archive, args.output, args.password,
                args.files, args.verbose
            )
        elif args.command == 'list':
            success = cli.list_archive(
                args.archive, args.password, args.detailed
            )
        elif args.command == 'test':
            success = cli.test_archive(args.archive, args.password)
        elif args.command == 'benchmark':
            success = cli.benchmark(args.files, args.output)
        else:
            parser.print_help()
            return 1
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⚠️ ユーザーによって中断されました")
        return 130
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())


# 公開API
__all__ = ['NXZipCLI', 'main', 'create_parser']
