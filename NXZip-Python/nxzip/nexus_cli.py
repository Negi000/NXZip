#!/usr/bin/env python3
"""
NEXUS Theory CLI - NEXUS理論完全実装CLI
全ての理論コンポーネントを統合したコマンドラインインターフェース
"""

import argparse
import sys
import time
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import json

# プロジェクトパス追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from engine.nexus_theory_engine import NEXUSTheoryEngine, DataFormat
from engine.nexus_advanced_optimizer import NEXUSAdvancedOptimizer
from engine.nexus_parallel_engine import NEXUSParallelEngine, ParallelConfig


class NEXUSCLIManager:
    """NEXUS CLI管理クラス"""
    
    def __init__(self):
        self.theory_engine = NEXUSTheoryEngine()
        self.optimizer = NEXUSAdvancedOptimizer(self.theory_engine)
        
        # 並列処理設定
        self.parallel_config = ParallelConfig(
            use_gpu=True,
            use_multiprocessing=True,
            use_threading=True,
            max_threads=8,
            max_processes=4,
            chunk_size_mb=2
        )
        self.parallel_engine = NEXUSParallelEngine(self.parallel_config)
        
        # 統計情報
        self.session_stats = {
            'files_processed': 0,
            'total_input_size': 0,
            'total_output_size': 0,
            'total_time': 0.0,
            'average_compression_ratio': 0.0
        }
    
    def compress_file(self, input_path: str, output_path: Optional[str] = None, 
                     mode: str = 'theory', quality: str = 'balanced') -> Dict[str, Any]:
        """ファイル圧縮"""
        input_file = Path(input_path)
        
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # 出力パス決定
        if output_path is None:
            output_path = str(input_file.with_suffix('.nxz'))
        
        output_file = Path(output_path)
        
        print(f"🔬 NEXUS圧縮開始")
        print(f"📁 入力: {input_file.name} ({input_file.stat().st_size // 1024:.1f}KB)")
        print(f"📁 出力: {output_file.name}")
        print(f"🎯 モード: {mode}, 品質: {quality}")
        
        # データ読み込み
        start_time = time.perf_counter()
        with open(input_file, 'rb') as f:
            data = f.read()
        
        input_size = len(data)
        
        # 圧縮実行
        if mode == 'theory':
            compressed_data = self.theory_engine.compress(data)
        elif mode == 'optimized':
            compressed_data = self.optimizer.optimize_compression(data, quality)
        elif mode == 'parallel':
            compressed_data = self.parallel_engine.parallel_compress(data, quality)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # 結果保存
        with open(output_file, 'wb') as f:
            f.write(compressed_data)
        
        total_time = time.perf_counter() - start_time
        output_size = len(compressed_data)
        compression_ratio = (1 - output_size / input_size) * 100
        
        # 結果情報
        result = {
            'input_file': str(input_file),
            'output_file': str(output_file),
            'input_size': input_size,
            'output_size': output_size,
            'compression_ratio': compression_ratio,
            'processing_time': total_time,
            'mode': mode,
            'quality': quality
        }
        
        # 統計更新
        self._update_stats(result)
        
        print(f"✅ 圧縮完了: {compression_ratio:.2f}% ({total_time:.2f}s)")
        return result
    
    def decompress_file(self, input_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """ファイル展開"""
        input_file = Path(input_path)
        
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # 出力パス決定
        if output_path is None:
            if input_file.suffix == '.nxz':
                output_path = str(input_file.with_suffix(''))
            else:
                output_path = str(input_file.with_suffix('.decompressed'))
        
        output_file = Path(output_path)
        
        print(f"🔓 NEXUS展開開始")
        print(f"📁 入力: {input_file.name} ({input_file.stat().st_size // 1024:.1f}KB)")
        print(f"📁 出力: {output_file.name}")
        
        # データ読み込み
        start_time = time.perf_counter()
        with open(input_file, 'rb') as f:
            compressed_data = f.read()
        
        # 展開実行（形式判定）
        try:
            # 並列形式チェック
            if compressed_data.startswith(b'NXPAR001'):
                decompressed_data = self.parallel_engine.parallel_decompress(compressed_data)
            else:
                decompressed_data = self.theory_engine.decompress(compressed_data)
        except Exception as e:
            print(f"❌ 展開エラー: {e}")
            raise
        
        # 結果保存
        with open(output_file, 'wb') as f:
            f.write(decompressed_data)
        
        total_time = time.perf_counter() - start_time
        
        result = {
            'input_file': str(input_file),
            'output_file': str(output_file),
            'input_size': len(compressed_data),
            'output_size': len(decompressed_data),
            'processing_time': total_time
        }
        
        print(f"✅ 展開完了: {len(decompressed_data) // 1024:.1f}KB ({total_time:.2f}s)")
        return result
    
    def benchmark_file(self, input_path: str) -> Dict[str, Any]:
        """ファイルベンチマーク"""
        input_file = Path(input_path)
        
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        print(f"📊 NEXUS理論ベンチマーク")
        print(f"📁 ファイル: {input_file.name} ({input_file.stat().st_size // 1024:.1f}KB)")
        print("=" * 60)
        
        with open(input_file, 'rb') as f:
            data = f.read()
        
        input_size = len(data)
        results = {}
        
        # テスト設定
        test_configs = [
            ('理論エンジン', 'theory', 'balanced'),
            ('最適化エンジン(高速)', 'optimized', 'fast'),
            ('最適化エンジン(バランス)', 'optimized', 'balanced'),
            ('最適化エンジン(最高)', 'optimized', 'max'),
            ('並列エンジン(高速)', 'parallel', 'fast'),
            ('並列エンジン(バランス)', 'parallel', 'balanced'),
            ('並列エンジン(最高)', 'parallel', 'max')
        ]
        
        for name, mode, quality in test_configs:
            print(f"\n🔬 テスト: {name}")
            
            try:
                start_time = time.perf_counter()
                
                if mode == 'theory':
                    compressed = self.theory_engine.compress(data)
                    decompressed = self.theory_engine.decompress(compressed)
                elif mode == 'optimized':
                    compressed = self.optimizer.optimize_compression(data, quality)
                    decompressed = self.theory_engine.decompress(compressed)
                elif mode == 'parallel':
                    compressed = self.parallel_engine.parallel_compress(data, quality)
                    decompressed = self.parallel_engine.parallel_decompress(compressed)
                
                total_time = time.perf_counter() - start_time
                
                # 正確性検証
                is_correct = data == decompressed
                compression_ratio = (1 - len(compressed) / input_size) * 100
                speed = input_size / (1024 * 1024) / total_time  # MB/s
                
                result = {
                    'compression_ratio': compression_ratio,
                    'speed_mbps': speed,
                    'time_seconds': total_time,
                    'is_correct': is_correct,
                    'compressed_size': len(compressed)
                }
                
                results[name] = result
                
                status = "✅" if is_correct else "❌"
                print(f"  圧縮率: {compression_ratio:.2f}%")
                print(f"  速度: {speed:.1f} MB/s")
                print(f"  時間: {total_time:.2f}s")
                print(f"  正確性: {status}")
                
            except Exception as e:
                print(f"  ❌ エラー: {e}")
                results[name] = {'error': str(e)}
        
        # 総合評価
        print(f"\n🏆 ベンチマーク結果サマリー")
        print("=" * 60)
        
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if successful_results:
            best_ratio = max(successful_results.values(), key=lambda x: x['compression_ratio'])
            best_speed = max(successful_results.values(), key=lambda x: x['speed_mbps'])
            
            best_ratio_name = [k for k, v in successful_results.items() if v == best_ratio][0]
            best_speed_name = [k for k, v in successful_results.items() if v == best_speed][0]
            
            print(f"🥇 最高圧縮率: {best_ratio_name} ({best_ratio['compression_ratio']:.2f}%)")
            print(f"🥇 最高速度: {best_speed_name} ({best_speed['speed_mbps']:.1f} MB/s)")
        
        return {
            'input_file': str(input_file),
            'input_size': input_size,
            'results': results
        }
    
    def batch_process(self, input_dir: str, output_dir: Optional[str] = None, 
                     mode: str = 'parallel', quality: str = 'balanced') -> List[Dict[str, Any]]:
        """バッチ処理"""
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        if output_dir is None:
            output_dir = str(input_path / "compressed")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # ファイル一覧取得
        files = list(input_path.glob("*"))
        files = [f for f in files if f.is_file()]
        
        print(f"📦 NEXUS バッチ処理")
        print(f"📁 入力ディレクトリ: {input_path}")
        print(f"📁 出力ディレクトリ: {output_path}")
        print(f"📊 ファイル数: {len(files)}")
        print(f"🎯 モード: {mode}, 品質: {quality}")
        print("=" * 60)
        
        results = []
        
        for i, file_path in enumerate(files, 1):
            print(f"\n📄 ({i}/{len(files)}) {file_path.name}")
            
            try:
                output_file_path = output_path / f"{file_path.name}.nxz"
                result = self.compress_file(str(file_path), str(output_file_path), mode, quality)
                results.append(result)
                
            except Exception as e:
                print(f"❌ エラー: {e}")
                error_result = {
                    'input_file': str(file_path),
                    'error': str(e)
                }
                results.append(error_result)
        
        # バッチ統計
        successful_results = [r for r in results if 'error' not in r]
        
        if successful_results:
            total_input = sum(r['input_size'] for r in successful_results)
            total_output = sum(r['output_size'] for r in successful_results)
            avg_ratio = (1 - total_output / total_input) * 100
            total_time = sum(r['processing_time'] for r in successful_results)
            
            print(f"\n🏆 バッチ処理完了")
            print(f"✅ 成功: {len(successful_results)}/{len(files)} ファイル")
            print(f"📊 総圧縮率: {avg_ratio:.2f}%")
            print(f"⏱️ 総時間: {total_time:.2f}s")
        
        return results
    
    def analyze_file(self, input_path: str) -> Dict[str, Any]:
        """ファイル分析"""
        input_file = Path(input_path)
        
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        print(f"🔍 NEXUS理論ファイル分析")
        print(f"📁 ファイル: {input_file.name}")
        print("=" * 60)
        
        with open(input_file, 'rb') as f:
            data = f.read()
        
        # データ形式分析
        data_format = self.theory_engine._analyze_data_format(data)
        print(f"📊 データ形式: {data_format.value}")
        
        # 特徴分析
        features = self.optimizer._analyze_data_features(data)
        print(f"📈 データ特徴:")
        for key, value in features.items():
            print(f"  {key}: {value:.4f}")
        
        # ML予測
        predictions = self.optimizer.ml_model.predict(features)
        print(f"🧠 ML予測:")
        for key, value in predictions.items():
            print(f"  {key}: {value}")
        
        # 圧縮可能性推定
        compressibility = features.get('compressibility', 0.5)
        if compressibility > 0.7:
            assessment = "高い"
        elif compressibility > 0.4:
            assessment = "中程度"
        else:
            assessment = "低い"
        
        print(f"🎯 圧縮可能性: {assessment} ({compressibility:.2f})")
        
        return {
            'file_path': str(input_file),
            'file_size': len(data),
            'data_format': data_format.value,
            'features': features,
            'predictions': predictions,
            'compressibility_assessment': assessment
        }
    
    def _update_stats(self, result: Dict[str, Any]):
        """統計更新"""
        if 'error' not in result:
            self.session_stats['files_processed'] += 1
            self.session_stats['total_input_size'] += result['input_size']
            self.session_stats['total_output_size'] += result['output_size']
            self.session_stats['total_time'] += result['processing_time']
            
            # 平均圧縮率更新
            if self.session_stats['total_input_size'] > 0:
                self.session_stats['average_compression_ratio'] = (
                    1 - self.session_stats['total_output_size'] / self.session_stats['total_input_size']
                ) * 100
    
    def get_session_stats(self) -> Dict[str, Any]:
        """セッション統計取得"""
        return self.session_stats.copy()


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="NEXUS Theory CLI - 理論的圧縮システム",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # ファイル圧縮
  python nexus_cli.py compress input.txt -o output.nxz -m parallel -q max
  
  # ファイル展開
  python nexus_cli.py decompress output.nxz -o restored.txt
  
  # ベンチマーク
  python nexus_cli.py benchmark input.txt
  
  # バッチ処理
  python nexus_cli.py batch ./files/ ./compressed/ -m optimized -q balanced
  
  # ファイル分析
  python nexus_cli.py analyze input.txt
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='利用可能なコマンド')
    
    # 圧縮コマンド
    compress_parser = subparsers.add_parser('compress', help='ファイル圧縮')
    compress_parser.add_argument('input', help='入力ファイルパス')
    compress_parser.add_argument('-o', '--output', help='出力ファイルパス')
    compress_parser.add_argument('-m', '--mode', choices=['theory', 'optimized', 'parallel'], 
                               default='parallel', help='圧縮モード')
    compress_parser.add_argument('-q', '--quality', choices=['fast', 'balanced', 'max'], 
                               default='balanced', help='圧縮品質')
    
    # 展開コマンド
    decompress_parser = subparsers.add_parser('decompress', help='ファイル展開')
    decompress_parser.add_argument('input', help='入力ファイルパス')
    decompress_parser.add_argument('-o', '--output', help='出力ファイルパス')
    
    # ベンチマークコマンド
    benchmark_parser = subparsers.add_parser('benchmark', help='ベンチマーク実行')
    benchmark_parser.add_argument('input', help='入力ファイルパス')
    
    # バッチ処理コマンド
    batch_parser = subparsers.add_parser('batch', help='バッチ処理')
    batch_parser.add_argument('input_dir', help='入力ディレクトリ')
    batch_parser.add_argument('output_dir', nargs='?', help='出力ディレクトリ')
    batch_parser.add_argument('-m', '--mode', choices=['theory', 'optimized', 'parallel'], 
                            default='parallel', help='圧縮モード')
    batch_parser.add_argument('-q', '--quality', choices=['fast', 'balanced', 'max'], 
                            default='balanced', help='圧縮品質')
    
    # 分析コマンド
    analyze_parser = subparsers.add_parser('analyze', help='ファイル分析')
    analyze_parser.add_argument('input', help='入力ファイルパス')
    
    # 統計コマンド
    stats_parser = subparsers.add_parser('stats', help='セッション統計表示')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # CLI管理器作成
    cli = NEXUSCLIManager()
    
    try:
        if args.command == 'compress':
            result = cli.compress_file(args.input, args.output, args.mode, args.quality)
            
        elif args.command == 'decompress':
            result = cli.decompress_file(args.input, args.output)
            
        elif args.command == 'benchmark':
            result = cli.benchmark_file(args.input)
            
        elif args.command == 'batch':
            result = cli.batch_process(args.input_dir, args.output_dir, args.mode, args.quality)
            
        elif args.command == 'analyze':
            result = cli.analyze_file(args.input)
            
        elif args.command == 'stats':
            stats = cli.get_session_stats()
            print(f"📊 セッション統計:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            return
        
        # 結果をJSONで出力（オプション）
        if '--json' in sys.argv:
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
