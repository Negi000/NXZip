#!/usr/bin/env python3
"""
NXZ統合圧縮ベンチマーク v1.0
SPE統合NXZ vs 7-Zip vs Zstandard 包括的性能比較
圧縮率・圧縮速度・展開速度・可逆性の全面評価
"""

import os
import sys
import time
import asyncio
import statistics
from typing import List, Dict, Any, Tuple
from pathlib import Path

# NXZip modules
sys.path.append(str(Path(__file__).parent.parent))
from nxzip.engine.nexus_nxz_unified import NXZUnifiedEngine, CompetitiveCompressionEngine

class ComprehensiveBenchmark:
    """包括的ベンチマーク実行システム"""
    
    def __init__(self):
        self.test_results = {
            'nxz_standard': [],
            'nxz_lightweight': [],
            'sevenz': [],
            'zstandard': []
        }
        
        self.test_files = []
        self.reversibility_results = {}
        
        print("🚀 NXZ統合圧縮ベンチマーク v1.0 初期化完了")
    
    def prepare_test_data(self) -> List[Tuple[str, bytes]]:
        """テストデータの準備"""
        test_data = []
        
        # 1. テキストデータ（高圧縮性）
        text_data = "The quick brown fox jumps over the lazy dog. " * 100
        test_data.append(("text_repetitive_1KB", text_data.encode('utf-8')))
        
        text_large = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 500
        test_data.append(("text_large_10KB", text_large.encode('utf-8')))
        
        # 2. 日本語テキスト
        japanese_text = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん。" * 100
        test_data.append(("japanese_text_5KB", japanese_text.encode('utf-8')))
        
        # 3. 数値データ（構造化）
        import struct
        float_array = [i * 0.1 for i in range(1000)]
        float_bytes = b''.join(struct.pack('f', f) for f in float_array)
        test_data.append(("float_array_4KB", float_bytes))
        
        # 4. 連続整数データ
        int_sequence = b''.join(i.to_bytes(4, 'little') for i in range(500))
        test_data.append(("int_sequence_2KB", int_sequence))
        
        # 5. バイナリデータ（低圧縮性）
        import random
        random.seed(42)  # 再現性のため
        random_bytes = bytes([random.randint(0, 255) for _ in range(5000)])
        test_data.append(("random_binary_5KB", random_bytes))
        
        # 6. パターン性バイナリ
        pattern_data = b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f' * 200
        test_data.append(("pattern_binary_3KB", pattern_data))
        
        # 7. 大容量データ（軽量モードテスト用）
        large_text = "This is a large text file for testing lightweight mode performance. " * 2000
        test_data.append(("large_text_100KB", large_text.encode('utf-8')))
        
        # 8. 実ファイルからのサンプル（存在する場合）
        sample_dir = Path(__file__).parent.parent / "sample"
        if sample_dir.exists():
            for file_path in sample_dir.glob("*.txt"):
                if file_path.stat().st_size < 50000:  # 50KB以下
                    try:
                        with open(file_path, 'rb') as f:
                            content = f.read()
                            test_data.append((f"real_file_{file_path.name}", content))
                    except:
                        pass
        
        print(f"📊 テストデータ準備完了: {len(test_data)}種類")
        for name, data in test_data:
            print(f"  - {name}: {len(data):,} bytes")
        
        return test_data
    
    async def run_nxz_benchmark(self, test_data: List[Tuple[str, bytes]], mode: str) -> List[Dict[str, Any]]:
        """NXZ統合エンジンベンチマーク"""
        results = []
        lightweight_mode = (mode == 'lightweight')
        
        print(f"\n🔥 NXZ統合エンジン ベンチマーク開始 ({mode}モード)")
        print("=" * 60)
        
        engine = NXZUnifiedEngine(lightweight_mode=lightweight_mode, encryption_enabled=True)
        # SPE（構造保持暗号化）はパスワード不要の構造変換のため、パスワードなしでテスト
        test_password = None
        
        for test_name, data in test_data:
            print(f"\n📁 テスト: {test_name} ({len(data):,} bytes)")
            
            try:
                # 圧縮テスト
                print("  🔄 圧縮実行中...")
                compress_start = time.time()
                compressed, comp_info = await engine.compress_nxz(data, test_password)
                compress_time = time.time() - compress_start
                
                # 解凍テスト
                print("  🔄 解凍実行中...")
                decompress_start = time.time()
                decompressed, decomp_info = await engine.decompress_nxz(compressed, test_password)
                decompress_time = time.time() - decompress_start
                
                # 可逆性確認
                is_reversible = decompressed == data
                
                # 結果記録
                result = {
                    'test_name': test_name,
                    'engine': f'NXZ-{mode}',
                    'original_size': len(data),
                    'compressed_size': len(compressed),
                    'compression_ratio': (1 - len(compressed) / len(data)) * 100 if len(data) > 0 else 0,
                    'compression_time': compress_time,
                    'decompression_time': decompress_time,
                    'compression_speed': (len(data) / (1024 * 1024) / compress_time) if compress_time > 0 else 0,  # MB/s
                    'decompression_speed': (len(data) / (1024 * 1024) / decompress_time) if decompress_time > 0 else 0,  # MB/s
                    'reversible': is_reversible,
                    'encryption_enabled': True,
                    'comp_info': comp_info,
                    'decomp_info': decomp_info
                }
                
                results.append(result)
                
                # 結果表示
                print(f"  ✅ 圧縮率: {result['compression_ratio']:.1f}%")
                print(f"  ⚡ 圧縮速度: {result['compression_speed']:.1f} MB/s")
                print(f"  ⚡ 展開速度: {result['decompression_speed']:.1f} MB/s")
                print(f"  🔄 可逆性: {'OK' if is_reversible else 'NG'}")
                
            except Exception as e:
                print(f"  ❌ エラー: {e}")
                results.append({
                    'test_name': test_name,
                    'engine': f'NXZ-{mode}',
                    'error': str(e)
                })
        
        return results
    
    def run_competitive_benchmark(self, test_data: List[Tuple[str, bytes]], engine_name: str) -> List[Dict[str, Any]]:
        """競合エンジンベンチマーク"""
        results = []
        
        print(f"\n🔥 {engine_name} ベンチマーク開始")
        print("=" * 60)
        
        for test_name, data in test_data:
            print(f"\n📁 テスト: {test_name} ({len(data):,} bytes)")
            
            try:
                # 圧縮テスト
                print("  🔄 圧縮実行中...")
                if engine_name == '7-Zip':
                    compressed, comp_info = CompetitiveCompressionEngine.compress_7zip(data)
                elif engine_name == 'Zstandard':
                    compressed, comp_info = CompetitiveCompressionEngine.compress_zstd(data)
                else:
                    raise ValueError(f"未サポートエンジン: {engine_name}")
                
                # 解凍テスト
                print("  🔄 解凍実行中...")
                if engine_name == '7-Zip':
                    decompressed, decomp_info = CompetitiveCompressionEngine.decompress_7zip(compressed)
                elif engine_name == 'Zstandard':
                    decompressed, decomp_info = CompetitiveCompressionEngine.decompress_zstd(compressed)
                
                # 可逆性確認
                is_reversible = decompressed == data
                
                # 結果記録
                result = {
                    'test_name': test_name,
                    'engine': engine_name,
                    'original_size': len(data),
                    'compressed_size': len(compressed),
                    'compression_ratio': (1 - len(compressed) / len(data)) * 100 if len(data) > 0 else 0,
                    'compression_time': comp_info['compression_time'],
                    'decompression_time': decomp_info['decompression_time'],
                    'compression_speed': comp_info['throughput_mbps'],
                    'decompression_speed': decomp_info['throughput_mbps'],
                    'reversible': is_reversible,
                    'encryption_enabled': False,
                    'comp_info': comp_info,
                    'decomp_info': decomp_info
                }
                
                results.append(result)
                
                # 結果表示
                print(f"  ✅ 圧縮率: {result['compression_ratio']:.1f}%")
                print(f"  ⚡ 圧縮速度: {result['compression_speed']:.1f} MB/s")
                print(f"  ⚡ 展開速度: {result['decompression_speed']:.1f} MB/s")
                print(f"  🔄 可逆性: {'OK' if is_reversible else 'NG'}")
                
            except Exception as e:
                print(f"  ❌ エラー: {e}")
                results.append({
                    'test_name': test_name,
                    'engine': engine_name,
                    'error': str(e)
                })
        
        return results
    
    async def run_full_benchmark(self) -> Dict[str, Any]:
        """完全ベンチマーク実行"""
        print("🚀 NXZ統合圧縮 包括的ベンチマーク開始")
        print("=" * 80)
        
        # テストデータ準備
        test_data = self.prepare_test_data()
        
        # 各エンジンでベンチマーク実行
        print("\n🎯 ベンチマーク実行中...")
        
        # NXZ統合エンジン（通常モード）
        nxz_standard_results = await self.run_nxz_benchmark(test_data, 'standard')
        self.test_results['nxz_standard'] = nxz_standard_results
        
        # NXZ統合エンジン（軽量モード）
        nxz_lightweight_results = await self.run_nxz_benchmark(test_data, 'lightweight')
        self.test_results['nxz_lightweight'] = nxz_lightweight_results
        
        # 7-Zip
        sevenz_results = self.run_competitive_benchmark(test_data, '7-Zip')
        self.test_results['sevenz'] = sevenz_results
        
        # Zstandard
        zstd_results = self.run_competitive_benchmark(test_data, 'Zstandard')
        self.test_results['zstandard'] = zstd_results
        
        # 結果分析
        analysis = self.analyze_results()
        
        return {
            'test_data_info': [(name, len(data)) for name, data in test_data],
            'results': self.test_results,
            'analysis': analysis
        }
    
    def analyze_results(self) -> Dict[str, Any]:
        """結果分析"""
        print("\n📊 結果分析中...")
        
        analysis = {
            'compression_ratio': {},
            'compression_speed': {},
            'decompression_speed': {},
            'reversibility': {},
            'overall_ranking': {}
        }
        
        # エンジン別統計
        for engine_key, results in self.test_results.items():
            valid_results = [r for r in results if 'error' not in r]
            
            if not valid_results:
                continue
            
            # 圧縮率統計
            ratios = [r['compression_ratio'] for r in valid_results]
            analysis['compression_ratio'][engine_key] = {
                'average': statistics.mean(ratios),
                'median': statistics.median(ratios),
                'best': max(ratios),
                'worst': min(ratios)
            }
            
            # 圧縮速度統計
            comp_speeds = [r['compression_speed'] for r in valid_results if r['compression_speed'] > 0]
            if comp_speeds:
                analysis['compression_speed'][engine_key] = {
                    'average': statistics.mean(comp_speeds),
                    'median': statistics.median(comp_speeds),
                    'best': max(comp_speeds),
                    'worst': min(comp_speeds)
                }
            
            # 展開速度統計
            decomp_speeds = [r['decompression_speed'] for r in valid_results if r['decompression_speed'] > 0]
            if decomp_speeds:
                analysis['decompression_speed'][engine_key] = {
                    'average': statistics.mean(decomp_speeds),
                    'median': statistics.median(decomp_speeds),
                    'best': max(decomp_speeds),
                    'worst': min(decomp_speeds)
                }
            
            # 可逆性統計
            reversible_count = sum(1 for r in valid_results if r['reversible'])
            analysis['reversibility'][engine_key] = {
                'success_count': reversible_count,
                'total_tests': len(valid_results),
                'success_rate': (reversible_count / len(valid_results)) * 100 if valid_results else 0
            }
        
        return analysis
    
    def print_summary_report(self, benchmark_results: Dict[str, Any]):
        """サマリーレポート出力"""
        print("\n" + "=" * 80)
        print("🏆 NXZ統合圧縮 ベンチマーク サマリーレポート")
        print("=" * 80)
        
        analysis = benchmark_results['analysis']
        
        # 1. 圧縮率比較
        print("\n📊 圧縮率比較 (平均値)")
        print("-" * 50)
        compression_ratios = analysis['compression_ratio']
        for engine, stats in sorted(compression_ratios.items(), key=lambda x: x[1]['average'], reverse=True):
            engine_name = self._format_engine_name(engine)
            print(f"{engine_name:20} {stats['average']:6.1f}% (最高: {stats['best']:5.1f}%)")
        
        # 2. 圧縮速度比較
        print("\n⚡ 圧縮速度比較 (平均値)")
        print("-" * 50)
        compression_speeds = analysis['compression_speed']
        for engine, stats in sorted(compression_speeds.items(), key=lambda x: x[1]['average'], reverse=True):
            engine_name = self._format_engine_name(engine)
            print(f"{engine_name:20} {stats['average']:6.1f} MB/s (最高: {stats['best']:5.1f} MB/s)")
        
        # 3. 展開速度比較
        print("\n⚡ 展開速度比較 (平均値)")
        print("-" * 50)
        decompression_speeds = analysis['decompression_speed']
        for engine, stats in sorted(decompression_speeds.items(), key=lambda x: x[1]['average'], reverse=True):
            engine_name = self._format_engine_name(engine)
            print(f"{engine_name:20} {stats['average']:6.1f} MB/s (最高: {stats['best']:5.1f} MB/s)")
        
        # 4. 可逆性比較
        print("\n🔄 可逆性比較")
        print("-" * 50)
        reversibility = analysis['reversibility']
        for engine, stats in sorted(reversibility.items(), key=lambda x: x[1]['success_rate'], reverse=True):
            engine_name = self._format_engine_name(engine)
            print(f"{engine_name:20} {stats['success_rate']:5.1f}% ({stats['success_count']}/{stats['total_tests']})")
        
        # 5. 総合評価
        print("\n🏆 総合評価")
        print("-" * 50)
        
        # 各カテゴリでのランキング計算
        rankings = {}
        for engine in compression_ratios.keys():
            rankings[engine] = {
                'compression_ratio_rank': 0,
                'compression_speed_rank': 0,
                'decompression_speed_rank': 0,
                'reversibility_rank': 0
            }
        
        # 圧縮率ランキング
        sorted_by_ratio = sorted(compression_ratios.items(), key=lambda x: x[1]['average'], reverse=True)
        for i, (engine, _) in enumerate(sorted_by_ratio):
            rankings[engine]['compression_ratio_rank'] = i + 1
        
        # 圧縮速度ランキング
        sorted_by_comp_speed = sorted(compression_speeds.items(), key=lambda x: x[1]['average'], reverse=True)
        for i, (engine, _) in enumerate(sorted_by_comp_speed):
            rankings[engine]['compression_speed_rank'] = i + 1
        
        # 展開速度ランキング
        sorted_by_decomp_speed = sorted(decompression_speeds.items(), key=lambda x: x[1]['average'], reverse=True)
        for i, (engine, _) in enumerate(sorted_by_decomp_speed):
            rankings[engine]['decompression_speed_rank'] = i + 1
        
        # 可逆性ランキング
        sorted_by_reversibility = sorted(reversibility.items(), key=lambda x: x[1]['success_rate'], reverse=True)
        for i, (engine, _) in enumerate(sorted_by_reversibility):
            rankings[engine]['reversibility_rank'] = i + 1
        
        # 総合スコア計算（順位の合計、低いほど良い）
        total_scores = {}
        for engine, ranks in rankings.items():
            total_score = sum(ranks.values())
            total_scores[engine] = total_score
        
        # 総合ランキング表示
        for engine, score in sorted(total_scores.items(), key=lambda x: x[1]):
            engine_name = self._format_engine_name(engine)
            ranks = rankings[engine]
            print(f"{engine_name:20} 総合スコア: {score:2d} (圧縮率:{ranks['compression_ratio_rank']}位, 圧縮速度:{ranks['compression_speed_rank']}位, 展開速度:{ranks['decompression_speed_rank']}位, 可逆性:{ranks['reversibility_rank']}位)")
        
        # 6. 特記事項
        print("\n📝 特記事項")
        print("-" * 50)
        print("• NXZ統合エンジンは SPE暗号化機能を含む")
        print("• 軽量モードは低リソース環境向け最適化")
        print("• 通常モードは最大圧縮率・性能追求")
        print("• 7-Zip/Zstandardは暗号化機能なし")
        print("• 測定環境・データサイズにより結果は変動")
    
    def _format_engine_name(self, engine_key: str) -> str:
        """エンジン名の整形"""
        name_map = {
            'nxz_standard': 'NXZ統合(通常)',
            'nxz_lightweight': 'NXZ統合(軽量)',
            'sevenz': '7-Zip',
            'zstandard': 'Zstandard'
        }
        return name_map.get(engine_key, engine_key)

async def main():
    """メイン実行関数"""
    benchmark = ComprehensiveBenchmark()
    
    try:
        # ベンチマーク実行
        results = await benchmark.run_full_benchmark()
        
        # サマリーレポート出力
        benchmark.print_summary_report(results)
        
        # 詳細結果の保存（オプション）
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = f"nxz_benchmark_report_{timestamp}.json"
        
        try:
            import json
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 詳細レポート保存: {report_file}")
        except Exception as e:
            print(f"\n⚠️ レポート保存エラー: {e}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ ベンチマーク中断されました")
    except Exception as e:
        print(f"\n❌ ベンチマークエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
