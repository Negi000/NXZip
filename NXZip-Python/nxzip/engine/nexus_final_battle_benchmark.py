#!/usr/bin/env python3
"""
NEXUS 最終決戦ベンチマーク
Ultimate Engine vs 7Z vs Zstandard - 最終対決
"""

import os
import sys
import time
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import statistics

# 究極エンジンインポート
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from nexus_ultimate_nxz_engine import NEXUSUltimateEngine
    ULTIMATE_AVAILABLE = True
    print("🚀 NEXUS Ultimate Engine 準備完了")
except ImportError:
    print("⚠️ NEXUS Ultimate Engine が利用できません")
    ULTIMATE_AVAILABLE = False

try:
    from nexus_spe_integrated_engine import NEXUSSPEIntegratedEngine
    SPE_AVAILABLE = True
    print("🔐 NEXUS SPE Integrated Engine 準備完了")
except ImportError:
    print("⚠️ NEXUS SPE Integrated Engine が利用できません")
    SPE_AVAILABLE = False

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
    print("⚡ Zstandard Engine 準備完了")
except ImportError:
    print("⚠️ Zstandard ライブラリが利用できません")
    ZSTD_AVAILABLE = False

import lzma


class NEXUSUltimateCompetitor:
    """NEXUS Ultimate版競合テスト"""
    
    def __init__(self):
        self.name = "NEXUS-Ultimate-NXZ"
        if ULTIMATE_AVAILABLE:
            self.engine = NEXUSUltimateEngine(max_workers=4, encryption_enabled=True)
        else:
            self.engine = None
    
    def compress(self, data: bytes, level: int = 6, password: str = None) -> Tuple[bytes, Dict[str, Any]]:
        """Ultimate NXZ圧縮"""
        if not self.engine:
            return data, {'error': 'ultimate_engine_not_available'}
        
        start_time = time.perf_counter()
        
        try:
            metadata = {
                'compression_level': level,
                'nexus_version': 'Ultimate_v1',
                'format': 'NXZ_Ultimate'
            }
            
            compressed_data, compression_info = self.engine.compress_to_nxz_ultimate(
                data, password=password, metadata=metadata
            )
            
            processing_time = time.perf_counter() - start_time
            
            result_info = {
                'original_size': len(data),
                'compressed_size': len(compressed_data),
                'compression_ratio': compression_info.get('total_compression_ratio', 0),
                'processing_time': processing_time,
                'throughput_mb_s': (len(data) / 1024 / 1024) / processing_time if processing_time > 0 else 0,
                'encrypted': compression_info.get('encrypted', False),
                'format': 'nxz_ultimate',
                'method': compression_info.get('compression_info', {}).get('best_method', 'unknown'),
                'nexus_info': compression_info
            }
            
            return compressed_data, result_info
            
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            return data, {
                'error': str(e),
                'processing_time': processing_time,
                'original_size': len(data)
            }


class FinalBattleBenchmark:
    """最終決戦ベンチマーク"""
    
    def __init__(self):
        self.competitors = {}
        
        # NEXUS Ultimate
        if ULTIMATE_AVAILABLE:
            self.competitors['NEXUS-Ultimate'] = NEXUSUltimateCompetitor()
        
        # NEXUS SPE Integrated
        if SPE_AVAILABLE:
            from nexus_spe_nxz_vs_competitors import NEXUSCompetitor
            self.competitors['NEXUS-SPE'] = NEXUSCompetitor()
        
        # Zstandard
        if ZSTD_AVAILABLE:
            from nexus_spe_nxz_vs_competitors import ZstdCompetitor
            self.competitors['Zstandard'] = ZstdCompetitor()
        
        # LZMA
        from nexus_spe_nxz_vs_competitors import LZMACompetitor
        self.competitors['LZMA'] = LZMACompetitor()
        
        print(f"🏟️ 競合エンジン準備完了: {list(self.competitors.keys())}")
    
    def generate_ultimate_test_datasets(self) -> Dict[str, bytes]:
        """究極テストデータセット生成"""
        datasets = {}
        
        # 1. 大型テキストデータ（高圧縮期待）
        large_text = (
            "The NEXUS Ultimate Engine represents the pinnacle of compression technology, "
            "integrating Transform-Model-Code (TMC) algorithms with Structure-Preserving Encryption (SPE). "
            "This revolutionary approach analyzes data structure patterns and applies optimal compression strategies "
            "while maintaining security through advanced cryptographic techniques. "
            "Performance benchmarks demonstrate superior compression ratios compared to traditional algorithms. "
        ) * 2000
        datasets['large_text'] = large_text.encode('utf-8')
        
        # 2. JSON構造化データ（構造特化圧縮テスト）
        json_data = '{"users": ['
        for i in range(1000):
            json_data += f'{{"id": {i}, "name": "user_{i}", "email": "user_{i}@example.com", "active": {str(i % 2 == 0).lower()}, "score": {i * 3.14159:.5f}}}'
            if i < 999:
                json_data += ', '
        json_data += '], "metadata": {"version": "1.0", "count": 1000, "generated": "2024-01-01"}}'
        datasets['structured_json'] = json_data.encode('utf-8')
        
        # 3. 高反復バイナリデータ
        repetitive_binary = b'NEXUS' * 5000 + b'\x00\x01\x02\x03' * 2500
        datasets['repetitive_binary'] = repetitive_binary
        
        # 4. 混合データ（テキスト + バイナリ）
        mixed_data = b'HEADER_SECTION_START\n'
        mixed_data += ("Configuration data for NEXUS engine testing. " * 200).encode('utf-8')
        mixed_data += b'\n\nBINARY_SECTION_START\n'
        mixed_data += bytes(range(256)) * 100
        mixed_data += b'\nEND_OF_DATA\n'
        datasets['mixed_content'] = mixed_data
        
        # 5. 低エントロピーデータ
        low_entropy = b'A' * 3000 + b'B' * 3000 + b'C' * 3000 + b'D' * 3000
        datasets['low_entropy'] = low_entropy
        
        return datasets
    
    def run_final_battle(self) -> Dict[str, Any]:
        """最終決戦実行"""
        print("\n🏆 NEXUS vs 全競合 最終決戦開始")
        print("=" * 80)
        
        datasets = self.generate_ultimate_test_datasets()
        results = {}
        
        # テスト設定
        test_configs = [
            {'level': 1, 'name': 'Fast', 'password': None},
            {'level': 6, 'name': 'Balanced', 'password': None},
            {'level': 9, 'name': 'Maximum', 'password': None},
            {'level': 6, 'name': 'Encrypted', 'password': 'nexus_ultimate_2024'}
        ]
        
        for data_type, test_data in datasets.items():
            print(f"\n📊 データセット: {data_type}")
            print(f"   サイズ: {len(test_data):,} bytes ({len(test_data)/1024:.1f} KB)")
            print("-" * 60)
            
            type_results = {}
            
            for config in test_configs:
                level = config['level']
                config_name = config['name']
                password = config['password']
                
                print(f"\n🔧 設定: {config_name} (Level {level})")
                if password:
                    print(f"   🔐 パスワード保護あり")
                
                config_results = {}
                
                # 各競合でテスト
                for competitor_name, competitor in self.competitors.items():
                    try:
                        if hasattr(competitor, 'available') and not competitor.available:
                            print(f"    {competitor_name:18}: ❌ Not Available")
                            config_results[competitor_name] = {'error': 'not_available'}
                            continue
                        
                        compressed_data, info = competitor.compress(test_data, level, password)
                        
                        if 'error' not in info:
                            compression_ratio = info['compression_ratio']
                            throughput = info['throughput_mb_s']
                            compressed_size = info['compressed_size']
                            method = info.get('method', 'unknown')
                            
                            # サイズ効果の表示
                            size_reduction = len(test_data) - compressed_size
                            
                            print(f"    {competitor_name:18}: {compression_ratio:6.2f}% | "
                                  f"{throughput:6.1f}MB/s | "
                                  f"{compressed_size:8,}B | "
                                  f"({method})")
                            
                            config_results[competitor_name] = info
                        else:
                            print(f"    {competitor_name:18}: ❌ {info['error']}")
                            config_results[competitor_name] = info
                            
                    except Exception as e:
                        print(f"    {competitor_name:18}: ❌ Exception: {str(e)}")
                        config_results[competitor_name] = {'error': str(e)}
                
                type_results[config_name] = config_results
            
            results[data_type] = type_results
        
        # 最終評価
        final_analysis = self._analyze_final_results(results)
        
        return {
            'detailed_results': results,
            'final_analysis': final_analysis,
            'test_timestamp': time.time(),
            'nexus_versions': ['Ultimate_v1', 'SPE_Integrated_v2']
        }
    
    def _analyze_final_results(self, results: Dict) -> Dict[str, Any]:
        """最終結果分析"""
        analysis = {
            'overall_champion': None,
            'category_winners': {
                'best_compression': None,
                'best_speed': None,
                'best_encrypted': None,
                'most_consistent': None
            },
            'nexus_performance': {},
            'detailed_metrics': {}
        }
        
        try:
            # 全結果からメトリクス収集
            all_metrics = {}
            
            for data_type, type_results in results.items():
                for config_name, config_results in type_results.items():
                    for competitor, info in config_results.items():
                        if 'error' not in info and 'compression_ratio' in info:
                            if competitor not in all_metrics:
                                all_metrics[competitor] = {
                                    'compression_ratios': [],
                                    'speeds': [],
                                    'encrypted_tests': 0,
                                    'successful_tests': 0
                                }
                            
                            all_metrics[competitor]['compression_ratios'].append(info['compression_ratio'])
                            all_metrics[competitor]['speeds'].append(info['throughput_mb_s'])
                            all_metrics[competitor]['successful_tests'] += 1
                            
                            if info.get('encrypted', False):
                                all_metrics[competitor]['encrypted_tests'] += 1
            
            # 平均値計算
            for competitor, metrics in all_metrics.items():
                if metrics['compression_ratios']:
                    avg_compression = statistics.mean(metrics['compression_ratios'])
                    avg_speed = statistics.mean(metrics['speeds'])
                    std_compression = statistics.stdev(metrics['compression_ratios']) if len(metrics['compression_ratios']) > 1 else 0
                    
                    analysis['detailed_metrics'][competitor] = {
                        'average_compression_ratio': avg_compression,
                        'average_speed_mb_s': avg_speed,
                        'compression_consistency': std_compression,
                        'successful_tests': metrics['successful_tests'],
                        'encrypted_capabilities': metrics['encrypted_tests'] > 0
                    }
            
            # カテゴリー別勝者決定
            if analysis['detailed_metrics']:
                # 最高圧縮率
                best_comp = max(analysis['detailed_metrics'].items(), 
                               key=lambda x: x[1]['average_compression_ratio'])
                analysis['category_winners']['best_compression'] = {
                    'winner': best_comp[0],
                    'ratio': best_comp[1]['average_compression_ratio']
                }
                
                # 最高速度
                best_speed = max(analysis['detailed_metrics'].items(), 
                                key=lambda x: x[1]['average_speed_mb_s'])
                analysis['category_winners']['best_speed'] = {
                    'winner': best_speed[0],
                    'speed': best_speed[1]['average_speed_mb_s']
                }
                
                # 一貫性（低い標準偏差）
                most_consistent = min(analysis['detailed_metrics'].items(), 
                                     key=lambda x: x[1]['compression_consistency'])
                analysis['category_winners']['most_consistent'] = {
                    'winner': most_consistent[0],
                    'consistency_score': most_consistent[1]['compression_consistency']
                }
                
                # 総合チャンピオン（圧縮率70% + 速度30%の重み付け）
                overall_scores = {}
                for competitor, metrics in analysis['detailed_metrics'].items():
                    score = (metrics['average_compression_ratio'] * 0.7 + 
                            metrics['average_speed_mb_s'] * 0.3)
                    overall_scores[competitor] = score
                
                if overall_scores:
                    champion = max(overall_scores.items(), key=lambda x: x[1])
                    analysis['overall_champion'] = {
                        'winner': champion[0],
                        'score': champion[1]
                    }
                
                # NEXUS特別分析
                nexus_competitors = [k for k in analysis['detailed_metrics'].keys() if 'NEXUS' in k]
                if nexus_competitors:
                    nexus_best = max([(k, v) for k, v in analysis['detailed_metrics'].items() if 'NEXUS' in k],
                                   key=lambda x: x[1]['average_compression_ratio'])
                    analysis['nexus_performance'] = {
                        'best_nexus_engine': nexus_best[0],
                        'nexus_compression_ratio': nexus_best[1]['average_compression_ratio'],
                        'nexus_speed': nexus_best[1]['average_speed_mb_s'],
                        'nexus_vs_best_overall': nexus_best[0] == analysis['overall_champion']['winner']
                    }
        
        except Exception as e:
            analysis['error'] = str(e)
        
        return analysis
    
    def print_final_championship_report(self, results: Dict[str, Any]):
        """最終チャンピオンシップレポート"""
        print("\n" + "=" * 80)
        print("🏆🏆🏆 NEXUS vs 全競合 最終決戦結果 🏆🏆🏆")
        print("=" * 80)
        
        analysis = results.get('final_analysis', {})
        
        # 総合チャンピオン
        if 'overall_champion' in analysis and analysis['overall_champion']:
            champion = analysis['overall_champion']
            print(f"\n👑 総合チャンピオン: {champion['winner']}")
            print(f"   総合スコア: {champion['score']:.2f}")
        
        # カテゴリー別勝者
        print(f"\n🏅 カテゴリー別勝者:")
        categories = analysis.get('category_winners', {})
        
        if categories.get('best_compression'):
            winner = categories['best_compression']
            print(f"   🗜️  最高圧縮率: {winner['winner']} ({winner['ratio']:.2f}%)")
        
        if categories.get('best_speed'):
            winner = categories['best_speed']
            print(f"   ⚡ 最高速度: {winner['winner']} ({winner['speed']:.1f}MB/s)")
        
        if categories.get('most_consistent'):
            winner = categories['most_consistent']
            print(f"   📊 最高一貫性: {winner['winner']} (偏差: {winner['consistency_score']:.2f})")
        
        # 詳細メトリクス
        print(f"\n📈 詳細パフォーマンス:")
        metrics = analysis.get('detailed_metrics', {})
        
        for competitor, data in metrics.items():
            encrypted_icon = "🔐" if data['encrypted_capabilities'] else "🔓"
            print(f"   {competitor:20}: {data['average_compression_ratio']:6.2f}% | "
                  f"{data['average_speed_mb_s']:6.1f}MB/s | "
                  f"テスト数:{data['successful_tests']:2d} {encrypted_icon}")
        
        # NEXUS特別レポート
        nexus_perf = analysis.get('nexus_performance', {})
        if nexus_perf:
            print(f"\n🚀 NEXUS特別レポート:")
            print(f"   最優秀NEXUS: {nexus_perf.get('best_nexus_engine', 'N/A')}")
            print(f"   NEXUS圧縮率: {nexus_perf.get('nexus_compression_ratio', 0):.2f}%")
            print(f"   NEXUS速度: {nexus_perf.get('nexus_speed', 0):.1f}MB/s")
            
            if nexus_perf.get('nexus_vs_best_overall', False):
                print(f"   🎉 NEXUSが総合優勝！！！")
            else:
                print(f"   📈 NEXUS改良の余地あり")
        
        print(f"\n🎯 NEXUS革新技術:")
        print(f"   ✓ TMC (Transform-Model-Code) アルゴリズム")
        print(f"   ✓ SPE (Structure-Preserving Encryption)")
        print(f"   ✓ NXZ次世代フォーマット")
        print(f"   ✓ 複数方式並列最適化")
        print(f"   ✓ 構造特化前処理/後処理")
        print(f"   ✓ 統合セキュリティ機能")


# メイン実行
if __name__ == "__main__":
    print("🔥🔥🔥 NEXUS 最終決戦ベンチマーク 🔥🔥🔥")
    print("Ultimate Engine vs Industry Standards - The Final Battle")
    print("=" * 80)
    
    benchmark = FinalBattleBenchmark()
    
    try:
        results = benchmark.run_final_battle()
        benchmark.print_final_championship_report(results)
        
        # 最終結果をJSONで保存
        output_file = Path(current_dir) / "nexus_final_battle_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 最終決戦結果を保存: {output_file}")
        
        print(f"\n🏁 最終決戦完了 - NEXUS Engine評価終了")
        
    except Exception as e:
        print(f"\n❌ 最終決戦エラー: {e}")
        import traceback
        traceback.print_exc()
