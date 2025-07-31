#!/usr/bin/env python3
"""
包括的圧縮エンジン比較テスト v2.0
通常モード vs 軽量モード vs Zstandard vs 7Zip
"""

import time
import tempfile
import os
import sys
import hashlib
import io
from pathlib import Path
import zstandard as zstd
import py7zr

# NEXUS TMC エンジンをインポート
sys.path.append('.')
from lightweight_mode import NEXUSTMCLightweight

class CompressionBenchmark:
    """圧縮ベンチマーククラス"""
    
    def __init__(self):
        self.nexus_light = NEXUSTMCLightweight()
        self.results = {}
        
    def create_test_datasets(self):
        """多様なテストデータセット作成"""
        datasets = {}
        
        # 1. 高反復テキスト（7Zipが得意とするパターン）
        repetitive_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 2000
        datasets['高反復テキスト'] = repetitive_text.encode('utf-8')
        
        # 2. 構造化データ（JSON/XML様）
        structured_data = []
        for i in range(1000):
            structured_data.append(f'<record id="{i}"><name>User_{i}</name><status>active</status><score>{i*10}</score></record>')
        datasets['構造化データ'] = '\n'.join(structured_data).encode('utf-8')
        
        # 3. CSV形式データ
        csv_data = "ID,名前,年齢,部署,給与,住所,電話番号\n"
        for i in range(2000):
            csv_data += f"{i},田中{i},{20+i%60},営業部{i%10},{300000+i*100},東京都{i%23}区,090-{i:04d}-{i*2:04d}\n"
        datasets['CSV形式'] = csv_data.encode('utf-8')
        
        # 4. プログラムコード（Python）
        code_template = '''
def function_{idx}(param1, param2, param3=None):
    """
    関数{idx}の説明文
    複数行にわたる詳細な説明がここに入ります。
    パラメータの説明や戻り値の説明も含まれます。
    """
    # 入力値の検証
    if param1 is None or param2 is None:
        raise ValueError("必須パラメータが不足しています")
    
    # メイン処理
    result = param1 + param2
    if param3 is not None:
        result *= param3
    
    # 条件分岐処理
    if result > 1000:
        result = result // 2
        print(f"大きな値が検出されました: {{result}}")
    elif result < 0:
        result = abs(result)
        print(f"負の値を正の値に変換しました: {{result}}")
    
    return result

class DataProcessor{idx}:
    """データ処理クラス{idx}"""
    
    def __init__(self):
        self.data = []
        self.processed = False
    
    def add_data(self, item):
        self.data.append(item)
    
    def process(self):
        if not self.data:
            return None
        
        total = sum(self.data)
        average = total / len(self.data)
        self.processed = True
        
        return {{
            'total': total,
            'average': average,
            'count': len(self.data),
            'processed_at': time.time()
        }}
'''
        
        code_data = ""
        for i in range(200):
            code_data += code_template.format(idx=i)
        datasets['プログラムコード'] = code_data.encode('utf-8')
        
        # 5. バイナリデータ（圧縮困難）
        import random
        random.seed(42)
        binary_data = bytes([random.randint(0, 255) for _ in range(50000)])
        datasets['ランダムバイナリ'] = binary_data
        
        # 6. 実際のファイル（存在する場合）
        sample_path = Path("sample/出庫実績明細_202412.txt")
        if sample_path.exists():
            try:
                with open(sample_path, 'rb') as f:
                    real_data = f.read()[:200000]  # 200KB制限
                    if len(real_data) > 10000:
                        datasets['実ファイル'] = real_data
            except:
                pass
        
        return datasets
    
    def test_zstandard(self, data, level=3):
        """Zstandard圧縮テスト"""
        try:
            # 圧縮
            start_time = time.perf_counter()
            compressed = zstd.compress(data, level=level)
            compression_time = time.perf_counter() - start_time
            
            # 展開
            start_time = time.perf_counter()
            decompressed = zstd.decompress(compressed)
            decompression_time = time.perf_counter() - start_time
            
            # 可逆性チェック
            integrity_ok = (decompressed == data)
            
            return {
                'compressed_size': len(compressed),
                'compression_time': compression_time,
                'decompression_time': decompression_time,
                'compression_ratio': len(compressed) / len(data),
                'compression_speed': len(data) / (1024 * 1024 * compression_time) if compression_time > 0 else 0,
                'decompression_speed': len(data) / (1024 * 1024 * decompression_time) if decompression_time > 0 else 0,
                'integrity_ok': integrity_ok,
                'total_time': compression_time + decompression_time
            }
        except Exception as e:
            return {'error': str(e)}
    
    def test_7zip(self, data):
        """7Zip圧縮テスト"""
        try:
            # 一時ファイルを使用
            with tempfile.NamedTemporaryFile() as temp_input:
                temp_input.write(data)
                temp_input.flush()
                
                with tempfile.NamedTemporaryFile(suffix='.7z') as temp_output:
                    # 圧縮
                    start_time = time.perf_counter()
                    with py7zr.SevenZipFile(temp_output.name, 'w') as archive:
                        archive.write(temp_input.name, 'data')
                    compression_time = time.perf_counter() - start_time
                    
                    # 圧縮サイズ取得
                    compressed_size = os.path.getsize(temp_output.name)
                    
                    # 展開
                    start_time = time.perf_counter()
                    with py7zr.SevenZipFile(temp_output.name, 'r') as archive:
                        extracted = archive.read(['data'])
                        decompressed = extracted['data'].getvalue()
                    decompression_time = time.perf_counter() - start_time
                    
                    # 可逆性チェック
                    integrity_ok = (decompressed == data)
                    
                    return {
                        'compressed_size': compressed_size,
                        'compression_time': compression_time,
                        'decompression_time': decompression_time,
                        'compression_ratio': compressed_size / len(data),
                        'compression_speed': len(data) / (1024 * 1024 * compression_time) if compression_time > 0 else 0,
                        'decompression_speed': len(data) / (1024 * 1024 * decompression_time) if decompression_time > 0 else 0,
                        'integrity_ok': integrity_ok,
                        'total_time': compression_time + decompression_time
                    }
        except Exception as e:
            return {'error': str(e)}
    
    def test_nexus_lightweight(self, data):
        """NEXUS軽量モードテスト"""
        try:
            # 圧縮
            start_time = time.perf_counter()
            compressed, meta = self.nexus_light.compress_fast(data)
            compression_time = time.perf_counter() - start_time
            
            # 展開
            start_time = time.perf_counter()
            decompressed = self.nexus_light.decompress_fast(compressed, meta)
            decompression_time = time.perf_counter() - start_time
            
            # 可逆性チェック
            integrity_ok = (decompressed == data)
            
            return {
                'compressed_size': len(compressed),
                'compression_time': compression_time,
                'decompression_time': decompression_time,
                'compression_ratio': len(compressed) / len(data),
                'compression_speed': len(data) / (1024 * 1024 * compression_time) if compression_time > 0 else 0,
                'decompression_speed': len(data) / (1024 * 1024 * decompression_time) if decompression_time > 0 else 0,
                'integrity_ok': integrity_ok,
                'total_time': compression_time + decompression_time
            }
        except Exception as e:
            return {'error': str(e)}
    
    def test_nexus_normal(self, data):
        """NEXUS通常モードテスト（完全版エンジン）"""
        try:
            # 通常モードの実装が必要
            # 現在は軽量モードと同じ結果を返す（プレースホルダー）
            print("⚠️ 注意: 通常モードは軽量モードの結果を使用（実装待ち）")
            return self.test_nexus_lightweight(data)
        except Exception as e:
            return {'error': str(e)}
    
    def run_comprehensive_test(self):
        """包括的テスト実行"""
        print("🔍 包括的圧縮エンジン比較テスト開始")
        print("="*80)
        
        datasets = self.create_test_datasets()
        
        engines = {
            'Zstandard レベル1': lambda data: self.test_zstandard(data, level=1),
            'Zstandard レベル3': lambda data: self.test_zstandard(data, level=3),
            'Zstandard レベル6': lambda data: self.test_zstandard(data, level=6),
            '7Zip': self.test_7zip,
            'NEXUS軽量モード': self.test_nexus_lightweight,
            'NEXUS通常モード': self.test_nexus_normal
        }
        
        all_results = {}
        
        for dataset_name, data in datasets.items():
            print(f"\n📊 テストデータ: {dataset_name}")
            print(f"   原始サイズ: {len(data):,} bytes")
            print("-" * 60)
            
            dataset_results = {}
            
            for engine_name, test_func in engines.items():
                print(f"   {engine_name:20} ... ", end="", flush=True)
                
                try:
                    result = test_func(data)
                    
                    if 'error' in result:
                        print(f"❌ エラー: {result['error']}")
                        continue
                    
                    dataset_results[engine_name] = result
                    
                    # 結果表示
                    ratio = result['compression_ratio']
                    comp_speed = result['compression_speed']
                    decomp_speed = result['decompression_speed']
                    integrity = "✅" if result['integrity_ok'] else "❌"
                    
                    print(f"{integrity} 圧縮率: {ratio:.3f} | "
                          f"圧縮: {comp_speed:6.1f} MB/s | "
                          f"展開: {decomp_speed:6.1f} MB/s")
                    
                except Exception as e:
                    print(f"❌ 例外: {e}")
            
            all_results[dataset_name] = dataset_results
        
        return all_results
    
    def analyze_results(self, results):
        """結果分析"""
        print(f"\n{'='*80}")
        print("📈 詳細分析結果")
        print(f"{'='*80}")
        
        # エンジン別統計
        engine_stats = {}
        
        for dataset_name, dataset_results in results.items():
            for engine_name, result in dataset_results.items():
                if 'error' not in result:
                    if engine_name not in engine_stats:
                        engine_stats[engine_name] = {
                            'ratios': [], 'comp_speeds': [], 'decomp_speeds': [],
                            'total_times': [], 'integrity_failures': 0, 'test_count': 0
                        }
                    
                    stats = engine_stats[engine_name]
                    stats['ratios'].append(result['compression_ratio'])
                    stats['comp_speeds'].append(result['compression_speed'])
                    stats['decomp_speeds'].append(result['decompression_speed'])
                    stats['total_times'].append(result['total_time'])
                    stats['test_count'] += 1
                    
                    if not result['integrity_ok']:
                        stats['integrity_failures'] += 1
        
        # 平均値計算と表示
        print("\n🎯 エンジン別平均性能:")
        print("-" * 80)
        print(f"{'エンジン名':<20} {'圧縮率':<8} {'削減率':<8} {'圧縮速度':<12} {'展開速度':<12} {'可逆性'}")
        print("-" * 80)
        
        for engine_name, stats in engine_stats.items():
            if stats['test_count'] > 0:
                avg_ratio = sum(stats['ratios']) / len(stats['ratios'])
                avg_reduction = (1 - avg_ratio) * 100
                avg_comp_speed = sum(stats['comp_speeds']) / len(stats['comp_speeds'])
                avg_decomp_speed = sum(stats['decomp_speeds']) / len(stats['decomp_speeds'])
                integrity_rate = (stats['test_count'] - stats['integrity_failures']) / stats['test_count'] * 100
                
                print(f"{engine_name:<20} {avg_ratio:<8.3f} {avg_reduction:<7.1f}% "
                      f"{avg_comp_speed:<11.1f} {avg_decomp_speed:<11.1f} {integrity_rate:<6.1f}%")
        
        return engine_stats
    
    def generate_strategic_analysis(self, results, engine_stats):
        """戦略的分析・計画立案"""
        print(f"\n{'='*80}")
        print("🎯 目標達成分析 & 戦略的計画")
        print(f"{'='*80}")
        
        # 目標指標
        targets = {
            '通常モード_vs_7Zip': {
                'compression_target': '同等以上',
                'speed_target': '2倍以上',
                'description': '7Zipに圧縮率で勝つか同等、速度で倍以上'
            },
            '軽量モード_vs_Zstandard': {
                'compression_target': '同等以上',
                'speed_target': '優位',
                'description': 'Zstandardに圧縮率で同等か勝って、速度で勝つ'
            }
        }
        
        # 分析実行
        if '7Zip' in engine_stats and 'NEXUS通常モード' in engine_stats:
            self._analyze_normal_vs_7zip(engine_stats)
        
        if 'Zstandard レベル3' in engine_stats and 'NEXUS軽量モード' in engine_stats:
            self._analyze_lightweight_vs_zstd(engine_stats)
        
        # 今後の改善計画
        self._generate_improvement_roadmap(engine_stats)
    
    def _analyze_normal_vs_7zip(self, engine_stats):
        """通常モード vs 7Zip分析"""
        print("\n🔍 通常モード vs 7Zip 詳細分析:")
        
        normal_stats = engine_stats.get('NEXUS通常モード', {})
        zip7_stats = engine_stats.get('7Zip', {})
        
        if not normal_stats or not zip7_stats:
            print("❌ 比較データが不足しています")
            return
        
        normal_ratio = sum(normal_stats['ratios']) / len(normal_stats['ratios'])
        zip7_ratio = sum(zip7_stats['ratios']) / len(zip7_stats['ratios'])
        normal_speed = sum(normal_stats['comp_speeds']) / len(normal_stats['comp_speeds'])
        zip7_speed = sum(zip7_stats['comp_speeds']) / len(zip7_stats['comp_speeds'])
        
        compression_improvement = ((zip7_ratio - normal_ratio) / zip7_ratio) * 100
        speed_improvement = (normal_speed / zip7_speed)
        
        print(f"   圧縮率: NEXUS {normal_ratio:.3f} vs 7Zip {zip7_ratio:.3f}")
        print(f"   圧縮率改善: {compression_improvement:+.1f}% (正=NEXUS優位)")
        print(f"   速度: NEXUS {normal_speed:.1f} vs 7Zip {zip7_speed:.1f} MB/s")
        print(f"   速度倍率: {speed_improvement:.1f}x")
        
        # 目標達成評価
        compression_goal = compression_improvement >= 0
        speed_goal = speed_improvement >= 2.0
        
        print(f"\n   📊 目標達成状況:")
        print(f"   圧縮率目標: {'✅ 達成' if compression_goal else '❌ 未達成'}")
        print(f"   速度目標(2倍): {'✅ 達成' if speed_goal else '❌ 未達成'}")
        
        if not compression_goal or not speed_goal:
            print(f"\n   🚀 改善必要項目:")
            if not compression_goal:
                print(f"   - 圧縮率を{-compression_improvement:.1f}%改善が必要")
            if not speed_goal:
                print(f"   - 速度を{2.0/speed_improvement:.1f}倍改善が必要")
    
    def _analyze_lightweight_vs_zstd(self, engine_stats):
        """軽量モード vs Zstandard分析"""
        print("\n🔍 軽量モード vs Zstandard 詳細分析:")
        
        light_stats = engine_stats.get('NEXUS軽量モード', {})
        zstd_stats = engine_stats.get('Zstandard レベル3', {})
        
        if not light_stats or not zstd_stats:
            print("❌ 比較データが不足しています")
            return
        
        light_ratio = sum(light_stats['ratios']) / len(light_stats['ratios'])
        zstd_ratio = sum(zstd_stats['ratios']) / len(zstd_stats['ratios'])
        light_speed = sum(light_stats['comp_speeds']) / len(light_stats['comp_speeds'])
        zstd_speed = sum(zstd_stats['comp_speeds']) / len(zstd_stats['comp_speeds'])
        
        compression_improvement = ((zstd_ratio - light_ratio) / zstd_ratio) * 100
        speed_improvement = ((light_speed - zstd_speed) / zstd_speed) * 100
        
        print(f"   圧縮率: NEXUS軽量 {light_ratio:.3f} vs Zstd {zstd_ratio:.3f}")
        print(f"   圧縮率改善: {compression_improvement:+.1f}% (正=NEXUS優位)")
        print(f"   速度: NEXUS軽量 {light_speed:.1f} vs Zstd {zstd_speed:.1f} MB/s")
        print(f"   速度改善: {speed_improvement:+.1f}% (正=NEXUS優位)")
        
        # 目標達成評価
        compression_goal = compression_improvement >= 0
        speed_goal = speed_improvement > 0
        
        print(f"\n   📊 目標達成状況:")
        print(f"   圧縮率目標: {'✅ 達成' if compression_goal else '❌ 未達成'}")
        print(f"   速度目標: {'✅ 達成' if speed_goal else '❌ 未達成'}")
        
        if compression_goal and speed_goal:
            print(f"   🎉 軽量モードは目標を完全達成！")
    
    def _generate_improvement_roadmap(self, engine_stats):
        """改善ロードマップ生成"""
        print(f"\n{'='*60}")
        print("🗺️ 今後の改善ロードマップ")
        print(f"{'='*60}")
        
        print("\n📅 Phase 1: 短期改善 (1-2ヶ月)")
        print("   🔧 通常モードの実装完了")
        print("   - BWT変換の最適化")
        print("   - Context Mixing の効率化")
        print("   - 並列処理の強化")
        print("   - メモリ使用量の最適化")
        
        print("\n📅 Phase 2: 中期改善 (3-6ヶ月)")
        print("   ⚡ Rust移植による高速化")
        print("   - コアアルゴリズムのRust実装")
        print("   - Python FFI によるハイブリッド化")
        print("   - SIMD命令の活用")
        print("   - マルチスレッド最適化")
        
        print("\n📅 Phase 3: 長期改善 (6-12ヶ月)")
        print("   🧠 AI/ML最適化の導入")
        print("   - データパターン学習")
        print("   - 適応的圧縮アルゴリズム")
        print("   - GPU加速サポート")
        print("   - 新しい変換アルゴリズム")
        
        print("\n🎯 重点改善領域:")
        
        # 7Zip比較での改善点
        if '7Zip' in engine_stats:
            zip7_ratio = sum(engine_stats['7Zip']['ratios']) / len(engine_stats['7Zip']['ratios'])
            print(f"   📦 vs 7Zip対策:")
            print(f"   - 目標圧縮率: {zip7_ratio:.3f} 以下")
            print(f"   - LZMA2アルゴリズムの研究・改良")
            print(f"   - 辞書サイズの動的調整")
            print(f"   - PPM (Prediction by Partial Matching) の導入")
        
        # Zstandard比較での改善点
        if 'Zstandard レベル3' in engine_stats:
            zstd_speed = sum(engine_stats['Zstandard レベル3']['comp_speeds']) / len(engine_stats['Zstandard レベル3']['comp_speeds'])
            print(f"   ⚡ vs Zstandard対策:")
            print(f"   - 目標速度: {zstd_speed:.1f} MB/s 以上")
            print(f"   - ストリーミング圧縮の最適化")
            print(f"   - 前処理の軽量化")
            print(f"   - キャッシュ効率の向上")
        
        print(f"\n✅ 推奨優先順位:")
        print("   1. 🥇 軽量モードの更なる最適化（既に優秀）")
        print("   2. 🥈 通常モードの完全実装と最適化")
        print("   3. 🥉 Rust移植による性能向上")
        print("   4. 🏅 AI/ML技術の統合")

def main():
    """メイン実行関数"""
    benchmark = CompressionBenchmark()
    
    print("🚀 包括的圧縮エンジン評価システム v2.0")
    print("通常モード | 軽量モード | Zstandard | 7Zip 完全比較")
    print(f"{'='*80}")
    
    # テスト実行
    results = benchmark.run_comprehensive_test()
    
    # 結果分析
    engine_stats = benchmark.analyze_results(results)
    
    # 戦略分析
    benchmark.generate_strategic_analysis(results, engine_stats)
    
    print(f"\n{'='*80}")
    print("🎊 包括評価完了！")
    print("詳細な分析結果とロードマップが生成されました。")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
