#!/usr/bin/env python3
"""
NXZip 包括的パフォーマンステスト
動作速度と圧縮率の徹底検証
"""

import os
import sys
import time
import hashlib
import statistics
import gc
from typing import List, Dict, Any, Tuple
import json

# テスト対象システムのインポート
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nxzip_complete import SuperNXZipFile, SPECore, EncryptionAlgorithm, CompressionAlgorithm, KDFAlgorithm

class ComprehensiveBenchmark:
    """包括的ベンチマークテストスイート"""
    
    def __init__(self):
        self.results = {
            'system_info': self._get_system_info(),
            'spe_tests': [],
            'compression_tests': [],
            'encryption_tests': [],
            'integration_tests': [],
            'memory_tests': [],
            'stress_tests': []
        }
        
        # テストデータの準備
        self.test_data = self._prepare_test_data()
        
    def _get_system_info(self) -> Dict[str, Any]:
        """システム情報を取得"""
        import platform
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'processor': platform.processor(),
            'timestamp': time.time(),
            'test_date': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _prepare_test_data(self) -> Dict[str, bytes]:
        """多様なテストデータを準備"""
        print("📊 テストデータ準備中...")
        
        data = {}
        
        # 1. 小容量データ (1KB)
        data['small'] = b"A" * 1024
        
        # 2. 中容量データ (100KB) - 繰り返しパターン
        pattern = b"The quick brown fox jumps over the lazy dog. " * 2048
        data['medium_repetitive'] = pattern[:100 * 1024]
        
        # 3. 中容量データ (100KB) - ランダム
        data['medium_random'] = os.urandom(100 * 1024)
        
        # 4. 大容量データ (1MB) - 混合パターン
        mixed_data = bytearray()
        for i in range(1024):
            if i % 4 == 0:
                mixed_data.extend(b"COMPRESSED" * 100)
            elif i % 4 == 1:
                mixed_data.extend(os.urandom(1000))
            elif i % 4 == 2:
                mixed_data.extend(b"X" * 1000)
            else:
                mixed_data.extend(str(i).encode() * 200)
        data['large_mixed'] = bytes(mixed_data[:1024 * 1024])
        
        # 5. 超大容量データ (10MB) - 高圧縮率期待
        huge_pattern = b"NXZip" * 200000 + b"Enterprise" * 200000 + b"Security" * 200000
        data['huge_compressible'] = (huge_pattern * 10)[:10 * 1024 * 1024]
        
        # 6. テキストデータ - JSON構造
        json_data = {
            "records": [
                {
                    "id": i,
                    "name": f"Record_{i}",
                    "data": "x" * 100,
                    "timestamp": time.time() + i,
                    "metadata": {"type": "test", "version": 1.0}
                } for i in range(1000)
            ]
        }
        data['structured_json'] = json.dumps(json_data, indent=2).encode('utf-8')
        
        print(f"✅ {len(data)} 種類のテストデータを準備完了")
        return data
    
    def run_all_tests(self) -> Dict[str, Any]:
        """全テストを実行"""
        print("🚀 NXZip 包括的パフォーマンステスト開始")
        print("=" * 60)
        
        # 1. SPE Core単体テスト
        print("\n🔐 SPE Core パフォーマンステスト")
        self._test_spe_performance()
        
        # 2. 圧縮性能テスト
        print("\n🗜️ 圧縮性能テスト")
        self._test_compression_performance()
        
        # 3. 暗号化性能テスト
        print("\n🔒 暗号化性能テスト")
        self._test_encryption_performance()
        
        # 4. 統合システムテスト
        print("\n⚡ 統合システム性能テスト")
        self._test_integration_performance()
        
        # 5. メモリ効率テスト
        print("\n💾 メモリ効率テスト")
        self._test_memory_efficiency()
        
        # 6. ストレステスト
        print("\n🔥 ストレステスト")
        self._test_stress_conditions()
        
        # 7. 結果の分析と表示
        print("\n📈 テスト結果分析")
        self._analyze_results()
        
        return self.results
    
    def _test_spe_performance(self):
        """SPE Core単体の性能テスト"""
        spe = SPECore()
        
        for name, data in self.test_data.items():
            print(f"  📊 テスト: {name} ({len(data):,} bytes)")
            
            times = []
            for trial in range(5):  # 5回測定して平均を取る
                gc.collect()  # ガベージコレクション
                
                # SPE変換テスト
                start_time = time.perf_counter()
                transformed = spe.apply_transform(data)
                transform_time = time.perf_counter() - start_time
                
                # SPE逆変換テスト
                start_time = time.perf_counter()
                restored = spe.reverse_transform(transformed)
                reverse_time = time.perf_counter() - start_time
                
                # 整合性確認
                if restored != data:
                    raise RuntimeError(f"SPE reversibility failed for {name}")
                
                times.append({
                    'transform_time': transform_time,
                    'reverse_time': reverse_time,
                    'total_time': transform_time + reverse_time,
                    'transformed_size': len(transformed)
                })
            
            # 統計計算
            avg_transform = statistics.mean([t['transform_time'] for t in times])
            avg_reverse = statistics.mean([t['reverse_time'] for t in times])
            avg_total = statistics.mean([t['total_time'] for t in times])
            transformed_size = times[0]['transformed_size']
            
            throughput_mb = (len(data) / (1024 * 1024)) / avg_total
            
            result = {
                'data_type': name,
                'original_size': len(data),
                'transformed_size': transformed_size,
                'avg_transform_time': avg_transform,
                'avg_reverse_time': avg_reverse,
                'avg_total_time': avg_total,
                'throughput_mb_s': throughput_mb,
                'size_overhead': (transformed_size - len(data)) / len(data) * 100
            }
            
            self.results['spe_tests'].append(result)
            
            print(f"    ⏱️  変換: {avg_transform*1000:.2f}ms")
            print(f"    ⏱️  復元: {avg_reverse*1000:.2f}ms") 
            print(f"    🚀 スループット: {throughput_mb:.2f} MB/s")
            print(f"    📊 サイズ変化: {result['size_overhead']:+.1f}%")
    
    def _test_compression_performance(self):
        """圧縮性能テスト"""
        algorithms = [
            CompressionAlgorithm.ZLIB,
            CompressionAlgorithm.LZMA2,
            CompressionAlgorithm.ZSTD,
            CompressionAlgorithm.AUTO
        ]
        
        for algo in algorithms:
            print(f"  🗜️ アルゴリズム: {algo}")
            
            for name, data in self.test_data.items():
                if len(data) > 5 * 1024 * 1024 and algo == CompressionAlgorithm.LZMA2:
                    continue  # LZMA2は大容量データで時間がかかりすぎるためスキップ
                
                nxzip = SuperNXZipFile(compression_algo=algo)
                
                times = []
                for trial in range(3):
                    gc.collect()
                    
                    # 圧縮テスト
                    start_time = time.perf_counter()
                    compressed = nxzip.compressor.compress(data, show_progress=False)
                    compress_time = time.perf_counter() - start_time
                    
                    # 展開テスト
                    start_time = time.perf_counter()
                    decompressed = nxzip.compressor.decompress(compressed[0], compressed[1], show_progress=False)
                    decompress_time = time.perf_counter() - start_time
                    
                    # 整合性確認
                    if decompressed != data:
                        raise RuntimeError(f"Compression reversibility failed for {name} with {algo}")
                    
                    times.append({
                        'compress_time': compress_time,
                        'decompress_time': decompress_time,
                        'compressed_size': len(compressed[0])
                    })
                
                # 統計計算
                avg_compress = statistics.mean([t['compress_time'] for t in times])
                avg_decompress = statistics.mean([t['decompress_time'] for t in times])
                compressed_size = times[0]['compressed_size']
                
                compression_ratio = (1 - compressed_size / len(data)) * 100
                compress_throughput = (len(data) / (1024 * 1024)) / avg_compress
                decompress_throughput = (len(data) / (1024 * 1024)) / avg_decompress
                
                result = {
                    'algorithm': algo,
                    'data_type': name,
                    'original_size': len(data),
                    'compressed_size': compressed_size,
                    'compression_ratio': compression_ratio,
                    'avg_compress_time': avg_compress,
                    'avg_decompress_time': avg_decompress,
                    'compress_throughput_mb_s': compress_throughput,
                    'decompress_throughput_mb_s': decompress_throughput
                }
                
                self.results['compression_tests'].append(result)
                
                print(f"    📊 {name}: {compression_ratio:.1f}% | "
                      f"圧縮: {compress_throughput:.1f} MB/s | "
                      f"展開: {decompress_throughput:.1f} MB/s")
    
    def _test_encryption_performance(self):
        """暗号化性能テスト"""
        algorithms = [
            EncryptionAlgorithm.AES_GCM,
            EncryptionAlgorithm.XCHACHA20_POLY1305
        ]
        
        kdf_algorithms = [
            KDFAlgorithm.PBKDF2,
            KDFAlgorithm.SCRYPT
        ]
        
        password = "TestPassword123"
        
        for enc_algo in algorithms:
            for kdf_algo in kdf_algorithms:
                print(f"  🔒 {enc_algo} + {kdf_algo}")
                
                from nxzip_complete import SuperCrypto
                crypto = SuperCrypto(enc_algo, kdf_algo)
                
                for name, data in self.test_data.items():
                    if len(data) > 1024 * 1024:  # 暗号化テストは1MB以下に制限
                        continue
                    
                    times = []
                    for trial in range(3):
                        gc.collect()
                        
                        # 暗号化テスト
                        start_time = time.perf_counter()
                        encrypted_data, metadata = crypto.encrypt(data, password, show_progress=False)
                        encrypt_time = time.perf_counter() - start_time
                        
                        # 復号化テスト
                        start_time = time.perf_counter()
                        decrypted = crypto.decrypt(encrypted_data, metadata, password, show_progress=False)
                        decrypt_time = time.perf_counter() - start_time
                        
                        # 整合性確認
                        if decrypted != data:
                            raise RuntimeError(f"Encryption reversibility failed for {name}")
                        
                        times.append({
                            'encrypt_time': encrypt_time,
                            'decrypt_time': decrypt_time,
                            'encrypted_size': len(encrypted_data)
                        })
                    
                    # 統計計算
                    avg_encrypt = statistics.mean([t['encrypt_time'] for t in times])
                    avg_decrypt = statistics.mean([t['decrypt_time'] for t in times])
                    encrypted_size = times[0]['encrypted_size']
                    
                    encrypt_throughput = (len(data) / (1024 * 1024)) / avg_encrypt
                    decrypt_throughput = (len(data) / (1024 * 1024)) / avg_decrypt
                    size_overhead = (encrypted_size - len(data)) / len(data) * 100
                    
                    result = {
                        'encryption_algorithm': enc_algo,
                        'kdf_algorithm': kdf_algo,
                        'data_type': name,
                        'original_size': len(data),
                        'encrypted_size': encrypted_size,
                        'size_overhead': size_overhead,
                        'avg_encrypt_time': avg_encrypt,
                        'avg_decrypt_time': avg_decrypt,
                        'encrypt_throughput_mb_s': encrypt_throughput,
                        'decrypt_throughput_mb_s': decrypt_throughput
                    }
                    
                    self.results['encryption_tests'].append(result)
                    
                    print(f"    📊 {name}: "
                          f"暗号化: {encrypt_throughput:.1f} MB/s | "
                          f"復号化: {decrypt_throughput:.1f} MB/s | "
                          f"オーバーヘッド: +{size_overhead:.1f}%")
    
    def _test_integration_performance(self):
        """統合システム性能テスト"""
        configurations = [
            {'compression': CompressionAlgorithm.AUTO, 'encryption': EncryptionAlgorithm.AES_GCM, 'kdf': KDFAlgorithm.PBKDF2},
            {'compression': CompressionAlgorithm.ZSTD, 'encryption': EncryptionAlgorithm.XCHACHA20_POLY1305, 'kdf': KDFAlgorithm.SCRYPT},
            {'compression': CompressionAlgorithm.LZMA2, 'encryption': EncryptionAlgorithm.AES_GCM, 'kdf': KDFAlgorithm.PBKDF2}
        ]
        
        password = "IntegrationTest123"
        
        for config in configurations:
            print(f"  ⚡ 構成: {config['compression']} + {config['encryption']} + {config['kdf']}")
            
            nxzip = SuperNXZipFile(
                compression_algo=config['compression'],
                encryption_algo=config['encryption'],
                kdf_algo=config['kdf']
            )
            
            for name, data in self.test_data.items():
                if len(data) > 2 * 1024 * 1024:  # 統合テストは2MB以下に制限
                    continue
                
                times = []
                for trial in range(3):
                    gc.collect()
                    
                    # アーカイブ作成テスト
                    start_time = time.perf_counter()
                    archive_data = nxzip.create_archive(data, password, show_progress=False)
                    create_time = time.perf_counter() - start_time
                    
                    # アーカイブ展開テスト
                    start_time = time.perf_counter()
                    extracted_data = nxzip.extract_archive(archive_data, password, show_progress=False)
                    extract_time = time.perf_counter() - start_time
                    
                    # 整合性確認
                    if extracted_data != data:
                        raise RuntimeError(f"Integration test failed for {name}")
                    
                    times.append({
                        'create_time': create_time,
                        'extract_time': extract_time,
                        'archive_size': len(archive_data)
                    })
                
                # 統計計算
                avg_create = statistics.mean([t['create_time'] for t in times])
                avg_extract = statistics.mean([t['extract_time'] for t in times])
                archive_size = times[0]['archive_size']
                
                total_compression = (1 - archive_size / len(data)) * 100
                create_throughput = (len(data) / (1024 * 1024)) / avg_create
                extract_throughput = (len(data) / (1024 * 1024)) / avg_extract
                
                result = {
                    'configuration': config,
                    'data_type': name,
                    'original_size': len(data),
                    'archive_size': archive_size,
                    'total_compression_ratio': total_compression,
                    'avg_create_time': avg_create,
                    'avg_extract_time': avg_extract,
                    'create_throughput_mb_s': create_throughput,
                    'extract_throughput_mb_s': extract_throughput
                }
                
                self.results['integration_tests'].append(result)
                
                print(f"    📊 {name}: {total_compression:.1f}% | "
                      f"作成: {create_throughput:.1f} MB/s | "
                      f"展開: {extract_throughput:.1f} MB/s")
    
    def _test_memory_efficiency(self):
        """メモリ効率テスト（psutil無しバージョン）"""
        print("  💾 メモリ効率テスト (簡易版)")
        
        nxzip = SuperNXZipFile()
        
        for name, data in self.test_data.items():
            if len(data) > 1024 * 1024:  # メモリテストは1MB以下に制限
                continue
            
            try:
                # メモリ使用量の代わりに処理時間を測定
                start_time = time.perf_counter()
                
                # アーカイブ作成
                archive_data = nxzip.create_archive(data, show_progress=False)
                create_time = time.perf_counter() - start_time
                
                start_time = time.perf_counter()
                # アーカイブ展開
                extracted_data = nxzip.extract_archive(archive_data, show_progress=False)
                extract_time = time.perf_counter() - start_time
                
                # 整合性確認
                if extracted_data != data:
                    raise RuntimeError(f"Memory test integrity failed for {name}")
                
                # 処理効率の計算
                total_time = create_time + extract_time
                efficiency_ratio = (len(data) / 1024 / 1024) / total_time if total_time > 0 else 0
                
                result = {
                    'data_type': name,
                    'data_size_mb': len(data) / 1024 / 1024,
                    'total_processing_time': total_time,
                    'processing_efficiency_mb_s': efficiency_ratio,
                    'archive_compression_ratio': (1 - len(archive_data) / len(data)) * 100
                }
                
                self.results['memory_tests'].append(result)
                
                print(f"    📊 {name}: "
                      f"処理時間: {total_time:.3f}s | "
                      f"効率: {efficiency_ratio:.2f} MB/s | "
                      f"圧縮率: {result['archive_compression_ratio']:.1f}%")
                      
            except Exception as e:
                print(f"    ❌ {name}: エラー - {e}")
                continue
    
    def _test_stress_conditions(self):
        """ストレス条件テスト"""
        print("  🔥 大容量データストレステスト")
        
        # 非常に大きなデータ（ストリーミング処理想定）
        stress_data = b"STRESS_TEST_DATA" * 1024 * 1024  # 16MB
        
        nxzip = SuperNXZipFile(compression_algo=CompressionAlgorithm.ZSTD)
        
        try:
            start_time = time.perf_counter()
            archive_data = nxzip.create_archive(stress_data, show_progress=False)
            create_time = time.perf_counter() - start_time
            
            start_time = time.perf_counter()
            extracted_data = nxzip.extract_archive(archive_data, show_progress=False)
            extract_time = time.perf_counter() - start_time
            
            # 整合性確認
            if extracted_data != stress_data:
                raise RuntimeError("Stress test integrity failed")
            
            compression_ratio = (1 - len(archive_data) / len(stress_data)) * 100
            create_throughput = (len(stress_data) / (1024 * 1024)) / create_time
            extract_throughput = (len(stress_data) / (1024 * 1024)) / extract_time
            
            result = {
                'test_type': 'large_data_stress',
                'data_size_mb': len(stress_data) / 1024 / 1024,
                'compression_ratio': compression_ratio,
                'create_time': create_time,
                'extract_time': extract_time,
                'create_throughput_mb_s': create_throughput,
                'extract_throughput_mb_s': extract_throughput,
                'success': True
            }
            
            print(f"    ✅ 16MB処理成功: {compression_ratio:.1f}% | "
                  f"作成: {create_throughput:.1f} MB/s | "
                  f"展開: {extract_throughput:.1f} MB/s")
            
        except Exception as e:
            result = {
                'test_type': 'large_data_stress',
                'error': str(e),
                'success': False
            }
            print(f"    ❌ 16MB処理失敗: {e}")
        
        self.results['stress_tests'].append(result)
    
    def _analyze_results(self):
        """結果の分析と表示"""
        print("\n📈 総合パフォーマンス分析")
        print("=" * 60)
        
        # SPE性能サマリー
        if self.results['spe_tests']:
            spe_throughputs = [t['throughput_mb_s'] for t in self.results['spe_tests']]
            print(f"🔐 SPE平均スループット: {statistics.mean(spe_throughputs):.2f} MB/s")
            print(f"   最高スループット: {max(spe_throughputs):.2f} MB/s")
            print(f"   最低スループット: {min(spe_throughputs):.2f} MB/s")
        
        # 圧縮性能サマリー
        if self.results['compression_tests']:
            by_algorithm = {}
            for test in self.results['compression_tests']:
                algo = test['algorithm']
                if algo not in by_algorithm:
                    by_algorithm[algo] = {'ratios': [], 'throughputs': []}
                by_algorithm[algo]['ratios'].append(test['compression_ratio'])
                by_algorithm[algo]['throughputs'].append(test['compress_throughput_mb_s'])
            
            print(f"\n🗜️ 圧縮性能サマリー:")
            for algo, data in by_algorithm.items():
                avg_ratio = statistics.mean(data['ratios'])
                avg_throughput = statistics.mean(data['throughputs'])
                print(f"   {algo}: {avg_ratio:.1f}% | {avg_throughput:.1f} MB/s")
        
        # 統合システム性能サマリー
        if self.results['integration_tests']:
            integration_throughputs = [t['create_throughput_mb_s'] for t in self.results['integration_tests']]
            integration_ratios = [t['total_compression_ratio'] for t in self.results['integration_tests']]
            print(f"\n⚡ 統合システム:")
            print(f"   平均総合圧縮率: {statistics.mean(integration_ratios):.1f}%")
            print(f"   平均処理速度: {statistics.mean(integration_throughputs):.1f} MB/s")
        
        # メモリ効率サマリー
        if self.results['memory_tests']:
            memory_efficiencies = [t.get('processing_efficiency_mb_s', 0) for t in self.results['memory_tests'] if t.get('processing_efficiency_mb_s', 0) > 0]
            if memory_efficiencies:
                print(f"\n💾 処理効率: {statistics.mean(memory_efficiencies):.2f} MB/s (データ処理速度)")
        
        print(f"\n✅ 全テスト完了 - 結果はself.resultsに保存されています")
    
    def save_results(self, filename: str = None):
        """結果をJSONファイルに保存"""
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f'nxzip_benchmark_results_{timestamp}.json'
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📊 ベンチマーク結果を保存: {filename}")


def main():
    """メイン実行関数"""
    print("🚀 NXZip 包括的パフォーマンステスト")
    print("=" * 60)
    
    benchmark = ComprehensiveBenchmark()
    
    try:
        results = benchmark.run_all_tests()
        benchmark.save_results()
        
        print("\n🎉 全てのテストが正常に完了しました!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ テストが中断されました")
        return 1
    except Exception as e:
        print(f"\n❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
