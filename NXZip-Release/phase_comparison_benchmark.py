#!/usr/bin/env python3
"""
Phase Comparison Benchmark - Compare Phase 1 vs Phase 2 Performance
==================================================================
"""
import os
import time
import subprocess
import sys

def benchmark_file(filename, versions, modes):
    """ファイルの速度比較ベンチマーク"""
    
    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        return
    
    file_size = os.path.getsize(filename) / 1024 / 1024  # MB
    print(f"\n📁 Testing: {filename} ({file_size:.2f} MB)")
    
    results = {}
    
    for version in versions:
        results[version] = {}
        print(f"\n🚀 Testing {version}...")
        
        for mode in modes:
            print(f"  🎯 Mode: {mode}")
            
            # 速度測定
            start_time = time.time()
            cmd = f'python {version} "{filename}" -o "test_{mode.lower()}_{version.replace(".", "_")}.nxz" -m {mode} --quiet'
            
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
                elapsed_time = time.time() - start_time
                
                if result.returncode == 0:
                    output_file = f"test_{mode.lower()}_{version.replace('.', '_')}.nxz"
                    if os.path.exists(output_file):
                        compressed_size = os.path.getsize(output_file)
                        compression_ratio = (1 - compressed_size / os.path.getsize(filename)) * 100
                        speed = file_size / elapsed_time if elapsed_time > 0 else 0
                        
                        results[version][mode] = {
                            'time': elapsed_time,
                            'speed': speed,
                            'ratio': compression_ratio,
                            'size': compressed_size
                        }
                        
                        print(f"    ✅ {elapsed_time:.3f}s, {speed:.1f} MB/s, ratio: {compression_ratio:.1f}%")
                        
                        # クリーンアップ
                        try:
                            os.remove(output_file)
                        except:
                            pass
                    else:
                        print(f"    ❌ Output file not created")
                        results[version][mode] = {'error': 'no_output'}
                else:
                    print(f"    ❌ Error: {result.stderr[:100]}")
                    results[version][mode] = {'error': result.stderr[:100]}
                    
            except subprocess.TimeoutExpired:
                print(f"    ⏰ Timeout (>60s)")
                results[version][mode] = {'error': 'timeout'}
            except Exception as e:
                print(f"    ❌ Exception: {e}")
                results[version][mode] = {'error': str(e)}
    
    return results

def print_comparison(results, modes):
    """結果比較表示"""
    
    print("\n" + "="*80)
    print("🏆 PERFORMANCE COMPARISON RESULTS")
    print("="*80)
    
    versions = list(results.keys())
    if len(versions) != 2:
        print("❌ Need exactly 2 versions to compare")
        return
    
    v1, v2 = versions
    
    for mode in modes:
        print(f"\n📊 {mode} Mode Comparison:")
        print("-" * 40)
        
        if mode in results[v1] and mode in results[v2]:
            r1 = results[v1][mode]
            r2 = results[v2][mode]
            
            if 'error' not in r1 and 'error' not in r2:
                # 速度比較
                speed_improvement = ((r2['speed'] - r1['speed']) / r1['speed']) * 100
                time_improvement = ((r1['time'] - r2['time']) / r1['time']) * 100
                ratio_diff = r2['ratio'] - r1['ratio']
                
                print(f"  {v1}: {r1['time']:.3f}s, {r1['speed']:.1f} MB/s, {r1['ratio']:.1f}%")
                print(f"  {v2}: {r2['time']:.3f}s, {r2['speed']:.1f} MB/s, {r2['ratio']:.1f}%")
                print(f"  📈 Speed:  {speed_improvement:+.1f}% ({r2['speed']:.1f} vs {r1['speed']:.1f} MB/s)")
                print(f"  ⏱️  Time:   {time_improvement:+.1f}% ({r2['time']:.3f} vs {r1['time']:.3f}s)")
                print(f"  📦 Ratio:  {ratio_diff:+.1f}% ({r2['ratio']:.1f}% vs {r1['ratio']:.1f}%)")
                
                if speed_improvement > 5:
                    print(f"  🚀 IMPROVEMENT: {speed_improvement:.1f}% faster!")
                elif speed_improvement > -5:
                    print(f"  ⚖️  SIMILAR: {speed_improvement:+.1f}% change")
                else:
                    print(f"  🐌 SLOWER: {speed_improvement:.1f}% decrease")
            else:
                print(f"  ❌ Error in results: {r1.get('error', 'ok')} vs {r2.get('error', 'ok')}")

def main():
    print("🚀 NXZip Phase Comparison Benchmark")
    print("=" * 50)
    
    # テストファイル
    test_files = ["COT-001.jpg", "COT-001.png"]
    available_files = [f for f in test_files if os.path.exists(f)]
    
    if not available_files:
        print(f"❌ No test files found: {test_files}")
        return
    
    # テストバージョン
    versions = ["nxzip_core_optimized.py", "nxzip_core_phase2_clean.py"]
    available_versions = [v for v in versions if os.path.exists(v)]
    
    if len(available_versions) < 2:
        print(f"❌ Need 2 versions to compare. Found: {available_versions}")
        return
    
    # テストモード
    modes = ["FAST", "MAXIMUM"]
    
    print(f"📁 Test files: {available_files}")
    print(f"🔧 Versions: {available_versions}")
    print(f"🎯 Modes: {modes}")
    
    # ベンチマーク実行
    all_results = {}
    
    for filename in available_files:
        print(f"\n{'='*60}")
        results = benchmark_file(filename, available_versions, modes)
        all_results[filename] = results
        print_comparison(results, modes)
    
    # 総合結果
    print(f"\n{'='*80}")
    print("🏁 OVERALL SUMMARY")
    print("="*80)
    
    for filename, results in all_results.items():
        print(f"\n📁 {filename}:")
        print_comparison(results, modes)

if __name__ == "__main__":
    main()
