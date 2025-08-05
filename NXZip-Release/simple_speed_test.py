#!/usr/bin/env python3
"""
Simple Performance Test - Manual Speed Measurement
================================================
"""
import time
import os
import sys

def test_compression():
    """シンプルな圧縮速度テスト"""
    
    test_file = "COT-001.jpg"
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return
    
    file_size = os.path.getsize(test_file) / 1024 / 1024  # MB
    print(f"📁 Testing: {test_file} ({file_size:.2f} MB)")
    
    # テスト実行
    versions = [
        ("Phase 1 Optimized", "nxzip_core_optimized.py"),
        ("Phase 2 Clean", "nxzip_core_phase2_clean.py")
    ]
    
    modes = ["FAST", "MAXIMUM"]
    
    for version_name, script_name in versions:
        if not os.path.exists(script_name):
            print(f"❌ Script not found: {script_name}")
            continue
            
        print(f"\n🚀 Testing {version_name}...")
        
        for mode in modes:
            print(f"  🎯 Mode: {mode}")
            
            # 手動実行
            import subprocess
            start_time = time.time()
            
            try:
                # 実行コマンド作成
                cmd = [
                    sys.executable, script_name, test_file,
                    "-o", f"test_{mode}_{version_name.replace(' ', '_')}.nxz",
                    "-m", mode, "--quiet"
                ]
                
                print(f"    🔧 Running: {' '.join(cmd)}")
                
                # プロセス実行（30秒タイムアウト）
                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                try:
                    stdout, stderr = process.communicate(timeout=30)
                    elapsed_time = time.time() - start_time
                    
                    if process.returncode == 0:
                        output_file = f"test_{mode}_{version_name.replace(' ', '_')}.nxz"
                        if os.path.exists(output_file):
                            compressed_size = os.path.getsize(output_file)
                            compression_ratio = (1 - compressed_size / os.path.getsize(test_file)) * 100
                            speed = file_size / elapsed_time if elapsed_time > 0 else 0
                            
                            print(f"    ✅ Success: {elapsed_time:.3f}s, {speed:.1f} MB/s, ratio: {compression_ratio:.1f}%")
                            
                            # クリーンアップ
                            try:
                                os.remove(output_file)
                            except:
                                pass
                        else:
                            print(f"    ❌ Output file not created")
                    else:
                        print(f"    ❌ Exit code: {process.returncode}")
                        if stderr:
                            print(f"    📝 Error: {stderr[:200]}")
                        
                except subprocess.TimeoutExpired:
                    process.kill()
                    print(f"    ⏰ Timeout (30s exceeded)")
                    
            except Exception as e:
                print(f"    ❌ Exception: {e}")

if __name__ == "__main__":
    print("🚀 Simple NXZip Performance Test")
    print("=" * 40)
    test_compression()
