#!/usr/bin/env python3
"""
SPE Core デバッグスクリプト
"""

import sys
sys.path.insert(0, '.')

from nxzip.engine.spe_core import SPECore

def debug_phase1():
    """フェーズ1のデバッグ"""
    spe = SPECore()
    test_data = b"test"
    
    print(f"Original data: {test_data}")
    print(f"Original length: {len(test_data)}")
    
    # フェーズ1のみテスト
    result = bytearray(test_data)
    processed = spe._phase1_preprocessing(result, len(test_data))
    print(f"After phase1: length={len(processed)}")
    print(f"Last 20 bytes: {processed[-20:].hex()}")
    
    # 逆変換テスト
    try:
        restored = spe._reverse_phase1_preprocessing(processed)
        print(f"Restored: {restored}")
        print(f"Match: {restored == test_data}")
    except Exception as e:
        print(f"Error in reverse: {e}")

if __name__ == "__main__":
    debug_phase1()
