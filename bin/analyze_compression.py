import numpy as np
from nexus_compression_engine import NEXUSCompressor

def analyze_compression_efficiency():
    """NEXUS圧縮効率の詳細分析"""
    print("🔍 NEXUS圧縮効率分析開始")
    
    # 異なるパターンのテストデータ
    test_patterns = {
        "高反復": np.array([1,2,3,4]*20, dtype=np.uint8),
        "低反復": np.array(list(range(50)), dtype=np.uint8),
        "部分反復": np.array([1,2,3,4,5]*5 + list(range(25)), dtype=np.uint8),
        "同一値": np.array([42]*50, dtype=np.uint8),
        "バイナリ": np.array([0,1]*25, dtype=np.uint8)
    }
    
    nc = NEXUSCompressor()
    
    for pattern_name, data in test_patterns.items():
        print(f"\n📊 パターン: {pattern_name}")
        print(f"データサイズ: {len(data)} bytes")
        
        # 圧縮前のエントロピー計算
        unique_vals, counts = np.unique(data, return_counts=True)
        probabilities = counts / len(data)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        print(f"シャノンエントロピー: {entropy:.2f} bits")
        
        # NEXUS圧縮
        try:
            compressed, nexus_state = nc.nexus_compress(data.tobytes())
            print(f"圧縮サイズ: {len(compressed)} bytes")
            print(f"圧縮率: {len(compressed)/len(data)*100:.1f}%")
            
            # NEXUS効果分析
            print(f"グループ数: {len(nexus_state.original_groups)}")
            print(f"ユニークグループ: {len(nexus_state.unique_groups)}")
            reduction = (1 - len(nexus_state.unique_groups)/len(nexus_state.original_groups))*100
            print(f"グループ削減: {reduction:.1f}%")
            
        except Exception as e:
            print(f"エラー: {e}")

if __name__ == "__main__":
    analyze_compression_efficiency()
