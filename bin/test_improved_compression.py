from nexus_compression_engine import NEXUSCompressor
import numpy as np
import time

def test_improved_compression():
    """改善されたNEXUS圧縮のテスト"""
    print("🚀 改善版NEXUS圧縮テスト開始")
    
    test_cases = [
        ("小データ(20B)", np.array([1,2,3,4]*5, dtype=np.uint8)),
        ("反復データ(100B)", np.array([1,2,3,4]*25, dtype=np.uint8)),
        ("同一値データ(50B)", np.array([42]*50, dtype=np.uint8)),
        ("バイナリデータ(100B)", np.array([0,1]*50, dtype=np.uint8)),
        ("低エントロピー(80B)", np.array([1,1,2,2,3,3,4,4]*10, dtype=np.uint8)),
        ("高エントロピー(100B)", np.array(range(100), dtype=np.uint8))
    ]
    
    nc = NEXUSCompressor()
    total_improvement = 0
    
    for name, data in test_cases:
        print(f"\n📋 {name}")
        print(f"元サイズ: {len(data)} bytes")
        
        start_time = time.time()
        try:
            compressed = nc.compress(data.tobytes())
            compression_time = time.time() - start_time
            
            print(f"圧縮サイズ: {len(compressed)} bytes")
            ratio = len(compressed) / len(data) * 100
            print(f"圧縮率: {ratio:.1f}%")
            print(f"圧縮時間: {compression_time*1000:.1f}ms")
            
            # 展開テスト
            start_time = time.time()
            decompressed = nc.decompress(compressed)
            decompress_time = time.time() - start_time
            
            success = np.array_equal(data, np.frombuffer(decompressed, dtype=np.uint8))
            print(f"展開時間: {decompress_time*1000:.1f}ms")
            print(f"✅ 成功: {success}" if success else f"❌ 失敗")
            
            if success and ratio < 100:
                improvement = 100 - ratio
                total_improvement += improvement
                print(f"📈 圧縮改善: {improvement:.1f}%")
        
        except Exception as e:
            print(f"❌ エラー: {e}")
    
    print(f"\n🎯 総合結果")
    print(f"平均圧縮改善: {total_improvement/len(test_cases):.1f}%")
    print("🏁 テスト完了")

if __name__ == "__main__":
    test_improved_compression()
