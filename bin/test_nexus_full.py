from nexus_compression_engine import NEXUSCompressor
import numpy as np

def test_nexus_full_cycle():
    """完全なNEXUS圧縮・展開サイクルのテスト"""
    print("🧪 NEXUS完全サイクルテスト開始")
    
    # テストデータ
    test_cases = [
        np.array([1,2,3,4]*5, dtype=np.uint8),
        np.array([1,1,2,2,3,3,4,4]*3, dtype=np.uint8),
        np.array(range(20), dtype=np.uint8),
        np.array([100,200,50,75]*8, dtype=np.uint8)
    ]
    
    nc = NEXUSCompressor()
    
    for i, data in enumerate(test_cases):
        print(f"\n📋 テストケース {i+1}")
        print(f"Original: {list(data[:10])}{'...' if len(data) > 10 else ''}")
        print(f"Size: {len(data.tobytes())} bytes")
        
        try:
            # 圧縮
            compressed = nc.compress(data.tobytes())
            print(f"Compressed: {len(compressed)} bytes")
            print(f"Ratio: {len(data.tobytes())/len(compressed):.2f}:1")
            
            # 展開
            decompressed = nc.decompress(compressed)
            decompressed_array = np.frombuffer(decompressed, dtype=np.uint8)
            
            # 検証
            success = np.array_equal(data, decompressed_array)
            print(f"✅ 成功: {success}" if success else f"❌ 失敗: {success}")
            
            if not success:
                print(f"Expected: {list(data[:5])}")
                print(f"Got: {list(decompressed_array[:5])}")
        
        except Exception as e:
            print(f"❌ エラー: {e}")
    
    print("\n🏁 テスト完了")

if __name__ == "__main__":
    test_nexus_full_cycle()
