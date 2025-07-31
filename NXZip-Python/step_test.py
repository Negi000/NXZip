#!/usr/bin/env python3
"""
NEXUS TMC v9.0 段階的テスト
"""

def test_step_by_step():
    """段階的にTMCエンジンをテスト"""
    import sys
    sys.path.insert(0, '.')
    
    print("🔍 NEXUS TMC v9.0 段階的テスト")
    print("=" * 50)
    
    try:
        print("Step 1: nxzipパッケージ...")
        import nxzip
        print("✅ nxzip OK")
        
        print("Step 2: engineパッケージ...")
        import nxzip.engine
        print("✅ nxzip.engine OK")
        
        print("Step 3: 各種依存モジュール確認...")
        
        # 必要な依存関係を段階的に確認
        try:
            from nxzip.engine.nexus_unified import NEXUSUnified
            print("✅ NEXUSUnified OK")
        except Exception as e:
            print(f"⚠️ NEXUSUnified: {e}")
        
        try:
            from nxzip.engine.nexus_target import NEXUSTargetAchievement
            print("✅ NEXUSTargetAchievement OK")
        except Exception as e:
            print(f"⚠️ NEXUSTargetAchievement: {e}")
        
        try:
            from nxzip.engine.nexus_breakthrough import NEXUSBreakthroughEngine
            print("✅ NEXUSBreakthroughEngine OK")
        except Exception as e:
            print(f"⚠️ NEXUSBreakthroughEngine: {e}")
        
        print("Step 4: TMCエンジン本体...")
        try:
            from nxzip.engine.nexus_tmc import NEXUSTMCEngineV9
            print("✅ NEXUSTMCEngineV9 クラス インポート成功")
            
            # エンジン初期化テスト
            engine = NEXUSTMCEngineV9(max_workers=1)
            print("✅ エンジン初期化成功")
            
            # 基本圧縮テスト
            test_data = b"NEXUS TMC v9.0 Engine Test"
            print(f"📄 テストデータ: {len(test_data)} bytes")
            
            compressed, meta = engine.compress_tmc(test_data)
            print(f"📦 圧縮完了: {len(compressed)} bytes")
            
            decompressed, decomp_meta = engine.decompress_tmc(compressed)
            print(f"📂 展開完了: {len(decompressed)} bytes")
            
            # 結果確認
            is_identical = test_data == decompressed
            compression_ratio = len(compressed) / len(test_data) * 100
            
            print(f"📊 圧縮率: {compression_ratio:.1f}%")
            print(f"🔄 可逆性: {'✅ OK' if is_identical else '❌ NG'}")
            
            if 'data_type' in meta:
                print(f"🔍 検出データ型: {meta['data_type']}")
            
            if is_identical:
                print("🎉 NEXUS TMC v9.0 エンジンテスト完全成功！")
                return True
            else:
                print("⚠️ 可逆性に問題があります")
                return False
                
        except Exception as e:
            print(f"❌ TMCエンジンエラー: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ パッケージインポートエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_step_by_step()
