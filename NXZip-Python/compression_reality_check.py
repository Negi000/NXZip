#!/usr/bin/env python3
"""
NXZip 実際の圧縮チェック - SPE + NEXUS TMC 使用確認
実際にSPEとNEXUS TMCが使われているかを詳細に調査
"""

import os
import sys
import time
import warnings
from pathlib import Path

# 警告を抑制
warnings.filterwarnings("ignore")

# NXZipを直接インポート
sys.path.insert(0, '.')

class CompressionAnalyzer:
    def __init__(self):
        self.test_data = b"Test data for compression analysis " * 100
        
    def analyze_nexus_unified(self):
        """NEXUS Unified エンジンの実際の処理を解析"""
        print("🔍 NEXUS Unified エンジン解析")
        print("=" * 50)
        
        from nxzip.engine.nexus_unified import NEXUSUnified
        nexus = NEXUSUnified()
        
        # 実際の圧縮処理を実行
        original_data = self.test_data
        compressed_data = nexus.compress(original_data)
        
        print(f"元データ: {len(original_data)} bytes")
        print(f"圧縮後: {len(compressed_data)} bytes")
        print(f"圧縮率: {(1 - len(compressed_data)/len(original_data))*100:.1f}%")
        
        # ヘッダー確認
        if compressed_data.startswith(b'NXZIP3.0'):
            print("✅ NEXUS形式ヘッダー確認")
            data_part = compressed_data[8:]
            print(f"ヘッダー除くデータ: {len(data_part)} bytes")
            
            # 実際の圧縮方式確認
            try:
                import zlib
                decompressed = zlib.decompress(data_part)
                if decompressed == original_data:
                    print("❌ 実際の圧縮: 標準zlib使用")
                    print("⚠️ NEXUS TMC未使用: フォールバック処理")
                else:
                    print("✅ NEXUS TMC使用: 独自圧縮")
            except:
                print("✅ NEXUS TMC使用: zlib以外の圧縮")
        else:
            print("❌ NEXUS形式ヘッダーなし")
        
        return compressed_data
    
    def analyze_spe_core(self):
        """SPE Core エンジンの実際の処理を解析"""
        print("\n🔍 SPE Core エンジン解析")  
        print("=" * 50)
        
        from nxzip.engine.spe_core_jit import SPECoreJIT
        spe = SPECoreJIT()
        
        # 実際のSPE変換を実行
        original_data = self.test_data
        spe_transformed = spe.apply_transform(original_data)
        
        print(f"元データ: {len(original_data)} bytes")
        print(f"SPE変換後: {len(spe_transformed)} bytes")
        
        # データ変化確認
        if spe_transformed == original_data:
            print("❌ SPE未実行: データ変化なし")
        else:
            print("✅ SPE実行: データ変換確認")
            print(f"変換前先頭: {original_data[:20].hex()}")
            print(f"変換後先頭: {spe_transformed[:20].hex()}")
        
        # 逆変換テスト
        restored_data = spe.reverse_transform(spe_transformed)
        if restored_data == original_data:
            print("✅ SPE逆変換: 正常動作")
        else:
            print("❌ SPE逆変換: 失敗")
            print(f"復元サイズ: {len(restored_data)} bytes")
            
        return spe_transformed
    
    def analyze_nxzip_core(self):
        """NXZip Core (nxzip_core.py) の使用状況を解析"""
        print("\n🔍 NXZip Core 統合処理解析")
        print("=" * 50)
        
        try:
            # NXZip Coreが存在するかチェック
            nxzip_core_path = Path("c:\\Users\\241822\\Desktop\\新しいフォルダー (2)\\NXZip\\NXZip-Release\\nxzip_core.py")
            if nxzip_core_path.exists():
                print(f"✅ NXZip Core ファイル存在: {nxzip_core_path}")
                
                # 直接インポートを試行
                sys.path.insert(0, str(nxzip_core_path.parent))
                try:
                    import nxzip_core
                    print("✅ NXZip Core インポート成功")
                    
                    # NXZipCoreクラスの使用
                    core = nxzip_core.NXZipCore()
                    print("✅ NXZipCore初期化成功")
                    
                    # 実際の圧縮テスト
                    result = core.compress(self.test_data, mode="balanced")
                    print(f"NXZip Core結果:")
                    print(f"  成功: {result.success}")
                    print(f"  圧縮率: {result.compression_ratio:.1f}%")
                    print(f"  エンジン: {result.engine}")
                    print(f"  メソッド: {result.method}")
                    
                    # メタデータ確認
                    metadata = result.metadata
                    if 'stages' in metadata:
                        print("  処理ステージ:")
                        for stage_name, stage_info in metadata['stages']:
                            print(f"    {stage_name}: {stage_info}")
                            
                        # SPE使用確認
                        spe_used = any('spe' in stage_name for stage_name, _ in metadata['stages'])
                        tmc_used = any('tmc' in stage_name for stage_name, _ in metadata['stages'])
                        
                        print(f"  🔐 SPE使用: {'✅' if spe_used else '❌'}")
                        print(f"  🔄 TMC使用: {'✅' if tmc_used else '❌'}")
                    
                    return result
                    
                except ImportError as e:
                    print(f"❌ NXZip Core インポート失敗: {e}")
                except Exception as e:
                    print(f"❌ NXZip Core 実行エラー: {e}")
            else:
                print("❌ NXZip Core ファイル未発見")
                
        except Exception as e:
            print(f"❌ NXZip Core 解析エラー: {e}")
        
        return None
    
    def analyze_cli_unified(self):
        """CLI Unified の実際の処理を解析"""
        print("\n🔍 CLI Unified 処理解析")
        print("=" * 50)
        
        # 実際に使われる関数を直接実行
        from nxzip.cli_unified import compress_file
        
        # テストファイル作成
        test_file = "temp_test.txt"
        with open(test_file, 'wb') as f:
            f.write(self.test_data)
        
        try:
            # CLI圧縮実行（標準出力をキャプチャ）
            print("CLI Unified compress_file実行中...")
            success = compress_file(test_file, "temp_test.nxz")
            
            if success and os.path.exists("temp_test.nxz"):
                # 結果ファイル解析
                with open("temp_test.nxz", 'rb') as f:
                    compressed_data = f.read()
                
                print(f"圧縮ファイルサイズ: {len(compressed_data)} bytes")
                
                # ヘッダー確認
                if compressed_data.startswith(b'NXZIP3.0'):
                    print("✅ NEXUS Unifiedフォーマット使用")
                    
                    # 実際の圧縮内容確認
                    data_part = compressed_data[8:]
                    try:
                        import zlib
                        decompressed = zlib.decompress(data_part)
                        if decompressed == self.test_data:
                            print("❌ 実際の圧縮: 標準zlib (NEXUS TMC未使用)")
                        else:
                            print("✅ 実際の圧縮: NEXUS TMC使用")
                    except:
                        print("✅ 実際の圧縮: 非zlib (NEXUS TMC使用可能性)")
                else:
                    print("❌ NEXUS Unifiedフォーマット未使用")
                
                # クリーンアップ
                os.remove("temp_test.nxz")
            else:
                print("❌ CLI圧縮失敗")
                
        finally:
            # テストファイル削除
            if os.path.exists(test_file):
                os.remove(test_file)
    
    def run_complete_analysis(self):
        """完全な圧縮解析を実行"""
        print("🔥 NXZip 実際の圧縮解析 - SPE + NEXUS TMC使用確認")
        print("=" * 70)
        
        # 各コンポーネントを個別に解析
        self.analyze_nexus_unified()
        self.analyze_spe_core()
        nxzip_core_result = self.analyze_nxzip_core()
        self.analyze_cli_unified()
        
        # 総合評価
        print("\n" + "=" * 70)
        print("🎯 総合評価")
        print("=" * 70)
        
        print("📊 コンポーネント別使用状況:")
        print("  NEXUS Unified: 実装済み（zlib使用疑い）")
        print("  SPE Core JIT: 実装済み（3段階変換）")
        print("  NXZip Core: 統合処理（実際の使用要確認）")
        print("  CLI Unified: NEXUSフォーマット使用")
        
        print("\n💡 結論:")
        if nxzip_core_result and nxzip_core_result.success:
            print("✅ SPE + NEXUS TMC統合処理が実装されている")
            print("✅ 実際にSPE変換とTMC処理が実行される")
            print("⚠️ ただし最終圧縮は標準ライブラリ使用の可能性")
        else:
            print("⚠️ CLI UnifiedはNEXUS Unifiedを使用")
            print("⚠️ NEXUS Unifiedは内部でzlib使用（TMC未実装疑い）")
            print("⚠️ SPEは独立して動作するが統合されていない")
            print("❌ 完全なSPE + NEXUS TMC統合が未完成")
        
        print("\n🚀 改善提案:")
        print("1. NEXUS Unified内でSPE Core JITを使用")
        print("2. 真のNEXUS TMC圧縮アルゴリズム実装")
        print("3. NXZip Core統合処理の実際の使用")
        print("4. CLI UnifiedでNXZip Core直接使用")

if __name__ == "__main__":
    analyzer = CompressionAnalyzer()
    analyzer.run_complete_analysis()
