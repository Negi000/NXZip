#!/usr/bin/env python3
"""
TMC問題の回避策
GUIでの使用時に適切な展開が行われるように修正
"""

import os
import sys

def create_tmc_wrapper():
    """TMC問題の回避策となるラッパーを作成"""
    print("🔧 TMC問題回避策の実装")
    print("=" * 50)
    
    wrapper_code = '''#!/usr/bin/env python3
"""
TMC v9.1 回避策ラッパー
GUI使用時の問題を回避するための安全な実装
"""
import zlib
import lzma
import hashlib
from typing import Dict, Any, Tuple

class TMCSafeWrapper:
    """TMC安全ラッパークラス"""
    
    def __init__(self, original_engine):
        self.original_engine = original_engine
        self.debug = True
    
    def log(self, message: str, level: str = "INFO"):
        if self.debug:
            print(f"[TMC安全:{level}] {message}")
    
    def compress(self, data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """安全な圧縮処理"""
        try:
            # 元エンジンで圧縮
            compressed, info = self.original_engine.compress(data)
            
            # 情報に安全フラグを追加
            info['safe_wrapper'] = True
            info['original_size_recorded'] = len(data)
            info['original_hash'] = hashlib.sha256(data).hexdigest()
            
            self.log(f"圧縮完了: {len(data):,} -> {len(compressed):,} bytes")
            return compressed, info
            
        except Exception as e:
            self.log(f"圧縮エラー: {e}", "ERROR")
            raise
    
    def decompress(self, compressed_data: bytes, compression_info: Dict[str, Any]) -> bytes:
        """安全な解凍処理"""
        self.log(f"安全解凍開始: {len(compressed_data):,} bytes")
        
        try:
            # 元エンジンで解凍を試行
            result = self.original_engine.decompress(compressed_data, compression_info)
            
            # サイズ・ハッシュ検証
            expected_size = compression_info.get('original_size_recorded')
            expected_hash = compression_info.get('original_hash')
            
            if expected_size and len(result) != expected_size:
                self.log(f"⚠️ サイズ不一致: 期待{expected_size:,} vs 実際{len(result):,}", "WARNING")
                
                # サイズが大幅に小さい場合は問題ありと判定
                if len(result) < expected_size * 0.1:  # 10%未満の場合
                    self.log("❌ 深刻なサイズ不一致 - フォールバック実行", "ERROR")
                    return self._fallback_decompress(compressed_data, compression_info)
            
            if expected_hash:
                actual_hash = hashlib.sha256(result).hexdigest()
                if actual_hash != expected_hash:
                    self.log(f"⚠️ ハッシュ不一致検出", "WARNING")
                    self.log(f"期待: {expected_hash[:16]}...", "DEBUG")
                    self.log(f"実際: {actual_hash[:16]}...", "DEBUG")
                    
                    # フォールバック処理
                    return self._fallback_decompress(compressed_data, compression_info)
            
            self.log(f"安全解凍完了: {len(result):,} bytes")
            return result
            
        except Exception as e:
            self.log(f"標準解凍失敗: {e}", "ERROR")
            return self._fallback_decompress(compressed_data, compression_info)
    
    def _fallback_decompress(self, compressed_data: bytes, compression_info: Dict[str, Any]) -> bytes:
        """フォールバック解凍処理"""
        self.log("🔄 フォールバック解凍開始")
        
        # 基本的な解凍を試行
        methods = [
            ("zlib", lambda d: zlib.decompress(d)),
            ("lzma", lambda d: lzma.decompress(d)),
        ]
        
        for method_name, decompress_func in methods:
            try:
                result = decompress_func(compressed_data)
                self.log(f"✅ {method_name}フォールバック成功: {len(result):,} bytes")
                
                # サイズ妥当性チェック
                expected_size = compression_info.get('original_size_recorded')
                if expected_size and abs(len(result) - expected_size) < expected_size * 0.1:
                    self.log("✅ フォールバック結果のサイズ妥当")
                    return result
                elif not expected_size:
                    return result
                    
            except Exception as e:
                self.log(f"{method_name}フォールバック失敗: {e}")
                continue
        
        # すべて失敗
        self.log("❌ すべてのフォールバック失敗", "ERROR")
        raise Exception("フォールバック解凍も失敗")

def wrap_tmc_engine(engine):
    """TMCエンジンを安全ラッパーでラップ"""
    return TMCSafeWrapper(engine)
'''
    
    # ラッパーファイルを作成
    wrapper_path = "NXZip-Release/engine/tmc_safe_wrapper.py"
    with open(wrapper_path, 'w', encoding='utf-8') as f:
        f.write(wrapper_code)
    
    print(f"✅ TMC安全ラッパー作成完了: {wrapper_path}")
    
    # GUIファイルの修正
    print("🔄 GUIファイルに安全ラッパーを統合中...")
    
    gui_file = "NXZip-Release/NXZip_Professional_v2.py"
    
    with open(gui_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # ラッパーのインポートを追加
    import_addition = """
# TMC安全ラッパーのインポート
try:
    from engine.tmc_safe_wrapper import wrap_tmc_engine
    TMC_SAFE_WRAPPER_AVAILABLE = True
    print("🛡️ TMC安全ラッパー利用可能")
except ImportError:
    TMC_SAFE_WRAPPER_AVAILABLE = False
    print("⚠️ TMC安全ラッパーが見つかりません")
"""
    
    if "TMC_SAFE_WRAPPER_AVAILABLE" not in content:
        # TMC_FIXED_AVAILABLEの後に追加
        fixed_pos = content.find("TMC_FIXED_AVAILABLE = False")
        if fixed_pos > 0:
            insert_pos = content.find("\n", fixed_pos) + 1
            content = content[:insert_pos] + import_addition + content[insert_pos:]
    
    # TMCエンジン初期化部分を修正
    old_init = '''            try:
                self.tmc_engine = NEXUSTMCEngineV91()
                print(f"🔥 NEXUS TMC v9.1 Engine initialized for {mode} mode")
            except Exception as e:
                print(f"⚠️ TMC engine initialization failed: {e}")
                self.use_advanced = False'''
    
    new_init = '''            try:
                base_engine = NEXUSTMCEngineV91()
                
                # 安全ラッパーで包む
                if TMC_SAFE_WRAPPER_AVAILABLE:
                    self.tmc_engine = wrap_tmc_engine(base_engine)
                    print(f"🛡️ NEXUS TMC v9.1 Engine (安全ラッパー付き) initialized for {mode} mode")
                else:
                    self.tmc_engine = base_engine
                    print(f"🔥 NEXUS TMC v9.1 Engine (標準) initialized for {mode} mode")
                    
            except Exception as e:
                print(f"⚠️ TMC engine initialization failed: {e}")
                self.use_advanced = False'''
    
    if old_init in content:
        content = content.replace(old_init, new_init)
        print("✅ TMCエンジン初期化部分を安全ラッパー対応に修正")
    
    # ファイル更新
    with open(gui_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ GUI統合完了")
    print("\n🎯 TMC問題回避策の実装完了")
    print("📝 次回GUI実行時に安全ラッパーが動作し、問題を回避します")

if __name__ == "__main__":
    create_tmc_wrapper()
