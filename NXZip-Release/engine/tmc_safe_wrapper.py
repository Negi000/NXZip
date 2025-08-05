#!/usr/bin/env python3
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
    
    def compress(self, data: bytes, chunk_callback=None, **kwargs) -> Tuple[bytes, Dict[str, Any]]:
        """安全な圧縮処理（コールバック対応）"""
        self.log(f"圧縮開始: {len(data):,} bytes, chunk_callback={chunk_callback is not None}")
        
        try:
            # chunk_callbackがある場合は元エンジンに渡す
            if chunk_callback:
                self.log("chunk_callback検出 - 元エンジンの対応確認中...")
                
                # 元エンジンの compress メソッドの引数を確認
                compress_method = getattr(self.original_engine, 'compress', None)
                if compress_method:
                    import inspect
                    sig = inspect.signature(compress_method)
                    param_names = list(sig.parameters.keys())
                    self.log(f"元エンジンのcompressパラメータ: {param_names}")
                    
                    if 'chunk_callback' in param_names:
                        self.log("chunk_callback対応エンジン - コールバック付きで実行")
                        compressed, info = self.original_engine.compress(data, chunk_callback=chunk_callback, **kwargs)
                    else:
                        self.log("chunk_callback非対応エンジン - コールバック無しで実行")
                        compressed, info = self.original_engine.compress(data, **kwargs)
                else:
                    self.log("元エンジンにcompressメソッドが見つかりません")
                    raise Exception("無効なエンジン")
            else:
                self.log("chunk_callback無し - 標準実行")
                compressed, info = self.original_engine.compress(data, **kwargs)
            
            # 情報に安全フラグを追加
            info['safe_wrapper'] = True
            info['original_size_recorded'] = len(data)
            info['original_hash'] = hashlib.sha256(data).hexdigest()
            
            self.log(f"圧縮完了: {len(data):,} -> {len(compressed):,} bytes")
            return compressed, info
            
        except Exception as e:
            self.log(f"圧縮エラー: {e}", "ERROR")
            import traceback
            self.log(f"スタックトレース: {traceback.format_exc()}", "ERROR")
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
