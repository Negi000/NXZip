"""
NEXUS TMC Engine - LeCo Transform Module

This module provides advanced LeCo (Learning Compression) transformation
with multiple model selection for optimal integer sequence compression.
"""

import numpy as np
import json
from typing import List, Tuple, Dict, Any

__all__ = ['LeCoTransformer']


class LeCoTransformer:
    """
    TMC v6.0 高度機械学習変換（マルチモデル対応）
    動的モデル選択による予測圧縮の最適化
    """
    
    def __init__(self, lightweight_mode: bool = False):
        self.lightweight_mode = lightweight_mode
        
        if lightweight_mode:
            print("⚡ LeCo軽量モード: 簡易処理")
        else:
            print("🧠 LeCo通常モード: 高精度予測")
    
    def transform(self, data: bytes) -> Tuple[List[bytes], Dict[str, Any]]:
        """LeCo v6.0変換：複数モデルの動的選択"""
        print("  [LeCo] TMC v6.0 マルチモデル変換を実行中...")
        info = {'method': 'leco_multimodel', 'original_size': len(data)}
        
        try:
            # 4バイト単位チェック
            if len(data) % 4 != 0:
                print("    [LeCo] データが4バイトの倍数ではないため、変換をスキップします。")
                return [data], info
            
            integers = np.frombuffer(data, dtype=np.int32)
            print(f"    [LeCo] {len(integers)}個の整数を処理します。")
            
            # 複数モデルの試行と最適選択
            best_model = self._select_optimal_model(integers)
            
            model_type = best_model['type']
            params = best_model['params']
            residuals = best_model['residuals']
            compression_score = best_model['score']
            
            print(f"    [LeCo] 最適モデル: {model_type}")
            print(f"    [LeCo] 圧縮スコア: {compression_score:.2f} bits/element")
            print(f"    [LeCo] 残差範囲: [{np.min(residuals)}, {np.max(residuals)}]")
            
            # モデル情報とパラメータのシリアライズ
            model_info = {
                'model_type': model_type,
                'params': params,
                'data_length': len(integers)
            }
            model_info_json = json.dumps(model_info, separators=(',', ':'))
            model_info_bytes = model_info_json.encode('utf-8')
            model_header = len(model_info_bytes).to_bytes(4, 'big') + model_info_bytes
            
            # 残差ストリーム生成
            residuals_stream = residuals.astype(np.int32).tobytes()
            
            # 統計情報更新
            info.update({
                'model_type': model_type,
                'compression_score': compression_score,
                'residual_variance': float(np.var(residuals)),
                'model_params': params
            })
            
            return [model_header, residuals_stream], info
            
        except Exception as e:
            print(f"    [LeCo] エラー: {e}")
            return [data], info
    
    def _select_optimal_model(self, integers: np.ndarray) -> Dict[str, Any]:
        """複数モデルを試行し、最適なものを動的選択"""
        models_to_try = []
        
        # 1. 定数モデル (Constant Model)
        try:
            const_result = self._try_constant_model(integers)
            models_to_try.append(const_result)
            print(f"    [LeCo] 定数モデル: {const_result['score']:.2f} bits/element")
        except Exception as e:
            print(f"    [LeCo] 定数モデルエラー: {e}")
        
        # 2. 線形モデル (Linear Model)
        try:
            linear_result = self._try_linear_model(integers)
            models_to_try.append(linear_result)
            print(f"    [LeCo] 線形モデル: {linear_result['score']:.2f} bits/element")
        except Exception as e:
            print(f"    [LeCo] 線形モデルエラー: {e}")
        
        # 3. 二次モデル (Quadratic Model) - オプション
        if len(integers) >= 10:  # 十分なデータ点がある場合のみ
            try:
                quad_result = self._try_quadratic_model(integers)
                models_to_try.append(quad_result)
                print(f"    [LeCo] 二次モデル: {quad_result['score']:.2f} bits/element")
            except Exception as e:
                print(f"    [LeCo] 二次モデルエラー: {e}")
        
        # 最適モデル選択（最小スコア）
        if not models_to_try:
            # フォールバック: 定数モデル
            mean_val = np.mean(integers)
            residuals = integers - int(mean_val)
            return {
                'type': 'constant_fallback',
                'params': {'c': float(mean_val)},
                'residuals': residuals,
                'score': 32.0  # ペナルティスコア
            }
        
        best_model = min(models_to_try, key=lambda x: x['score'])
        return best_model
    
    def _try_constant_model(self, integers: np.ndarray) -> Dict[str, Any]:
        """定数モデル: y = c (Frame-of-Reference圧縮相当)"""
        mean_val = np.mean(integers)
        constant = int(round(mean_val))
        
        residuals = integers - constant
        max_abs_residual = int(np.max(np.abs(residuals))) if len(residuals) > 0 else 0
        
        # 残差を格納するのに必要なビット数を計算
        bits_needed = max_abs_residual.bit_length() + 1 if max_abs_residual > 0 else 1  # 符号ビット含む
        compression_score = float(bits_needed)
        
        return {
            'type': 'constant',
            'params': {'c': float(constant)},
            'residuals': residuals,
            'score': compression_score
        }
    
    def _try_linear_model(self, integers: np.ndarray) -> Dict[str, Any]:
        """線形モデル: y = ax + b"""
        x = np.arange(len(integers))
        slope, intercept = np.polyfit(x, integers, 1)
        
        predicted_values = (slope * x + intercept).astype(np.int32)
        residuals = integers - predicted_values
        
        max_abs_residual = int(np.max(np.abs(residuals))) if len(residuals) > 0 else 0
        bits_needed = max_abs_residual.bit_length() + 1 if max_abs_residual > 0 else 1
        
        # パラメータ格納コストも考慮（簡易版）
        param_cost = 64  # slope + intercept (各32bit想定)
        total_bits = bits_needed * len(integers) + param_cost
        compression_score = float(total_bits) / len(integers)
        
        return {
            'type': 'linear',
            'params': {'slope': float(slope), 'intercept': float(intercept)},
            'residuals': residuals,
            'score': compression_score
        }
    
    def _try_quadratic_model(self, integers: np.ndarray) -> Dict[str, Any]:
        """二次モデル: y = ax^2 + bx + c"""
        x = np.arange(len(integers))
        coeffs = np.polyfit(x, integers, 2)  # [a, b, c]
        
        predicted_values = np.polyval(coeffs, x).astype(np.int32)
        residuals = integers - predicted_values
        
        max_abs_residual = int(np.max(np.abs(residuals))) if len(residuals) > 0 else 0
        bits_needed = max_abs_residual.bit_length() + 1 if max_abs_residual > 0 else 1
        
        # パラメータ格納コストも考慮
        param_cost = 96  # a + b + c (各32bit想定)
        total_bits = bits_needed * len(integers) + param_cost
        compression_score = float(total_bits) / len(integers)
        
        return {
            'type': 'quadratic',
            'params': {'a': float(coeffs[0]), 'b': float(coeffs[1]), 'c': float(coeffs[2])},
            'residuals': residuals,
            'score': compression_score
        }
    
    def inverse_transform(self, streams: List[bytes], info: Dict[str, Any]) -> bytes:
        """LeCo v6.0マルチモデル逆変換"""
        print("  [LeCo] TMC v6.0 マルチモデル逆変換を実行中...")
        try:
            if len(streams) != 2:
                return streams[0] if streams else b''
            
            # モデル情報の復元
            model_header = streams[0]
            residuals_stream = streams[1]
            
            # モデル情報ヘッダーの解析
            model_info_size = int.from_bytes(model_header[:4], 'big')
            model_info_json = model_header[4:4+model_info_size].decode('utf-8')
            model_info = json.loads(model_info_json)
            
            model_type = model_info['model_type']
            params = model_info['params']
            data_length = model_info['data_length']
            
            # 残差の復元
            residuals = np.frombuffer(residuals_stream, dtype=np.int32)
            
            print(f"    [LeCo] モデルタイプ: {model_type}")
            print(f"    [LeCo] データ長: {data_length}")
            
            # モデルタイプ別の逆変換
            if model_type == 'constant' or model_type == 'constant_fallback':
                constant = int(params['c'])
                original_integers = residuals + constant
                
            elif model_type == 'linear':
                slope = params['slope']
                intercept = params['intercept']
                x = np.arange(len(residuals))
                predicted_values = (slope * x + intercept).astype(np.int32)
                original_integers = predicted_values + residuals
                
            elif model_type == 'quadratic':
                a, b, c = params['a'], params['b'], params['c']
                x = np.arange(len(residuals))
                predicted_values = (a * x*x + b * x + c).astype(np.int32)
                original_integers = predicted_values + residuals
                
            else:
                print(f"    [LeCo] 未知のモデルタイプ: {model_type}")
                return b''.join(streams)
            
            return original_integers.tobytes()
            
        except Exception as e:
            print(f"    [LeCo] 逆変換エラー: {e}")
            return b''.join(streams)
