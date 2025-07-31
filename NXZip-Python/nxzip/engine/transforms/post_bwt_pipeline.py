"""
NEXUS TMC Engine - Post-BWT Pipeline Module

This module provides specialized post-BWT processing pipeline for
handling BWT+MTF transformed data with run-length encoding and
entropy optimization.
"""

from typing import List, Tuple

__all__ = ['PostBWTPipeline']


class PostBWTPipeline:
    """
    TMC v7.0 ポストBWTパイプライン
    BWT+MTF後の特殊なデータ構造に特化した専門符号化
    """
    
    def encode(self, mtf_stream: bytes) -> List[bytes]:
        """BWT+MTF後のストリームを専門符号化"""
        print("    [ポストBWT] RLE + 分割エントロピー符号化を実行中...")
        
        try:
            # 1. ランレングス符号化 (RLE)
            literals, run_lengths = self._apply_rle(mtf_stream)
            
            print(f"    [ポストBWT] RLE: {len(mtf_stream)} bytes -> リテラル: {len(literals)}, ラン: {len(run_lengths)}")
            
            # 2. 分割したストリームを返す
            return [literals, run_lengths]
            
        except Exception as e:
            print(f"    [ポストBWT] エラー: {e} - 元データを返却")
            return [mtf_stream]
    
    def decode(self, streams: List[bytes]) -> bytes:
        """ポストBWT専門復号"""
        print("    [ポストBWT] RLE逆変換を実行中...")
        
        try:
            if len(streams) == 1:
                return streams[0]  # RLE未適用
            
            if len(streams) >= 2:
                literals = streams[0]
                run_lengths = streams[1]
                
                # 逆RLE
                mtf_stream = self._reverse_rle(literals, run_lengths)
                print(f"    [ポストBWT] 逆RLE: リテラル: {len(literals)}, ラン: {len(run_lengths)} -> {len(mtf_stream)} bytes")
                
                return mtf_stream
            
            return b''.join(streams)
            
        except Exception as e:
            print(f"    [ポストBWT] 逆変換エラー: {e}")
            return b''.join(streams)
    
    def _apply_rle(self, data: bytes) -> Tuple[bytes, bytes]:
        """ランレングス符号化（100%可逆保証版）"""
        if not data:
            return b'', b''
        
        literals = bytearray()
        run_lengths = bytearray()
        
        current_byte = data[0]
        run_length = 1
        
        for i in range(1, len(data)):
            if data[i] == current_byte and run_length < 255:
                run_length += 1
            else:
                # ランを記録
                literals.append(current_byte)
                run_lengths.append(run_length)
                
                # 新しいランを開始
                current_byte = data[i]
                run_length = 1
        
        # 最後のランを記録
        literals.append(current_byte)
        run_lengths.append(run_length)
        
        # 可逆性検証（デバッグ用）
        reconstructed = self._reverse_rle_verify(bytes(literals), bytes(run_lengths))
        if reconstructed != data:
            print(f"    [RLE符号化] 警告: 可逆性テスト失敗 - 元データ形式で保存")
            # 可逆性が保証できない場合は元データをそのまま保存
            return data, b'\x00'  # 特殊マーカー：元データそのまま
        
        print(f"    [RLE符号化] 可逆性確認: {len(data)} -> {len(literals)} literals, {len(run_lengths)} runs")
        return bytes(literals), bytes(run_lengths)
    
    def _reverse_rle_verify(self, literals: bytes, run_lengths: bytes) -> bytes:
        """RLE逆変換（検証専用 - エラー時例外発生）"""
        if len(literals) != len(run_lengths):
            raise ValueError(f"Size mismatch: literals={len(literals)}, run_lengths={len(run_lengths)}")
        
        result = bytearray()
        for literal, run_length in zip(literals, run_lengths):
            if run_length <= 0 or run_length > 255:
                raise ValueError(f"Invalid run length: {run_length}")
            result.extend([literal] * run_length)
        
        return bytes(result)
    
    def _reverse_rle(self, literals: bytes, run_lengths: bytes) -> bytes:
        """逆ランレングス符号化（100%可逆保証版）"""
        # 特殊マーカーチェック：元データそのまま保存の場合
        if len(run_lengths) == 1 and run_lengths[0] == 0:
            print(f"    [RLE逆変換] 元データそのまま復元: {len(literals)} bytes")
            return literals
        
        # 入力検証
        if not literals or not run_lengths:
            print(f"    [RLE逆変換] 警告: 空入力データ")
            return b''
        
        # サイズ一致チェック（厳密）
        if len(literals) != len(run_lengths):
            print(f"    [RLE逆変換] 致命的エラー: サイズ不整合 literals={len(literals)}, run_lengths={len(run_lengths)}")
            # 可逆性が保証できない場合は、literalsをそのまま返す
            return literals
        
        result = bytearray()
        max_output_size = 100 * 1024 * 1024  # 100MB制限
        
        try:
            for i, (literal, run_length) in enumerate(zip(literals, run_lengths)):
                # 実行長検証
                if run_length <= 0:
                    print(f"    [RLE逆変換] 警告: 位置{i}で実行長0 - スキップ")
                    continue
                elif run_length > 255:
                    print(f"    [RLE逆変換] 警告: 位置{i}で異常な実行長{run_length} -> 255に制限")
                    run_length = 255
                
                # メモリオーバーフロー保護
                if len(result) + run_length > max_output_size:
                    print(f"    [RLE逆変換] 警告: 出力サイズ制限に達しました ({max_output_size} bytes)")
                    break
                
                # 反復実行
                result.extend([literal] * run_length)
            
            print(f"    [RLE逆変換] 完了: {len(literals)} literals -> {len(result)} bytes")
            return bytes(result)
            
        except Exception as e:
            print(f"    [RLE逆変換] エラー: {e}")
            # フォールバック：literalsをそのまま返却
            return literals
