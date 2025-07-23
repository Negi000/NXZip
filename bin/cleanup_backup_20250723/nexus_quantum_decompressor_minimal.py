#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚛️ NEXUS Quantum Decompressor MINIMAL FIX
元の高性能エンジン用の最小修正デコンプレッサー
"""

import os
import sys
import struct
import hashlib
import lzma
from pathlib import Path
from typing import Dict, Any, List

class MinimalQuantumDecompressor:
    """最小修正量子解凍エンジン"""
    
    def decompress_file(self, input_path: str, output_path: str = None) -> Dict[str, Any]:
        """量子解凍実行"""
        if not os.path.exists(input_path):
            return {'error': f'入力ファイルが見つかりません: {input_path}'}
        
        if output_path is None:
            input_file = Path(input_path)
            output_path = str(input_file.with_suffix('.restored' + input_file.suffix.replace('.nxz', '')))
        
        try:
            with open(input_path, 'rb') as f:
                compressed_data = f.read()
            
            # ヘッダー確認
            if compressed_data.startswith(b'NXQNT_PNG_V1'):
                header_size = 12
                format_type = 'PNG'
            elif compressed_data.startswith(b'NXQNT_JPEG_V1'):
                header_size = 13
                format_type = 'JPEG'
            else:
                return {'error': '不明な量子圧縮フォーマット'}
            
            # 最小メタデータ読み取り (ハッシュ16bytes)
            metadata_start = header_size
            original_hash = compressed_data[metadata_start:metadata_start + 16]
            
            # 量子ヘッダーをスキップ (量子位相4 + ペア数2 = 6bytes)
            quantum_start = metadata_start + 16
            compressed_data = compressed_data[quantum_start + 6:]
            
            # ハイブリッド処理フラグ確認
            compression_flag = compressed_data[0]
            payload = compressed_data[1:]
            
            if compression_flag == 0x01:
                # 量子圧縮データの場合（非可逆だが高圧縮率）
                print("   ⚡ 量子圧縮データを検出 - 高圧縮率モード（非可逆）")
                final_data = lzma.decompress(payload)
                # 注意：量子圧縮は可逆性なし
            else:
                # 元データ圧縮の場合（可逆）
                print("   🔒 元データ圧縮を検出 - 完全可逆モード")
                # 量子サイズ情報をスキップ（4バイト）
                quantum_size_data = payload[:4]
                quantum_size = struct.unpack('>I', quantum_size_data)[0]
                lzma_data = payload[4:]
                final_data = lzma.decompress(lzma_data)
            
            # ハッシュ検証（MD5で統一）
            restored_hash = hashlib.md5(final_data).digest()
            hash_match = restored_hash == original_hash
            
            with open(output_path, 'wb') as f:
                f.write(final_data)
            
            return {
                'input_file': input_path,
                'output_file': output_path,
                'restored_size': len(final_data),
                'format_type': format_type,
                'hash_match': hash_match,
                'success': True
            }
            
        except Exception as e:
            return {'error': f'解凍エラー: {str(e)}'}
    
    def _reverse_quantum_pixel_entanglement(self, data: bytes) -> List[bytes]:
        """量子ピクセルもつれの逆変換（可逆版）"""
        # 加算平均から4チャンネルのデータを復元
        # 元の処理: ((r + g + b + a) // 4) % 256 -> entangled_value
        # 逆変換: entangled_value * 4 を各チャンネルに分散
        channels = [[], [], [], []]  # R, G, B, A
        
        for byte in data:
            # 各バイトを4チャンネルに復元（近似）
            avg_value = byte
            channels[0].append(avg_value)
            channels[1].append(avg_value)
            channels[2].append(avg_value)
            channels[3].append(avg_value)
        
        return [bytes(channel) for channel in channels]
    
    def _reverse_quantum_channel_separation(self, channels: List[bytes]) -> bytes:
        """量子チャンネル分離の逆変換"""
        # 4チャンネルデータから元のインターリーブ形式に復元
        result = bytearray()
        min_len = min(len(ch) for ch in channels) if channels else 0
        
        for i in range(min_len):
            for channel in channels:
                if i < len(channel):
                    result.append(channel[i])
        
        return bytes(result)

def main():
    if len(sys.argv) < 2:
        print("使用法: python nexus_quantum_decompressor_minimal.py <圧縮ファイル> [出力ファイル]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    engine = MinimalQuantumDecompressor()
    result = engine.decompress_file(input_file, output_file)
    
    if 'error' in result:
        print(f"❌ エラー: {result['error']}")
        sys.exit(1)
    
    print("⚛ 量子解凍完了（最小修正版）")
    print(f"入力: {result['input_file']}")
    print(f"出力: {result['output_file']}")
    print(f"復元サイズ: {result['restored_size']:,} bytes")
    print(f"形式: {result['format_type']}")
    print(f"ハッシュ一致: {'はい' if result['hash_match'] else 'いいえ'}")

if __name__ == "__main__":
    main()
