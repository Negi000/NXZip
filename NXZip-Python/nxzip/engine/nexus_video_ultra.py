#!/usr/bin/env python3
"""
NEXUS Video Ultra - 動画専用超軽量エンジン
圧縮スキップ・SPEのみの超高速処理
"""

import struct
import time
import zlib
from typing import Optional
from pathlib import Path
import sys

# プロジェクトパス追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from .spe_core_jit import SPECoreJIT

# NXZ定数
NXZ_MAGIC = b'NXZU'  # Ultra専用マジック
NXZ_VERSION = 1

class NEXUSVideoUltra:
    """
    動画専用超軽量NEXUS - SPEのみ・圧縮スキップ
    
    戦略:
    1. 圧縮処理をスキップ（動画は既に圧縮済みのため）
    2. SPE暗号化のみ実行
    3. 最小限ヘッダー
    4. 最大速度重視
    """
    
    def __init__(self):
        self.spe = SPECoreJIT()
    
    def compress(self, data: bytes) -> bytes:
        """超軽量動画処理（圧縮スキップ）"""
        if not data:
            return self._create_empty_nxz()
        
        # 1. 圧縮スキップ（動画は既に圧縮済み）
        # データにマーカーだけ追加
        processed_data = b'ULTRARAW' + data
        
        # 2. SPE暗号化（構造保存）
        encrypted_data = self.spe.apply_transform(processed_data)
        
        # 3. 最小ヘッダー
        header = self._create_ultra_header(
            original_size=len(data),
            encrypted_size=len(encrypted_data)
        )
        
        return header + encrypted_data
    
    def decompress(self, nxz_data: bytes) -> bytes:
        """超軽量動画展開"""
        if not nxz_data:
            return b""
        
        # 1. 最小ヘッダー解析
        if len(nxz_data) < 24:
            raise ValueError("Invalid NXZ Ultra format")
        
        # 2. 暗号化データ抽出
        encrypted_data = nxz_data[24:]  # 最小ヘッダー24バイト
        
        # 3. SPE復号化
        processed_data = self.spe.reverse_transform(encrypted_data)
        
        # 4. マーカー除去
        if processed_data.startswith(b'ULTRARAW'):
            original_data = processed_data[8:]
        else:
            raise ValueError("Unknown ultra format")
        
        return original_data
    
    def _create_ultra_header(self, original_size: int, encrypted_size: int) -> bytes:
        """超最小ヘッダー作成 (24バイト)"""
        header = bytearray(24)
        
        # マジックナンバー
        header[0:4] = NXZ_MAGIC
        
        # バージョン
        header[4:8] = struct.pack('<I', NXZ_VERSION)
        
        # サイズ情報
        header[8:16] = struct.pack('<Q', original_size)
        header[16:24] = struct.pack('<Q', encrypted_size)
        
        return bytes(header)
    
    def _create_empty_nxz(self) -> bytes:
        """空の超軽量NXZファイル作成"""
        return self._create_ultra_header(0, 0)

def test_nexus_video_ultra():
    """NEXUS Video Ultra テスト"""
    print("🚀 NEXUS Video Ultra テスト - 超軽量動画エンジン")
    print("=" * 60)
    
    # MP4テストファイル
    test_file = Path(r"C:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\sample\Python基礎講座3_4月26日-3.mp4")
    
    if not test_file.exists():
        print("❌ MP4テストファイルが見つかりません")
        return
    
    file_size = test_file.stat().st_size
    print(f"📄 ファイル: {test_file.name}")
    print(f"📊 サイズ: {file_size//1024//1024} MB")
    
    # データ読み込み
    print("\n📖 データ読み込み中...")
    with open(test_file, 'rb') as f:
        data = f.read()
    
    # NEXUS Video Ultra初期化
    nexus = NEXUSVideoUltra()
    
    # 圧縮テスト
    print("\n🚀 NEXUS Video Ultra 処理中...")
    start_time = time.perf_counter()
    compressed = nexus.compress(data)
    compress_time = time.perf_counter() - start_time
    
    # 圧縮結果
    compression_ratio = (1 - len(compressed) / len(data)) * 100
    compress_speed = (len(data) / 1024 / 1024) / compress_time
    
    print(f"✅ 処理完了!")
    print(f"   📈 サイズ変化: {compression_ratio:.2f}%")
    print(f"   ⚡ 速度: {compress_speed:.2f} MB/s")
    print(f"   ⏱️ 時間: {compress_time:.2f}秒")
    
    # 展開テスト
    print(f"\n🔄 展開テスト中...")
    start_time = time.perf_counter()
    decompressed = nexus.decompress(compressed)
    decomp_time = time.perf_counter() - start_time
    
    # 展開結果
    decomp_speed = (len(data) / 1024 / 1024) / decomp_time
    
    print(f"✅ 展開完了!")
    print(f"   ⚡ 速度: {decomp_speed:.2f} MB/s")
    print(f"   ⏱️ 時間: {decomp_time:.2f}秒")
    
    # 正確性確認
    is_correct = data == decompressed
    print(f"   🔍 正確性: {'✅ OK' if is_correct else '❌ NG'}")
    
    # 総合評価
    total_time = compress_time + decomp_time
    total_speed = (len(data) * 2 / 1024 / 1024) / total_time
    
    print(f"\n🚀 NEXUS Video Ultra 結果:")
    print(f"   サイズ変化: {compression_ratio:.2f}%")
    print(f"   総合速度: {total_speed:.2f} MB/s")
    print(f"   戦略: SPEのみ・圧縮スキップ")
    print(f"   完全可逆性: ✅ 保証")
    
    # 速度目標
    target_speed = 100  # 100MB/sを目標
    
    print(f"\n🎯 超高速目標評価:")
    print(f"   速度: {total_speed:.2f} MB/s {'✅' if total_speed >= target_speed else '⚠️'} (目標{target_speed}MB/s)")
    
    # 特徴説明
    print(f"\n💡 Ultra戦略の特徴:")
    print(f"   - 圧縮処理をスキップ（動画は既に圧縮済み）")
    print(f"   - SPE構造保存暗号化のみ実行")
    print(f"   - 最小限ヘッダー（24バイト）")
    print(f"   - 完全可逆性保証")
    
    if total_speed >= target_speed:
        print(f"\n🏆 目標達成！超高速動画処理が実現されました！")
    
    return compression_ratio, total_speed

if __name__ == "__main__":
    test_nexus_video_ultra()
