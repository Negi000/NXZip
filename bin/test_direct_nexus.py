from nexus_compression_engine import NEXUSCompressor
import numpy as np

# メモリ上での直接テスト（エンコード/デコードなし）
class DirectNEXUS(NEXUSCompressor):
    def __init__(self):
        super().__init__()
        self._state = None
    
    def compress_direct(self, data: bytes) -> 'DirectNEXUS':
        """メモリ上での直接圧縮"""
        _, self._state = self.nexus_compress(data)
        return self
    
    def decompress_direct(self) -> bytes:
        """メモリ上での直接展開"""
        if not self._state:
            return b''
        
        print("🔄 直接NEXUS展開開始...")
        
        # 元のグループ情報を使用（エンコード/デコードなし）
        restored_groups = self._state.original_groups
        print(f"Restored groups: {len(restored_groups)}")
        
        # グリッド復元
        grid = self._restore_grid(restored_groups, self._state.grid_dimensions)
        print(f"Grid restored: {grid}")
        
        # 要素統合（元のサイズ情報を使用）
        original_size = self._state.compression_metadata.get('original_size', 0)
        elements = self._flatten_grid(grid, self._state.grid_dimensions, original_size)
        print(f"Original size: {original_size}")
        print(f"Final elements: {elements}")
        
        return bytes(elements)

if __name__ == "__main__":
    nc = DirectNEXUS()
    data = np.array([1,2,3,4]*3, dtype=np.uint8)
    print("Original:", list(data))
    
    # 直接メモリ圧縮・展開
    nc.compress_direct(data.tobytes())
    decompressed = nc.decompress_direct()
    decompressed_array = np.frombuffer(decompressed, dtype=np.uint8)
    
    print("Decompressed:", list(decompressed_array))
    print("Success:", np.array_equal(data, decompressed_array))
