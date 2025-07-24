from nexus_compression_engine import NEXUSCompressor
import numpy as np

class DebugNEXUS(NEXUSCompressor):
    def nexus_decompress(self, compressed_data):
        print("🔄 NEXUS展開開始...")
        nexus_state = self._decode_nexus_state(compressed_data)
        print(f"Unique groups: {len(nexus_state.unique_groups)}")
        for i, group in enumerate(nexus_state.unique_groups):
            print(f"  Group {i}: elements={group.elements}, positions={group.positions}")
        
        restored_groups = self._restore_groups(nexus_state)
        print(f"Restored groups: {len(restored_groups)}")
        
        grid = self._restore_grid(restored_groups, nexus_state.grid_dimensions)
        print(f"Grid dimensions: {nexus_state.grid_dimensions}")
        print(f"Grid restored: {grid}")
        
        original_size = nexus_state.compression_metadata.get('original_size', 0)
        elements = self._flatten_grid(grid, nexus_state.grid_dimensions, original_size)
        print(f"Original size: {original_size}")
        print(f"Final elements: {elements}")
        return bytes(elements)

if __name__ == "__main__":
    nc = DebugNEXUS()
    data = np.array([1,2,3,4]*3, dtype=np.uint8)
    print("Original:", list(data))
    compressed = nc.compress(data.tobytes())
    decompressed = nc.decompress(compressed)
