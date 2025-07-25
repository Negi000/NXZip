#!/usr/bin/env python3
"""
🔥 NEXUS INFECTION SHAPE TRANSFORMATION DEBUG
Debug the specific issue with group_id_stream being empty
"""

import json
from nexus_advanced_engine import NexusAdvancedCompressor

def debug_shape_transformation():
    print("🔥 NEXUS SHAPE TRANSFORMATION DEBUG")
    print("===================================")
    
    # Simple test data
    test_data = b"ABC"
    print(f"Original: {list(test_data)}")
    
    # Create compressor with debug mode
    compressor = NexusAdvancedCompressor()
    
    # Add debugging to shape transformation method
    original_apply_shape_transformation = compressor._apply_shape_transformation
    
    def debug_apply_shape_transformation(blocks, consolidation_map, normalized_groups):
        print(f"\n🔥 SHAPE TRANSFORMATION DEBUG:")
        print(f"Blocks count: {len(blocks)}")
        print(f"Consolidation map entries: {len(consolidation_map)}")
        print(f"Normalized groups entries: {len(normalized_groups)}")
        
        result = original_apply_shape_transformation(blocks, consolidation_map, normalized_groups)
        
        print(f"Result group_ids length: {len(result[0])}")
        print(f"Result perm_maps length: {len(result[1])}")
        print(f"Group IDs: {result[0]}")
        print(f"First few perm maps: {result[1][:3]}")
        
        return result
    
    # Monkey patch for debugging
    compressor._apply_shape_transformation = debug_apply_shape_transformation
    
    try:
        compressed = compressor.compress(test_data, silent=False)
        print(f"\nCompressed length: {len(compressed)} bytes")
    except Exception as e:
        print(f"❌ Compression error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_shape_transformation()
