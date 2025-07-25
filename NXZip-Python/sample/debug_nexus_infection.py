#!/usr/bin/env python3
"""
🔥 NEXUS INFECTION DEBUG
Debug the compression/decompression issue
"""

import json
from nexus_advanced_engine import NexusAdvancedCompressor, NexusAdvancedDecompressor

def debug_nexus_infection():
    print("🔥 NEXUS INFECTION DEBUG")
    print("========================")
    
    # Test data
    test_data = b"NEXUS INFECTED!"
    print(f"Original: {list(test_data)}")
    
    # Compress
    compressor = NexusAdvancedCompressor()
    compressed = compressor.compress(test_data, silent=True)
    
    print(f"Compressed length: {len(compressed)} bytes")
    
    # Decompress and debug
    decompressor = NexusAdvancedDecompressor()
    
    # Extract payload for debugging
    import lzma
    decompressed_payload = lzma.decompress(compressed)
    
    try:
        payload = json.loads(decompressed_payload.decode('utf-8'))
        print(f"\n🔥 PAYLOAD DEBUG:")
        print(f"Header: {payload['header']}")
        print(f"Unique groups count: {len(payload['unique_groups'])}")
        print(f"Group IDs stream length: {len(payload['encoded_streams']['group_ids'])}")
        print(f"Perm IDs stream length: {len(payload['encoded_streams']['perm_ids'])}")
        print(f"Element consolidation map entries: {len(payload.get('element_consolidation_map', {}))}")
        
        # Try manual Huffman decode
        print(f"\n🔥 HUFFMAN DEBUG:")
        print(f"Group IDs encoded stream: {payload['encoded_streams']['group_ids']}")
        print(f"Perm IDs encoded stream: {payload['encoded_streams']['perm_ids']}")
        
    except Exception as e:
        print(f"❌ Payload debug error: {e}")
    
    # Full decompression
    result = decompressor.decompress(compressed)
    print(f"\n🔥 RESULT:")
    print(f"Decompressed: {list(result)}")
    print(f"Match: {list(test_data) == list(result)}")

if __name__ == "__main__":
    debug_nexus_infection()
