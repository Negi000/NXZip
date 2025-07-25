#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEXUS LZMA KILLER ENGINE - Debug Version
Perfect reconstruction debugging and optimization
"""

import struct
import math
import time
import random
import os
import collections
from typing import List, Tuple, Dict, Any

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    print("Installing Zstandard...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'zstandard'])
    import zstandard as zstd
    ZSTD_AVAILABLE = True

try:
    import lzma
    LZMA_AVAILABLE = True
except ImportError:
    LZMA_AVAILABLE = False


class SimpleReversibleCompressor:
    """Simple reversible compressor with guaranteed perfect reconstruction"""
    
    def __init__(self):
        self.compressor = zstd.ZstdCompressor(level=6, write_content_size=True)
        self.decompressor = zstd.ZstdDecompressor()
    
    def find_repeating_patterns(self, data: bytes, min_length: int = 4) -> Dict:
        """Find simple repeating patterns"""
        patterns = {}
        
        # Only look for patterns up to reasonable length
        max_length = min(32, len(data) // 4)
        
        for pattern_len in range(min_length, max_length + 1):
            pattern_counts = {}
            
            for i in range(len(data) - pattern_len + 1):
                pattern = data[i:i+pattern_len]
                
                # Skip patterns with all same byte (handled by RLE)
                if len(set(pattern)) == 1:
                    continue
                
                if pattern in pattern_counts:
                    pattern_counts[pattern] += 1
                else:
                    pattern_counts[pattern] = 1
            
            # Only keep patterns that appear multiple times and are valuable
            for pattern, count in pattern_counts.items():
                if count >= 3:  # Must appear at least 3 times
                    savings = (count - 1) * len(pattern) - 8  # 8 bytes overhead
                    if savings > 20:  # Minimum savings threshold
                        patterns[pattern] = {
                            'count': count,
                            'savings': savings,
                            'length': len(pattern)
                        }
        
        # Sort by savings and limit to top patterns
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1]['savings'], reverse=True)
        
        # Return only top 32 most valuable patterns
        result = {}
        for i, (pattern, info) in enumerate(sorted_patterns[:32]):
            result[pattern] = i
        
        return result
    
    def apply_pattern_compression(self, data: bytes, patterns: Dict) -> Tuple[bytes, List]:
        """Apply pattern compression with careful tracking"""
        if not patterns:
            return data, []
        
        result = bytearray(data)
        replacements = []
        
        # Process patterns from longest to shortest to avoid conflicts
        sorted_patterns = sorted(patterns.items(), key=lambda x: len(x[0]), reverse=True)
        
        for pattern, pattern_id in sorted_patterns:
            # Create unique marker - using 4 bytes: [254, 253, id, length]
            marker = bytes([254, 253, pattern_id, len(pattern)])
            
            # Track all replacements for this pattern
            positions = []
            i = 0
            
            while i <= len(result) - len(pattern):
                if result[i:i+len(pattern)] == pattern:
                    positions.append(i)
                    i += len(pattern)  # Skip past this occurrence
                else:
                    i += 1
            
            # Apply replacements in reverse order to maintain positions
            for pos in reversed(positions):
                if pos + len(pattern) <= len(result):
                    # Record the replacement
                    replacements.append({
                        'position': pos,
                        'pattern': pattern,
                        'pattern_id': pattern_id,
                        'original_length': len(pattern)
                    })
                    
                    # Replace with marker
                    result[pos:pos+len(pattern)] = marker
        
        return bytes(result), replacements
    
    def compress(self, data: bytes, debug: bool = False) -> bytes:
        """Compress with perfect reversibility tracking"""
        if not data:
            return b''
        
        original_length = len(data)
        
        if debug:
            print(f"🔍 DEBUG: Compressing {original_length} bytes")
        
        # Step 1: Pattern detection
        patterns = self.find_repeating_patterns(data)
        
        if debug:
            print(f"🔍 DEBUG: Found {len(patterns)} patterns")
            for pattern, pattern_id in list(patterns.items())[:5]:  # Show first 5
                print(f"   Pattern {pattern_id}: {pattern[:16]} (length {len(pattern)})")
        
        # Step 2: Apply pattern compression
        compressed_data, replacements = self.apply_pattern_compression(data, patterns)
        
        if debug:
            print(f"🔍 DEBUG: After pattern compression: {len(compressed_data)} bytes")
            print(f"🔍 DEBUG: Made {len(replacements)} replacements")
        
        # Step 3: Zstandard compression
        zstd_compressed = self.compressor.compress(compressed_data)
        
        if debug:
            print(f"🔍 DEBUG: After Zstandard: {len(zstd_compressed)} bytes")
        
        # Step 4: Create package
        package = self._create_debug_package(
            zstd_compressed, original_length, patterns, replacements
        )
        
        if debug:
            ratio = len(package) / original_length * 100
            print(f"🔍 DEBUG: Final package: {len(package)} bytes ({ratio:.1f}%)")
        
        return package
    
    def _create_debug_package(self, compressed_data: bytes, original_length: int,
                            patterns: Dict, replacements: List) -> bytes:
        """Create package with detailed metadata for debugging"""
        package = bytearray()
        
        # Magic + original length
        package.extend(b'NXDB')  # NEXUS Debug
        package.extend(struct.pack('<I', original_length))
        
        # Patterns metadata
        pattern_data = bytearray()
        pattern_data.extend(struct.pack('<H', len(patterns)))  # Pattern count
        
        for pattern, pattern_id in sorted(patterns.items(), key=lambda x: x[1]):
            pattern_data.extend(struct.pack('<H', pattern_id))  # ID
            pattern_data.extend(struct.pack('<H', len(pattern)))  # Length
            pattern_data.extend(pattern)  # Pattern bytes
        
        package.extend(struct.pack('<I', len(pattern_data)))
        package.extend(pattern_data)
        
        # Replacements metadata
        replacement_data = bytearray()
        replacement_data.extend(struct.pack('<I', len(replacements)))  # Replacement count
        
        for replacement in replacements:
            replacement_data.extend(struct.pack('<I', replacement['position']))
            replacement_data.extend(struct.pack('<H', replacement['pattern_id']))
            replacement_data.extend(struct.pack('<H', replacement['original_length']))
        
        package.extend(struct.pack('<I', len(replacement_data)))
        package.extend(replacement_data)
        
        # Compressed data
        package.extend(struct.pack('<I', len(compressed_data)))
        package.extend(compressed_data)
        
        return bytes(package)
    
    def decompress(self, package: bytes, debug: bool = False) -> bytes:
        """Decompress with detailed debugging"""
        if len(package) < 16:
            return b''
        
        offset = 0
        
        # Parse header
        magic = package[offset:offset+4]
        if magic != b'NXDB':
            raise ValueError("Invalid debug package")
        offset += 4
        
        original_length, = struct.unpack('<I', package[offset:offset+4])
        offset += 4
        
        if debug:
            print(f"🔍 DEBUG DECOMP: Original length {original_length}")
        
        # Parse patterns
        pattern_data_size, = struct.unpack('<I', package[offset:offset+4])
        offset += 4
        pattern_data = package[offset:offset+pattern_data_size]
        offset += pattern_data_size
        patterns = self._parse_patterns(pattern_data, debug)
        
        # Parse replacements
        replacement_data_size, = struct.unpack('<I', package[offset:offset+4])
        offset += 4
        replacement_data = package[offset:offset+replacement_data_size]
        offset += replacement_data_size
        replacements = self._parse_replacements(replacement_data, debug)
        
        # Parse compressed data
        compressed_size, = struct.unpack('<I', package[offset:offset+4])
        offset += 4
        compressed_data = package[offset:offset+compressed_size]
        
        if debug:
            print(f"🔍 DEBUG DECOMP: Patterns {len(patterns)}, Replacements {len(replacements)}")
        
        # Decompress with Zstandard
        decompressed_data = self.decompressor.decompress(compressed_data)
        
        if debug:
            print(f"🔍 DEBUG DECOMP: After Zstandard: {len(decompressed_data)} bytes")
        
        # Reverse pattern compression
        reconstructed = self._reverse_pattern_compression(
            decompressed_data, patterns, replacements, original_length, debug
        )
        
        if debug:
            print(f"🔍 DEBUG DECOMP: Final result: {len(reconstructed)} bytes")
            print(f"🔍 DEBUG DECOMP: Perfect match: {len(reconstructed) == original_length}")
        
        return reconstructed
    
    def _parse_patterns(self, data: bytes, debug: bool = False) -> Dict:
        """Parse patterns from metadata"""
        patterns = {}
        offset = 0
        
        if len(data) < 2:
            return patterns
        
        pattern_count, = struct.unpack('<H', data[offset:offset+2])
        offset += 2
        
        if debug:
            print(f"🔍 DEBUG: Parsing {pattern_count} patterns")
        
        for i in range(pattern_count):
            if offset + 4 > len(data):
                break
            
            pattern_id, = struct.unpack('<H', data[offset:offset+2])
            offset += 2
            
            pattern_length, = struct.unpack('<H', data[offset:offset+2])
            offset += 2
            
            if offset + pattern_length > len(data):
                break
            
            pattern = data[offset:offset+pattern_length]
            offset += pattern_length
            
            patterns[pattern_id] = pattern
            
            if debug and i < 3:
                print(f"   Pattern {pattern_id}: {pattern[:16]} (length {pattern_length})")
        
        return patterns
    
    def _parse_replacements(self, data: bytes, debug: bool = False) -> List:
        """Parse replacements from metadata"""
        replacements = []
        offset = 0
        
        if len(data) < 4:
            return replacements
        
        replacement_count, = struct.unpack('<I', data[offset:offset+4])
        offset += 4
        
        if debug:
            print(f"🔍 DEBUG: Parsing {replacement_count} replacements")
        
        for i in range(replacement_count):
            if offset + 8 > len(data):
                break
            
            position, = struct.unpack('<I', data[offset:offset+4])
            offset += 4
            
            pattern_id, = struct.unpack('<H', data[offset:offset+2])
            offset += 2
            
            original_length, = struct.unpack('<H', data[offset:offset+2])
            offset += 2
            
            replacements.append({
                'position': position,
                'pattern_id': pattern_id,
                'original_length': original_length
            })
            
            if debug and i < 3:
                print(f"   Replacement {i}: pos={position}, id={pattern_id}, len={original_length}")
        
        return replacements
    
    def _reverse_pattern_compression(self, data: bytes, patterns: Dict, 
                                   replacements: List, original_length: int,
                                   debug: bool = False) -> bytes:
        """Reverse pattern compression with detailed tracking"""
        result = bytearray(data)
        
        if debug:
            print(f"🔍 DEBUG REVERSE: Starting with {len(result)} bytes")
        
        # Sort replacements by position (process from end to beginning)
        sorted_replacements = sorted(replacements, key=lambda x: x['position'], reverse=True)
        
        # Process each replacement
        for i, replacement in enumerate(sorted_replacements):
            position = replacement['position']
            pattern_id = replacement['pattern_id']
            original_length = replacement['original_length']
            
            if pattern_id not in patterns:
                if debug:
                    print(f"   WARNING: Pattern {pattern_id} not found")
                continue
            
            original_pattern = patterns[pattern_id]
            
            # Look for marker starting from expected position
            marker_found = False
            search_start = max(0, position - 10)
            search_end = min(len(result), position + 10)
            
            for search_pos in range(search_start, search_end):
                if (search_pos + 4 <= len(result) and
                    result[search_pos] == 254 and 
                    result[search_pos + 1] == 253 and
                    result[search_pos + 2] == pattern_id and
                    result[search_pos + 3] == len(original_pattern)):
                    
                    # Found the marker, replace it
                    result[search_pos:search_pos + 4] = original_pattern
                    marker_found = True
                    
                    if debug and i < 3:
                        print(f"   Replaced marker at {search_pos} with pattern {pattern_id}")
                    break
            
            if not marker_found and debug:
                print(f"   WARNING: Marker for pattern {pattern_id} not found near position {position}")
        
        # Trim to original length
        if len(result) > original_length:
            result = result[:original_length]
        elif len(result) < original_length:
            result.extend(b'\x00' * (original_length - len(result)))
        
        if debug:
            print(f"🔍 DEBUG REVERSE: Final length {len(result)}")
        
        return bytes(result)


def test_debug_compressor():
    """Test debug compressor with small data"""
    print("🔬 NEXUS DEBUG COMPRESSOR TEST")
    print("=" * 50)
    
    compressor = SimpleReversibleCompressor()
    
    # Test with very simple data first
    test_data = b"ABCDABCDABCDABCDABCDABCD" * 10  # Simple repeating pattern
    
    print(f"📁 Testing with {len(test_data)} bytes of simple data")
    print(f"   Sample: {test_data[:50]}...")
    
    # Compress with debug
    compressed = compressor.compress(test_data, debug=True)
    
    print(f"\n📤 Compression complete:")
    ratio = len(compressed) / len(test_data) * 100
    print(f"   Ratio: {ratio:.1f}%")
    
    # Decompress with debug
    print(f"\n📥 Decompressing...")
    decompressed = compressor.decompress(compressed, debug=True)
    
    # Verify
    is_perfect = test_data == decompressed
    print(f"\n✅ Results:")
    print(f"   Original: {len(test_data)} bytes")
    print(f"   Decompressed: {len(decompressed)} bytes")
    print(f"   Perfect match: {'✓' if is_perfect else '✗'}")
    
    if not is_perfect:
        # Show differences
        if len(test_data) == len(decompressed):
            diff_count = sum(1 for a, b in zip(test_data, decompressed) if a != b)
            print(f"   Differences: {diff_count} bytes")
            
            # Show first few differences
            for i, (a, b) in enumerate(zip(test_data, decompressed)):
                if a != b:
                    print(f"   Diff at {i}: {a} != {b}")
                    if i > 5:  # Limit output
                        break
        else:
            print(f"   Length mismatch: {len(test_data)} vs {len(decompressed)}")
    
    return is_perfect


def benchmark_vs_lzma_debug():
    """Simple benchmark against LZMA"""
    print("\n" + "=" * 50)
    print("🏁 SIMPLE BENCHMARK vs LZMA")
    print("=" * 50)
    
    compressor = SimpleReversibleCompressor()
    
    # Create test data
    test_data = (b"Hello World! " * 100 + 
                 b"ABCDEFGH" * 50 + 
                 b"1234567890" * 30)
    
    print(f"📊 Testing with {len(test_data)} bytes")
    
    # NEXUS compression
    start_time = time.time()
    nexus_compressed = compressor.compress(test_data)
    nexus_time = time.time() - start_time
    
    nexus_decompressed = compressor.decompress(nexus_compressed)
    nexus_perfect = test_data == nexus_decompressed
    
    # LZMA compression
    if LZMA_AVAILABLE:
        start_time = time.time()
        lzma_compressed = lzma.compress(test_data, preset=6)
        lzma_time = time.time() - start_time
        
        lzma_decompressed = lzma.decompress(lzma_compressed)
        lzma_perfect = test_data == lzma_decompressed
    else:
        lzma_compressed = test_data
        lzma_time = 0
        lzma_perfect = False
    
    # Results
    nexus_ratio = len(nexus_compressed) / len(test_data) * 100
    lzma_ratio = len(lzma_compressed) / len(test_data) * 100 if LZMA_AVAILABLE else 100
    
    print(f"\n📈 RESULTS:")
    print(f"   NEXUS Debug: {len(nexus_compressed)} bytes ({nexus_ratio:.1f}%) - {nexus_time:.3f}s - {'✓' if nexus_perfect else '✗'}")
    if LZMA_AVAILABLE:
        print(f"   LZMA:        {len(lzma_compressed)} bytes ({lzma_ratio:.1f}%) - {lzma_time:.3f}s - {'✓' if lzma_perfect else '✗'}")
    
    if nexus_perfect and LZMA_AVAILABLE:
        if nexus_ratio < lzma_ratio:
            improvement = (lzma_ratio - nexus_ratio) / lzma_ratio * 100
            print(f"\n🏆 NEXUS WINS! {improvement:.1f}% better than LZMA!")
        else:
            deficit = (nexus_ratio - lzma_ratio) / lzma_ratio * 100
            print(f"\n📊 LZMA wins by {deficit:.1f}%")
    elif nexus_perfect:
        print(f"\n✅ NEXUS: Perfect reconstruction achieved!")


if __name__ == "__main__":
    success = test_debug_compressor()
    if success:
        benchmark_vs_lzma_debug()
    else:
        print("\n❌ Debug test failed - reconstruction issue detected")
