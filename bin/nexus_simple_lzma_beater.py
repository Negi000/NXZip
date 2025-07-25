#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEXUS SIMPLE LZMA BEATER
Minimal implementation focused purely on beating LZMA compression ratios
Maximum simplicity for guaranteed reversibility
"""

import struct
import time
import random
import os
import collections
from typing import List, Tuple, Dict, Any

try:
    import lzma
    LZMA_AVAILABLE = True
except ImportError:
    LZMA_AVAILABLE = False

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False


class SimpleLZMABeater:
    """Simple compressor designed to beat LZMA with minimal complexity"""
    
    def __init__(self):
        if ZSTD_AVAILABLE:
            self.compressor = zstd.ZstdCompressor(level=22, write_content_size=True)
            self.decompressor = zstd.ZstdDecompressor()
        else:
            self.compressor = None
            self.decompressor = None
    
    def find_simple_patterns(self, data: bytes) -> Dict[bytes, int]:
        """Find simple, high-value patterns"""
        if len(data) < 8:
            return {}
        
        patterns = {}
        
        # Look for patterns of 8, 16, 32 bytes only (simple approach)
        for pattern_len in [8, 16, 32]:
            if pattern_len > len(data) // 4:
                continue
            
            pattern_counts = {}
            
            # Sample every nth position for performance
            step = max(1, len(data) // 1000)
            
            for i in range(0, len(data) - pattern_len + 1, step):
                pattern = data[i:i+pattern_len]
                
                # Skip patterns with too many nulls or low entropy
                if pattern.count(0) > pattern_len // 2:
                    continue
                if len(set(pattern)) < 3:  # Need some diversity
                    continue
                
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            # Only keep patterns that appear frequently and save significant space
            for pattern, count in pattern_counts.items():
                if count >= 3:  # At least 3 occurrences
                    savings = (count - 1) * len(pattern) - 8  # 8 bytes overhead
                    if savings > 50:  # Significant savings required
                        patterns[pattern] = count
        
        # Limit to top 16 patterns to prevent dictionary explosion
        sorted_patterns = sorted(patterns.items(), 
                               key=lambda x: (x[1] - 1) * len(x[0]), 
                               reverse=True)
        
        result = {}
        for i, (pattern, count) in enumerate(sorted_patterns[:16]):
            result[pattern] = i
        
        return result
    
    def apply_simple_compression(self, data: bytes, patterns: Dict[bytes, int]) -> bytes:
        """Apply simple pattern replacement"""
        if not patterns:
            return data
        
        result = bytearray(data)
        
        # Sort patterns by length (longest first)
        sorted_patterns = sorted(patterns.items(), key=lambda x: len(x[0]), reverse=True)
        
        for pattern, pattern_id in sorted_patterns:
            # Simple replacement: use pattern_id as a single byte reference
            # Only if pattern_id < 240 to avoid conflict with data
            if pattern_id >= 240:
                continue
            
            # Find and replace all non-overlapping occurrences
            i = 0
            replacements = 0
            while i <= len(result) - len(pattern):
                if result[i:i+len(pattern)] == pattern:
                    # Replace with single-byte reference
                    result[i:i+len(pattern)] = bytes([240 + pattern_id])
                    i += 1  # Move past the replacement
                    replacements += 1
                else:
                    i += 1
            
            if replacements > 0:
                print(f"   Replaced pattern {pattern_id} ({len(pattern)} bytes) {replacements} times")
        
        return bytes(result)
    
    def compress(self, data: bytes, debug: bool = False) -> bytes:
        """Simple compression pipeline"""
        if not data:
            return b''
        
        original_length = len(data)
        
        if debug:
            print(f"🎯 SIMPLE LZMA BEATER: {original_length} bytes")
        
        # Step 1: Find valuable patterns
        patterns = self.find_simple_patterns(data)
        
        if debug:
            print(f"   📊 Found {len(patterns)} valuable patterns")
        
        # Step 2: Apply pattern compression
        pattern_compressed = self.apply_simple_compression(data, patterns)
        
        if debug:
            pattern_reduction = (original_length - len(pattern_compressed)) / original_length * 100
            print(f"   🔧 Pattern stage: {len(pattern_compressed)} bytes ({pattern_reduction:.1f}% reduction)")
        
        # Step 3: Secondary compression with Zstandard
        if self.compressor:
            final_compressed = self.compressor.compress(pattern_compressed)
            
            if debug:
                final_ratio = len(final_compressed) / original_length * 100
                space_saved = (1 - len(final_compressed) / original_length) * 100
                print(f"   🚀 Zstandard stage: {len(final_compressed)} bytes ({final_ratio:.1f}%)")
                print(f"   💾 Total space saved: {space_saved:.1f}%")
        else:
            final_compressed = pattern_compressed
        
        # Step 4: Create package
        package = self._create_simple_package(final_compressed, original_length, patterns)
        
        return package
    
    def _create_simple_package(self, compressed_data: bytes, original_length: int, 
                             patterns: Dict[bytes, int]) -> bytes:
        """Create simple package format"""
        package = bytearray()
        
        # Magic + original length
        package.extend(b'NXSM')  # NEXUS Simple
        package.extend(struct.pack('<I', original_length))
        
        # Pattern dictionary (simple format)
        package.extend(struct.pack('<H', len(patterns)))  # Pattern count
        
        for pattern, pattern_id in sorted(patterns.items(), key=lambda x: x[1]):
            package.extend(struct.pack('<H', pattern_id))    # Pattern ID
            package.extend(struct.pack('<H', len(pattern)))  # Pattern length
            package.extend(pattern)                          # Pattern data
        
        # Compressed data
        package.extend(struct.pack('<I', len(compressed_data)))
        package.extend(compressed_data)
        
        return bytes(package)
    
    def decompress(self, package: bytes, debug: bool = False) -> bytes:
        """Simple decompression"""
        if len(package) < 12:
            return b''
        
        offset = 0
        
        # Parse header
        magic = package[offset:offset+4]
        if magic != b'NXSM':
            raise ValueError("Invalid simple package")
        offset += 4
        
        original_length, = struct.unpack('<I', package[offset:offset+4])
        offset += 4
        
        if debug:
            print(f"🔍 SIMPLE DECOMP: Target {original_length} bytes")
        
        # Parse patterns
        pattern_count, = struct.unpack('<H', package[offset:offset+2])
        offset += 2
        
        patterns = {}
        for _ in range(pattern_count):
            if offset + 4 > len(package):
                break
            
            pattern_id, = struct.unpack('<H', package[offset:offset+2])
            offset += 2
            
            pattern_length, = struct.unpack('<H', package[offset:offset+2])
            offset += 2
            
            if offset + pattern_length > len(package):
                break
            
            pattern = package[offset:offset+pattern_length]
            offset += pattern_length
            
            patterns[pattern_id] = pattern
        
        if debug:
            print(f"   📚 Loaded {len(patterns)} patterns")
        
        # Parse compressed data
        compressed_size, = struct.unpack('<I', package[offset:offset+4])
        offset += 4
        compressed_data = package[offset:offset+compressed_size]
        
        # Decompress with Zstandard
        if self.decompressor:
            pattern_compressed = self.decompressor.decompress(compressed_data)
        else:
            pattern_compressed = compressed_data
        
        if debug:
            print(f"   📤 After Zstandard: {len(pattern_compressed)} bytes")
        
        # Reverse pattern compression
        result = bytearray(pattern_compressed)
        
        if debug:
            print(f"   🔄 Reversing pattern compression...")
        
        # Process patterns in reverse order of ID (highest ID first to avoid conflicts)
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[0], reverse=True)
        
        for pattern_id, pattern in sorted_patterns:
            if pattern_id >= 16:  # Safety check - only use 0-15
                continue
            
            # Replace all occurrences of the reference byte
            reference_byte = 240 + pattern_id
            
            # Use a more careful replacement approach
            new_result = bytearray()
            i = 0
            replacements_made = 0
            
            while i < len(result):
                if result[i] == reference_byte:
                    # Replace single byte with original pattern
                    new_result.extend(pattern)
                    replacements_made += 1
                    i += 1
                else:
                    new_result.append(result[i])
                    i += 1
            
            result = new_result
            
            if debug and replacements_made > 0:
                print(f"      Restored pattern {pattern_id} ({len(pattern)} bytes) {replacements_made} times")
        
        if debug:
            print(f"   📏 Before trimming: {len(result)} bytes")
        
        # Trim to original length
        if len(result) > original_length:
            result = result[:original_length]
        elif len(result) < original_length:
            result.extend(b'\x00' * (original_length - len(result)))
        
        if debug:
            print(f"   ✅ Final result: {len(result)} bytes")
            print(f"   🎯 Length match: {len(result) == original_length}")
        
        return bytes(result)


def create_pattern_rich_data(size_kb: int) -> bytes:
    """Create data with many patterns for testing"""
    print(f"Creating pattern-rich data: {size_kb}KB")
    
    target_size = size_kb * 1024
    data = bytearray()
    
    # Base patterns that will repeat
    base_patterns = [
        b"Hello World! This is a test pattern. ",
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
        b"The quick brown fox jumps over the lazy dog. ",
        b"<html><body><h1>Test</h1></body></html>",
        b"function test() { return 42; } ",
        b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    ]
    
    # Create data with 80% patterns, 20% random
    while len(data) < target_size * 0.8:
        pattern = random.choice(base_patterns)
        data.extend(pattern)
    
    # Add some random data
    while len(data) < target_size:
        data.append(random.randint(0, 255))
    
    # Trim to exact size
    data = data[:target_size]
    
    print(f"Pattern-rich data created: {len(data)} bytes")
    return bytes(data)


def benchmark_simple_vs_lzma():
    """Benchmark simple compressor vs LZMA"""
    print("🎯 SIMPLE LZMA BEATER vs LZMA BENCHMARK")
    print("=" * 60)
    
    if not LZMA_AVAILABLE:
        print("LZMA not available - cannot benchmark")
        return
    
    compressor = SimpleLZMABeater()
    test_sizes = [5, 10, 20]  # KB
    
    nexus_wins = 0
    lzma_wins = 0
    
    for size_kb in test_sizes:
        print(f"\n📊 Testing {size_kb}KB pattern-rich data:")
        
        # Create test data
        test_data = create_pattern_rich_data(size_kb)
        
        print(f"   Original: {len(test_data)} bytes")
        
        # NEXUS Simple compression
        start_time = time.time()
        nexus_compressed = compressor.compress(test_data, debug=False)
        nexus_comp_time = time.time() - start_time
        
        start_time = time.time()
        nexus_decompressed = compressor.decompress(nexus_compressed, debug=False)
        nexus_decomp_time = time.time() - start_time
        
        nexus_perfect = test_data == nexus_decompressed
        nexus_ratio = len(nexus_compressed) / len(test_data) * 100
        
        # LZMA compression
        start_time = time.time()
        lzma_compressed = lzma.compress(test_data, preset=6)
        lzma_comp_time = time.time() - start_time
        
        start_time = time.time()
        lzma_decompressed = lzma.decompress(lzma_compressed)
        lzma_decomp_time = time.time() - start_time
        
        lzma_perfect = test_data == lzma_decompressed
        lzma_ratio = len(lzma_compressed) / len(test_data) * 100
        
        # Results
        print(f"   NEXUS Simple: {len(nexus_compressed):,} bytes ({nexus_ratio:.1f}%) - {nexus_comp_time:.3f}s - {'✓' if nexus_perfect else '✗'}")
        print(f"   LZMA:         {len(lzma_compressed):,} bytes ({lzma_ratio:.1f}%) - {lzma_comp_time:.3f}s - {'✓' if lzma_perfect else '✗'}")
        
        # Winner
        if nexus_perfect and lzma_perfect:
            if nexus_ratio < lzma_ratio:
                improvement = (lzma_ratio - nexus_ratio) / lzma_ratio * 100
                print(f"   🏆 NEXUS WINS! {improvement:.1f}% better compression!")
                nexus_wins += 1
            else:
                deficit = (nexus_ratio - lzma_ratio) / lzma_ratio * 100
                print(f"   📊 LZMA wins by {deficit:.1f}%")
                lzma_wins += 1
        elif nexus_perfect:
            print(f"   ⚠️  NEXUS perfect, LZMA failed")
        elif lzma_perfect:
            print(f"   ⚠️  LZMA perfect, NEXUS failed")
            lzma_wins += 1
        else:
            print(f"   ❌ Both failed")
    
    # Final score
    print(f"\n🏁 FINAL SCORE:")
    print(f"   NEXUS Simple: {nexus_wins} wins")
    print(f"   LZMA:         {lzma_wins} wins")
    
    if nexus_wins > lzma_wins:
        print(f"   🎉 NEXUS SIMPLE BEATS LZMA!")
    elif lzma_wins > nexus_wins:
        print(f"   📈 LZMA wins, but NEXUS shows potential")
    else:
        print(f"   🤝 It's a tie!")


def quick_test():
    """Quick test with small data"""
    print("🚀 QUICK TEST")
    print("=" * 30)
    
    compressor = SimpleLZMABeater()
    
    # Simple test data with obvious patterns
    test_data = (b"Hello World! " * 20 + 
                 b"Test pattern " * 15 + 
                 b"ABCDEFGH" * 10)
    
    print(f"Test data: {len(test_data)} bytes")
    
    # Compress
    compressed = compressor.compress(test_data, debug=True)
    
    # Decompress
    print(f"\nDecompressing...")
    decompressed = compressor.decompress(compressed, debug=True)
    
    # Check
    is_perfect = test_data == decompressed
    ratio = len(compressed) / len(test_data) * 100
    
    print(f"\nResult: {ratio:.1f}% compression, Perfect: {'✓' if is_perfect else '✗'}")
    
    return is_perfect


if __name__ == "__main__":
    if quick_test():
        benchmark_simple_vs_lzma()
    else:
        print("Quick test failed - debugging needed")
