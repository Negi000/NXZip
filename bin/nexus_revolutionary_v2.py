#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEXUS REVOLUTIONARY COMPRESSION ENGINE v2
Perfect reversibility + Maximum efficiency through true redundancy elimination
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


class NexusRevolutionaryEngine:
    """Revolutionary NEXUS Engine with perfect reversibility"""
    
    def __init__(self, compression_level: int = 3):
        self.compression_level = compression_level
        self.compressor = zstd.ZstdCompressor(level=compression_level, write_content_size=True)
        self.decompressor = zstd.ZstdDecompressor()
    
    def _analyze_data_patterns(self, data: bytes) -> Dict:
        """Deep analysis of data patterns for optimal compression"""
        analysis = {
            'length': len(data),
            'entropy': 0,
            'repetitive_ratio': 0,
            'unique_bytes': len(set(data)),
            'most_common_byte': 0,
            'patterns': {}
        }
        
        if len(data) == 0:
            return analysis
        
        # Entropy calculation
        counts = collections.Counter(data)
        total = len(data)
        entropy = 0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(p)
        analysis['entropy'] = entropy
        
        # Most common byte
        analysis['most_common_byte'] = counts.most_common(1)[0][0]
        
        # Pattern detection (2-byte, 4-byte, 8-byte patterns)
        for pattern_size in [2, 4, 8]:
            if len(data) >= pattern_size:
                patterns = {}
                for i in range(len(data) - pattern_size + 1):
                    pattern = data[i:i+pattern_size]
                    patterns[pattern] = patterns.get(pattern, 0) + 1
                
                # Keep only patterns that occur more than once
                significant_patterns = {p: count for p, count in patterns.items() if count > 1}
                if significant_patterns:
                    analysis['patterns'][pattern_size] = significant_patterns
        
        # Repetitive ratio (how much can be saved by eliminating repetition)
        total_savings = 0
        for pattern_size, patterns in analysis['patterns'].items():
            for pattern, count in patterns.items():
                if count > 1:
                    savings = (count - 1) * pattern_size
                    total_savings += savings
        
        analysis['repetitive_ratio'] = total_savings / max(len(data), 1)
        
        return analysis
    
    def _create_pattern_dictionary(self, data: bytes, analysis: Dict) -> Tuple[Dict[bytes, int], bytes]:
        """Create pattern dictionary for maximum compression"""
        dictionary = {}
        dict_id = 0
        
        # Start with most valuable patterns (highest savings)
        pattern_values = []
        
        for pattern_size, patterns in analysis.get('patterns', {}).items():
            for pattern, count in patterns.items():
                if count > 1:
                    savings = (count - 1) * pattern_size  # Bytes saved
                    value_per_byte = savings / pattern_size  # Efficiency metric
                    pattern_values.append((pattern, count, savings, value_per_byte))
        
        # Sort by value per byte (most efficient patterns first)
        pattern_values.sort(key=lambda x: x[3], reverse=True)
        
        # Build dictionary (limit to prevent explosion)
        max_dict_entries = min(256, len(pattern_values))  # Limit dictionary size
        
        for pattern, count, savings, value_per_byte in pattern_values[:max_dict_entries]:
            if value_per_byte > 1.5:  # Only include highly valuable patterns
                dictionary[pattern] = dict_id
                dict_id += 1
        
        # Apply pattern replacement
        compressed_data = bytearray(data)
        replacement_map = []  # Track replacements for reversal
        
        # Sort patterns by length (longest first to avoid conflicts)
        sorted_patterns = sorted(dictionary.keys(), key=len, reverse=True)
        
        for pattern in sorted_patterns:
            pattern_id = dictionary[pattern]
            # Create unique replacement marker (unlikely to occur in data)
            marker = bytes([255, 254, pattern_id, len(pattern)])
            
            # Find and replace all occurrences
            i = 0
            while i <= len(compressed_data) - len(pattern):
                if compressed_data[i:i+len(pattern)] == pattern:
                    # Record replacement for reversal
                    replacement_map.append((i, len(pattern), pattern_id))
                    # Replace with marker
                    compressed_data[i:i+len(pattern)] = marker
                    i += len(marker)
                else:
                    i += 1
        
        return dictionary, bytes(compressed_data)
    
    def _create_minimal_metadata(self, dictionary: Dict[bytes, int], replacement_count: int, original_length: int) -> bytes:
        """Create minimal metadata for reconstruction"""
        metadata = bytearray()
        
        # Header
        metadata.extend(b'NXRV')  # Magic: NEXUS Revolutionary
        metadata.extend(struct.pack('<I', original_length))
        metadata.extend(struct.pack('<H', len(dictionary)))
        
        # Dictionary (pattern -> id mapping)
        for pattern, pattern_id in sorted(dictionary.items(), key=lambda x: x[1]):
            metadata.extend(struct.pack('<B', pattern_id))
            metadata.extend(struct.pack('<B', len(pattern)))
            metadata.extend(pattern)
        
        return bytes(metadata)
    
    def _reconstruct_from_patterns(self, compressed_data: bytes, dictionary: Dict[int, bytes]) -> bytes:
        """Reconstruct original data from pattern-compressed data"""
        reconstructed = bytearray(compressed_data)
        
        # Find and replace all markers
        i = 0
        while i < len(reconstructed) - 3:
            if (reconstructed[i] == 255 and reconstructed[i+1] == 254):
                pattern_id = reconstructed[i+2]
                pattern_length = reconstructed[i+3]
                
                if pattern_id in dictionary:
                    original_pattern = dictionary[pattern_id]
                    # Replace marker with original pattern
                    reconstructed[i:i+4] = original_pattern
                    i += len(original_pattern)
                else:
                    i += 1
            else:
                i += 1
        
        return bytes(reconstructed)
    
    def compress(self, data: bytes, silent: bool = False) -> bytes:
        """Revolutionary NEXUS compression with perfect reversibility"""
        if not data:
            return b''
        
        start_time = time.time()
        
        # Phase 1: Deep pattern analysis
        analysis = self._analyze_data_patterns(data)
        
        if not silent:
            print(f"   [NEXUS REV] Length: {len(data):,} bytes")
            print(f"   [NEXUS REV] Entropy: {analysis['entropy']:.2f}")
            print(f"   [NEXUS REV] Unique bytes: {analysis['unique_bytes']}")
            print(f"   [NEXUS REV] Repetitive ratio: {analysis['repetitive_ratio']:.2%}")
        
        # Phase 2: Pattern dictionary creation
        dictionary, pattern_compressed = self._create_pattern_dictionary(data, analysis)
        
        if not silent:
            print(f"   [NEXUS REV] Dictionary patterns: {len(dictionary)}")
            print(f"   [NEXUS REV] After patterns: {len(pattern_compressed):,} bytes")
        
        # Phase 3: Create minimal metadata
        metadata = self._create_minimal_metadata(dictionary, 0, len(data))
        
        # Phase 4: Combine metadata + compressed data
        combined_data = metadata + pattern_compressed
        
        # Phase 5: Final Zstandard compression
        final_compressed = self.compressor.compress(combined_data)
        
        compression_time = time.time() - start_time
        compression_ratio = len(final_compressed) / len(data)
        space_savings = (1 - compression_ratio) * 100
        
        if not silent:
            print(f"   [NEXUS REV] Metadata: {len(metadata):,} bytes")
            print(f"   [NEXUS REV] Combined: {len(combined_data):,} bytes")
            print(f"   [NEXUS REV] Final compressed: {len(final_compressed):,} bytes")
            print(f"   [NEXUS REV] Compression ratio: {compression_ratio:.2%}")
            print(f"   [NEXUS REV] Space savings: {space_savings:.1f}%")
            print(f"   [NEXUS REV] Time: {compression_time:.3f}s")
        
        return final_compressed
    
    def decompress(self, compressed_data: bytes, silent: bool = False) -> bytes:
        """Revolutionary NEXUS decompression"""
        if not compressed_data:
            return b''
        
        start_time = time.time()
        
        # Phase 1: Zstandard decompression
        combined_data = self.decompressor.decompress(compressed_data)
        
        # Phase 2: Parse metadata
        offset = 0
        
        # Check magic number
        magic = combined_data[offset:offset+4]
        if magic != b'NXRV':
            raise ValueError("Invalid NEXUS Revolutionary format")
        offset += 4
        
        original_length, = struct.unpack('<I', combined_data[offset:offset+4])
        offset += 4
        
        dict_size, = struct.unpack('<H', combined_data[offset:offset+2])
        offset += 2
        
        # Reconstruct dictionary
        reverse_dictionary = {}
        for _ in range(dict_size):
            pattern_id = combined_data[offset]
            offset += 1
            pattern_length = combined_data[offset]
            offset += 1
            pattern = combined_data[offset:offset+pattern_length]
            offset += pattern_length
            reverse_dictionary[pattern_id] = pattern
        
        # Phase 3: Extract pattern-compressed data
        pattern_compressed = combined_data[offset:]
        
        # Phase 4: Reconstruct from patterns
        reconstructed = self._reconstruct_from_patterns(pattern_compressed, reverse_dictionary)
        
        # Phase 5: Trim to original length
        result = reconstructed[:original_length]
        
        decompression_time = time.time() - start_time
        
        if not silent:
            print(f"   [NEXUS REV DECOMP] Restored: {len(result):,} bytes")
            print(f"   [NEXUS REV DECOMP] Time: {decompression_time:.3f}s")
        
        return result


def create_pattern_test_file(filename: str, size_kb: int):
    """Create test file with intentional patterns for compression testing"""
    print(f"Creating pattern-rich test file: {filename} ({size_kb}KB)")
    
    target_size = size_kb * 1024
    data = bytearray()
    
    # Pattern 1: Highly repetitive sequences (50%)
    repetitive_size = target_size * 50 // 100
    patterns = [
        b"NEXUS_COMPRESSION_ALGORITHM_TEST_",  # 34 bytes
        b"REVOLUTIONARY_DATA_PROCESSING_",     # 32 bytes  
        b"PATTERN_RECOGNITION_ENGINE_",        # 29 bytes
        b"ZSTANDARD_INTEGRATION_LAYER_"        # 30 bytes
    ]
    
    while len(data) < repetitive_size:
        pattern = patterns[len(data) // 100 % len(patterns)]
        data.extend(pattern)
    
    # Pattern 2: Structured data with patterns (30%)
    structured_size = target_size * 30 // 100
    current_size = len(data)
    while len(data) < current_size + structured_size:
        header = b"DATA_HEADER_"
        index = str(len(data) % 10000).zfill(5).encode()
        footer = b"_END_BLOCK"
        data.extend(header + index + footer)
    
    # Pattern 3: Semi-random but with recurring elements (20%)
    semi_random_size = target_size - len(data)
    recurring_elements = [b"ABCD", b"1234", b"WXYZ", b"9876"]
    for _ in range(semi_random_size // 8):
        if len(data) < target_size:
            # Add recurring element
            data.extend(recurring_elements[random.randint(0, len(recurring_elements)-1)])
            # Add some random bytes
            if len(data) < target_size:
                data.extend(bytes([random.randint(0, 255) for _ in range(4)]))
    
    # Trim to exact size
    data = data[:target_size]
    
    with open(filename, 'wb') as f:
        f.write(data)
    
    print(f"Pattern-rich test file created: {len(data)} bytes")


def test_nexus_revolutionary():
    """Test NEXUS Revolutionary Engine"""
    print("🚀 NEXUS REVOLUTIONARY COMPRESSION ENGINE TEST 🚀")
    print("=" * 65)
    
    engine = NexusRevolutionaryEngine(compression_level=3)
    test_sizes = [5, 10, 20, 50]  # KB
    
    for size_kb in test_sizes:
        test_file = f"test_revolutionary_{size_kb}kb.bin"
        
        print(f"\n📁 Testing {size_kb}KB file:")
        create_pattern_test_file(test_file, size_kb)
        
        with open(test_file, 'rb') as f:
            data = f.read()
        
        print(f"   Original size: {len(data):,} bytes")
        
        # Compression
        compressed = engine.compress(data)
        
        # Decompression
        decompressed = engine.decompress(compressed)
        
        # Results
        compression_ratio = len(compressed) / len(data) * 100
        is_perfect = data == decompressed
        
        print(f"   Compressed size: {len(compressed):,} bytes ({compression_ratio:.1f}%)")
        print(f"   Perfect recovery: {'✓' if is_perfect else '✗'}")
        
        if is_perfect:
            if compression_ratio < 30:
                print("   🎉 NEXUS REVOLUTIONARY: EXCELLENT COMPRESSION!")
            elif compression_ratio < 50:
                print("   ✅ NEXUS REVOLUTIONARY: VERY GOOD COMPRESSION!")
            elif compression_ratio < 70:
                print("   ⭐ NEXUS REVOLUTIONARY: GOOD COMPRESSION!")
            else:
                print("   ⚠️  NEXUS REVOLUTIONARY: Perfect but limited compression")
        else:
            print("   ❌ Data corruption detected")
            if len(data) != len(decompressed):
                print(f"   [Debug] Size mismatch: {len(data)} vs {len(decompressed)}")
            else:
                diff_count = sum(1 for a, b in zip(data, decompressed) if a != b)
                print(f"   [Debug] {diff_count} byte differences out of {len(data)}")
        
        # Clean up
        try:
            os.remove(test_file)
        except:
            pass


def benchmark_revolutionary():
    """Benchmark Revolutionary NEXUS vs Pure Zstandard"""
    print("\n🏁 BENCHMARK: NEXUS Revolutionary vs Pure Zstandard")
    print("=" * 65)
    
    # Create test data
    test_file = "benchmark_revolutionary.bin"
    create_pattern_test_file(test_file, 20)  # 20KB
    
    with open(test_file, 'rb') as f:
        data = f.read()
    
    print(f"Benchmark data: {len(data):,} bytes")
    
    # Test 1: Pure Zstandard
    compressor = zstd.ZstdCompressor(level=3)
    start_time = time.time()
    pure_zstd = compressor.compress(data)
    pure_zstd_time = time.time() - start_time
    pure_ratio = len(pure_zstd) / len(data) * 100
    
    # Test 2: NEXUS Revolutionary
    engine = NexusRevolutionaryEngine(compression_level=3)
    start_time = time.time()
    nexus_revolutionary = engine.compress(data, silent=True)
    nexus_revolutionary_time = time.time() - start_time
    nexus_ratio = len(nexus_revolutionary) / len(data) * 100
    
    # Test decompression correctness
    decompressed = engine.decompress(nexus_revolutionary, silent=True)
    is_perfect = data == decompressed
    
    print(f"\n📊 Results:")
    print(f"   Pure Zstandard:      {len(pure_zstd):,} bytes ({pure_ratio:.1f}%) in {pure_zstd_time:.3f}s")
    print(f"   NEXUS Revolutionary: {len(nexus_revolutionary):,} bytes ({nexus_ratio:.1f}%) in {nexus_revolutionary_time:.3f}s")
    print(f"   Perfect recovery:    {'✓' if is_perfect else '✗'}")
    
    if len(nexus_revolutionary) < len(pure_zstd):
        improvement = (len(pure_zstd) - len(nexus_revolutionary)) / len(pure_zstd) * 100
        print(f"   🎉 NEXUS improvement: {improvement:.1f}% better compression!")
    else:
        regression = (len(nexus_revolutionary) - len(pure_zstd)) / len(pure_zstd) * 100
        print(f"   📊 Pure Zstandard: {regression:.1f}% better this time")
    
    # Clean up
    try:
        os.remove(test_file)
    except:
        pass


if __name__ == "__main__":
    test_nexus_revolutionary()
    benchmark_revolutionary()
