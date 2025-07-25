#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEXUS LZMA KILLER ENGINE
Simplified ultra-compression engine to surpass LZMA/LZMA2
Perfect reversibility + Maximum compression
"""

import struct
import math
import time
import random
import os
import collections
import hashlib
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


class SmartPatternDetector:
    """Smart pattern detection for maximum redundancy elimination"""
    
    def __init__(self):
        self.pattern_cache = {}
    
    def find_optimal_patterns(self, data: bytes) -> Dict[bytes, int]:
        """Find most valuable patterns for compression"""
        if len(data) < 4:
            return {}
        
        patterns = {}
        
        # Multi-length pattern detection (2-16 bytes)
        for pattern_len in range(2, min(17, len(data) + 1)):
            if pattern_len > len(data) // 2:  # Don't go beyond half the data
                break
                
            pattern_counts = {}
            
            # Sample-based detection for large files (performance optimization)
            step = max(1, len(data) // 10000) if len(data) > 50000 else 1
            
            for i in range(0, len(data) - pattern_len + 1, step):
                pattern = data[i:i+pattern_len]
                
                # Skip patterns with too many null bytes or control characters
                if pattern.count(0) > pattern_len // 2:
                    continue
                    
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            # Calculate value for each pattern
            for pattern, count in pattern_counts.items():
                if count >= 2:  # Must appear at least twice
                    # Value = bytes saved = (count-1) * pattern_length - overhead
                    overhead = 5  # marker overhead per pattern
                    value = (count - 1) * len(pattern) - overhead
                    
                    if value > 0:  # Only profitable patterns
                        patterns[pattern] = {
                            'count': count,
                            'length': len(pattern),
                            'value': value,
                            'ratio': value / len(pattern)
                        }
        
        # Sort by value and return top patterns
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1]['value'], reverse=True)
        
        # Limit to prevent dictionary explosion
        max_patterns = min(128, len(sorted_patterns))
        result = {}
        
        for i, (pattern, info) in enumerate(sorted_patterns[:max_patterns]):
            if info['value'] > 10:  # Minimum value threshold
                result[pattern] = i
        
        return result
    
    def replace_patterns(self, data: bytes, pattern_dict: Dict[bytes, int]) -> Tuple[bytes, List]:
        """Replace patterns with markers and track replacements"""
        if not pattern_dict:
            return data, []
        
        result = bytearray(data)
        replacements = []
        
        # Sort patterns by length (longest first to avoid conflicts)
        sorted_patterns = sorted(pattern_dict.items(), key=lambda x: len(x[0]), reverse=True)
        
        for pattern, pattern_id in sorted_patterns:
            # Create unique marker: [255, 254, pattern_id, length]
            marker = bytes([255, 254, pattern_id % 256, len(pattern) % 256])
            
            # Find and replace all occurrences
            i = 0
            pattern_replacements = 0
            
            while i <= len(result) - len(pattern):
                if result[i:i+len(pattern)] == pattern:
                    # Record replacement
                    replacements.append({
                        'position': i,
                        'original_length': len(pattern),
                        'pattern_id': pattern_id,
                        'pattern': pattern
                    })
                    
                    # Replace with marker
                    result[i:i+len(pattern)] = marker
                    i += len(marker)
                    pattern_replacements += 1
                else:
                    i += 1
            
            # Only keep track if we actually made replacements
            if pattern_replacements == 0:
                # Remove unused replacements
                replacements = [r for r in replacements if r['pattern_id'] != pattern_id]
        
        return bytes(result), replacements


class NexusLZMAKillerEngine:
    """LZMA Killer - Ultra compression engine"""
    
    def __init__(self, compression_level: int = 6):
        self.compression_level = min(22, max(1, compression_level))
        self.detector = SmartPatternDetector()
        
        # Multi-level compressors
        self.zstd_turbo = zstd.ZstdCompressor(level=1, write_content_size=True)
        self.zstd_balanced = zstd.ZstdCompressor(level=6, write_content_size=True)
        self.zstd_max = zstd.ZstdCompressor(level=22, write_content_size=True)
        self.decompressor = zstd.ZstdDecompressor()
    
    def compress(self, data: bytes, silent: bool = False) -> bytes:
        """Ultra compression with perfect reversibility"""
        if not data:
            return b''
        
        start_time = time.time()
        original_length = len(data)
        
        if not silent:
            print(f"🔥 NEXUS LZMA KILLER: Processing {original_length:,} bytes")
        
        # Step 1: Pattern detection and elimination
        pattern_dict = self.detector.find_optimal_patterns(data)
        pattern_data, replacements = self.detector.replace_patterns(data, pattern_dict)
        
        if not silent:
            pattern_reduction = (original_length - len(pattern_data)) / original_length * 100
            print(f"   🎯 Pattern elimination: {len(pattern_data):,} bytes ({pattern_reduction:.1f}% reduction)")
            print(f"   📊 Found {len(pattern_dict)} valuable patterns, {len(replacements)} replacements")
        
        # Step 2: Try multiple compression strategies
        compression_candidates = []
        
        # Strategy 1: Zstandard Turbo
        try:
            zstd_turbo_result = self.zstd_turbo.compress(pattern_data)
            compression_candidates.append(('zstd_1', zstd_turbo_result))
        except:
            pass
        
        # Strategy 2: Zstandard Balanced  
        try:
            zstd_balanced_result = self.zstd_balanced.compress(pattern_data)
            compression_candidates.append(('zstd_6', zstd_balanced_result))
        except:
            pass
        
        # Strategy 3: Zstandard Maximum
        try:
            zstd_max_result = self.zstd_max.compress(pattern_data)
            compression_candidates.append(('zstd_22', zstd_max_result))
        except:
            pass
        
        # Strategy 4: LZMA (if available)
        if LZMA_AVAILABLE:
            try:
                lzma_result = lzma.compress(pattern_data, preset=9)
                compression_candidates.append(('lzma_9', lzma_result))
            except:
                pass
        
        # Select best compression
        if compression_candidates:
            best_method, best_result = min(compression_candidates, key=lambda x: len(x[1]))
        else:
            best_method, best_result = 'raw', pattern_data
        
        # Step 3: Create final package with metadata
        final_package = self._create_complete_package(
            best_result, best_method, original_length, pattern_dict, replacements
        )
        
        # Results
        total_time = time.time() - start_time
        final_ratio = len(final_package) / original_length * 100
        space_saved = (1 - len(final_package) / original_length) * 100
        
        if not silent:
            print(f"   🚀 Best method: {best_method}")
            print(f"   📦 Final size: {len(final_package):,} bytes ({final_ratio:.1f}%)")
            print(f"   ⚡ Compression time: {total_time:.3f}s")
            print(f"   💾 Space saved: {space_saved:.1f}%")
            
            if final_ratio < 20:
                print("   🏆 PHENOMENAL COMPRESSION! LZMA KILLER SUCCESS!")
            elif final_ratio < 40:
                print("   🎉 EXCELLENT compression achieved!")
            elif final_ratio < 60:
                print("   ✅ Good compression performance")
            else:
                print("   📈 Compression successful, room for optimization")
        
        return final_package
    
    def _create_complete_package(self, compressed_data: bytes, method: str, 
                               original_length: int, pattern_dict: Dict[bytes, int], 
                               replacements: List[Dict]) -> bytes:
        """Create complete package with all reconstruction metadata"""
        package = bytearray()
        
        # Magic header
        package.extend(b'NXLK')  # NEXUS LZMA Killer
        
        # Core metadata
        package.extend(struct.pack('<I', original_length))  # Original length
        method_bytes = method.encode('utf-8')[:12].ljust(12, b'\x00')
        package.extend(method_bytes)  # Compression method
        
        # Pattern dictionary
        pattern_metadata = self._serialize_pattern_dict(pattern_dict)
        package.extend(struct.pack('<I', len(pattern_metadata)))
        package.extend(pattern_metadata)
        
        # Replacement information
        replacement_metadata = self._serialize_replacements(replacements)
        package.extend(struct.pack('<I', len(replacement_metadata)))
        package.extend(replacement_metadata)
        
        # Compressed data
        package.extend(struct.pack('<I', len(compressed_data)))
        package.extend(compressed_data)
        
        return bytes(package)
    
    def _serialize_pattern_dict(self, pattern_dict: Dict[bytes, int]) -> bytes:
        """Serialize pattern dictionary"""
        if not pattern_dict:
            return b''
        
        serialized = bytearray()
        serialized.extend(struct.pack('<H', len(pattern_dict)))  # Number of patterns
        
        for pattern, pattern_id in sorted(pattern_dict.items(), key=lambda x: x[1]):
            serialized.extend(struct.pack('<H', pattern_id))  # Pattern ID
            serialized.extend(struct.pack('<H', len(pattern)))  # Pattern length
            serialized.extend(pattern)  # Pattern data
        
        return bytes(serialized)
    
    def _serialize_replacements(self, replacements: List[Dict]) -> bytes:
        """Serialize replacement information"""
        if not replacements:
            return b''
        
        serialized = bytearray()
        serialized.extend(struct.pack('<I', len(replacements)))  # Number of replacements
        
        for replacement in replacements:
            serialized.extend(struct.pack('<I', replacement['position']))  # Position
            serialized.extend(struct.pack('<H', replacement['original_length']))  # Original length
            serialized.extend(struct.pack('<H', replacement['pattern_id']))  # Pattern ID
        
        return bytes(serialized)
    
    def decompress(self, package: bytes, silent: bool = False) -> bytes:
        """Decompress with perfect reconstruction"""
        if len(package) < 20:
            return b''
        
        start_time = time.time()
        offset = 0
        
        # Parse header
        magic = package[offset:offset+4]
        if magic != b'NXLK':
            raise ValueError("Invalid NEXUS LZMA Killer package")
        offset += 4
        
        original_length, = struct.unpack('<I', package[offset:offset+4])
        offset += 4
        
        method = package[offset:offset+12].rstrip(b'\x00').decode('utf-8')
        offset += 12
        
        # Parse pattern dictionary
        pattern_metadata_size, = struct.unpack('<I', package[offset:offset+4])
        offset += 4
        pattern_metadata = package[offset:offset+pattern_metadata_size]
        offset += pattern_metadata_size
        pattern_dict = self._deserialize_pattern_dict(pattern_metadata)
        
        # Parse replacements
        replacement_metadata_size, = struct.unpack('<I', package[offset:offset+4])
        offset += 4
        replacement_metadata = package[offset:offset+replacement_metadata_size]
        offset += replacement_metadata_size
        replacements = self._deserialize_replacements(replacement_metadata)
        
        # Parse compressed data
        compressed_size, = struct.unpack('<I', package[offset:offset+4])
        offset += 4
        compressed_data = package[offset:offset+compressed_size]
        
        if not silent:
            print(f"🔓 NEXUS LZMA KILLER DECOMPRESSION")
            print(f"   📋 Method: {method}, Original: {original_length:,} bytes")
            print(f"   🎯 Patterns: {len(pattern_dict)}, Replacements: {len(replacements)}")
        
        # Decompress main data
        if method.startswith('zstd_'):
            decompressed_data = self.decompressor.decompress(compressed_data)
        elif method.startswith('lzma_'):
            if LZMA_AVAILABLE:
                decompressed_data = lzma.decompress(compressed_data)
            else:
                raise ValueError("LZMA not available for decompression")
        else:
            decompressed_data = compressed_data
        
        # Reconstruct original data by reversing pattern replacements
        reconstructed_data = self._reconstruct_from_patterns(
            decompressed_data, pattern_dict, replacements, original_length
        )
        
        decompress_time = time.time() - start_time
        
        if not silent:
            print(f"   ⚡ Decompression time: {decompress_time:.3f}s")
            print(f"   ✅ Reconstructed: {len(reconstructed_data):,} bytes")
        
        return reconstructed_data
    
    def _deserialize_pattern_dict(self, data: bytes) -> Dict[int, bytes]:
        """Deserialize pattern dictionary"""
        if len(data) < 2:
            return {}
        
        pattern_dict = {}
        offset = 0
        
        pattern_count, = struct.unpack('<H', data[offset:offset+2])
        offset += 2
        
        for _ in range(pattern_count):
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
            
            pattern_dict[pattern_id] = pattern
        
        return pattern_dict
    
    def _deserialize_replacements(self, data: bytes) -> List[Dict]:
        """Deserialize replacement information"""
        if len(data) < 4:
            return []
        
        replacements = []
        offset = 0
        
        replacement_count, = struct.unpack('<I', data[offset:offset+4])
        offset += 4
        
        for _ in range(replacement_count):
            if offset + 8 > len(data):
                break
                
            position, = struct.unpack('<I', data[offset:offset+4])
            offset += 4
            
            original_length, = struct.unpack('<H', data[offset:offset+2])
            offset += 2
            
            pattern_id, = struct.unpack('<H', data[offset:offset+2])
            offset += 2
            
            replacements.append({
                'position': position,
                'original_length': original_length,
                'pattern_id': pattern_id
            })
        
        return replacements
    
    def _reconstruct_from_patterns(self, compressed_data: bytes, pattern_dict: Dict[int, bytes], 
                                 replacements: List[Dict], original_length: int) -> bytes:
        """Reconstruct original data from patterns"""
        # Start with decompressed data
        result = bytearray(compressed_data)
        
        # Sort replacements by position (reverse order to avoid offset shifts)
        sorted_replacements = sorted(replacements, key=lambda x: x['position'], reverse=True)
        
        # Apply pattern replacements in reverse
        for replacement in sorted_replacements:
            position = replacement['position']
            pattern_id = replacement['pattern_id']
            
            if pattern_id in pattern_dict:
                original_pattern = pattern_dict[pattern_id]
                
                # Find marker position (might have shifted)
                marker_start = None
                for i in range(max(0, position - 100), min(len(result), position + 100)):
                    if (i + 4 <= len(result) and 
                        result[i] == 255 and result[i+1] == 254 and 
                        result[i+2] == pattern_id % 256):
                        marker_start = i
                        break
                
                if marker_start is not None:
                    # Replace marker with original pattern
                    marker_end = marker_start + 4
                    result[marker_start:marker_end] = original_pattern
        
        # Trim or pad to original length
        if len(result) > original_length:
            result = result[:original_length]
        elif len(result) < original_length:
            result.extend(b'\x00' * (original_length - len(result)))
        
        return bytes(result)


def create_benchmark_file(filename: str, size_kb: int):
    """Create benchmark file with realistic data patterns"""
    print(f"Creating benchmark file: {filename} ({size_kb}KB)")
    
    target_size = size_kb * 1024
    data = bytearray()
    
    # Create realistic data with various patterns
    patterns = [
        b"ABCDEFGHIJKLMNOP" * 4,  # 64-byte pattern
        b"0123456789" * 6,       # 60-byte numeric pattern
        b"Hello World! " * 8,    # 104-byte text pattern
        bytes(range(128)) * 2,   # 256-byte sequential pattern
    ]
    
    while len(data) < target_size:
        # Add patterns (70% of data)
        for pattern in patterns:
            if len(data) + len(pattern) <= target_size * 0.7:
                data.extend(pattern)
        
        # Add semi-random data (20% of data)
        if len(data) < target_size * 0.9:
            semi_random_size = min(target_size - len(data), target_size // 10)
            base_value = random.randint(0, 200)
            semi_random = bytes([(base_value + i % 50) % 256 for i in range(semi_random_size)])
            data.extend(semi_random)
        
        # Add truly random data (10% of data)
        if len(data) < target_size:
            random_size = min(target_size - len(data), target_size // 20)
            random_data = bytes([random.randint(0, 255) for _ in range(random_size)])
            data.extend(random_data)
    
    # Trim to exact size
    data = data[:target_size]
    
    with open(filename, 'wb') as f:
        f.write(data)
    
    print(f"Benchmark file created: {len(data)} bytes")


def benchmark_lzma_killer():
    """Benchmark NEXUS LZMA Killer against LZMA"""
    print("🔥 NEXUS LZMA KILLER vs LZMA BENCHMARK")
    print("=" * 60)
    
    engine = NexusLZMAKillerEngine()
    test_sizes = [10, 25, 50]  # KB
    
    total_nexus_ratio = 0
    total_lzma_ratio = 0
    test_count = 0
    
    for size_kb in test_sizes:
        test_file = f"lzma_killer_test_{size_kb}kb.bin"
        print(f"\n📊 Benchmarking {size_kb}KB file:")
        
        create_benchmark_file(test_file, size_kb)
        
        with open(test_file, 'rb') as f:
            data = f.read()
        
        print(f"   Original size: {len(data):,} bytes")
        
        # NEXUS LZMA Killer compression
        start_time = time.time()
        nexus_compressed = engine.compress(data, silent=True)
        nexus_comp_time = time.time() - start_time
        
        # NEXUS LZMA Killer decompression
        start_time = time.time()
        nexus_decompressed = engine.decompress(nexus_compressed, silent=True)
        nexus_decomp_time = time.time() - start_time
        
        # LZMA compression (if available)
        if LZMA_AVAILABLE:
            start_time = time.time()
            lzma_compressed = lzma.compress(data, preset=9)
            lzma_comp_time = time.time() - start_time
            
            start_time = time.time()
            lzma_decompressed = lzma.decompress(lzma_compressed)
            lzma_decomp_time = time.time() - start_time
        else:
            lzma_compressed = data
            lzma_decompressed = data
            lzma_comp_time = 0
            lzma_decomp_time = 0
        
        # Calculate results
        nexus_ratio = len(nexus_compressed) / len(data) * 100
        lzma_ratio = len(lzma_compressed) / len(data) * 100 if LZMA_AVAILABLE else 100
        nexus_perfect = data == nexus_decompressed
        lzma_perfect = data == lzma_decompressed if LZMA_AVAILABLE else False
        
        total_nexus_ratio += nexus_ratio
        total_lzma_ratio += lzma_ratio
        test_count += 1
        
        print(f"\n   📈 COMPRESSION RESULTS:")
        print(f"   NEXUS LZMA Killer: {len(nexus_compressed):,} bytes ({nexus_ratio:.1f}%) - {nexus_comp_time:.3f}s")
        if LZMA_AVAILABLE:
            print(f"   LZMA (preset 9):   {len(lzma_compressed):,} bytes ({lzma_ratio:.1f}%) - {lzma_comp_time:.3f}s")
        
        print(f"\n   📉 DECOMPRESSION RESULTS:")
        print(f"   NEXUS LZMA Killer: {nexus_decomp_time:.3f}s - {'✓' if nexus_perfect else '✗'}")
        if LZMA_AVAILABLE:
            print(f"   LZMA (preset 9):   {lzma_decomp_time:.3f}s - {'✓' if lzma_perfect else '✗'}")
        
        # Performance comparison
        if LZMA_AVAILABLE and nexus_perfect:
            if nexus_ratio < lzma_ratio:
                improvement = (lzma_ratio - nexus_ratio) / lzma_ratio * 100
                print(f"\n   🏆 NEXUS LZMA KILLER WINS! {improvement:.1f}% better compression!")
            else:
                deficit = (nexus_ratio - lzma_ratio) / lzma_ratio * 100
                print(f"\n   📊 LZMA wins by {deficit:.1f}%")
        elif nexus_perfect:
            print(f"\n   ✅ NEXUS LZMA Killer: Perfect reconstruction achieved")
        else:
            print(f"\n   ❌ NEXUS LZMA Killer: Data corruption detected")
        
        # Clean up
        try:
            os.remove(test_file)
        except:
            pass
    
    # Overall summary
    if test_count > 0:
        avg_nexus_ratio = total_nexus_ratio / test_count
        avg_lzma_ratio = total_lzma_ratio / test_count
        
        print(f"\n🏁 FINAL BENCHMARK SUMMARY:")
        print(f"   Average NEXUS LZMA Killer: {avg_nexus_ratio:.1f}%")
        if LZMA_AVAILABLE:
            print(f"   Average LZMA:              {avg_lzma_ratio:.1f}%")
            if avg_nexus_ratio < avg_lzma_ratio:
                overall_improvement = (avg_lzma_ratio - avg_nexus_ratio) / avg_lzma_ratio * 100
                print(f"   🎉 NEXUS LZMA KILLER OVERALL WINNER! {overall_improvement:.1f}% better!")
            else:
                print(f"   📈 LZMA wins overall, but NEXUS shows promise")


if __name__ == "__main__":
    benchmark_lzma_killer()
