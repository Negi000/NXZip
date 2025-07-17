#!/usr/bin/env python3
"""
Test Python script for NXZip format detection
"""

import os
import sys
import hashlib

def calculate_hash(data):
    """Calculate SHA256 hash of data"""
    return hashlib.sha256(data).hexdigest()

def process_file(filename):
    """Process a file and return statistics"""
    try:
        with open(filename, 'rb') as f:
            data = f.read()
        
        stats = {
            'filename': filename,
            'size': len(data),
            'hash': calculate_hash(data),
            'format': filename.split('.')[-1] if '.' in filename else 'unknown'
        }
        
        return stats
    except Exception as e:
        return {'error': str(e)}

def main():
    """Main function for testing"""
    print("NXZip Format Detection Test")
    print("-" * 30)
    
    test_files = [
        'test.txt',
        'sample.json',
        'document.md'
    ]
    
    for filename in test_files:
        if os.path.exists(filename):
            stats = process_file(filename)
            print(f"File: {stats.get('filename', 'N/A')}")
            print(f"Size: {stats.get('size', 'N/A')} bytes")
            print(f"Format: {stats.get('format', 'N/A')}")
            print(f"Hash: {stats.get('hash', 'N/A')[:16]}...")
            print()
        else:
            print(f"File not found: {filename}")

if __name__ == "__main__":
    main()
