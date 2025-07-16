#!/usr/bin/env python3
"""
🚀 NXZip - Next-generation eXtreme Universal Zip Archive System

Entry Point for NXZip Tool

Usage:
    python -m nxzip create archive.nxz file1.txt file2.txt
    python -m nxzip extract archive.nxz -o output_dir
    python -m nxzip list archive.nxz
    python -m nxzip test archive.nxz
    python -m nxzip benchmark file1.txt file2.txt

Copyright (c) 2025 NXZip Project
"""

import sys
from .cli.main import main

if __name__ == '__main__':
    sys.exit(main())
