#!/usr/bin/env python3
"""
🚀 NXZip Tool Entry Point

NXZip - Next-generation eXtreme Universal Zip Archive System
Usage: python nxzip_tool.py [command] [options]
"""

import sys
import os

# パッケージパスを追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nxzip.cli.main import main

if __name__ == '__main__':
    sys.exit(main())
