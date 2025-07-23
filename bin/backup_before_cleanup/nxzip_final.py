#!/usr/bin/env python3
"""
NXZip CLI Tool - 最終統合版
97.31%圧縮率と139.80MB/sの性能を持つ統合ツール
"""

import sys
import os
from pathlib import Path

# プロジェクトパス追加
current_dir = Path(__file__).parent
project_root = current_dir.parent / "NXZip-Python"
sys.path.insert(0, str(project_root))

from nxzip.cli_final import main

if __name__ == "__main__":
    main()
