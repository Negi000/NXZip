#!/usr/bin/env python3
"""
NXZip CLI Tool - 統合圧縮システム
AV1/SRLA/AVIF制約除去技術による次世代圧縮
"""

import sys
import os
from pathlib import Path

# プロジェクトパス追加
current_dir = Path(__file__).parent
project_root = current_dir.parent / "NXZip-Python"
sys.path.insert(0, str(project_root))

from nxzip.cli_unified import main

if __name__ == "__main__":
    main()
