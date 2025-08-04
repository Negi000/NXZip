#!/usr/bin/env python3
"""
NXZip Professional Launcher v2.0
"""

import sys
import os
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# メインアプリケーションを起動
if __name__ == "__main__":
    try:
        print("🚀 Starting NXZip Professional v2.0...")
        from NXZip_Professional import main
        main()
    except Exception as e:
        print(f"❌ Launch failed: {e}")
        input("Press Enter to exit...")
