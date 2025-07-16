#!/usr/bin/env python3
"""
Progress Bar Utilities
プログレスバー表示のためのユーティリティ
"""

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class ProgressBar:
    """プログレスバー表示クラス"""
    
    def __init__(self, total: int, desc: str = "処理中"):
        self.total = total
        self.desc = desc
        self.current = 0
        if TQDM_AVAILABLE:
            self.pbar = tqdm(total=total, desc=desc, unit='B', unit_scale=True)
        else:
            self.pbar = None
    
    def update(self, amount: int):
        """進捗を更新"""
        self.current += amount
        if self.pbar:
            self.pbar.update(amount)
        else:
            percent = (self.current / self.total) * 100 if self.total > 0 else 0
            print(f"\r{self.desc}: {percent:.1f}%", end='', flush=True)
    
    def close(self):
        """プログレスバーを閉じる"""
        if self.pbar:
            self.pbar.close()
        else:
            print()  # 改行
