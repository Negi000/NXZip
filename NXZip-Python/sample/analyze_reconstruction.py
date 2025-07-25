#!/usr/bin/env python3
"""データ復元プロセスの詳細分析"""

def analyze_data_reconstruction():
    """データ復元プロセスを詳細に分析"""
    
    print("=== 元データとブロック分析 ===")
    original_data = b"ABCDEFGH"  # [65, 66, 67, 68, 69, 70, 71, 72]
    print(f"元データ: {list(original_data)}")
    
    # 実際に生成されるブロック（trace_blocks.pyの結果より）
    generated_blocks = [
        (65, 66, 67, 69, 71, 72, 0),  # Block 0
        (68, 69, 70, 72, 0, 0, 0),   # Block 1  
        (71, 72, 0, 0, 0, 0, 0)      # Block 2
    ]
    
    print(f"生成されたブロック:")
    for i, block in enumerate(generated_blocks):
        print(f"  Block {i}: {block}")
    
    # フラット化した時のデータ
    flat_from_blocks = []
    for block in generated_blocks:
        flat_from_blocks.extend(block)
    
    print(f"ブロックをフラット化: {flat_from_blocks}")
    print(f"元の長さ8にトリミング: {flat_from_blocks[:8]}")
    
    # 問題：元データと一致しない！
    print(f"元データ  : [65, 66, 67, 68, 69, 70, 71, 72]")
    print(f"復元データ: {flat_from_blocks[:8]}")
    
    print("\n=== 位置別詳細分析 ===")
    # 各位置の元データがどのブロックのどの位置にあるかを調べる
    
    # H-7形状での位置マッピング
    # 元データの各バイトがグリッド上のどこに配置されるか
    original_positions = [
        (0, 0), (0, 1), (0, 2),  # A, B, C
        (1, 0), (1, 1), (1, 2),  # D, E, F  
        (2, 0), (2, 1)           # G, H
    ]
    
    # H-7形状の座標
    h7_coords = [(0, 0), (0, 1), (0, 2), (1, 1), (2, 0), (2, 1), (2, 2)]
    
    print("H-7形状でのデータ配置:")
    grid = [[None for _ in range(3)] for _ in range(6)]  # 6行3列（パディング込み）
    
    # 元データを配置
    for i, (r, c) in enumerate(original_positions):
        if i < len(original_data):
            grid[r][c] = chr(original_data[i])
    
    for r in range(6):
        row_str = ""
        for c in range(3):
            if grid[r][c] is not None:
                row_str += f"{grid[r][c]:2}"
            else:
                row_str += " ."
        print(f"  Row {r}: {row_str}")
    
    print(f"\nH-7形状座標: {h7_coords}")
    
    # 各ブロックがどの位置から抽出されるか
    print("\n=== ブロック抽出位置分析 ===")
    
    # H-7形状は3x3の窓で、7個の座標を持つ
    # Grid width = 3なので、水平方向のスライドは1回だけ可能
    # (6-3+1) * (3-3+1) = 4 * 1 = 4個のブロックが理論上可能だが、
    # 実際には3個しか生成されていない
    
    for block_id in range(3):
        print(f"\nBlock {block_id}:")
        start_row = block_id
        start_col = 0
        
        extracted_values = []
        for dr, dc in h7_coords:
            r, c = start_row + dr, start_col + dc
            if r < 6 and c < 3:
                if r < len(original_positions) and original_positions[r * 3 + c] == (r, c):
                    # 元データの範囲内
                    if r * 3 + c < len(original_data):
                        extracted_values.append(original_data[r * 3 + c])
                    else:
                        extracted_values.append(0)  # パディング
                else:
                    extracted_values.append(0)  # パディング
            else:
                extracted_values.append(0)  # 範囲外
        
        print(f"  抽出位置: {[(start_row + dr, start_col + dc) for dr, dc in h7_coords]}")
        print(f"  理論値: {extracted_values}")
        print(f"  実際: {generated_blocks[block_id]}")

if __name__ == "__main__":
    analyze_data_reconstruction()
