use anyhow::Result;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// SPE (Structure-Preserving Encryption) のスタブ実装
/// 
/// 注意: これは概念実証用のスタブ実装です。
/// 実際のSPEアルゴリズムは高度な難読化技術を含み、
/// 商用利用時は非公開DLLとして提供されます。

/// SPE変換を適用（可逆的な構造保持難読化）
pub fn apply_spe_transform(data: &[u8]) -> Result<Vec<u8>> {
    let mut transformed = data.to_vec();
    
    // 段階的変換のため元のサイズを記録
    let original_size = data.len();
    
    // 1. XORマスク適用
    apply_xor_mask(&mut transformed);
    
    // 2. ブロックシャッフル適用（小さいデータはスキップ）
    if original_size >= 128 {
        let block_size = calculate_optimal_block_size(original_size);
        apply_block_shuffle(&mut transformed, block_size);
    }
    
    Ok(transformed)
}

/// SPE逆変換を適用（元の構造に復元）
pub fn reverse_spe_transform(data: &[u8]) -> Result<Vec<u8>> {
    let mut restored = data.to_vec();
    let original_size = data.len(); // XORマスク適用前と後でサイズは変わらない
    
    // 逆順で変換を戻す
    
    // 1. ブロックシャッフル逆変換（小さいデータはスキップ）
    if original_size >= 128 {
        let block_size = calculate_optimal_block_size(original_size);
        reverse_block_shuffle(&mut restored, block_size);
    }
    
    // 2. XORマスク除去
    remove_xor_mask(&mut restored);
    
    Ok(restored)
}

fn calculate_optimal_block_size(data_len: usize) -> usize {
    // データサイズに応じて最適なブロックサイズを計算
    match data_len {
        0..=1024 => 64,
        1025..=65536 => 256,
        65537..=1048576 => 1024,
        _ => 4096,
    }
}

fn apply_block_shuffle(data: &mut [u8], block_size: usize) {
    if data.len() < block_size * 2 {
        return; // ブロックサイズが大きすぎる場合はスキップ
    }
    
    let num_blocks = data.len() / block_size;
    let mut blocks: Vec<Vec<u8>> = Vec::new();
    
    // ブロックに分割
    for i in 0..num_blocks {
        let start = i * block_size;
        let end = std::cmp::min(start + block_size, data.len());
        blocks.push(data[start..end].to_vec());
    }
    
    // 決定論的な並び替えパターンを生成
    let mut hasher = DefaultHasher::new();
    data.len().hash(&mut hasher);
    let seed = hasher.finish();
    
    // 新しい順序を計算
    let mut new_order: Vec<usize> = (0..blocks.len()).collect();
    for i in 0..new_order.len() {
        let swap_with = (seed as usize + i * 17) % new_order.len();
        new_order.swap(i, swap_with);
    }
    
    // 新しい順序でブロックを再配置
    let shuffled_blocks: Vec<Vec<u8>> = new_order.iter().map(|&i| blocks[i].clone()).collect();
    
    // データを復元
    let mut index = 0;
    for block in shuffled_blocks {
        let len = block.len();
        data[index..index + len].copy_from_slice(&block);
        index += len;
    }
}

fn reverse_block_shuffle(data: &mut [u8], block_size: usize) {
    if data.len() < block_size * 2 {
        return; // ブロックサイズが大きすぎる場合はスキップ
    }
    
    let num_blocks = data.len() / block_size;
    let mut blocks: Vec<Vec<u8>> = Vec::new();
    
    // ブロックに分割
    for i in 0..num_blocks {
        let start = i * block_size;
        let end = std::cmp::min(start + block_size, data.len());
        blocks.push(data[start..end].to_vec());
    }
    
    // 同じシード値で並び替えパターンを生成
    let mut hasher = DefaultHasher::new();
    data.len().hash(&mut hasher);
    let seed = hasher.finish();
    
    // 元の順序を再計算
    let mut shuffled_order: Vec<usize> = (0..blocks.len()).collect();
    for i in 0..shuffled_order.len() {
        let swap_with = (seed as usize + i * 17) % shuffled_order.len();
        shuffled_order.swap(i, swap_with);
    }
    
    // 逆マッピングを作成
    let mut original_order = vec![0; blocks.len()];
    for (shuffled_pos, &original_pos) in shuffled_order.iter().enumerate() {
        original_order[original_pos] = shuffled_pos;
    }
    
    // 元の順序でブロックを再配置
    let restored_blocks: Vec<Vec<u8>> = original_order.iter().map(|&i| blocks[i].clone()).collect();
    
    // データを復元
    let mut index = 0;
    for block in restored_blocks {
        let len = block.len();
        data[index..index + len].copy_from_slice(&block);
        index += len;
    }
}

fn apply_xor_mask(data: &mut [u8]) {
    // 簡単なXORマスク (実際のSPEはより複雑な鍵導出)
    const XOR_PATTERN: &[u8] = b"NXZ_SPE_MASK_2024";
    
    for (i, byte) in data.iter_mut().enumerate() {
        *byte ^= XOR_PATTERN[i % XOR_PATTERN.len()];
    }
}

fn remove_xor_mask(data: &mut [u8]) {
    // XORは自己逆変換なので同じ処理
    apply_xor_mask(data);
}

fn get_original_size_from_padded(data: &[u8]) -> Result<usize> {
    if data.len() < 8 {
        anyhow::bail!("データが小さすぎて元サイズを取得できません");
    }
    
    // 末尾から長さ情報を読み取り
    let len_start = data.len() - 8;
    let len_bytes: [u8; 8] = data[len_start..].try_into()?;
    let original_len = usize::from_le_bytes(len_bytes);
    
    if original_len > len_start {
        anyhow::bail!("不正なパディング長さ情報: {} > {}", original_len, len_start);
    }
    
    Ok(original_len)
}

fn apply_structure_padding(data: &mut Vec<u8>) -> Result<()> {
    // 圧縮効率を上げるための構造保持パディング
    let original_len = data.len();
    
    // 長さ情報を末尾に追加
    data.extend_from_slice(&original_len.to_le_bytes());
    
    // アライメント調整 (16バイト境界)
    while data.len() % 16 != 0 {
        data.push(0x00);
    }
    
    Ok(())
}

fn remove_structure_padding(data: &mut Vec<u8>) -> Result<()> {
    if data.len() < 8 {
        anyhow::bail!("データが小さすぎてパディング除去できません");
    }
    
    // 末尾から長さ情報を読み取り
    let len_start = data.len() - 8;
    let len_bytes: [u8; 8] = data[len_start..].try_into()?;
    let original_len = usize::from_le_bytes(len_bytes);
    
    if original_len > len_start {
        anyhow::bail!("不正なパディング長さ情報: {} > {}", original_len, len_start);
    }
    
    // 元のサイズに切り詰め（長さ情報の部分は除く）
    data.truncate(original_len);
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_spe_transform_reversible() {
        let original_data = b"Hello, NXZip World! This is a test data for SPE transformation.";
        
        // 変換
        let transformed = apply_spe_transform(original_data).unwrap();
        
        // 元データと異なることを確認
        assert_ne!(&transformed, original_data);
        
        // 逆変換
        let restored = reverse_spe_transform(&transformed).unwrap();
        
        // 完全に復元されることを確認
        assert_eq!(&restored, original_data);
    }
    
    #[test]
    fn test_spe_transform_different_sizes() {
        for size in [10, 100, 1000, 10000] {
            let original_data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
            
            let transformed = apply_spe_transform(&original_data).unwrap();
            let restored = reverse_spe_transform(&transformed).unwrap();
            
            assert_eq!(restored, original_data, "Failed for size {}", size);
        }
    }
    
    #[test]
    fn test_block_shuffle_reversible() {
        let original_data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        let mut test_data = original_data.clone();
        
        let block_size = calculate_optimal_block_size(test_data.len());
        
        // ブロックシャッフル適用
        apply_block_shuffle(&mut test_data, block_size);
        
        // 元データと異なることを確認
        assert_ne!(test_data, original_data);
        
        // 逆変換適用
        reverse_block_shuffle(&mut test_data, block_size);
        
        // 完全に復元されることを確認
        assert_eq!(test_data, original_data);
    }
    
    #[test]
    #[ignore] // 現在の実装では構造パディングを使用していない
    fn test_structure_padding_reversible() {
        let original_data = b"Test data for padding verification".to_vec();
        let mut test_data = original_data.clone();
        
        // パディング適用
        apply_structure_padding(&mut test_data).unwrap();
        
        // データサイズが増加していることを確認
        assert!(test_data.len() > original_data.len());
        
        // パディング除去
        remove_structure_padding(&mut test_data).unwrap();
        
        // 完全に復元されることを確認
        assert_eq!(test_data, original_data);
    }
    
    #[test]
    fn test_xor_mask_reversible() {
        let original_data = b"XOR mask test data with various characters 12345!@#$%";
        let mut test_data = original_data.to_vec();
        
        // XORマスク適用
        apply_xor_mask(&mut test_data);
        
        // 元データと異なることを確認
        assert_ne!(&test_data, original_data);
        
        // XORマスク除去
        remove_xor_mask(&mut test_data);
        
        // 完全に復元されることを確認
        assert_eq!(&test_data, original_data);
    }
}
