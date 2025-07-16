use std::convert::TryInto;

// SPE (Structure-Preserving Encryption) の基本テスト
fn main() {
    println!("🎯 NXZip SPE Core System Test");
    
    // 基本的な可逆性テスト
    test_basic_reversibility();
    
    // データ構造保持テスト
    test_structure_preservation();
    
    // パフォーマンステスト
    test_performance();
    
    // 統合テスト
    integration_test();
    
    println!("✅ All SPE Core tests passed!");
}

fn test_basic_reversibility() {
    println!("\n📋 Testing Basic Reversibility...");
    
    let test_data = b"Hello, NXZip SPE Core System!";
    println!("Original data: {:?}", String::from_utf8_lossy(test_data));
    
    // 簡易SPE変換（XOR + ブロックシャッフル）
    let transformed = apply_simple_spe(test_data);
    println!("Transformed: {:02X?}", &transformed[..16.min(transformed.len())]);
    
    // 逆変換
    let restored = reverse_simple_spe(&transformed);
    println!("Restored: {:?}", String::from_utf8_lossy(&restored));
    
    assert_eq!(test_data.to_vec(), restored, "可逆性テスト失敗");
    println!("✅ Reversibility test passed");
}

fn test_structure_preservation() {
    println!("\n🏗️ Testing Structure Preservation...");
    
    // 異なるサイズのデータでテスト
    let test_sizes = vec![10, 100, 1000, 5000];
    
    for size in test_sizes {
        let test_data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        
        let transformed = apply_simple_spe(&test_data);
        let restored = reverse_simple_spe(&transformed);
        
        assert_eq!(test_data, restored, "Structure preservation failed for size {}", size);
        
        let ratio = transformed.len() as f64 / test_data.len() as f64;
        println!("Size {}: Original {} -> Transformed {} (ratio: {:.2})", 
            size, test_data.len(), transformed.len(), ratio);
    }
    
    println!("✅ Structure preservation test passed");
}

fn test_performance() {
    println!("\n⚡ Testing Performance...");
    
    let data_size = 10000;
    let test_data: Vec<u8> = (0..data_size).map(|i| (i % 256) as u8).collect();
    
    let start = std::time::Instant::now();
    let transformed = apply_simple_spe(&test_data);
    let transform_time = start.elapsed();
    
    let start = std::time::Instant::now();
    let restored = reverse_simple_spe(&transformed);
    let restore_time = start.elapsed();
    
    assert_eq!(test_data, restored);
    
    let throughput = data_size as f64 / transform_time.as_secs_f64() / 1024.0 / 1024.0;
    
    println!("Performance for {} bytes:", data_size);
    println!("  Transform: {:?} ({:.2} MB/s)", transform_time, throughput);
    println!("  Restore: {:?}", restore_time);
    
    println!("✅ Performance test completed");
}

// 簡易SPE実装（デモンストレーション用）
fn apply_simple_spe(data: &[u8]) -> Vec<u8> {
    let mut result = data.to_vec();
    
    // 1. 構造保持パディング
    let original_len = result.len();
    let padded_len = ((original_len + 15) / 16) * 16; // 16バイト境界
    result.resize(padded_len, 0);
    
    // 元の長さを末尾に記録
    result.extend_from_slice(&(original_len as u64).to_le_bytes());
    
    // 2. ブロックシャッフル（16バイトブロック）
    if result.len() >= 32 {
        let block_size = 16;
        let num_blocks = result.len() / block_size;
        
        for i in 0..num_blocks {
            let swap_with = (i * 7 + 3) % num_blocks; // 簡単な決定論的パターン
            if i != swap_with {
                let start1 = i * block_size;
                let start2 = swap_with * block_size;
                
                for j in 0..block_size {
                    if start1 + j < result.len() && start2 + j < result.len() {
                        result.swap(start1 + j, start2 + j);
                    }
                }
            }
        }
    }
    
    // 3. XOR難読化
    let xor_key = b"NXZip_SPE_2024";
    for (i, byte) in result.iter_mut().enumerate() {
        *byte ^= xor_key[i % xor_key.len()];
    }
    
    result
}

fn reverse_simple_spe(data: &[u8]) -> Vec<u8> {
    let mut result = data.to_vec();
    
    // 逆順で処理
    
    // 1. XOR除去
    let xor_key = b"NXZip_SPE_2024";
    for (i, byte) in result.iter_mut().enumerate() {
        *byte ^= xor_key[i % xor_key.len()];
    }
    
    // 2. ブロックシャッフル逆変換
    if result.len() >= 32 {
        let block_size = 16;
        let num_blocks = result.len() / block_size;
        
        // 同じパターンで逆変換（XORと同様、シャッフルも自己逆変換）
        for i in 0..num_blocks {
            let swap_with = (i * 7 + 3) % num_blocks;
            if i != swap_with {
                let start1 = i * block_size;
                let start2 = swap_with * block_size;
                
                for j in 0..block_size {
                    if start1 + j < result.len() && start2 + j < result.len() {
                        result.swap(start1 + j, start2 + j);
                    }
                }
            }
        }
    }
    
    // 3. パディング除去
    if result.len() >= 8 {
        let len_start = result.len() - 8;
        let len_bytes: [u8; 8] = result[len_start..].try_into().unwrap();
        let original_len = u64::from_le_bytes(len_bytes) as usize;
        
        if original_len <= len_start {
            result.truncate(original_len);
        }
    }
    
    result
}

// 統合テスト関数
fn integration_test() {
    println!("\n🔧 Running Integration Tests...");
    
    // 複数のデータパターンをテスト
    let test_patterns = vec![
        b"".to_vec(),                                    // 空データ
        b"A".to_vec(),                                   // 1バイト
        b"Hello".to_vec(),                               // 短文
        "これは日本語のテストデータです。".as_bytes().to_vec(), // 日本語
        (0..255).collect::<Vec<u8>>(),                   // バイナリデータ
        b"A".repeat(1000),                               // 反復データ
    ];
    
    for (i, pattern) in test_patterns.iter().enumerate() {
        println!("Pattern {}: {} bytes", i + 1, pattern.len());
        
        let transformed = apply_simple_spe(pattern);
        let restored = reverse_simple_spe(&transformed);
        
        assert_eq!(pattern, &restored, "Pattern {} failed", i + 1);
        
        if pattern.len() > 0 {
            let compression_ratio = transformed.len() as f64 / pattern.len() as f64;
            println!("  Ratio: {:.2}x", compression_ratio);
        }
    }
    
    println!("✅ Integration tests passed");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_spe_core_functionality() {
        let data = b"Test data for SPE core";
        let transformed = apply_simple_spe(data);
        let restored = reverse_simple_spe(&transformed);
        assert_eq!(data.to_vec(), restored);
    }
    
    #[test]
    fn test_empty_data() {
        let data = b"";
        let transformed = apply_simple_spe(data);
        let restored = reverse_simple_spe(&transformed);
        assert_eq!(data.to_vec(), restored);
    }
    
    #[test]
    fn test_large_data() {
        let data: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
        let transformed = apply_simple_spe(&data);
        let restored = reverse_simple_spe(&transformed);
        assert_eq!(data, restored);
    }
}
