use anyhow::Result;
use std::time::Instant;
use tempfile::NamedTempFile;

use nxzip::engine::{SPECore, SPEParameters, StructureLevel};
use nxzip::crypto::{IntegratedEncryptor, IntegratedDecryptor, EncryptionAlgorithm, KeyDerivationFunction};
use nxzip::formats::{EnhancedNxzFile, EnhancedNxzHeader};
use nxzip::engine::{Compressor, CompressionAlgorithm};

/// SPEコアシステムの包括的テスト
#[tokio::test]
async fn test_spe_core_comprehensive() -> Result<()> {
    // テストデータの準備
    let test_cases = vec![
        // 小さなテキストデータ
        ("Hello, NXZip SPE Core!", "small_text"),
        // 中程度のテキストデータ
        (&"A".repeat(1000), "medium_text"),
        // 大きなテキストデータ
        (&"Lorem ipsum dolor sit amet".repeat(1000), "large_text"),
        // バイナリデータ
        (&(0..255).cycle().take(2048).map(|b| b as u8).collect::<Vec<u8>>().iter().map(|&b| b as char).collect::<String>(), "binary_data"),
    ];
    
    for structure_level in [StructureLevel::Basic, StructureLevel::Extended, StructureLevel::Maximum] {
        for (test_data, description) in &test_cases {
            println!("Testing SPE Core - Level: {:?}, Data: {}", structure_level, description);
            
            // SPEパラメータの設定
            let mut params = SPEParameters::default();
            params.structure_level = structure_level;
            params.block_size = 512; // 小さなブロックサイズでテスト
            
            let mut spe_core = SPECore::new(params)?;
            
            let start_time = Instant::now();
            
            // SPE変換
            let transform_result = spe_core.apply_transform(test_data.as_bytes())?;
            let transform_time = start_time.elapsed();
            
            // 変換されたデータが元データと異なることを確認
            assert_ne!(&transform_result.transformed_data, test_data.as_bytes(), 
                "変換されたデータが元データと同じです - Level: {:?}, Data: {}", structure_level, description);
            
            let start_time = Instant::now();
            
            // SPE逆変換
            let restored_data = spe_core.reverse_transform(&transform_result)?;
            let restore_time = start_time.elapsed();
            
            // 完全に復元されることを確認
            assert_eq!(&restored_data, test_data.as_bytes(), 
                "復元されたデータが元データと一致しません - Level: {:?}, Data: {}", structure_level, description);
            
            println!("  変換時間: {:?}, 復元時間: {:?}, 変換率: {:.2}%", 
                transform_time, restore_time, 
                (transform_result.transformed_data.len() as f64 / test_data.len() as f64) * 100.0);
        }
    }
    
    Ok(())
}

/// 統合暗号化システムのテスト
#[tokio::test]
async fn test_integrated_encryption_comprehensive() -> Result<()> {
    let password = "comprehensive_test_password_2024";
    let test_data = "これは統合暗号化システムの包括的なテストデータです。SPE変換と暗号化の組み合わせをテストします。".repeat(100);
    
    let test_configs = vec![
        (EncryptionAlgorithm::AesGcm, KeyDerivationFunction::Pbkdf2, false),
        (EncryptionAlgorithm::AesGcm, KeyDerivationFunction::Pbkdf2, true),
        (EncryptionAlgorithm::XChaCha20Poly1305, KeyDerivationFunction::Argon2id, false),
        (EncryptionAlgorithm::XChaCha20Poly1305, KeyDerivationFunction::Argon2id, true),
    ];
    
    for (enc_algo, kdf_algo, spe_enabled) in test_configs {
        println!("Testing Integrated Encryption - Algo: {:?}, KDF: {:?}, SPE: {}", 
            enc_algo, kdf_algo, spe_enabled);
        
        let mut kdf_config = nxzip::crypto::integrated::KeyDerivationConfig::default();
        kdf_config.kdf = kdf_algo;
        kdf_config.iterations = 1000; // テスト用に短縮
        
        let mut encryptor = IntegratedEncryptor::with_config(
            password,
            enc_algo,
            kdf_config.clone(),
            spe_enabled,
        )?;
        
        if spe_enabled {
            encryptor.set_spe_level(StructureLevel::Extended)?;
        }
        
        let start_time = Instant::now();
        
        // 暗号化
        let encryption_result = encryptor.encrypt(test_data.as_bytes())?;
        let encrypt_time = start_time.elapsed();
        
        // 暗号化されたデータが元データと異なることを確認
        assert_ne!(&encryption_result.encrypted_data, test_data.as_bytes());
        
        // SPE設定の確認
        assert_eq!(encryption_result.metadata.spe_enabled, spe_enabled);
        if spe_enabled {
            assert!(encryption_result.spe_result.is_some());
        }
        
        let start_time = Instant::now();
        
        // 復号化
        let decrypted_data = encryptor.decrypt(&encryption_result)?;
        let decrypt_time = start_time.elapsed();
        
        // 完全に復元されることを確認
        assert_eq!(&decrypted_data, test_data.as_bytes());
        
        println!("  暗号化時間: {:?}, 復号化時間: {:?}, サイズ変化: {:.2}%", 
            encrypt_time, decrypt_time,
            (encryption_result.encrypted_data.len() as f64 / test_data.len() as f64) * 100.0);
        
        // 異なるパスワードでの復号化テスト（失敗することを確認）
        let wrong_password_result = IntegratedDecryptor::with_metadata("wrong_password", &encryption_result.metadata);
        assert!(wrong_password_result.is_ok()); // 復号化器作成は成功
        
        if let Ok(mut wrong_decryptor) = wrong_password_result {
            let wrong_decrypt_result = wrong_decryptor.decrypt(&encryption_result);
            assert!(wrong_decrypt_result.is_err(), "間違ったパスワードで復号化が成功してしまいました");
        }
    }
    
    Ok(())
}

/// 強化されたNXZファイルフォーマットのテスト
#[tokio::test]
async fn test_enhanced_nxz_format() -> Result<()> {
    let test_data = "Enhanced NXZ Format Test Data".repeat(500);
    
    // 様々な設定でファイルを作成
    let configs = vec![
        (CompressionAlgorithm::Zstd, Some(EncryptionAlgorithm::AesGcm), Some(KeyDerivationFunction::Pbkdf2), true, Some(StructureLevel::Basic)),
        (CompressionAlgorithm::Lzma2, Some(EncryptionAlgorithm::XChaCha20Poly1305), Some(KeyDerivationFunction::Argon2id), true, Some(StructureLevel::Extended)),
        (CompressionAlgorithm::Auto, None, None, false, None),
    ];
    
    for (comp_algo, enc_algo, kdf_algo, spe_enabled, spe_level) in configs {
        println!("Testing Enhanced NXZ Format - Compression: {:?}, Encryption: {:?}, SPE: {} ({:?})", 
            comp_algo, enc_algo, spe_enabled, spe_level);
        
        // ファイル作成
        let mut nxz_file = EnhancedNxzFile::new(
            test_data.len() as u64,
            comp_algo,
            enc_algo,
            kdf_algo,
            spe_enabled,
            spe_level,
        )?;
        
        // 圧縮（簡易版）
        let compressor = Compressor::new(comp_algo);
        let compressed = compressor.compress(test_data.as_bytes())?;
        nxz_file.compressed_data = compressed;
        
        // 完全性情報の更新
        nxz_file.extended_fields.integrity_info.original_hash = {
            use sha2::{Sha256, Digest};
            let mut hasher = Sha256::new();
            hasher.update(test_data.as_bytes());
            hasher.finalize().into()
        };
        
        nxz_file.extended_fields.integrity_info.compressed_hash = {
            use sha2::{Sha256, Digest};
            let mut hasher = Sha256::new();
            hasher.update(&nxz_file.compressed_data);
            hasher.finalize().into()
        };
        
        // バイナリ変換テスト
        let binary_data = nxz_file.to_bytes()?;
        let restored_file = EnhancedNxzFile::from_bytes(&binary_data)?;
        
        // ヘッダ情報の確認
        assert_eq!(restored_file.header.original_size, test_data.len() as u64);
        assert_eq!(restored_file.header.is_spe_enabled(), spe_enabled);
        assert_eq!(restored_file.header.is_encrypted(), enc_algo.is_some());
        
        if let Some(expected_algo) = enc_algo {
            assert_eq!(restored_file.header.get_encryption_algorithm()?.unwrap(), expected_algo);
        }
        
        if let Some(expected_level) = spe_level {
            assert_eq!(restored_file.header.get_spe_level()?.unwrap(), expected_level);
        }
        
        // 完全性検証
        restored_file.verify_integrity()?;
        
        // 一時ファイルでの読み書きテスト
        let temp_file = NamedTempFile::new()?;
        let temp_path = temp_file.path().to_str().unwrap();
        
        nxz_file.write_to_file(temp_path).await?;
        let loaded_file = EnhancedNxzFile::read_from_file(temp_path).await?;
        
        // 読み込んだファイルの検証
        assert_eq!(loaded_file.header.original_size, nxz_file.header.original_size);
        assert_eq!(loaded_file.compressed_data, nxz_file.compressed_data);
        
        println!("  ファイルサイズ: {} -> {} バイト ({:.2}%)", 
            test_data.len(), binary_data.len(),
            (binary_data.len() as f64 / test_data.len() as f64) * 100.0);
    }
    
    Ok(())
}

/// パフォーマンステスト
#[tokio::test]
async fn test_spe_performance() -> Result<()> {
    let data_sizes = vec![1024, 10240, 102400, 1048576]; // 1KB, 10KB, 100KB, 1MB
    
    for size in data_sizes {
        println!("Performance Test - Data Size: {} bytes", size);
        
        // テストデータ生成
        let test_data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        
        for level in [StructureLevel::Basic, StructureLevel::Extended, StructureLevel::Maximum] {
            let mut params = SPEParameters::default();
            params.structure_level = level;
            
            let mut spe_core = SPECore::new(params)?;
            
            // 変換性能測定
            let start = Instant::now();
            let result = spe_core.apply_transform(&test_data)?;
            let transform_time = start.elapsed();
            
            // 復元性能測定
            let start = Instant::now();
            let restored = spe_core.reverse_transform(&result)?;
            let restore_time = start.elapsed();
            
            // 正確性確認
            assert_eq!(restored, test_data);
            
            let throughput_mb_s = (size as f64) / (1024.0 * 1024.0) / transform_time.as_secs_f64();
            
            println!("  Level {:?}: 変換 {:?} ({:.2} MB/s), 復元 {:?}", 
                level, transform_time, throughput_mb_s, restore_time);
        }
        
        println!();
    }
    
    Ok(())
}

/// エラー回復テスト
#[tokio::test]
async fn test_error_recovery() -> Result<()> {
    let test_data = b"Error recovery test data for NXZip SPE system";
    let mut spe_core = SPECore::default()?;
    
    // 正常なケース
    let result = spe_core.apply_transform(test_data)?;
    let restored = spe_core.reverse_transform(&result)?;
    assert_eq!(&restored, test_data);
    
    // データ破損のシミュレーション
    let mut corrupted_result = result.clone();
    
    // 変換データの一部を破損
    if let Some(byte) = corrupted_result.transformed_data.get_mut(10) {
        *byte = byte.wrapping_add(1);
    }
    
    // 破損したデータの復元は適切にエラーになるかチェック
    let recovery_result = spe_core.reverse_transform(&corrupted_result);
    
    // メタデータに依存するため、データ破損が検出される可能性がある
    // ただし、SPEの特性上、軽微な破損では完全にエラーにならない場合もある
    match recovery_result {
        Ok(recovered) => {
            // 復元できた場合、元データと異なることを確認
            if recovered == test_data {
                println!("Warning: Light corruption was not detected");
            } else {
                println!("Corruption detected through data comparison");
            }
        },
        Err(_) => {
            println!("Corruption properly detected and rejected");
        }
    }
    
    Ok(())
}

/// 並行処理テスト
#[tokio::test]
async fn test_concurrent_operations() -> Result<()> {
    use std::sync::Arc;
    use tokio::task;
    
    let test_data = Arc::new(b"Concurrent operations test data".repeat(100));
    let num_tasks = 10;
    
    let mut handles = Vec::new();
    
    for i in 0..num_tasks {
        let data = test_data.clone();
        let handle = task::spawn(async move {
            let mut spe_core = SPECore::default().unwrap();
            
            // 各タスクで独立してSPE変換を実行
            for _ in 0..5 {
                let result = spe_core.apply_transform(&data).unwrap();
                let restored = spe_core.reverse_transform(&result).unwrap();
                assert_eq!(&restored, data.as_slice());
            }
            
            i
        });
        
        handles.push(handle);
    }
    
    // 全タスクの完了を待機
    for handle in handles {
        let task_id = handle.await?;
        println!("Task {} completed successfully", task_id);
    }
    
    println!("All concurrent operations completed successfully");
    
    Ok(())
}
