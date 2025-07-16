use anyhow::Result;
use aes_gcm::{Aes256Gcm, KeyInit, Nonce};
use aes_gcm::aead::{Aead, generic_array::GenericArray};
use chacha20poly1305::{XChaCha20Poly1305, XNonce};
use pbkdf2::pbkdf2_hmac;
use sha2::Sha256;
use rand::{thread_rng, RngCore};
use argon2::{Argon2, PasswordHash, PasswordHasher, PasswordVerifier, Salt};
use argon2::password_hash::{rand_core::OsRng, SaltString};
use serde::{Deserialize, Serialize};

use crate::engine::spe_core::{SPECore, SPEParameters, SPETransformResult, StructureLevel};

/// 暗号化アルゴリズム
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EncryptionAlgorithm {
    /// AES-256-GCM
    AesGcm,
    /// XChaCha20-Poly1305
    XChaCha20Poly1305,
}

/// キー導出機能
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum KeyDerivationFunction {
    /// PBKDF2-SHA256
    Pbkdf2,
    /// Argon2id
    Argon2id,
}

/// キー導出設定
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyDerivationConfig {
    /// KDF種類
    pub kdf: KeyDerivationFunction,
    /// ソルト
    pub salt: Vec<u8>,
    /// イテレーション回数 (PBKDF2) または時間コスト (Argon2)
    pub iterations: u32,
    /// メモリコスト (Argon2のみ)
    pub memory_cost: Option<u32>,
    /// 並列度 (Argon2のみ)
    pub parallelism: Option<u32>,
}

/// 統合暗号化器 (SPE + 暗号化)
#[derive(Debug)]
pub struct IntegratedEncryptor {
    /// 暗号化キー
    key: [u8; 32],
    /// 暗号化アルゴリズム
    algorithm: EncryptionAlgorithm,
    /// キー導出設定
    kdf_config: KeyDerivationConfig,
    /// SPEコア
    spe_core: Option<SPECore>,
    /// SPE有効フラグ
    spe_enabled: bool,
}

/// 暗号化結果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionResult {
    /// 暗号化されたデータ
    pub encrypted_data: Vec<u8>,
    /// 暗号化メタデータ
    pub metadata: EncryptionMetadata,
    /// SPE変換結果（SPE有効時のみ）
    pub spe_result: Option<SPETransformResult>,
}

/// 暗号化メタデータ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionMetadata {
    /// 暗号化アルゴリズム
    pub algorithm: EncryptionAlgorithm,
    /// キー導出設定
    pub kdf_config: KeyDerivationConfig,
    /// SPE有効フラグ
    pub spe_enabled: bool,
    /// SPEパラメータ（SPE有効時のみ）
    pub spe_params: Option<SPEParameters>,
    /// 元のデータサイズ
    pub original_size: usize,
    /// 完全性チェックサム
    pub integrity_checksum: [u8; 32],
}

impl Default for EncryptionAlgorithm {
    fn default() -> Self {
        EncryptionAlgorithm::AesGcm
    }
}

impl std::fmt::Display for EncryptionAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EncryptionAlgorithm::AesGcm => write!(f, "AES-256-GCM"),
            EncryptionAlgorithm::XChaCha20Poly1305 => write!(f, "XChaCha20-Poly1305"),
        }
    }
}

impl Default for KeyDerivationFunction {
    fn default() -> Self {
        KeyDerivationFunction::Pbkdf2
    }
}

impl Default for KeyDerivationConfig {
    fn default() -> Self {
        let mut salt = vec![0u8; 16];
        thread_rng().fill_bytes(&mut salt);
        
        Self {
            kdf: KeyDerivationFunction::default(),
            salt,
            iterations: 100_000,
            memory_cost: Some(65536), // 64MB
            parallelism: Some(4),
        }
    }
}

impl IntegratedEncryptor {
    /// パスワードから暗号化器を作成
    pub fn new(password: &str) -> Result<Self> {
        Self::with_config(password, EncryptionAlgorithm::default(), KeyDerivationConfig::default(), false)
    }
    
    /// 設定を指定して暗号化器を作成
    pub fn with_config(
        password: &str,
        algorithm: EncryptionAlgorithm,
        kdf_config: KeyDerivationConfig,
        spe_enabled: bool,
    ) -> Result<Self> {
        let key = Self::derive_key(password, &kdf_config)?;
        
        let spe_core = if spe_enabled {
            Some(SPECore::from_key(&key)?)
        } else {
            None
        };
        
        Ok(Self {
            key,
            algorithm,
            kdf_config,
            spe_core,
            spe_enabled,
        })
    }
    
    /// キーから暗号化器を作成
    pub fn from_key(
        key: &[u8; 32],
        algorithm: EncryptionAlgorithm,
        kdf_config: KeyDerivationConfig,
        spe_enabled: bool,
    ) -> Result<Self> {
        let spe_core = if spe_enabled {
            Some(SPECore::from_key(key)?)
        } else {
            None
        };
        
        Ok(Self {
            key: *key,
            algorithm,
            kdf_config,
            spe_core,
            spe_enabled,
        })
    }
    
    /// SPE強度を設定
    pub fn set_spe_level(&mut self, level: StructureLevel) -> Result<()> {
        if let Some(ref mut spe_core) = self.spe_core {
            let mut params = SPEParameters::default();
            params.structure_level = level;
            *spe_core = SPECore::new(params)?;
        }
        Ok(())
    }
    
    /// 統合暗号化を実行
    pub fn encrypt(&mut self, data: &[u8]) -> Result<EncryptionResult> {
        // 完全性チェックサムを計算
        let integrity_checksum = self.calculate_integrity_checksum(data);
        
        // SPE変換（有効時のみ）
        let (processed_data, spe_result) = if self.spe_enabled {
            if let Some(ref mut spe_core) = self.spe_core {
                let spe_result = spe_core.apply_transform(data)?;
                let processed_data = spe_result.transformed_data.clone();
                (processed_data, Some(spe_result))
            } else {
                return Err(anyhow::anyhow!("SPEが有効ですがSPEコアが初期化されていません"));
            }
        } else {
            (data.to_vec(), None)
        };
        
        // 暗号化
        let encrypted_data = match self.algorithm {
            EncryptionAlgorithm::AesGcm => self.encrypt_aes_gcm(&processed_data)?,
            EncryptionAlgorithm::XChaCha20Poly1305 => self.encrypt_xchacha20(&processed_data)?,
        };
        
        // メタデータの構築
        let metadata = EncryptionMetadata {
            algorithm: self.algorithm,
            kdf_config: self.kdf_config.clone(),
            spe_enabled: self.spe_enabled,
            spe_params: spe_result.as_ref().map(|r| r.metadata.parameters.clone()),
            original_size: data.len(),
            integrity_checksum,
        };
        
        Ok(EncryptionResult {
            encrypted_data,
            metadata,
            spe_result,
        })
    }
    
    /// 統合復号化を実行
    pub fn decrypt(&mut self, result: &EncryptionResult) -> Result<Vec<u8>> {
        // 暗号化アルゴリズムの検証
        if result.metadata.algorithm as u8 != self.algorithm as u8 {
            return Err(anyhow::anyhow!("暗号化アルゴリズムが一致しません"));
        }
        
        // 復号化
        let decrypted_data = match self.algorithm {
            EncryptionAlgorithm::AesGcm => self.decrypt_aes_gcm(&result.encrypted_data)?,
            EncryptionAlgorithm::XChaCha20Poly1305 => self.decrypt_xchacha20(&result.encrypted_data)?,
        };
        
        // SPE逆変換（有効時のみ）
        let restored_data = if result.metadata.spe_enabled {
            if let Some(ref spe_result) = result.spe_result {
                if let Some(ref mut spe_core) = self.spe_core {
                    // SPE変換結果を復元
                    let mut temp_result = spe_result.clone();
                    temp_result.transformed_data = decrypted_data;
                    spe_core.reverse_transform(&temp_result)?
                } else {
                    return Err(anyhow::anyhow!("SPE結果がありますがSPEコアが初期化されていません"));
                }
            } else {
                return Err(anyhow::anyhow!("SPEが有効ですがSPE結果がありません"));
            }
        } else {
            decrypted_data
        };
        
        // サイズ検証
        if restored_data.len() != result.metadata.original_size {
            return Err(anyhow::anyhow!("復元データサイズが不正: {} != {}", 
                restored_data.len(), result.metadata.original_size));
        }
        
        // 完全性検証
        let calculated_checksum = self.calculate_integrity_checksum(&restored_data);
        if calculated_checksum != result.metadata.integrity_checksum {
            return Err(anyhow::anyhow!("データ完全性チェックに失敗しました"));
        }
        
        Ok(restored_data)
    }
    
    /// キー導出
    fn derive_key(password: &str, config: &KeyDerivationConfig) -> Result<[u8; 32]> {
        let mut key = [0u8; 32];
        
        match config.kdf {
            KeyDerivationFunction::Pbkdf2 => {
                pbkdf2_hmac::<Sha256>(
                    password.as_bytes(),
                    &config.salt,
                    config.iterations,
                    &mut key,
                );
            },
            KeyDerivationFunction::Argon2id => {
                let argon2 = Argon2::default();
                let salt = Salt::from_b64(&base64::encode(&config.salt))
                    .map_err(|e| anyhow::anyhow!("Argon2ソルトエラー: {}", e))?;
                
                let password_hash = argon2.hash_password(password.as_bytes(), &salt)
                    .map_err(|e| anyhow::anyhow!("Argon2ハッシュエラー: {}", e))?;
                
                key.copy_from_slice(&password_hash.hash.unwrap().as_bytes()[..32]);
            },
        }
        
        Ok(key)
    }
    
    /// AES-GCM暗号化
    fn encrypt_aes_gcm(&self, data: &[u8]) -> Result<Vec<u8>> {
        let cipher = Aes256Gcm::new(GenericArray::from_slice(&self.key));
        
        // ランダムなnonce生成
        let mut nonce_bytes = [0u8; 12];
        thread_rng().fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);
        
        // 暗号化
        let ciphertext = cipher.encrypt(nonce, data)
            .map_err(|e| anyhow::anyhow!("AES-GCM暗号化エラー: {}", e))?;
        
        // nonce + ciphertext の形式で結合
        let mut result = Vec::with_capacity(12 + ciphertext.len());
        result.extend_from_slice(&nonce_bytes);
        result.extend_from_slice(&ciphertext);
        
        Ok(result)
    }
    
    /// AES-GCM復号化
    fn decrypt_aes_gcm(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.len() < 12 {
            return Err(anyhow::anyhow!("AES-GCMデータが小さすぎます"));
        }
        
        let cipher = Aes256Gcm::new(GenericArray::from_slice(&self.key));
        
        // nonceと暗号文を分離
        let (nonce_bytes, ciphertext) = data.split_at(12);
        let nonce = Nonce::from_slice(nonce_bytes);
        
        // 復号化
        let plaintext = cipher.decrypt(nonce, ciphertext)
            .map_err(|e| anyhow::anyhow!("AES-GCM復号化エラー: {}", e))?;
        
        Ok(plaintext)
    }
    
    /// XChaCha20-Poly1305暗号化
    fn encrypt_xchacha20(&self, data: &[u8]) -> Result<Vec<u8>> {
        let cipher = XChaCha20Poly1305::new(GenericArray::from_slice(&self.key));
        
        // ランダムなnonce生成 (XChaCha20は24バイトnonce)
        let mut nonce_bytes = [0u8; 24];
        thread_rng().fill_bytes(&mut nonce_bytes);
        let nonce = XNonce::from_slice(&nonce_bytes);
        
        // 暗号化
        let ciphertext = cipher.encrypt(nonce, data)
            .map_err(|e| anyhow::anyhow!("XChaCha20暗号化エラー: {}", e))?;
        
        // nonce + ciphertext の形式で結合
        let mut result = Vec::with_capacity(24 + ciphertext.len());
        result.extend_from_slice(&nonce_bytes);
        result.extend_from_slice(&ciphertext);
        
        Ok(result)
    }
    
    /// XChaCha20-Poly1305復号化
    fn decrypt_xchacha20(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.len() < 24 {
            return Err(anyhow::anyhow!("XChaCha20データが小さすぎます"));
        }
        
        let cipher = XChaCha20Poly1305::new(GenericArray::from_slice(&self.key));
        
        // nonceと暗号文を分離
        let (nonce_bytes, ciphertext) = data.split_at(24);
        let nonce = XNonce::from_slice(nonce_bytes);
        
        // 復号化
        let plaintext = cipher.decrypt(nonce, ciphertext)
            .map_err(|e| anyhow::anyhow!("XChaCha20復号化エラー: {}", e))?;
        
        Ok(plaintext)
    }
    
    /// 完全性チェックサムを計算
    fn calculate_integrity_checksum(&self, data: &[u8]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hasher.update(&self.key);
        hasher.update(b"NXZIP_INTEGRITY_CHECK");
        hasher.finalize().into()
    }
}

/// 復号化器
#[derive(Debug)]
pub struct IntegratedDecryptor {
    encryptor: IntegratedEncryptor,
}

impl IntegratedDecryptor {
    /// パスワードから復号化器を作成
    pub fn new(password: &str) -> Result<Self> {
        Ok(Self {
            encryptor: IntegratedEncryptor::new(password)?,
        })
    }
    
    /// メタデータを指定して復号化器を作成
    pub fn with_metadata(password: &str, metadata: &EncryptionMetadata) -> Result<Self> {
        let mut encryptor = IntegratedEncryptor::with_config(
            password,
            metadata.algorithm,
            metadata.kdf_config.clone(),
            metadata.spe_enabled,
        )?;
        
        // SPEパラメータがある場合は設定
        if let Some(ref spe_params) = metadata.spe_params {
            if let Some(ref mut spe_core) = encryptor.spe_core {
                *spe_core = SPECore::new(spe_params.clone())?;
            }
        }
        
        Ok(Self { encryptor })
    }
    
    /// 復号化を実行
    pub fn decrypt(&mut self, result: &EncryptionResult) -> Result<Vec<u8>> {
        self.encryptor.decrypt(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_integrated_encryption_without_spe() -> Result<()> {
        let password = "test_password_123";
        let mut encryptor = IntegratedEncryptor::new(password)?;
        
        let original_data = b"Hello, NXZip Integrated Encryption!";
        
        // 暗号化
        let result = encryptor.encrypt(original_data)?;
        
        // 暗号化されたデータが元データと異なることを確認
        assert_ne!(&result.encrypted_data, original_data);
        
        // 復号化
        let decrypted = encryptor.decrypt(&result)?;
        
        // 完全に復元されることを確認
        assert_eq!(&decrypted, original_data);
        
        Ok(())
    }
    
    #[test]
    fn test_integrated_encryption_with_spe() -> Result<()> {
        let password = "spe_test_password_456";
        let mut encryptor = IntegratedEncryptor::with_config(
            password,
            EncryptionAlgorithm::XChaCha20Poly1305,
            KeyDerivationConfig::default(),
            true, // SPE有効
        )?;
        
        let original_data = b"This is a test of integrated SPE + encryption system with various data patterns and structures.";
        
        // 暗号化
        let result = encryptor.encrypt(original_data)?;
        
        // SPEが適用されていることを確認
        assert!(result.metadata.spe_enabled);
        assert!(result.spe_result.is_some());
        
        // 復号化
        let decrypted = encryptor.decrypt(&result)?;
        
        // 完全に復元されることを確認
        assert_eq!(&decrypted, original_data);
        
        Ok(())
    }
    
    #[test]
    fn test_different_encryption_algorithms() -> Result<()> {
        let password = "algorithm_test_789";
        let test_data = b"Algorithm comparison test data";
        
        for algorithm in [EncryptionAlgorithm::AesGcm, EncryptionAlgorithm::XChaCha20Poly1305] {
            let mut encryptor = IntegratedEncryptor::with_config(
                password,
                algorithm,
                KeyDerivationConfig::default(),
                false,
            )?;
            
            let result = encryptor.encrypt(test_data)?;
            let decrypted = encryptor.decrypt(&result)?;
            
            assert_eq!(&decrypted, test_data, "Failed for algorithm {:?}", algorithm);
        }
        
        Ok(())
    }
    
    #[test]
    fn test_spe_structure_levels() -> Result<()> {
        let password = "structure_level_test";
        let test_data = b"Structure level testing with comprehensive data patterns and various byte sequences.";
        
        for level in [StructureLevel::Basic, StructureLevel::Extended, StructureLevel::Maximum] {
            let mut encryptor = IntegratedEncryptor::with_config(
                password,
                EncryptionAlgorithm::AesGcm,
                KeyDerivationConfig::default(),
                true,
            )?;
            
            encryptor.set_spe_level(level)?;
            
            let result = encryptor.encrypt(test_data)?;
            let decrypted = encryptor.decrypt(&result)?;
            
            assert_eq!(&decrypted, test_data, "Failed for SPE level {:?}", level);
        }
        
        Ok(())
    }
    
    #[test]
    fn test_integrity_verification() -> Result<()> {
        let password = "integrity_test_password";
        let mut encryptor = IntegratedEncryptor::new(password)?;
        
        let original_data = b"Integrity verification test data";
        let mut result = encryptor.encrypt(original_data)?;
        
        // データを破損
        if let Some(byte) = result.encrypted_data.get_mut(10) {
            *byte = byte.wrapping_add(1);
        }
        
        // 破損したデータの復号化は失敗するはず
        let decrypt_result = encryptor.decrypt(&result);
        assert!(decrypt_result.is_err());
        
        Ok(())
    }
}
