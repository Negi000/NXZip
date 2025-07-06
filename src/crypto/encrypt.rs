use anyhow::Result;
use aes_gcm::{Aes256Gcm, KeyInit, Nonce};
use aes_gcm::aead::{Aead, generic_array::GenericArray};
use chacha20poly1305::{XChaCha20Poly1305, XNonce};
use pbkdf2::pbkdf2_hmac;
use sha2::Sha256;
use rand::{thread_rng, RngCore};

#[derive(Debug, Clone, Copy)]
pub enum EncryptionAlgorithm {
    AesGcm,
    XChaCha20Poly1305,
}

impl std::fmt::Display for EncryptionAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EncryptionAlgorithm::AesGcm => write!(f, "AES-256-GCM"),
            EncryptionAlgorithm::XChaCha20Poly1305 => write!(f, "XChaCha20-Poly1305"),
        }
    }
}

impl Default for EncryptionAlgorithm {
    fn default() -> Self {
        EncryptionAlgorithm::AesGcm
    }
}

pub struct Encryptor {
    key: [u8; 32],
    algorithm: EncryptionAlgorithm,
}

impl Encryptor {
    pub fn new(password: &str) -> Result<Self> {
        Self::with_algorithm(password, EncryptionAlgorithm::default())
    }
    
    pub fn with_algorithm(password: &str, algorithm: EncryptionAlgorithm) -> Result<Self> {
        let mut key = [0u8; 32];
        let salt = b"nxzip_salt_2024"; // 実際の実装では動的ソルト
        
        pbkdf2_hmac::<Sha256>(
            password.as_bytes(),
            salt,
            100_000, // イテレーション回数
            &mut key,
        );
        
        Ok(Self { key, algorithm })
    }
    
    pub fn from_key(key: &[u8; 32], algorithm: EncryptionAlgorithm) -> Result<Self> {
        Ok(Self {
            key: *key,
            algorithm,
        })
    }
    
    pub fn encrypt(&self, data: &[u8]) -> Result<Vec<u8>> {
        match self.algorithm {
            EncryptionAlgorithm::AesGcm => self.encrypt_aes_gcm(data),
            EncryptionAlgorithm::XChaCha20Poly1305 => self.encrypt_xchacha20(data),
        }
    }
    
    fn encrypt_aes_gcm(&self, data: &[u8]) -> Result<Vec<u8>> {
        let cipher = Aes256Gcm::new(GenericArray::from_slice(&self.key));
        
        // ランダムなnonce生成
        let mut nonce_bytes = [0u8; 12];
        thread_rng().fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);
        
        // 暗号化
        let ciphertext = cipher.encrypt(nonce, data)
            .map_err(|e| anyhow::anyhow!("暗号化エラー: {}", e))?;
        
        // nonce + ciphertext の形式で結合
        let mut result = Vec::with_capacity(12 + ciphertext.len());
        result.extend_from_slice(&nonce_bytes);
        result.extend_from_slice(&ciphertext);
        
        Ok(result)
    }
    
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
}
