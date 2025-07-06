use anyhow::Result;
use aes_gcm::{Aes256Gcm, KeyInit, Nonce};
use aes_gcm::aead::{Aead, generic_array::GenericArray};
use pbkdf2::pbkdf2_hmac;
use sha2::Sha256;
use rand::{thread_rng, RngCore};

pub struct Encryptor {
    key: [u8; 32],
}

impl Encryptor {
    pub fn new(password: &str) -> Result<Self> {
        let mut key = [0u8; 32];
        let salt = b"nxzip_salt_2024"; // 実際の実装では動的ソルト
        
        pbkdf2_hmac::<Sha256>(
            password.as_bytes(),
            salt,
            100_000, // イテレーション回数
            &mut key,
        );
        
        Ok(Self { key })
    }
    
    pub fn encrypt(&self, data: &[u8]) -> Result<Vec<u8>> {
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
}
