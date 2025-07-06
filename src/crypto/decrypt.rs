use anyhow::Result;
use aes_gcm::{Aes256Gcm, KeyInit, Nonce};
use aes_gcm::aead::{Aead, generic_array::GenericArray};
use chacha20poly1305::{XChaCha20Poly1305, XNonce};
use pbkdf2::pbkdf2_hmac;
use sha2::Sha256;

use super::encrypt::EncryptionAlgorithm;

pub struct Decryptor {
    key: [u8; 32],
    algorithm: EncryptionAlgorithm,
}

impl Decryptor {
    pub fn new(password: &str) -> Result<Self> {
        Self::with_algorithm(password, EncryptionAlgorithm::default())
    }
    
    pub fn with_algorithm(password: &str, algorithm: EncryptionAlgorithm) -> Result<Self> {
        let mut key = [0u8; 32];
        let salt = b"nxzip_salt_2024"; // Encryptorと同じソルト
        
        pbkdf2_hmac::<Sha256>(
            password.as_bytes(),
            salt,
            100_000,
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
    
    pub fn decrypt(&self, encrypted_data: &[u8]) -> Result<Vec<u8>> {
        match self.algorithm {
            EncryptionAlgorithm::AesGcm => self.decrypt_aes_gcm(encrypted_data),
            EncryptionAlgorithm::XChaCha20Poly1305 => self.decrypt_xchacha20(encrypted_data),
        }
    }
    
    fn decrypt_aes_gcm(&self, encrypted_data: &[u8]) -> Result<Vec<u8>> {
        if encrypted_data.len() < 12 {
            anyhow::bail!("暗号化データが短すぎます");
        }
        
        let cipher = Aes256Gcm::new(GenericArray::from_slice(&self.key));
        
        // nonce と ciphertext を分離
        let nonce = Nonce::from_slice(&encrypted_data[..12]);
        let ciphertext = &encrypted_data[12..];
        
        // 復号
        let plaintext = cipher.decrypt(nonce, ciphertext)
            .map_err(|e| anyhow::anyhow!("復号エラー: パスワードが間違っているか、データが破損している可能性があります - {}", e))?;
        
        Ok(plaintext)
    }
    
    fn decrypt_xchacha20(&self, encrypted_data: &[u8]) -> Result<Vec<u8>> {
        if encrypted_data.len() < 24 {
            anyhow::bail!("XChaCha20暗号化データが短すぎます");
        }
        
        let cipher = XChaCha20Poly1305::new(GenericArray::from_slice(&self.key));
        
        // nonce と ciphertext を分離 (XChaCha20は24バイトnonce)
        let nonce = XNonce::from_slice(&encrypted_data[..24]);
        let ciphertext = &encrypted_data[24..];
        
        // 復号
        let plaintext = cipher.decrypt(nonce, ciphertext)
            .map_err(|e| anyhow::anyhow!("XChaCha20復号エラー: パスワードが間違っているか、データが破損している可能性があります - {}", e))?;
        
        Ok(plaintext)
    }
}
