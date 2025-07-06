use anyhow::Result;
use aes_gcm::{Aes256Gcm, KeyInit, Nonce};
use aes_gcm::aead::{Aead, generic_array::GenericArray};
use pbkdf2::pbkdf2_hmac;
use sha2::Sha256;

pub struct Decryptor {
    key: [u8; 32],
}

impl Decryptor {
    pub fn new(password: &str) -> Result<Self> {
        let mut key = [0u8; 32];
        let salt = b"nxzip_salt_2024"; // Encryptorと同じソルト
        
        pbkdf2_hmac::<Sha256>(
            password.as_bytes(),
            salt,
            100_000,
            &mut key,
        );
        
        Ok(Self { key })
    }
    
    pub fn decrypt(&self, encrypted_data: &[u8]) -> Result<Vec<u8>> {
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
}
