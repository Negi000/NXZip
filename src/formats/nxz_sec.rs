use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::fs;
use crate::crypto::EncryptionAlgorithm;
use super::nxz::NxzFile;

/// .nxz.secファイル形式の暗号化パラメータ
#[derive(Debug, Serialize, Deserialize)]
pub struct NxzSecParams {
    /// 暗号化方式
    pub cipher_type: u8,
    /// 鍵導出方式 (PBKDF2=0, Argon2=1)
    pub kdf_type: u8,
    /// KDF反復回数
    pub kdf_iterations: u32,
    /// ソルト値 (16バイト)
    pub salt: [u8; 16],
    /// ナンス値 (12バイト)
    pub nonce: [u8; 12],
    /// 予約領域
    pub reserved: [u8; 2],
}

impl NxzSecParams {
    const SIZE: usize = 34; // 1+1+4+16+12 = 34バイト (予約領域を除く)
    
    pub fn new(cipher_type: EncryptionAlgorithm, kdf_type: KdfType) -> Self {
        use rand::RngCore;
        let mut rng = rand::thread_rng();
        
        let cipher_byte = match cipher_type {
            EncryptionAlgorithm::AesGcm => 0,
            EncryptionAlgorithm::XChaCha20Poly1305 => 1,
        };
        
        let kdf_byte = match kdf_type {
            KdfType::Pbkdf2 => 0,
            KdfType::Argon2 => 1,
        };
        
        let mut salt = [0u8; 16];
        let mut nonce = [0u8; 12];
        rng.fill_bytes(&mut salt);
        rng.fill_bytes(&mut nonce);
        
        Self {
            cipher_type: cipher_byte,
            kdf_type: kdf_byte,
            kdf_iterations: 100_000, // デフォルト反復回数
            salt,
            nonce,
            reserved: [0; 2],
        }
    }
    
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::with_capacity(Self::SIZE);
        
        bytes.push(self.cipher_type);
        bytes.push(self.kdf_type);
        bytes.extend_from_slice(&self.kdf_iterations.to_le_bytes());
        bytes.extend_from_slice(&self.salt);
        bytes.extend_from_slice(&self.nonce);
        // 予約領域は省略（34バイトに収める）
        
        Ok(bytes)
    }
    
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < Self::SIZE {
            anyhow::bail!("暗号化パラメータのサイズが不正です: {} < {}", bytes.len(), Self::SIZE);
        }
        
        let cipher_type = bytes[0];
        let kdf_type = bytes[1];
        let kdf_iterations = u32::from_le_bytes(bytes[2..6].try_into()?);
        let salt: [u8; 16] = bytes[6..22].try_into()?;
        let nonce: [u8; 12] = bytes[22..34].try_into()?;
        let reserved = [0; 2]; // デフォルト値
        
        Ok(Self {
            cipher_type,
            kdf_type,
            kdf_iterations,
            salt,
            nonce,
            reserved,
        })
    }
    
    pub fn get_cipher_type(&self) -> EncryptionAlgorithm {
        match self.cipher_type {
            0 => EncryptionAlgorithm::AesGcm,
            1 => EncryptionAlgorithm::XChaCha20Poly1305,
            _ => EncryptionAlgorithm::AesGcm, // デフォルト
        }
    }
    
    pub fn get_kdf_type(&self) -> KdfType {
        match self.kdf_type {
            0 => KdfType::Pbkdf2,
            1 => KdfType::Argon2,
            _ => KdfType::Pbkdf2, // デフォルト
        }
    }
}

/// 鍵導出方式
#[derive(Debug, Clone, Copy)]
pub enum KdfType {
    Pbkdf2,
    Argon2,
}

/// .nxz.secファイル全体の構造
pub struct NxzSecFile {
    params: NxzSecParams,
    encrypted_nxz_data: Vec<u8>,
}

impl NxzSecFile {
    pub fn new(
        nxz_file: &NxzFile,
        password: &str,
        cipher_type: EncryptionAlgorithm,
        kdf_type: KdfType,
    ) -> Result<Self> {
        let params = NxzSecParams::new(cipher_type, kdf_type);
        
        // NXZファイルをバイナリに変換
        let nxz_data = nxz_file.to_bytes()?;
        
        // 暗号化
        let encrypted_nxz_data = Self::encrypt_data(&nxz_data, password, &params)?;
        
        Ok(Self {
            params,
            encrypted_nxz_data,
        })
    }
    
    pub async fn write_to_file(&self, path: &str) -> Result<()> {
        let params_bytes = self.params.to_bytes()?;
        
        // ファイル構造: [暗号化パラメータ] + [暗号化されたNXZデータ]
        let mut file_data = Vec::new();
        file_data.extend_from_slice(&params_bytes);
        file_data.extend_from_slice(&self.encrypted_nxz_data);
        
        fs::write(path, file_data).await?;
        Ok(())
    }
    
    pub async fn read_from_file(path: &str) -> Result<Self> {
        let file_data = fs::read(path).await?;
        
        if file_data.len() < NxzSecParams::SIZE {
            anyhow::bail!(".nxz.secファイルのサイズが不正です");
        }
        
        // 暗号化パラメータを読み込み
        let params = NxzSecParams::from_bytes(&file_data[0..NxzSecParams::SIZE])?;
        
        // 暗号化されたデータ部を取得
        let encrypted_nxz_data = file_data[NxzSecParams::SIZE..].to_vec();
        
        Ok(Self {
            params,
            encrypted_nxz_data,
        })
    }
    
    pub fn decrypt_nxz(&self, password: &str) -> Result<NxzFile> {
        // 復号化
        let nxz_data = Self::decrypt_data(&self.encrypted_nxz_data, password, &self.params)?;
        
        // NXZファイルとして再構築
        NxzFile::from_bytes(&nxz_data)
    }
    
    fn encrypt_data(data: &[u8], password: &str, params: &NxzSecParams) -> Result<Vec<u8>> {
        use crate::crypto::Encryptor;
        
        // 鍵導出
        let key = Self::derive_key(password, &params.salt, params.kdf_iterations, params.get_kdf_type())?;
        
        // 暗号化器を作成
        let encryptor = Encryptor::from_key(&key, params.get_cipher_type())?;
        encryptor.encrypt(data)
    }
    
    fn decrypt_data(encrypted_data: &[u8], password: &str, params: &NxzSecParams) -> Result<Vec<u8>> {
        use crate::crypto::Decryptor;
        
        // 鍵導出
        let key = Self::derive_key(password, &params.salt, params.kdf_iterations, params.get_kdf_type())?;
        
        // 復号化器を作成
        let decryptor = Decryptor::from_key(&key, params.get_cipher_type())?;
        decryptor.decrypt(encrypted_data)
    }
    
    fn derive_key(password: &str, salt: &[u8], iterations: u32, kdf_type: KdfType) -> Result<[u8; 32]> {
        match kdf_type {
            KdfType::Pbkdf2 => {
                use pbkdf2::pbkdf2_hmac;
                use sha2::Sha256;
                
                let mut key = [0u8; 32];
                pbkdf2_hmac::<Sha256>(password.as_bytes(), salt, iterations, &mut key);
                Ok(key)
            }
            KdfType::Argon2 => {
                use argon2::{Argon2, Algorithm, Version, Params};
                use argon2::password_hash::{PasswordHash, PasswordHasher, SaltString};
                
                let params = Params::new(65536, 3, 4, Some(32))
                    .map_err(|e| anyhow::anyhow!("Argon2パラメータエラー: {}", e))?;
                let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
                
                // Base64エンコードされたソルト文字列を作成
                let salt_string = SaltString::encode_b64(salt)
                    .map_err(|e| anyhow::anyhow!("ソルトエンコードエラー: {}", e))?;
                
                let hash = argon2.hash_password(password.as_bytes(), &salt_string)
                    .map_err(|e| anyhow::anyhow!("Argon2ハッシュエラー: {}", e))?;
                
                // ハッシュからバイト列を取得
                let hash_string = hash.to_string();
                let parsed_hash = PasswordHash::new(&hash_string)
                    .map_err(|e| anyhow::anyhow!("ハッシュパースエラー: {}", e))?;
                
                if let Some(hash_bytes) = parsed_hash.hash {
                    if hash_bytes.len() != 32 {
                        anyhow::bail!("Argon2ハッシュのサイズが不正です: {}", hash_bytes.len());
                    }
                    let mut key = [0u8; 32];
                    key.copy_from_slice(hash_bytes.as_bytes());
                    Ok(key)
                } else {
                    anyhow::bail!("Argon2ハッシュの取得に失敗しました");
                }
            }
        }
    }
    
    pub fn params(&self) -> &NxzSecParams {
        &self.params
    }
    
    pub fn is_encrypted(&self) -> bool {
        true // .nxz.secは常に暗号化されている
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    
    #[tokio::test]
    async fn test_nxz_sec_roundtrip() {
        // テストデータでNXZファイルを作成
        let test_data = b"Hello, NXZip Secure! This is test data for .nxz.sec format.";
        let nxz_file = NxzFile::new(
            test_data,
            test_data.len() as u64,
            CompressionAlgorithm::Zstd,
            false,
            None,
            6,
        ).unwrap();
        
        // .nxz.secファイルを作成
        let password = "secure_password_123";
        let nxz_sec = NxzSecFile::new(
            &nxz_file,
            password,
            EncryptionAlgorithm::AesGcm,
            KdfType::Pbkdf2,
        ).unwrap();
        
        // 一時ファイルに書き込み
        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().to_str().unwrap();
        
        nxz_sec.write_to_file(temp_path).await.unwrap();
        
        // 読み込み・復号化
        let loaded_nxz_sec = NxzSecFile::read_from_file(temp_path).await.unwrap();
        let decrypted_nxz = loaded_nxz_sec.decrypt_nxz(password).unwrap();
        
        // 検証
        assert_eq!(decrypted_nxz.data(), test_data);
        assert_eq!(decrypted_nxz.original_size(), test_data.len() as u64);
    }
}
