use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokio::fs;
use crate::engine::CompressionAlgorithm;
use crate::crypto::EncryptionAlgorithm;
use crate::utils::metadata::NxzMetadata;

/// NXZファイル形式のヘッダ構造
#[derive(Debug, Serialize, Deserialize)]
pub struct NxzHeader {
    /// ファイルシグネチャ "NXZ\0"
    pub signature: [u8; 4],
    /// バージョン情報
    pub version: u16,
    /// ヘッダサイズ
    pub header_size: u32,
    /// ブロックサイズ
    pub block_size: u32,
    /// 元ファイルサイズ
    pub original_size: u64,
    /// 圧縮アルゴリズム
    pub compression_algorithm: u8,
    /// 暗号化フラグ
    pub encryption_flag: u8,
    /// 暗号化アルゴリズム
    pub encryption_algorithm: u8,
    /// 予約領域
    pub reserved: [u8; 15],
    /// メタデータサイズ
    pub metadata_size: u32,
}

impl NxzHeader {
    const SIGNATURE: [u8; 4] = *b"NXZ\0";
    const VERSION: u16 = 0x0100;
    const HEADER_SIZE: u32 = 64; // 固定ヘッダサイズ
    
    pub fn new(
        original_size: u64,
        compression_algorithm: CompressionAlgorithm,
        is_encrypted: bool,
        encryption_algorithm: Option<EncryptionAlgorithm>,
        metadata_size: u32,
    ) -> Self {
        let compression_algo_byte = match compression_algorithm {
            CompressionAlgorithm::Zstd => 0,
            CompressionAlgorithm::Lzma2 => 1,
            CompressionAlgorithm::Auto => 2,
        };
        
        let encryption_algo_byte = if let Some(algo) = encryption_algorithm {
            match algo {
                EncryptionAlgorithm::AesGcm => 0,
                EncryptionAlgorithm::XChaCha20Poly1305 => 1,
            }
        } else {
            0
        };
        
        Self {
            signature: Self::SIGNATURE,
            version: Self::VERSION,
            header_size: Self::HEADER_SIZE,
            block_size: 256 * 1024, // 256KB
            original_size,
            compression_algorithm: compression_algo_byte,
            encryption_flag: if is_encrypted { 1 } else { 0 },
            encryption_algorithm: encryption_algo_byte,
            reserved: [0; 15],
            metadata_size,
        }
    }
    
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::with_capacity(64);
        
        // 手動でバイナリ形式に変換
        bytes.extend_from_slice(&self.signature);
        bytes.extend_from_slice(&self.version.to_le_bytes());
        bytes.extend_from_slice(&self.header_size.to_le_bytes());
        bytes.extend_from_slice(&self.block_size.to_le_bytes());
        bytes.extend_from_slice(&self.original_size.to_le_bytes());
        bytes.push(self.compression_algorithm);
        bytes.push(self.encryption_flag);
        bytes.push(self.encryption_algorithm);
        bytes.extend_from_slice(&self.reserved);
        bytes.extend_from_slice(&self.metadata_size.to_le_bytes());
        
        // 64バイトまでパディング
        while bytes.len() < 64 {
            bytes.push(0);
        }
        
        Ok(bytes)
    }
    
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 64 {
            anyhow::bail!("ヘッダサイズが不正です");
        }
        
        // 手動でバイナリ形式から変換
        let signature: [u8; 4] = bytes[0..4].try_into()?;
        if signature != Self::SIGNATURE {
            anyhow::bail!("NXZファイルではありません");
        }
        
        let version = u16::from_le_bytes(bytes[4..6].try_into()?);
        let header_size = u32::from_le_bytes(bytes[6..10].try_into()?);
        let block_size = u32::from_le_bytes(bytes[10..14].try_into()?);
        let original_size = u64::from_le_bytes(bytes[14..22].try_into()?);
        let compression_algorithm = bytes[22];
        let encryption_flag = bytes[23];
        let encryption_algorithm = bytes[24];
        let reserved: [u8; 15] = bytes[25..40].try_into()?;
        let metadata_size = u32::from_le_bytes(bytes[40..44].try_into()?);
        
        Ok(Self {
            signature,
            version,
            header_size,
            block_size,
            original_size,
            compression_algorithm,
            encryption_flag,
            encryption_algorithm,
            reserved,
            metadata_size,
        })
    }
    
    pub fn get_compression_algorithm(&self) -> CompressionAlgorithm {
        match self.compression_algorithm {
            0 => CompressionAlgorithm::Zstd,
            1 => CompressionAlgorithm::Lzma2,
            _ => CompressionAlgorithm::Auto,
        }
    }
    
    pub fn get_encryption_algorithm(&self) -> Option<EncryptionAlgorithm> {
        if self.is_encrypted() {
            match self.encryption_algorithm {
                0 => Some(EncryptionAlgorithm::AesGcm),
                1 => Some(EncryptionAlgorithm::XChaCha20Poly1305),
                _ => Some(EncryptionAlgorithm::AesGcm), // デフォルト
            }
        } else {
            None
        }
    }
    
    pub fn is_encrypted(&self) -> bool {
        self.encryption_flag != 0
    }
}

/// NXZファイル全体の構造
pub struct NxzFile {
    header: NxzHeader,
    metadata: NxzMetadata,
    data: Vec<u8>,
}

impl NxzFile {
    pub fn new(
        data: &[u8],
        original_size: u64,
        compression_algorithm: CompressionAlgorithm,
        is_encrypted: bool,
        encryption_algorithm: Option<EncryptionAlgorithm>,
        compression_level: u8,
    ) -> Result<Self> {
        let metadata = NxzMetadata::new(
            original_size,
            data.len() as u64,
            compression_algorithm,
            compression_level,
            is_encrypted,
        );
        
        let metadata_bytes = metadata.to_bytes()?;
        let header = NxzHeader::new(
            original_size,
            compression_algorithm,
            is_encrypted,
            encryption_algorithm,
            metadata_bytes.len() as u32,
        );
        
        Ok(Self {
            header,
            metadata,
            data: data.to_vec(),
        })
    }
    
    pub async fn write_to_file(&self, path: &str) -> Result<()> {
        let file_data = self.to_bytes()?;
        fs::write(path, file_data).await?;
        Ok(())
    }
    
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let header_bytes = self.header.to_bytes()?;
        let metadata_bytes = self.metadata.to_bytes()?;
        
        // ファイル構造: [ヘッダ] + [メタデータ] + [データ部]
        let mut file_data = Vec::new();
        file_data.extend_from_slice(&header_bytes);
        file_data.extend_from_slice(&metadata_bytes);
        file_data.extend_from_slice(&self.data);
        
        Ok(file_data)
    }
    
    pub async fn read_from_file(path: &str) -> Result<Self> {
        let file_data = fs::read(path).await?;
        Self::from_bytes(&file_data)
    }
    
    pub fn from_bytes(file_data: &[u8]) -> Result<Self> {
        if file_data.len() < NxzHeader::HEADER_SIZE as usize {
            anyhow::bail!("ファイルサイズが小さすぎます");
        }
        
        // ヘッダ読み込み
        let header = NxzHeader::from_bytes(file_data)?;
        
        // メタデータ読み込み
        let metadata_start = NxzHeader::HEADER_SIZE as usize;
        let metadata_end = metadata_start + header.metadata_size as usize;
        
        if file_data.len() < metadata_end {
            anyhow::bail!("メタデータが読み込めません");
        }
        
        let metadata = NxzMetadata::from_bytes(&file_data[metadata_start..metadata_end])?;
        
        // データ部分読み込み
        let data = file_data[metadata_end..].to_vec();
        
        Ok(Self {
            header,
            metadata,
            data,
        })
    }
    
    // Getters
    pub fn is_encrypted(&self) -> bool {
        self.header.is_encrypted()
    }
    
    pub fn encryption_algorithm(&self) -> Option<EncryptionAlgorithm> {
        self.header.get_encryption_algorithm()
    }
    
    pub fn compression_algorithm(&self) -> CompressionAlgorithm {
        self.header.get_compression_algorithm()
    }
    
    pub fn data(&self) -> &[u8] {
        &self.data
    }
    
    pub fn compressed_data(&self) -> &[u8] {
        &self.data
    }
    
    pub fn original_size(&self) -> u64 {
        self.header.original_size
    }
    
    pub fn metadata(&self) -> &NxzMetadata {
        &self.metadata
    }
    
    pub fn version(&self) -> u16 {
        self.header.version
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    
    #[tokio::test]
    async fn test_nxz_file_roundtrip() {
        let test_data = b"Hello, NXZip! This is test data.";
        let nxz_file = NxzFile::new(
            test_data,
            test_data.len() as u64,
            CompressionAlgorithm::Zstd,
            false,
            None,
            6,
        ).unwrap();
        
        // 一時ファイルに書き込み
        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().to_str().unwrap();
        
        nxz_file.write_to_file(temp_path).await.unwrap();
        
        // 読み込み
        let loaded_nxz = NxzFile::read_from_file(temp_path).await.unwrap();
        
        // 検証
        assert_eq!(loaded_nxz.data(), test_data);
        assert_eq!(loaded_nxz.original_size(), test_data.len() as u64);
        assert!(!loaded_nxz.is_encrypted());
    }
}
