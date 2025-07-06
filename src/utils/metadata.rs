use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};
use crate::engine::CompressionAlgorithm;

#[derive(Debug, Serialize, Deserialize)]
pub struct NxzMetadata {
    pub version: String,
    pub created_at: u64,
    pub original_size: u64,
    pub compressed_size: u64,
    pub compression_algorithm: CompressionAlgorithm,
    pub compression_level: u8,
    pub is_encrypted: bool,
    pub file_hash: Option<Vec<u8>>,
    pub original_filename: Option<String>,
}

impl NxzMetadata {
    pub fn new(
        original_size: u64,
        compressed_size: u64,
        compression_algorithm: CompressionAlgorithm,
        compression_level: u8,
        is_encrypted: bool,
    ) -> Self {
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        Self {
            version: "1.0".to_string(),
            created_at,
            original_size,
            compressed_size,
            compression_algorithm,
            compression_level,
            is_encrypted,
            file_hash: None,
            original_filename: None,
        }
    }
    
    pub fn with_hash(mut self, hash: Vec<u8>) -> Self {
        self.file_hash = Some(hash);
        self
    }
    
    pub fn with_filename(mut self, filename: String) -> Self {
        self.original_filename = Some(filename);
        self
    }
    
    pub fn to_bytes(&self) -> anyhow::Result<Vec<u8>> {
        bincode::serialize(self).map_err(Into::into)
    }
    
    pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        bincode::deserialize(bytes).map_err(Into::into)
    }
}
