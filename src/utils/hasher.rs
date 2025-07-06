use anyhow::Result;
use blake3::Hasher as Blake3Hasher;
use sha2::{Sha256, Digest};

pub struct FileHasher {
    algorithm: HashAlgorithm,
}

#[derive(Debug, Clone)]
pub enum HashAlgorithm {
    Blake3,
    Sha256,
}

impl FileHasher {
    pub fn new() -> Self {
        Self {
            algorithm: HashAlgorithm::Blake3,
        }
    }
    
    pub fn with_algorithm(algorithm: HashAlgorithm) -> Self {
        Self { algorithm }
    }
    
    pub fn calculate_hash(&self, data: &[u8]) -> Result<Vec<u8>> {
        match self.algorithm {
            HashAlgorithm::Blake3 => {
                let mut hasher = Blake3Hasher::new();
                hasher.update(data);
                Ok(hasher.finalize().as_bytes().to_vec())
            }
            HashAlgorithm::Sha256 => {
                let mut hasher = Sha256::new();
                hasher.update(data);
                Ok(hasher.finalize().to_vec())
            }
        }
    }
    
    pub fn verify_hash(&self, data: &[u8], expected_hash: &[u8]) -> Result<bool> {
        let calculated_hash = self.calculate_hash(data)?;
        Ok(calculated_hash == expected_hash)
    }
}
