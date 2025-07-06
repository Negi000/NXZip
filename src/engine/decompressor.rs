use anyhow::Result;
use crate::engine::CompressionAlgorithm;

pub struct Decompressor {
    algorithm: CompressionAlgorithm,
}

impl Decompressor {
    pub fn new(algorithm: CompressionAlgorithm) -> Self {
        Self { algorithm }
    }
    
    pub fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        match self.algorithm {
            CompressionAlgorithm::Zstd => self.decompress_zstd(data),
            CompressionAlgorithm::Lzma2 => self.decompress_lzma2(data),
            CompressionAlgorithm::Auto => {
                // Autoの場合、まずZstdを試し、失敗したらLZMA2を試す
                self.decompress_zstd(data)
                    .or_else(|_| self.decompress_lzma2(data))
            }
        }
    }
    
    fn decompress_zstd(&self, data: &[u8]) -> Result<Vec<u8>> {
        let decompressed = zstd::bulk::decompress(data, 1024 * 1024 * 100)?; // 最大100MB
        Ok(decompressed)
    }
    
    fn decompress_lzma2(&self, data: &[u8]) -> Result<Vec<u8>> {
        use std::io::Read;
        let mut decoder = xz2::read::XzDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)?;
        Ok(decompressed)
    }
}
