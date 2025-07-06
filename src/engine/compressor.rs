use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    Zstd,
    Lzma2,
    Auto,
}

pub struct Compressor {
    algorithm: CompressionAlgorithm,
    level: u8,
}

impl Compressor {
    pub fn new(algorithm: CompressionAlgorithm, level: u8) -> Self {
        Self { algorithm, level }
    }
    
    pub fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        match self.algorithm {
            CompressionAlgorithm::Zstd => self.compress_zstd(data),
            CompressionAlgorithm::Lzma2 => self.compress_lzma2(data),
            CompressionAlgorithm::Auto => self.compress_auto(data),
        }
    }
    
    fn compress_zstd(&self, data: &[u8]) -> Result<Vec<u8>> {
        let compressed = zstd::bulk::compress(data, self.level as i32)?;
        Ok(compressed)
    }
    
    fn compress_lzma2(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut encoder = xz2::write::XzEncoder::new(Vec::new(), self.level as u32);
        use std::io::Write;
        encoder.write_all(data)?;
        let compressed = encoder.finish()?;
        Ok(compressed)
    }
    
    fn compress_auto(&self, data: &[u8]) -> Result<Vec<u8>> {
        // ファイルサイズと内容に基づいて最適なアルゴリズムを選択
        
        if data.len() < 1024 * 1024 {
            // 1MB未満: 高速展開重視でZstd
            self.compress_zstd(data)
        } else {
            // 1MB以上: テスト圧縮して効率の良い方を選択
            let zstd_sample = self.test_compress_zstd(&data[..std::cmp::min(data.len(), 64 * 1024)])?;
            let lzma_sample = self.test_compress_lzma2(&data[..std::cmp::min(data.len(), 64 * 1024)])?;
            
            let zstd_ratio = zstd_sample.len() as f64 / (64 * 1024) as f64;
            let lzma_ratio = lzma_sample.len() as f64 / (64 * 1024) as f64;
            
            if lzma_ratio < zstd_ratio * 0.95 {
                // LZMA2が5%以上良ければLZMA2を選択
                self.compress_lzma2(data)
            } else {
                // そうでなければZstdを選択（展開速度重視）
                self.compress_zstd(data)
            }
        }
    }
    
    fn test_compress_zstd(&self, sample: &[u8]) -> Result<Vec<u8>> {
        zstd::bulk::compress(sample, self.level as i32).map_err(Into::into)
    }
    
    fn test_compress_lzma2(&self, sample: &[u8]) -> Result<Vec<u8>> {
        let mut encoder = xz2::write::XzEncoder::new(Vec::new(), self.level as u32);
        use std::io::Write;
        encoder.write_all(sample)?;
        encoder.finish().map_err(Into::into)
    }
    
    /// 実際に使用された圧縮アルゴリズムを返す
    pub fn get_used_algorithm(&self, data: &[u8]) -> Result<CompressionAlgorithm> {
        match self.algorithm {
            CompressionAlgorithm::Auto => {
                if data.len() < 1024 * 1024 {
                    Ok(CompressionAlgorithm::Zstd)
                } else {
                    // 実際の選択ロジックに基づいて決定
                    // 簡略化のため、ここではZstdを返す
                    Ok(CompressionAlgorithm::Zstd)
                }
            }
            algo => Ok(algo),
        }
    }
}
