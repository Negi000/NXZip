//! # NXZip - 次世代統合アーカイブシステム
//!
//! NXZipは、SPE (Structure-Preserving Encryption) × 高効率可逆圧縮 × セキュリティ強化機構を
//! 統合した次世代アーカイブシステムです。
//!
//! ## 特徴
//!
//! - **SPE暗号化**: データ構造を保持しながら難読化
//! - **高効率圧縮**: LZMA2/Zstdを上回る圧縮率
//! - **多層セキュリティ**: AES-GCM/XChaCha20による強固な暗号化
//! - **高速展開**: SSD最適化による高速解凍
//!
//! ## 基本的な使用例
//!
//! ```rust
//! use nxzip::engine::{Compressor, Decompressor, CompressionAlgorithm};
//! use nxzip::crypto::{Encryptor, Decryptor, EncryptionAlgorithm};
//! use nxzip::formats::nxz::NxzFile;
//!
//! # tokio_test::block_on(async {
//! // データ準備
//! let data = b"Hello, NXZip!";
//!
//! // 圧縮
//! let compressor = Compressor::new(CompressionAlgorithm::Zstd, 6);
//! let compressed = compressor.compress(data).unwrap();
//!
//! // NXZファイル作成
//! let nxz_file = NxzFile::new(
//!     &compressed,
//!     data.len() as u64,
//!     CompressionAlgorithm::Zstd,
//!     false,
//!     None,
//!     6,
//! ).unwrap();
//!
//! // ファイル保存
//! nxz_file.write_to_file("example.nxz").await.unwrap();
//!
//! // 読み込み・展開
//! let loaded = NxzFile::read_from_file("example.nxz").await.unwrap();
//! let decompressor = Decompressor::new(loaded.compression_algorithm());
//! let decompressed = decompressor.decompress(loaded.data()).unwrap();
//! # });
//! ```
//!
//! ## 暗号化付きの例
//!
//! ```rust
//! use nxzip::crypto::{Encryptor, Decryptor, EncryptionAlgorithm};
//!
//! # tokio_test::block_on(async {
//! let data = b"Secret data";
//! let password = "secure_password";
//!
//! // 暗号化
//! let encryptor = Encryptor::with_algorithm(password, EncryptionAlgorithm::AesGcm).unwrap();
//! let encrypted = encryptor.encrypt(data).unwrap();
//!
//! // 復号化
//! let decryptor = Decryptor::with_algorithm(password, EncryptionAlgorithm::AesGcm).unwrap();
//! let decrypted = decryptor.decrypt(&encrypted).unwrap();
//!
//! assert_eq!(data, &decrypted[..]);
//! # });
//! ```

pub mod engine;
pub mod crypto;
pub mod formats;
pub mod utils;

// 頻繁に使用される型の再エクスポート
pub use engine::{Compressor, Decompressor, CompressionAlgorithm};
pub use crypto::{Encryptor, Decryptor, EncryptionAlgorithm};
pub use formats::nxz::NxzFile;
pub use formats::nxz_sec::{NxzSecFile, KdfType};

/// NXZipライブラリのバージョン情報
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// ライブラリの初期化（ログ設定など）
pub fn init() {
    env_logger::init();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_basic_workflow() {
        let data = b"Test data for basic workflow";
        
        // 圧縮
        let compressor = Compressor::new(CompressionAlgorithm::Zstd, 6);
        let compressed = compressor.compress(data).unwrap();
        
        // NXZファイル作成
        let nxz_file = NxzFile::new(
            &compressed,
            data.len() as u64,
            CompressionAlgorithm::Zstd,
            false,
            None,
            6,
        ).unwrap();
        
        // 展開
        let decompressor = Decompressor::new(nxz_file.compression_algorithm());
        let decompressed = decompressor.decompress(nxz_file.data()).unwrap();
        
        assert_eq!(data, &decompressed[..]);
    }

    #[tokio::test]
    async fn test_encrypted_workflow() {
        let data = b"Secret test data for encryption workflow";
        let password = "test_password_123";
        
        // SPE変換
        let spe_data = engine::spe_stub::apply_spe_transform(data).unwrap();
        
        // 圧縮
        let compressor = Compressor::new(CompressionAlgorithm::Zstd, 6);
        let compressed = compressor.compress(&spe_data).unwrap();
        
        // 暗号化
        let encryptor = Encryptor::with_algorithm(password, EncryptionAlgorithm::AesGcm).unwrap();
        let encrypted = encryptor.encrypt(&compressed).unwrap();
        
        // NXZファイル作成
        let nxz_file = NxzFile::new(
            &encrypted,
            data.len() as u64,
            CompressionAlgorithm::Zstd,
            true,
            Some(EncryptionAlgorithm::AesGcm),
            6,
        ).unwrap();
        
        // 復号化・展開・SPE逆変換
        let decryptor = Decryptor::with_algorithm(password, EncryptionAlgorithm::AesGcm).unwrap();
        let decrypted = decryptor.decrypt(nxz_file.data()).unwrap();
        
        let decompressor = Decompressor::new(nxz_file.compression_algorithm());
        let decompressed = decompressor.decompress(&decrypted).unwrap();
        
        let original = engine::spe_stub::reverse_spe_transform(&decompressed).unwrap();
        
        assert_eq!(data, &original[..]);
    }
}
