use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::fs;
use sha2::{Sha256, Digest};

use crate::engine::{CompressionAlgorithm, SPEParameters, StructureLevel};
use crate::crypto::{EncryptionAlgorithm, EncryptionMetadata, KeyDerivationFunction};
use crate::utils::metadata::NxzMetadata;

/// 強化されたNXZファイル形式のヘッダ構造（v2.0）
#[derive(Debug, Serialize, Deserialize)]
pub struct EnhancedNxzHeader {
    /// ファイルシグネチャ "NXZ2" (v2.0)
    pub signature: [u8; 4],
    /// バージョン情報
    pub version: u16,
    /// ヘッダサイズ
    pub header_size: u32,
    /// 元ファイルサイズ
    pub original_size: u64,
    /// 圧縮アルゴリズム
    pub compression_algorithm: u8,
    /// 暗号化フラグ
    pub encryption_flag: u8,
    /// 暗号化アルゴリズム
    pub encryption_algorithm: u8,
    /// キー導出機能
    pub kdf_algorithm: u8,
    /// SPE有効フラグ
    pub spe_enabled: u8,
    /// SPE構造レベル
    pub spe_level: u8,
    /// 完全性チェック有効フラグ
    pub integrity_check: u8,
    /// 拡張フィールドサイズ
    pub extended_fields_size: u32,
    /// 予約領域
    pub reserved: [u8; 13],
    /// メタデータサイズ
    pub metadata_size: u32,
    /// ヘッダチェックサム
    pub header_checksum: [u8; 32],
}

/// 拡張フィールド構造
#[derive(Debug, Serialize, Deserialize)]
pub struct ExtendedFields {
    /// SPEパラメータ（SPE有効時のみ）
    pub spe_params: Option<SPEParameters>,
    /// 暗号化メタデータ
    pub encryption_metadata: Option<EncryptionMetadata>,
    /// カスタムメタデータ
    pub custom_metadata: Vec<CustomMetadataEntry>,
    /// 圧縮統計
    pub compression_stats: CompressionStats,
    /// 完全性情報
    pub integrity_info: IntegrityInfo,
}

/// カスタムメタデータエントリ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomMetadataEntry {
    /// エントリキー
    pub key: String,
    /// エントリ値
    pub value: MetadataValue,
}

/// メタデータ値
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetadataValue {
    /// 文字列値
    String(String),
    /// 整数値
    Integer(i64),
    /// 浮動小数点値
    Float(f64),
    /// バイナリ値
    Binary(Vec<u8>),
    /// ブール値
    Boolean(bool),
}

/// 圧縮統計
#[derive(Debug, Serialize, Deserialize)]
pub struct CompressionStats {
    /// 圧縮率
    pub compression_ratio: f64,
    /// 圧縮時間（ミリ秒）
    pub compression_time_ms: u64,
    /// 使用メモリ量（バイト）
    pub memory_usage: u64,
    /// CPUコア使用数
    pub cpu_cores_used: u8,
}

/// 完全性情報
#[derive(Debug, Serialize, Deserialize)]
pub struct IntegrityInfo {
    /// 元データのハッシュ
    pub original_hash: [u8; 32],
    /// 圧縮データのハッシュ
    pub compressed_hash: [u8; 32],
    /// 暗号化データのハッシュ（暗号化時のみ）
    pub encrypted_hash: Option<[u8; 32]>,
    /// ブロックレベルハッシュ
    pub block_hashes: Vec<[u8; 32]>,
    /// 作成タイムスタンプ
    pub created_timestamp: u64,
}

/// 強化されたNXZファイル
#[derive(Debug)]
pub struct EnhancedNxzFile {
    /// ヘッダ
    pub header: EnhancedNxzHeader,
    /// 拡張フィールド
    pub extended_fields: ExtendedFields,
    /// メタデータ
    pub metadata: NxzMetadata,
    /// 圧縮データ
    pub compressed_data: Vec<u8>,
}

impl EnhancedNxzHeader {
    const SIGNATURE_V2: [u8; 4] = *b"NXZ2";
    const VERSION_V2: u16 = 0x0200;
    const HEADER_SIZE_V2: u32 = 128; // 128バイト固定ヘッダ
    
    /// 新しいv2.0ヘッダを作成
    pub fn new(
        original_size: u64,
        compression_algorithm: CompressionAlgorithm,
        encryption_algorithm: Option<EncryptionAlgorithm>,
        kdf_algorithm: Option<KeyDerivationFunction>,
        spe_enabled: bool,
        spe_level: Option<StructureLevel>,
        metadata_size: u32,
        extended_fields_size: u32,
    ) -> Result<Self> {
        let compression_algo_byte = match compression_algorithm {
            CompressionAlgorithm::Zstd => 0,
            CompressionAlgorithm::Lzma2 => 1,
            CompressionAlgorithm::Auto => 2,
        };
        
        let (encryption_flag, encryption_algo_byte) = if let Some(algo) = encryption_algorithm {
            let algo_byte = match algo {
                EncryptionAlgorithm::AesGcm => 0,
                EncryptionAlgorithm::XChaCha20Poly1305 => 1,
            };
            (1, algo_byte)
        } else {
            (0, 0)
        };
        
        let kdf_algo_byte = if let Some(kdf) = kdf_algorithm {
            match kdf {
                KeyDerivationFunction::Pbkdf2 => 0,
                KeyDerivationFunction::Argon2id => 1,
            }
        } else {
            0
        };
        
        let spe_level_byte = if let Some(level) = spe_level {
            match level {
                StructureLevel::Basic => 0,
                StructureLevel::Extended => 1,
                StructureLevel::Maximum => 2,
            }
        } else {
            0
        };
        
        let mut header = Self {
            signature: Self::SIGNATURE_V2,
            version: Self::VERSION_V2,
            header_size: Self::HEADER_SIZE_V2,
            original_size,
            compression_algorithm: compression_algo_byte,
            encryption_flag,
            encryption_algorithm: encryption_algo_byte,
            kdf_algorithm: kdf_algo_byte,
            spe_enabled: if spe_enabled { 1 } else { 0 },
            spe_level: spe_level_byte,
            integrity_check: 1, // 常に有効
            extended_fields_size,
            reserved: [0; 13],
            metadata_size,
            header_checksum: [0; 32], // 後で計算
        };
        
        // ヘッダチェックサムを計算
        header.header_checksum = header.calculate_checksum()?;
        
        Ok(header)
    }
    
    /// ヘッダチェックサムを計算
    fn calculate_checksum(&self) -> Result<[u8; 32]> {
        let mut hasher = Sha256::new();
        
        // チェックサム以外の全フィールドをハッシュ
        hasher.update(&self.signature);
        hasher.update(&self.version.to_le_bytes());
        hasher.update(&self.header_size.to_le_bytes());
        hasher.update(&self.original_size.to_le_bytes());
        hasher.update(&[self.compression_algorithm]);
        hasher.update(&[self.encryption_flag]);
        hasher.update(&[self.encryption_algorithm]);
        hasher.update(&[self.kdf_algorithm]);
        hasher.update(&[self.spe_enabled]);
        hasher.update(&[self.spe_level]);
        hasher.update(&[self.integrity_check]);
        hasher.update(&self.extended_fields_size.to_le_bytes());
        hasher.update(&self.reserved);
        hasher.update(&self.metadata_size.to_le_bytes());
        
        Ok(hasher.finalize().into())
    }
    
    /// バイナリ形式に変換
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::with_capacity(128);
        
        bytes.extend_from_slice(&self.signature);
        bytes.extend_from_slice(&self.version.to_le_bytes());
        bytes.extend_from_slice(&self.header_size.to_le_bytes());
        bytes.extend_from_slice(&self.original_size.to_le_bytes());
        bytes.push(self.compression_algorithm);
        bytes.push(self.encryption_flag);
        bytes.push(self.encryption_algorithm);
        bytes.push(self.kdf_algorithm);
        bytes.push(self.spe_enabled);
        bytes.push(self.spe_level);
        bytes.push(self.integrity_check);
        bytes.extend_from_slice(&self.extended_fields_size.to_le_bytes());
        bytes.extend_from_slice(&self.reserved);
        bytes.extend_from_slice(&self.metadata_size.to_le_bytes());
        bytes.extend_from_slice(&self.header_checksum);
        
        // 128バイトまでパディング
        while bytes.len() < 128 {
            bytes.push(0);
        }
        
        Ok(bytes)
    }
    
    /// バイナリから復元
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 128 {
            anyhow::bail!("ヘッダサイズが不正です: {}", bytes.len());
        }
        
        let mut offset = 0;
        
        // シグネチャ確認
        let signature: [u8; 4] = bytes[offset..offset + 4].try_into()?;
        offset += 4;
        
        if signature != Self::SIGNATURE_V2 {
            anyhow::bail!("不正なファイルシグネチャ: {:?}", signature);
        }
        
        // フィールドを順次読み取り
        let version = u16::from_le_bytes(bytes[offset..offset + 2].try_into()?);
        offset += 2;
        
        let header_size = u32::from_le_bytes(bytes[offset..offset + 4].try_into()?);
        offset += 4;
        
        let original_size = u64::from_le_bytes(bytes[offset..offset + 8].try_into()?);
        offset += 8;
        
        let compression_algorithm = bytes[offset];
        offset += 1;
        
        let encryption_flag = bytes[offset];
        offset += 1;
        
        let encryption_algorithm = bytes[offset];
        offset += 1;
        
        let kdf_algorithm = bytes[offset];
        offset += 1;
        
        let spe_enabled = bytes[offset];
        offset += 1;
        
        let spe_level = bytes[offset];
        offset += 1;
        
        let integrity_check = bytes[offset];
        offset += 1;
        
        let extended_fields_size = u32::from_le_bytes(bytes[offset..offset + 4].try_into()?);
        offset += 4;
        
        let reserved: [u8; 13] = bytes[offset..offset + 13].try_into()?;
        offset += 13;
        
        let metadata_size = u32::from_le_bytes(bytes[offset..offset + 4].try_into()?);
        offset += 4;
        
        let header_checksum: [u8; 32] = bytes[offset..offset + 32].try_into()?;
        
        let header = Self {
            signature,
            version,
            header_size,
            original_size,
            compression_algorithm,
            encryption_flag,
            encryption_algorithm,
            kdf_algorithm,
            spe_enabled,
            spe_level,
            integrity_check,
            extended_fields_size,
            reserved,
            metadata_size,
            header_checksum,
        };
        
        // チェックサム検証
        header.verify_checksum()?;
        
        Ok(header)
    }
    
    /// チェックサム検証
    fn verify_checksum(&self) -> Result<()> {
        let calculated = self.calculate_checksum()?;
        if calculated != self.header_checksum {
            anyhow::bail!("ヘッダチェックサムが一致しません");
        }
        Ok(())
    }
    
    /// SPE有効かどうか
    pub fn is_spe_enabled(&self) -> bool {
        self.spe_enabled == 1
    }
    
    /// 暗号化有効かどうか
    pub fn is_encrypted(&self) -> bool {
        self.encryption_flag == 1
    }
    
    /// 完全性チェック有効かどうか
    pub fn has_integrity_check(&self) -> bool {
        self.integrity_check == 1
    }
    
    /// 圧縮アルゴリズムを取得
    pub fn get_compression_algorithm(&self) -> Result<CompressionAlgorithm> {
        match self.compression_algorithm {
            0 => Ok(CompressionAlgorithm::Zstd),
            1 => Ok(CompressionAlgorithm::Lzma2),
            2 => Ok(CompressionAlgorithm::Auto),
            _ => Err(anyhow::anyhow!("不明な圧縮アルゴリズム: {}", self.compression_algorithm)),
        }
    }
    
    /// 暗号化アルゴリズムを取得
    pub fn get_encryption_algorithm(&self) -> Result<Option<EncryptionAlgorithm>> {
        if !self.is_encrypted() {
            return Ok(None);
        }
        
        match self.encryption_algorithm {
            0 => Ok(Some(EncryptionAlgorithm::AesGcm)),
            1 => Ok(Some(EncryptionAlgorithm::XChaCha20Poly1305)),
            _ => Err(anyhow::anyhow!("不明な暗号化アルゴリズム: {}", self.encryption_algorithm)),
        }
    }
    
    /// キー導出機能を取得
    pub fn get_kdf_algorithm(&self) -> Result<Option<KeyDerivationFunction>> {
        if !self.is_encrypted() {
            return Ok(None);
        }
        
        match self.kdf_algorithm {
            0 => Ok(Some(KeyDerivationFunction::Pbkdf2)),
            1 => Ok(Some(KeyDerivationFunction::Argon2id)),
            _ => Err(anyhow::anyhow!("不明なKDFアルゴリズム: {}", self.kdf_algorithm)),
        }
    }
    
    /// SPE構造レベルを取得
    pub fn get_spe_level(&self) -> Result<Option<StructureLevel>> {
        if !self.is_spe_enabled() {
            return Ok(None);
        }
        
        match self.spe_level {
            0 => Ok(Some(StructureLevel::Basic)),
            1 => Ok(Some(StructureLevel::Extended)),
            2 => Ok(Some(StructureLevel::Maximum)),
            _ => Err(anyhow::anyhow!("不明なSPEレベル: {}", self.spe_level)),
        }
    }
}

impl Default for ExtendedFields {
    fn default() -> Self {
        Self {
            spe_params: None,
            encryption_metadata: None,
            custom_metadata: Vec::new(),
            compression_stats: CompressionStats {
                compression_ratio: 0.0,
                compression_time_ms: 0,
                memory_usage: 0,
                cpu_cores_used: 1,
            },
            integrity_info: IntegrityInfo {
                original_hash: [0; 32],
                compressed_hash: [0; 32],
                encrypted_hash: None,
                block_hashes: Vec::new(),
                created_timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            },
        }
    }
}

impl ExtendedFields {
    /// バイナリ形式にシリアライズ
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        bincode::serialize(self)
            .map_err(|e| anyhow::anyhow!("拡張フィールドのシリアライズエラー: {}", e))
    }
    
    /// バイナリから復元
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        bincode::deserialize(bytes)
            .map_err(|e| anyhow::anyhow!("拡張フィールドのデシリアライズエラー: {}", e))
    }
}

impl EnhancedNxzFile {
    /// 新しいファイルを作成
    pub fn new(
        original_size: u64,
        compression_algorithm: CompressionAlgorithm,
        encryption_algorithm: Option<EncryptionAlgorithm>,
        kdf_algorithm: Option<KeyDerivationFunction>,
        spe_enabled: bool,
        spe_level: Option<StructureLevel>,
    ) -> Result<Self> {
        let extended_fields = ExtendedFields::default();
        let metadata = NxzMetadata::new();
        
        let extended_fields_bytes = extended_fields.to_bytes()?;
        let metadata_bytes = metadata.to_bytes()?;
        
        let header = EnhancedNxzHeader::new(
            original_size,
            compression_algorithm,
            encryption_algorithm,
            kdf_algorithm,
            spe_enabled,
            spe_level,
            metadata_bytes.len() as u32,
            extended_fields_bytes.len() as u32,
        )?;
        
        Ok(Self {
            header,
            extended_fields,
            metadata,
            compressed_data: Vec::new(),
        })
    }
    
    /// ファイルから読み込み
    pub async fn read_from_file(path: &str) -> Result<Self> {
        let data = fs::read(path).await?;
        Self::from_bytes(&data)
    }
    
    /// ファイルに書き込み
    pub async fn write_to_file(&self, path: &str) -> Result<()> {
        let data = self.to_bytes()?;
        fs::write(path, data).await?;
        Ok(())
    }
    
    /// バイナリ形式に変換
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        
        // ヘッダ
        bytes.extend_from_slice(&self.header.to_bytes()?);
        
        // 拡張フィールド
        let extended_bytes = self.extended_fields.to_bytes()?;
        bytes.extend_from_slice(&extended_bytes);
        
        // メタデータ
        let metadata_bytes = self.metadata.to_bytes()?;
        bytes.extend_from_slice(&metadata_bytes);
        
        // 圧縮データ
        bytes.extend_from_slice(&self.compressed_data);
        
        Ok(bytes)
    }
    
    /// バイナリから復元
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut offset = 0;
        
        // ヘッダの読み込み
        if bytes.len() < 128 {
            anyhow::bail!("ファイルが小さすぎます");
        }
        
        let header = EnhancedNxzHeader::from_bytes(&bytes[offset..offset + 128])?;
        offset += 128;
        
        // 拡張フィールドの読み込み
        let extended_fields_size = header.extended_fields_size as usize;
        if bytes.len() < offset + extended_fields_size {
            anyhow::bail!("拡張フィールドが不完全です");
        }
        
        let extended_fields = ExtendedFields::from_bytes(&bytes[offset..offset + extended_fields_size])?;
        offset += extended_fields_size;
        
        // メタデータの読み込み
        let metadata_size = header.metadata_size as usize;
        if bytes.len() < offset + metadata_size {
            anyhow::bail!("メタデータが不完全です");
        }
        
        let metadata = NxzMetadata::from_bytes(&bytes[offset..offset + metadata_size])?;
        offset += metadata_size;
        
        // 圧縮データの読み込み
        let compressed_data = bytes[offset..].to_vec();
        
        Ok(Self {
            header,
            extended_fields,
            metadata,
            compressed_data,
        })
    }
    
    /// 完全性検証
    pub fn verify_integrity(&self) -> Result<()> {
        // ヘッダチェックサム検証
        self.header.verify_checksum()?;
        
        // データ完全性検証
        if self.header.has_integrity_check() {
            let calculated_hash = self.calculate_data_hash();
            if calculated_hash != self.extended_fields.integrity_info.compressed_hash {
                anyhow::bail!("データ完全性チェックに失敗");
            }
        }
        
        Ok(())
    }
    
    /// データハッシュを計算
    fn calculate_data_hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(&self.compressed_data);
        hasher.finalize().into()
    }
}
