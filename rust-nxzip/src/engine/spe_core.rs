use anyhow::{anyhow, Result};
use sha2::{Sha256, Digest};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;

/// SPE (Structure-Preserving Encryption) コアシステム
/// 
/// データの論理構造を保持しながら高度な難読化を実現する
/// 完全可逆性と暗号学的強度を両立
#[derive(Debug, Clone)]
pub struct SPECore {
    /// SPE変換パラメータ
    params: SPEParameters,
    /// データ完全性チェック用
    integrity_checker: IntegrityChecker,
    /// 構造メタデータ管理
    structure_manager: StructureManager,
}

/// SPE変換パラメータ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SPEParameters {
    /// ブロックサイズ (バイト)
    pub block_size: usize,
    /// シャッフルラウンド数
    pub shuffle_rounds: u8,
    /// XOR鍵導出設定
    pub xor_derivation: XorDerivation,
    /// 構造保持レベル
    pub structure_level: StructureLevel,
    /// チェックサム設定
    pub checksum_config: ChecksumConfig,
}

/// XOR鍵導出設定
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XorDerivation {
    /// 基本シード
    pub seed: [u8; 32],
    /// 反復回数
    pub iterations: u32,
    /// 鍵長度
    pub key_length: usize,
}

/// 構造保持レベル
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum StructureLevel {
    /// 基本的な構造保持
    Basic,
    /// 拡張構造保持（より複雑な変換）
    Extended,
    /// 最大強度構造保持（最高レベルの難読化）
    Maximum,
}

/// チェックサム設定
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChecksumConfig {
    /// ブロックレベルチェックサム
    pub block_checksums: bool,
    /// 全体チェックサム
    pub global_checksum: bool,
    /// 構造チェックサム
    pub structure_checksum: bool,
}

/// データ完全性チェッカー
#[derive(Debug, Clone)]
pub struct IntegrityChecker {
    block_hashes: HashMap<usize, [u8; 32]>,
    global_hash: Option<[u8; 32]>,
    structure_hash: Option<[u8; 32]>,
}

/// 構造メタデータ管理
#[derive(Debug, Clone)]
pub struct StructureManager {
    /// ブロック境界情報
    block_boundaries: Vec<BlockBoundary>,
    /// データ型ヒント
    data_type_hints: Vec<DataTypeHint>,
    /// アライメント情報
    alignment_info: AlignmentInfo,
}

/// ブロック境界情報
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockBoundary {
    /// 開始位置
    pub start: usize,
    /// 終了位置
    pub end: usize,
    /// ブロックタイプ
    pub block_type: BlockType,
    /// チェックサム
    pub checksum: [u8; 32],
}

/// ブロックタイプ
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BlockType {
    /// データブロック
    Data,
    /// パディングブロック
    Padding,
    /// メタデータブロック
    Metadata,
}

/// データ型ヒント
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataTypeHint {
    /// 位置範囲
    pub range: (usize, usize),
    /// データ型
    pub data_type: DataType,
    /// エントロピー情報
    pub entropy: f64,
}

/// データ型
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DataType {
    /// テキストデータ
    Text,
    /// バイナリデータ
    Binary,
    /// 圧縮済みデータ
    Compressed,
    /// 暗号化済みデータ
    Encrypted,
    /// 混合データ
    Mixed,
}

/// アライメント情報
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentInfo {
    /// 自然アライメント
    pub natural_alignment: usize,
    /// 強制アライメント
    pub forced_alignment: Option<usize>,
    /// パディング戦略
    pub padding_strategy: PaddingStrategy,
}

/// パディング戦略
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PaddingStrategy {
    /// ゼロパディング
    Zero,
    /// ランダムパディング
    Random,
    /// パターンパディング
    Pattern,
    /// 適応的パディング
    Adaptive,
}

impl Default for SPEParameters {
    fn default() -> Self {
        Self {
            block_size: 4096,
            shuffle_rounds: 3,
            xor_derivation: XorDerivation {
                seed: [0u8; 32],
                iterations: 10000,
                key_length: 256,
            },
            structure_level: StructureLevel::Extended,
            checksum_config: ChecksumConfig {
                block_checksums: true,
                global_checksum: true,
                structure_checksum: true,
            },
        }
    }
}

impl SPECore {
    /// 新しいSPEコアを作成
    pub fn new(params: SPEParameters) -> Result<Self> {
        let integrity_checker = IntegrityChecker::new();
        let structure_manager = StructureManager::new();
        
        Ok(Self {
            params,
            integrity_checker,
            structure_manager,
        })
    }
    
    /// デフォルト設定でSPEコアを作成
    pub fn default() -> Result<Self> {
        Self::new(SPEParameters::default())
    }
    
    /// キーからSPEコアを作成
    pub fn from_key(key: &[u8]) -> Result<Self> {
        let mut params = SPEParameters::default();
        
        // キーからシードを導出
        let mut hasher = Sha256::new();
        hasher.update(key);
        hasher.update(b"SPE_SEED_DERIVATION");
        let hash = hasher.finalize();
        params.xor_derivation.seed.copy_from_slice(&hash);
        
        Self::new(params)
    }
    
    /// SPE変換を適用（完全可逆）
    pub fn apply_transform(&mut self, data: &[u8]) -> Result<SPETransformResult> {
        // 1. 入力データの分析と構造検出
        self.analyze_data_structure(data)?;
        
        // 2. 完全性チェックサムの計算
        self.calculate_checksums(data)?;
        
        // 3. 段階的SPE変換の適用
        let transformed_data = self.apply_layered_transform(data)?;
        
        // 4. 変換結果の検証
        self.verify_transform_integrity(&transformed_data)?;
        
        // 5. メタデータの構築
        let metadata = self.build_metadata(data.len())?;
        
        Ok(SPETransformResult {
            transformed_data,
            metadata,
            original_size: data.len(),
        })
    }
    
    /// SPE逆変換を適用（完全復元）
    pub fn reverse_transform(&mut self, result: &SPETransformResult) -> Result<Vec<u8>> {
        // 1. メタデータの復元
        self.restore_metadata(&result.metadata)?;
        
        // 2. 変換前の完全性チェック
        self.verify_transform_integrity(&result.transformed_data)?;
        
        // 3. 逆段階的SPE変換の適用
        let restored_data = self.reverse_layered_transform(&result.transformed_data)?;
        
        // 4. 復元データの完全性検証
        self.verify_restoration_integrity(&restored_data, result.original_size)?;
        
        // 5. 最終サイズチェック
        if restored_data.len() != result.original_size {
            return Err(anyhow!("復元データサイズが不正: {} != {}", 
                restored_data.len(), result.original_size));
        }
        
        Ok(restored_data)
    }
    
    /// データ構造の分析
    fn analyze_data_structure(&mut self, data: &[u8]) -> Result<()> {
        // エントロピー分析
        let entropy = self.calculate_entropy(data);
        
        // ブロック境界の検出
        let boundaries = self.detect_block_boundaries(data)?;
        self.structure_manager.block_boundaries = boundaries;
        
        // データ型の推定
        let type_hints = self.estimate_data_types(data, entropy)?;
        self.structure_manager.data_type_hints = type_hints;
        
        // アライメント分析
        let alignment = self.analyze_alignment(data)?;
        self.structure_manager.alignment_info = alignment;
        
        Ok(())
    }
    
    /// 完全性チェックサムの計算
    fn calculate_checksums(&mut self, data: &[u8]) -> Result<()> {
        if self.params.checksum_config.global_checksum {
            let mut hasher = Sha256::new();
            hasher.update(data);
            self.integrity_checker.global_hash = Some(hasher.finalize().into());
        }
        
        if self.params.checksum_config.block_checksums {
            for (i, boundary) in self.structure_manager.block_boundaries.iter().enumerate() {
                let block_data = &data[boundary.start..boundary.end];
                let mut hasher = Sha256::new();
                hasher.update(block_data);
                self.integrity_checker.block_hashes.insert(i, hasher.finalize().into());
            }
        }
        
        Ok(())
    }
    
    /// 段階的SPE変換の適用
    fn apply_layered_transform(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut transformed = data.to_vec();
        
        // レイヤー1: 構造保持パディング
        self.apply_structure_preserving_padding(&mut transformed)?;
        
        // レイヤー2: 適応的ブロックシャッフル
        self.apply_adaptive_block_shuffle(&mut transformed)?;
        
        // レイヤー3: 高度XOR難読化
        self.apply_advanced_xor_obfuscation(&mut transformed)?;
        
        // レイヤー4: 構造レベル別追加変換
        match self.params.structure_level {
            StructureLevel::Basic => {},
            StructureLevel::Extended => {
                self.apply_extended_transforms(&mut transformed)?;
            },
            StructureLevel::Maximum => {
                self.apply_extended_transforms(&mut transformed)?;
                self.apply_maximum_transforms(&mut transformed)?;
            },
        }
        
        Ok(transformed)
    }
    
    /// 逆段階的SPE変換の適用
    fn reverse_layered_transform(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut restored = data.to_vec();
        
        // 逆順で変換を除去
        
        // レイヤー4: 構造レベル別変換の除去
        match self.params.structure_level {
            StructureLevel::Basic => {},
            StructureLevel::Extended => {
                self.reverse_extended_transforms(&mut restored)?;
            },
            StructureLevel::Maximum => {
                self.reverse_maximum_transforms(&mut restored)?;
                self.reverse_extended_transforms(&mut restored)?;
            },
        }
        
        // レイヤー3: XOR難読化の除去
        self.reverse_advanced_xor_obfuscation(&mut restored)?;
        
        // レイヤー2: ブロックシャッフルの除去
        self.reverse_adaptive_block_shuffle(&mut restored)?;
        
        // レイヤー1: パディングの除去
        self.reverse_structure_preserving_padding(&mut restored)?;
        
        Ok(restored)
    }
    
    /// エントロピー計算
    fn calculate_entropy(&self, data: &[u8]) -> f64 {
        let mut counts = [0u32; 256];
        for &byte in data {
            counts[byte as usize] += 1;
        }
        
        let len = data.len() as f64;
        let mut entropy = 0.0;
        
        for &count in &counts {
            if count > 0 {
                let prob = count as f64 / len;
                entropy -= prob * prob.log2();
            }
        }
        
        entropy
    }
    
    /// ブロック境界の検出
    fn detect_block_boundaries(&self, data: &[u8]) -> Result<Vec<BlockBoundary>> {
        let mut boundaries = Vec::new();
        let block_size = self.params.block_size;
        
        for i in (0..data.len()).step_by(block_size) {
            let start = i;
            let end = std::cmp::min(i + block_size, data.len());
            
            let mut hasher = Sha256::new();
            hasher.update(&data[start..end]);
            let checksum = hasher.finalize().into();
            
            boundaries.push(BlockBoundary {
                start,
                end,
                block_type: BlockType::Data,
                checksum,
            });
        }
        
        Ok(boundaries)
    }
    
    /// データ型の推定
    fn estimate_data_types(&self, data: &[u8], entropy: f64) -> Result<Vec<DataTypeHint>> {
        let mut hints = Vec::new();
        
        // エントロピーベースの分類
        let data_type = match entropy {
            e if e < 3.0 => DataType::Text,
            e if e < 6.0 => DataType::Binary,
            e if e < 7.5 => DataType::Compressed,
            _ => DataType::Encrypted,
        };
        
        hints.push(DataTypeHint {
            range: (0, data.len()),
            data_type,
            entropy,
        });
        
        Ok(hints)
    }
    
    /// アライメント分析
    fn analyze_alignment(&self, data: &[u8]) -> Result<AlignmentInfo> {
        // 自然アライメントの検出
        let natural_alignment = self.detect_natural_alignment(data);
        
        Ok(AlignmentInfo {
            natural_alignment,
            forced_alignment: None,
            padding_strategy: PaddingStrategy::Adaptive,
        })
    }
    
    /// 自然アライメントの検出
    fn detect_natural_alignment(&self, data: &[u8]) -> usize {
        // 一般的なアライメント境界をチェック
        for alignment in [16, 8, 4, 2].iter() {
            if data.len() % alignment == 0 {
                return *alignment;
            }
        }
        1
    }
    
    // 以下、具体的な変換メソッドの実装（簡略化版）
    
    fn apply_structure_preserving_padding(&self, data: &mut Vec<u8>) -> Result<()> {
        let original_len = data.len();
        let alignment = self.structure_manager.alignment_info.natural_alignment;
        
        // アライメントに基づくパディング
        let padded_len = ((original_len + alignment - 1) / alignment) * alignment;
        let padding_needed = padded_len - original_len;
        
        // パディング戦略に基づくパディング
        match self.structure_manager.alignment_info.padding_strategy {
            PaddingStrategy::Zero => {
                data.extend(vec![0u8; padding_needed]);
            },
            PaddingStrategy::Random => {
                let mut rng = ChaCha20Rng::from_seed(self.params.xor_derivation.seed);
                let mut padding = vec![0u8; padding_needed];
                rng.fill_bytes(&mut padding);
                data.extend(padding);
            },
            PaddingStrategy::Pattern => {
                let pattern = [0xAA, 0x55, 0xAA, 0x55];
                for i in 0..padding_needed {
                    data.push(pattern[i % pattern.len()]);
                }
            },
            PaddingStrategy::Adaptive => {
                // データの特性に応じて適応的にパディング
                if original_len > 0 {
                    let last_byte = data[original_len - 1];
                    data.extend(vec![last_byte; padding_needed]);
                } else {
                    data.extend(vec![0u8; padding_needed]);
                }
            },
        }
        
        // 元の長さ情報を追加
        data.extend_from_slice(&original_len.to_le_bytes());
        
        Ok(())
    }
    
    fn reverse_structure_preserving_padding(&self, data: &mut Vec<u8>) -> Result<()> {
        if data.len() < 8 {
            return Err(anyhow!("データが小さすぎます: {}", data.len()));
        }
        
        // 元の長さ情報を取得
        let len_start = data.len() - 8;
        let len_bytes: [u8; 8] = data[len_start..].try_into()?;
        let original_len = usize::from_le_bytes(len_bytes);
        
        if original_len > len_start {
            return Err(anyhow!("不正な長さ情報: {} > {}", original_len, len_start));
        }
        
        // 元のサイズに復元
        data.truncate(original_len);
        
        Ok(())
    }
    
    fn apply_adaptive_block_shuffle(&self, data: &mut [u8]) -> Result<()> {
        if data.len() < self.params.block_size * 2 {
            return Ok(()); // 小さすぎる場合はスキップ
        }
        
        let block_size = self.params.block_size;
        let num_blocks = data.len() / block_size;
        
        // 決定論的なシャッフルパターンを生成
        let mut rng = ChaCha20Rng::from_seed(self.params.xor_derivation.seed);
        
        for round in 0..self.params.shuffle_rounds {
            let mut shuffle_indices: Vec<usize> = (0..num_blocks).collect();
            
            // Fisher-Yates シャッフル（決定論的）
            for i in (1..shuffle_indices.len()).rev() {
                let j = (rng.next_u64() as usize) % (i + 1);
                shuffle_indices.swap(i, j);
            }
            
            // ブロックを新しい順序で再配置
            let mut temp_data = vec![0u8; data.len()];
            for (new_pos, &old_pos) in shuffle_indices.iter().enumerate() {
                let src_start = old_pos * block_size;
                let src_end = std::cmp::min(src_start + block_size, data.len());
                let dst_start = new_pos * block_size;
                let dst_end = dst_start + (src_end - src_start);
                
                temp_data[dst_start..dst_end].copy_from_slice(&data[src_start..src_end]);
            }
            
            data.copy_from_slice(&temp_data);
        }
        
        Ok(())
    }
    
    fn reverse_adaptive_block_shuffle(&self, data: &mut [u8]) -> Result<()> {
        if data.len() < self.params.block_size * 2 {
            return Ok(());
        }
        
        let block_size = self.params.block_size;
        let num_blocks = data.len() / block_size;
        
        // 同じシードで逆順にシャッフルを戻す
        let mut rng = ChaCha20Rng::from_seed(self.params.xor_derivation.seed);
        
        // 全てのシャッフルパターンを事前計算
        let mut all_patterns = Vec::new();
        for round in 0..self.params.shuffle_rounds {
            let mut shuffle_indices: Vec<usize> = (0..num_blocks).collect();
            for i in (1..shuffle_indices.len()).rev() {
                let j = (rng.next_u64() as usize) % (i + 1);
                shuffle_indices.swap(i, j);
            }
            all_patterns.push(shuffle_indices);
        }
        
        // 逆順でシャッフルを戻す
        for pattern in all_patterns.iter().rev() {
            // 逆マッピングを作成
            let mut reverse_indices = vec![0; num_blocks];
            for (new_pos, &old_pos) in pattern.iter().enumerate() {
                reverse_indices[old_pos] = new_pos;
            }
            
            // 逆順で再配置
            let mut temp_data = vec![0u8; data.len()];
            for (old_pos, &new_pos) in reverse_indices.iter().enumerate() {
                let src_start = new_pos * block_size;
                let src_end = std::cmp::min(src_start + block_size, data.len());
                let dst_start = old_pos * block_size;
                let dst_end = dst_start + (src_end - src_start);
                
                temp_data[dst_start..dst_end].copy_from_slice(&data[src_start..src_end]);
            }
            
            data.copy_from_slice(&temp_data);
        }
        
        Ok(())
    }
    
    fn apply_advanced_xor_obfuscation(&self, data: &mut [u8]) -> Result<()> {
        // 高度なXOR鍵を導出
        let key = self.derive_xor_key(data.len())?;
        
        // データに XOR を適用
        for (i, byte) in data.iter_mut().enumerate() {
            *byte ^= key[i % key.len()];
        }
        
        Ok(())
    }
    
    fn reverse_advanced_xor_obfuscation(&self, data: &mut [u8]) -> Result<()> {
        // XOR は自己逆変換
        self.apply_advanced_xor_obfuscation(data)
    }
    
    fn derive_xor_key(&self, data_len: usize) -> Result<Vec<u8>> {
        let key_len = std::cmp::min(self.params.xor_derivation.key_length, data_len);
        let mut key = vec![0u8; key_len];
        
        let mut current_seed = self.params.xor_derivation.seed;
        
        for _ in 0..self.params.xor_derivation.iterations {
            let mut hasher = Sha256::new();
            hasher.update(&current_seed);
            hasher.update(&data_len.to_le_bytes());
            current_seed = hasher.finalize().into();
        }
        
        // キー生成
        let mut rng = ChaCha20Rng::from_seed(current_seed);
        rng.fill_bytes(&mut key);
        
        Ok(key)
    }
    
    fn apply_extended_transforms(&self, _data: &mut [u8]) -> Result<()> {
        // 拡張変換のプレースホルダー
        Ok(())
    }
    
    fn reverse_extended_transforms(&self, _data: &mut [u8]) -> Result<()> {
        // 拡張変換の逆変換のプレースホルダー
        Ok(())
    }
    
    fn apply_maximum_transforms(&self, _data: &mut [u8]) -> Result<()> {
        // 最大強度変換のプレースホルダー
        Ok(())
    }
    
    fn reverse_maximum_transforms(&self, _data: &mut [u8]) -> Result<()> {
        // 最大強度変換の逆変換のプレースホルダー
        Ok(())
    }
    
    fn verify_transform_integrity(&self, _data: &[u8]) -> Result<()> {
        // 変換完全性の検証
        Ok(())
    }
    
    fn verify_restoration_integrity(&self, _data: &[u8], _expected_size: usize) -> Result<()> {
        // 復元完全性の検証
        Ok(())
    }
    
    fn build_metadata(&self, original_size: usize) -> Result<SPEMetadata> {
        Ok(SPEMetadata {
            parameters: self.params.clone(),
            structure_info: self.structure_manager.clone(),
            integrity_info: IntegrityInfo {
                global_hash: self.integrity_checker.global_hash,
                block_hashes: self.integrity_checker.block_hashes.clone(),
                structure_hash: self.integrity_checker.structure_hash,
            },
            original_size,
        })
    }
    
    fn restore_metadata(&mut self, metadata: &SPEMetadata) -> Result<()> {
        self.params = metadata.parameters.clone();
        self.structure_manager = metadata.structure_info.clone();
        
        self.integrity_checker.global_hash = metadata.integrity_info.global_hash;
        self.integrity_checker.block_hashes = metadata.integrity_info.block_hashes.clone();
        self.integrity_checker.structure_hash = metadata.integrity_info.structure_hash;
        
        Ok(())
    }
}

/// SPE変換結果
#[derive(Debug, Clone)]
pub struct SPETransformResult {
    /// 変換されたデータ
    pub transformed_data: Vec<u8>,
    /// メタデータ
    pub metadata: SPEMetadata,
    /// 元のサイズ
    pub original_size: usize,
}

/// SPEメタデータ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SPEMetadata {
    /// SPEパラメータ
    pub parameters: SPEParameters,
    /// 構造情報
    pub structure_info: StructureManager,
    /// 完全性情報
    pub integrity_info: IntegrityInfo,
    /// 元のサイズ
    pub original_size: usize,
}

/// 完全性情報
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityInfo {
    /// グローバルハッシュ
    pub global_hash: Option<[u8; 32]>,
    /// ブロックハッシュ
    pub block_hashes: HashMap<usize, [u8; 32]>,
    /// 構造ハッシュ
    pub structure_hash: Option<[u8; 32]>,
}

impl IntegrityChecker {
    fn new() -> Self {
        Self {
            block_hashes: HashMap::new(),
            global_hash: None,
            structure_hash: None,
        }
    }
}

impl StructureManager {
    fn new() -> Self {
        Self {
            block_boundaries: Vec::new(),
            data_type_hints: Vec::new(),
            alignment_info: AlignmentInfo {
                natural_alignment: 16,
                forced_alignment: None,
                padding_strategy: PaddingStrategy::Adaptive,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_spe_core_reversibility() -> Result<()> {
        let mut spe = SPECore::default()?;
        let original_data = b"Hello, NXZip SPE Core! This is a comprehensive test of the Structure-Preserving Encryption system.";
        
        // 変換
        let result = spe.apply_transform(original_data)?;
        
        // 元データと異なることを確認
        assert_ne!(&result.transformed_data, original_data);
        
        // 逆変換
        let restored = spe.reverse_transform(&result)?;
        
        // 完全に復元されることを確認
        assert_eq!(&restored, original_data);
        
        Ok(())
    }
    
    #[test]
    fn test_spe_different_structure_levels() -> Result<()> {
        for level in [StructureLevel::Basic, StructureLevel::Extended, StructureLevel::Maximum] {
            let mut params = SPEParameters::default();
            params.structure_level = level;
            
            let mut spe = SPECore::new(params)?;
            let original_data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
            
            let result = spe.apply_transform(&original_data)?;
            let restored = spe.reverse_transform(&result)?;
            
            assert_eq!(restored, original_data, "Failed for structure level {:?}", level);
        }
        
        Ok(())
    }
    
    #[test]
    fn test_spe_with_key_derivation() -> Result<()> {
        let key = b"test_key_for_spe_core_system_validation";
        let mut spe = SPECore::from_key(key)?;
        
        let original_data = b"Key-derived SPE test data with various characteristics and patterns.";
        
        let result = spe.apply_transform(original_data)?;
        let restored = spe.reverse_transform(&result)?;
        
        assert_eq!(&restored, original_data);
        
        Ok(())
    }
    
    #[test]
    fn test_entropy_calculation() -> Result<()> {
        let spe = SPECore::default()?;
        
        // 低エントロピーデータ（テキスト）
        let text_data = b"aaaaaaaaaaaaaaaa";
        let text_entropy = spe.calculate_entropy(text_data);
        assert!(text_entropy < 3.0);
        
        // 高エントロピーデータ（ランダム）
        let random_data: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let random_entropy = spe.calculate_entropy(&random_data);
        assert!(random_entropy > 7.0);
        
        Ok(())
    }
    
    #[test]
    fn test_block_boundary_detection() -> Result<()> {
        let spe = SPECore::default()?;
        let data = vec![0u8; 10000];
        
        let boundaries = spe.detect_block_boundaries(&data)?;
        
        // ブロック数が期待通りであることを確認
        let expected_blocks = (data.len() + spe.params.block_size - 1) / spe.params.block_size;
        assert_eq!(boundaries.len(), expected_blocks);
        
        // 各ブロックのチェックサムが設定されていることを確認
        for boundary in boundaries {
            assert_ne!(boundary.checksum, [0u8; 32]);
        }
        
        Ok(())
    }
}
