use anyhow::Result;
use std::path::Path;
use tokio::fs;
use indicatif::{ProgressBar, ProgressStyle};

use crate::engine::{Compressor, Decompressor, CompressionAlgorithm};
use crate::crypto::{Encryptor, Decryptor, EncryptionAlgorithm};
use crate::formats::nxz::NxzFile;
use crate::formats::nxz_sec::{NxzSecFile, KdfType};
use crate::utils::hasher::FileHasher;

/// ファイル圧縮コマンドの実行
pub async fn compress_command(
    input: &str,
    output: &str,
    encrypt: bool,
    password: Option<String>,
    encryption_algo: &str,
    compression_algo: &str,
    level: u8,
) -> Result<()> {
    println!("🚀 NXZip圧縮を開始します...");
    println!("入力: {}", input);
    println!("出力: {}", output);
    
    // 入力ファイルの検証
    let input_path = Path::new(input);
    if !input_path.exists() {
        anyhow::bail!("入力ファイルが存在しません: {}", input);
    }
    
    // 圧縮アルゴリズムの選択
    let compression_algo = match compression_algo {
        "zstd" => CompressionAlgorithm::Zstd,
        "lzma2" => CompressionAlgorithm::Lzma2,
        "auto" => CompressionAlgorithm::Auto,
        _ => {
            println!("⚠️  不明な圧縮アルゴリズム '{}', 自動選択を使用します", compression_algo);
            CompressionAlgorithm::Auto
        }
    };
    
    // 暗号化アルゴリズムの選択
    let encryption_algorithm = match encryption_algo {
        "aes-gcm" => EncryptionAlgorithm::AesGcm,
        "xchacha20" => EncryptionAlgorithm::XChaCha20Poly1305,
        _ => {
            println!("⚠️  不明な暗号化アルゴリズム '{}', AES-GCMを使用します", encryption_algo);
            EncryptionAlgorithm::AesGcm
        }
    };
    
    // ファイルサイズ取得とプログレスバー設定
    let file_size = fs::metadata(input_path).await?.len();
    let pb = ProgressBar::new(file_size);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})"
        )?
        .progress_chars("#>-"),
    );
    
    // 1. ファイル読み込み
    pb.set_message("ファイル読み込み中...");
    let original_data = fs::read(input_path).await?;
    pb.inc(file_size / 4);
    
    // 2. SPE変換 (構造保持難読化)
    pb.set_message("SPE変換中...");
    let spe_data = crate::engine::spe_stub::apply_spe_transform(&original_data)?;
    pb.inc(file_size / 4);
    
    // 3. 圧縮処理
    pb.set_message("圧縮処理中...");
    let compressor = Compressor::new(compression_algo, level);
    let compressed_data = compressor.compress(&spe_data)?;
    pb.inc(file_size / 4);
    
    // 4. セキュリティ強化暗号化 (オプション)
    let final_data = if encrypt {
        pb.set_message("暗号化処理中...");
        let password = password.ok_or_else(|| anyhow::anyhow!("暗号化にはパスワードが必要です"))?;
        let encryptor = Encryptor::with_algorithm(&password, encryption_algorithm)?;
        encryptor.encrypt(&compressed_data)?
    } else {
        compressed_data
    };
    pb.inc(file_size / 4);
    
    // 5. NXZファイル形式で出力
    pb.set_message("ファイル書き込み中...");
    let nxz_file = NxzFile::new(
        &final_data,
        file_size,
        compression_algo,
        encrypt,
        if encrypt { Some(encryption_algorithm) } else { None },
        level,
    )?;
    
    nxz_file.write_to_file(output).await?;
    pb.finish_with_message("✅ 圧縮完了!");
    
    // 圧縮結果の表示
    let output_size = fs::metadata(output).await?.len();
    let compression_ratio = (file_size as f64 - output_size as f64) / file_size as f64 * 100.0;
    
    println!();
    println!("📊 圧縮結果:");
    println!("  元サイズ: {} bytes", file_size);
    println!("  圧縮後: {} bytes", output_size);
    println!("  圧縮率: {:.2}%", compression_ratio);
    println!("  アルゴリズム: {:?}", compression_algo);
    if encrypt {
        let encryption_name = match encryption_algorithm {
            EncryptionAlgorithm::AesGcm => "AES-GCM",
            EncryptionAlgorithm::XChaCha20Poly1305 => "XChaCha20-Poly1305",
        };
        println!("  暗号化: 有効 ({})", encryption_name);
    }
    
    Ok(())
}

/// ファイル展開コマンドの実行
pub async fn extract_command(
    input: &str,
    output: Option<String>,
    password: Option<String>,
    encryption_algo: &str,
) -> Result<()> {
    println!("📦 NXZip展開を開始します...");
    println!("入力: {}", input);
    
    // 入力ファイルの検証
    let input_path = Path::new(input);
    if !input_path.exists() {
        anyhow::bail!("入力ファイルが存在しません: {}", input);
    }
    
    // 暗号化アルゴリズムの選択
    let encryption_algorithm = match encryption_algo {
        "aes-gcm" => EncryptionAlgorithm::AesGcm,
        "xchacha20" => EncryptionAlgorithm::XChaCha20Poly1305,
        _ => {
            println!("⚠️  不明な暗号化アルゴリズム '{}', AES-GCMを使用します", encryption_algo);
            EncryptionAlgorithm::AesGcm
        }
    };
    
    // NXZファイル読み込みとプログレスバー設定
    let pb = ProgressBar::new_spinner();
    pb.set_message("NXZファイル解析中...");
    let nxz_file = NxzFile::read_from_file(input).await?;
    
    // セキュリティ強化暗号化の復号 (必要な場合)
    pb.set_message("復号処理中...");
    let compressed_data = if nxz_file.is_encrypted() {
        let password = password.ok_or_else(|| anyhow::anyhow!("暗号化ファイルの復号にはパスワードが必要です"))?;
        let decryptor = Decryptor::with_algorithm(&password, encryption_algorithm)?;
        decryptor.decrypt(&nxz_file.data())?
    } else {
        nxz_file.data().to_vec()
    };
    
    // 圧縮の展開
    pb.set_message("展開処理中...");
    let decompressor = Decompressor::new(nxz_file.compression_algorithm());
    let spe_data = decompressor.decompress(&compressed_data)?;
    
    // SPE逆変換 (構造復元)
    pb.set_message("SPE構造復元中...");
    let original_data = crate::engine::spe_stub::reverse_spe_transform(&spe_data)?;
    
    // 出力先の決定
    let output_path = if let Some(output) = output {
        output
    } else {
        // .nxz拡張子を除去して元のファイル名を推定
        let input_stem = input_path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("extracted_file");
        input_stem.to_string()
    };
    
    // ファイル書き込み
    pb.set_message("ファイル書き込み中...");
    fs::write(&output_path, &original_data).await?;
    
    pb.finish_with_message("✅ 展開完了!");
    
    // 結果表示
    println!();
    println!("📊 展開結果:");
    println!("  出力ファイル: {}", output_path);
    println!("  ファイルサイズ: {} bytes", original_data.len());
    
    // 整合性チェック
    let hasher = FileHasher::new();
    let checksum = hasher.calculate_hash(&original_data)?;
    println!("  SHA256チェックサム: {}", hex::encode(&checksum));
    
    Ok(())
}

/// アーカイブ情報表示コマンドの実行
pub async fn info_command(input: &str) -> Result<()> {
    println!("📋 NXZipアーカイブ情報を取得中...");
    println!("入力: {}", input);
    
    // 入力ファイルの検証
    let input_path = Path::new(input);
    if !input_path.exists() {
        anyhow::bail!("入力ファイルが存在しません: {}", input);
    }
    
    // NXZファイル読み込み
    let nxz_file = NxzFile::read_from_file(input).await?;
    
    // ファイル基本情報
    let file_data = fs::read(input_path).await?;
    println!();
    println!("📊 ファイル情報:");
    println!("  ファイル名: {}", input_path.file_name().unwrap_or_default().to_string_lossy());
    println!("  ファイルサイズ: {} bytes", file_data.len());
    println!("  フォーマット: NXZ v{}", nxz_file.version());
    
    // 圧縮情報
    println!();
    println!("🗜️  圧縮情報:");
    println!("  圧縮アルゴリズム: {:?}", nxz_file.compression_algorithm());
    println!("  元サイズ: {} bytes", nxz_file.original_size());
    println!("  圧縮後サイズ: {} bytes", nxz_file.compressed_data().len());
    
    let compression_ratio = if nxz_file.original_size() > 0 {
        let original = nxz_file.original_size() as i64;
        let compressed = nxz_file.compressed_data().len() as i64;
        ((original - compressed) as f64 / original as f64) * 100.0
    } else {
        0.0
    };
    println!("  圧縮率: {:.2}%", compression_ratio);
    
    // セキュリティ情報
    println!();
    println!("🔒 セキュリティ情報:");
    println!("  暗号化: {}", if nxz_file.is_encrypted() { "有効" } else { "無効" });
    println!("  SPE変換: 有効");
    
    if nxz_file.is_encrypted() {
        if let Some(encryption_algo) = nxz_file.encryption_algorithm() {
            println!("  暗号化アルゴリズム: {}", encryption_algo);
        } else {
            println!("  暗号化アルゴリズム: 不明");
        }
    }
    
    // メタデータ情報
    println!();
    println!("📝 メタデータ:");
    println!("  作成日時: {}", nxz_file.metadata().created_at().format("%Y-%m-%d %H:%M:%S"));
    
    // ハッシュ情報
    let hasher = FileHasher::new();
    let file_hash = hasher.calculate_hash(&file_data)?;
    println!("  ファイルハッシュ: {}", hex::encode(&file_hash));
    
    println!();
    println!("✅ 情報取得完了!");
    
    Ok(())
}

/// セキュリティ強化型(.nxz.sec)ファイル圧縮コマンドの実行
pub async fn sec_compress_command(
    input: &str,
    output: &str,
    password: &str,
    encryption_algo: &str,
    kdf_type: &str,
    compression_algo: &str,
    level: u8,
) -> Result<()> {
    println!("🔒 NXZip セキュリティ強化圧縮を開始します...");
    println!("入力: {}", input);
    println!("出力: {}", output);
    
    // 入力ファイルの検証
    let input_path = Path::new(input);
    if !input_path.exists() {
        anyhow::bail!("入力ファイルが存在しません: {}", input);
    }
    
    // 暗号化アルゴリズムの選択
    let encryption_algorithm = match encryption_algo {
        "aes-gcm" => EncryptionAlgorithm::AesGcm,
        "xchacha20" => EncryptionAlgorithm::XChaCha20Poly1305,
        _ => {
            println!("⚠️  不明な暗号化アルゴリズム '{}', AES-GCMを使用します", encryption_algo);
            EncryptionAlgorithm::AesGcm
        }
    };
    
    // KDF方式の選択
    let kdf = match kdf_type {
        "pbkdf2" => KdfType::Pbkdf2,
        "argon2" => KdfType::Argon2,
        _ => {
            println!("⚠️  不明なKDF方式 '{}', PBKDF2を使用します", kdf_type);
            KdfType::Pbkdf2
        }
    };
    
    // 圧縮アルゴリズムの選択
    let compression_algo = match compression_algo {
        "zstd" => CompressionAlgorithm::Zstd,
        "lzma2" => CompressionAlgorithm::Lzma2,
        "auto" => CompressionAlgorithm::Auto,
        _ => {
            println!("⚠️  不明な圧縮アルゴリズム '{}', 自動選択を使用します", compression_algo);
            CompressionAlgorithm::Auto
        }
    };
    
    // ファイルサイズ取得とプログレスバー設定
    let file_size = fs::metadata(input_path).await?.len();
    let pb = ProgressBar::new(file_size);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})"
        )?
        .progress_chars("#>-"),
    );
    
    // 1. ファイル読み込み
    pb.set_message("ファイル読み込み中...");
    let original_data = fs::read(input_path).await?;
    pb.inc(file_size / 5);
    
    // 2. SPE変換
    pb.set_message("SPE変換中...");
    let spe_data = crate::engine::spe_stub::apply_spe_transform(&original_data)?;
    pb.inc(file_size / 5);
    
    // 3. 圧縮処理
    pb.set_message("圧縮処理中...");
    let compressor = Compressor::new(compression_algo, level);
    let compressed_data = compressor.compress(&spe_data)?;
    pb.inc(file_size / 5);
    
    // 4. NXZファイル作成
    pb.set_message("NXZファイル作成中...");
    let nxz_file = NxzFile::new(
        &compressed_data,
        file_size,
        compression_algo,
        false, // .nxz.secでは内部NXZは暗号化しない
        None,
        level,
    )?;
    pb.inc(file_size / 5);
    
    // 5. .nxz.secファイル作成・暗号化
    pb.set_message("セキュリティ強化暗号化中...");
    let nxz_sec_file = NxzSecFile::new(&nxz_file, password, encryption_algorithm, kdf)?;
    nxz_sec_file.write_to_file(output).await?;
    pb.inc(file_size / 5);
    
    pb.finish_with_message("✅ セキュリティ強化圧縮完了!");
    
    // 結果表示
    let output_size = fs::metadata(output).await?.len();
    let compression_ratio = if file_size > 0 {
        ((file_size as i64 - output_size as i64) as f64 / file_size as f64) * 100.0
    } else {
        0.0
    };
    
    println!();
    println!("📊 圧縮結果:");
    println!("  元サイズ: {} bytes", file_size);
    println!("  圧縮後: {} bytes", output_size);
    println!("  圧縮率: {:.2}%", compression_ratio);
    println!("  アルゴリズム: {:?}", compression_algo);
    println!("  暗号化: 有効 ({})", encryption_algorithm);
    println!("  KDF: {:?}", kdf);
    
    Ok(())
}

/// セキュリティ強化型(.nxz.sec)ファイル展開コマンドの実行
pub async fn sec_extract_command(
    input: &str,
    output: Option<String>,
    password: &str,
) -> Result<()> {
    println!("🔓 NXZip セキュリティ強化展開を開始します...");
    println!("入力: {}", input);
    
    // 入力ファイルの検証
    let input_path = Path::new(input);
    if !input_path.exists() {
        anyhow::bail!("入力ファイルが存在しません: {}", input);
    }
    
    // 出力ファイル名の決定
    let output_file = if let Some(out) = output {
        out
    } else {
        // .nxz.secから元のファイル名を推測
        let input_stem = input_path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("extracted");
        
        if input_stem.ends_with(".nxz") {
            input_stem.trim_end_matches(".nxz").to_string()
        } else {
            format!("{}_extracted", input_stem)
        }
    };
    
    println!("出力: {}", output_file);
    
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("{spinner:.green} {msg}")?.tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ")
    );
    
    // 1. .nxz.secファイル読み込み・復号化
    pb.set_message("セキュリティ強化復号化中...");
    let nxz_sec_file = NxzSecFile::read_from_file(input).await?;
    let nxz_file = nxz_sec_file.decrypt_nxz(password)?;
    
    // 2. データ取得・展開
    pb.set_message("展開処理中...");
    let compressed_data = nxz_file.data();
    let decompressor = Decompressor::new(nxz_file.compression_algorithm());
    let spe_data = decompressor.decompress(compressed_data)?;
    
    // 3. SPE逆変換
    pb.set_message("SPE逆変換中...");
    let original_data = crate::engine::spe_stub::reverse_spe_transform(&spe_data)?;
    
    // 4. ファイル書き込み
    pb.set_message("ファイル書き込み中...");
    fs::write(&output_file, &original_data).await?;
    
    pb.finish_with_message("✅ セキュリティ強化展開完了!");
    
    // 結果表示
    println!();
    println!("📊 展開結果:");
    println!("  元サイズ: {} bytes", nxz_file.original_size());
    println!("  展開後: {} bytes", original_data.len());
    println!("  アルゴリズム: {:?}", nxz_file.compression_algorithm());
    println!("  暗号化: セキュリティ強化型(.nxz.sec)");
    
    // 整合性検証
    let expected_size = nxz_file.original_size() as usize;
    if original_data.len() != expected_size {
        println!("⚠️  警告: ファイルサイズが期待値と異なります");
    } else {
        println!("✅ 整合性検証: 正常");
    }
    
    Ok(())
}
