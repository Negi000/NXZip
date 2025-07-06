use anyhow::Result;
use std::path::Path;
use tokio::fs;
use indicatif::{ProgressBar, ProgressStyle};

use crate::engine::{Compressor, Decompressor, CompressionAlgorithm};
use crate::crypto::{Encryptor, Decryptor};
use crate::formats::nxz::NxzFile;
use crate::utils::hasher::FileHasher;

/// ファイル圧縮コマンドの実行
pub async fn compress_command(
    input: &str,
    output: &str,
    encrypt: bool,
    password: Option<String>,
    algorithm: &str,
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
    let compression_algo = match algorithm {
        "zstd" => CompressionAlgorithm::Zstd,
        "lzma2" => CompressionAlgorithm::Lzma2,
        "auto" => CompressionAlgorithm::Auto,
        _ => {
            println!("⚠️  不明な圧縮アルゴリズム '{}', 自動選択を使用します", algorithm);
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
    
    // 1. 元ファイル読み込み
    pb.set_message("ファイル読み込み中...");
    let input_data = fs::read(input_path).await?;
    pb.inc(file_size / 4);
    
    // 2. SPE暗号化 (構造保持暗号) - スタブ実装
    pb.set_message("SPE構造保持暗号処理中...");
    let spe_data = crate::engine::spe_stub::apply_spe_transform(&input_data)?;
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
        let encryptor = Encryptor::new(&password)?;
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
        println!("  暗号化: 有効 (AES-GCM)");
    }
    
    Ok(())
}

/// ファイル展開コマンドの実行
pub async fn extract_command(
    input: &str,
    output: Option<String>,
    password: Option<String>,
) -> Result<()> {
    println!("📦 NXZip展開を開始します...");
    println!("入力: {}", input);
    
    // 入力ファイルの検証
    let input_path = Path::new(input);
    if !input_path.exists() {
        anyhow::bail!("入力ファイルが存在しません: {}", input);
    }
    
    // NXZファイルの読み込み
    let pb = ProgressBar::new_spinner();
    pb.set_message("NXZファイル解析中...");
    let nxz_file = NxzFile::read_from_file(input).await?;
    
    // セキュリティ強化暗号化の復号 (必要な場合)
    pb.set_message("復号処理中...");
    let compressed_data = if nxz_file.is_encrypted() {
        let password = password.ok_or_else(|| anyhow::anyhow!("暗号化ファイルの復号にはパスワードが必要です"))?;
        let decryptor = Decryptor::new(&password)?;
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
