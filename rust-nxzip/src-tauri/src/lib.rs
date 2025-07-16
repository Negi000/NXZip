use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use tauri::command;
use anyhow::Result;

// NXZipライブラリのインポート
use nxzip::{
    engine::{Compressor, CompressionAlgorithm, Decompressor},
    crypto::EncryptionAlgorithm,
    formats::{nxz::NxzFile, nxz_sec::{NxzSecFile, KdfType}},
};

#[derive(Debug, Serialize, Deserialize)]
pub struct CompressionOptions {
    pub algorithm: String,
    pub level: u8,
    pub encryption: Option<String>,
    pub password: Option<String>,
    pub kdf: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CompressionResult {
    pub success: bool,
    pub message: String,
    pub output_path: Option<String>,
    pub compression_ratio: Option<f64>,
    pub original_size: Option<u64>,
    pub compressed_size: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExtractionResult {
    pub success: bool,
    pub message: String,
    pub output_path: Option<String>,
    pub extracted_size: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FileInfo {
    pub filename: String,
    pub original_size: u64,
    pub compressed_size: u64,
    pub compression_ratio: f64,
    pub algorithm: String,
    pub is_encrypted: bool,
    pub encryption_algorithm: Option<String>,
    pub created_at: String,
}

/// ファイルを圧縮する
#[command]
async fn compress_file(
    input_path: String,
    output_path: String,
    options: CompressionOptions,
) -> Result<CompressionResult, String> {
    tokio::task::spawn_blocking(move || {
        let input_data = std::fs::read(&input_path)
            .map_err(|e| format!("ファイル読み込みエラー: {}", e))?;

        let algorithm = match options.algorithm.as_str() {
            "zstd" => CompressionAlgorithm::Zstd,
            "lzma2" => CompressionAlgorithm::Lzma2,
            _ => CompressionAlgorithm::Auto,
        };

        let compressor = Compressor::new(algorithm, options.level);
        let compressed_data = compressor.compress(&input_data)
            .map_err(|e| format!("圧縮エラー: {}", e))?;

        let original_size = input_data.len() as u64;

        // 暗号化が指定されている場合
        if let (Some(encryption), Some(password)) = (&options.encryption, &options.password) {
            let encryption_algo = match encryption.as_str() {
                "xchacha20" => EncryptionAlgorithm::XChaCha20Poly1305,
                _ => EncryptionAlgorithm::AesGcm,
            };

            let kdf = match options.kdf.as_deref() {
                Some("argon2") => KdfType::Argon2,
                _ => KdfType::Pbkdf2,
            };

            // 通常のNXZファイルを作成
            let nxz_file = NxzFile::new(
                &compressed_data,
                original_size,
                algorithm,
                false, // セキュア形式なので暗号化フラグはfalse
                None,
                options.level,
            ).map_err(|e| format!("NXZファイル作成エラー: {}", e))?;

            // セキュア圧縮ファイルとして保存
            let sec_file = NxzSecFile::new(&nxz_file, password, encryption_algo, kdf)
                .map_err(|e| format!("セキュアファイル作成エラー: {}", e))?;
            
            tokio::runtime::Handle::current().block_on(async {
                sec_file.write_to_file(&output_path).await
            }).map_err(|e| format!("暗号化保存エラー: {}", e))?;
        } else {
            // 通常の圧縮ファイルとして保存
            let nxz_file = NxzFile::new(
                &compressed_data,
                original_size,
                algorithm,
                false,
                None,
                options.level,
            ).map_err(|e| format!("NXZファイル作成エラー: {}", e))?;

            tokio::runtime::Handle::current().block_on(async {
                nxz_file.write_to_file(&output_path).await
            }).map_err(|e| format!("保存エラー: {}", e))?;
        }

        let compressed_size = std::fs::metadata(&output_path)
            .map(|m| m.len())
            .unwrap_or(0);

        let compression_ratio = if original_size > 0 {
            (compressed_size as f64 / original_size as f64) * 100.0
        } else {
            0.0
        };

        Ok(CompressionResult {
            success: true,
            message: "圧縮が完了しました".to_string(),
            output_path: Some(output_path),
            compression_ratio: Some(compression_ratio),
            original_size: Some(original_size),
            compressed_size: Some(compressed_size),
        })
    })
    .await
    .map_err(|e| format!("タスク実行エラー: {}", e))?
}

/// ファイルを展開する
#[command]
async fn extract_file(
    input_path: String,
    output_path: String,
    password: Option<String>,
) -> Result<ExtractionResult, String> {
    tokio::task::spawn_blocking(move || {
        // ファイル拡張子で判定
        if input_path.ends_with(".nxz.sec") {
            // セキュア形式の展開
            let password = password.ok_or("暗号化ファイルにはパスワードが必要です")?;
            
            let sec_file = tokio::runtime::Handle::current().block_on(async {
                NxzSecFile::read_from_file(&input_path).await
            }).map_err(|e| format!("暗号化ファイル読み込みエラー: {}", e))?;

            let nxz_file = sec_file.decrypt_nxz(&password)
                .map_err(|e| format!("復号化エラー: {}", e))?;

            let decompressor = Decompressor::new(nxz_file.compression_algorithm());
            let decompressed_data = decompressor.decompress(nxz_file.data())
                .map_err(|e| format!("展開エラー: {}", e))?;

            std::fs::write(&output_path, &decompressed_data)
                .map_err(|e| format!("ファイル書き込みエラー: {}", e))?;

            Ok(ExtractionResult {
                success: true,
                message: "セキュア展開が完了しました".to_string(),
                output_path: Some(output_path),
                extracted_size: Some(decompressed_data.len() as u64),
            })
        } else {
            // 通常形式の展開
            let nxz_file = tokio::runtime::Handle::current().block_on(async {
                NxzFile::read_from_file(&input_path).await
            }).map_err(|e| format!("ファイル読み込みエラー: {}", e))?;

            let decompressor = Decompressor::new(nxz_file.compression_algorithm());
            let decompressed_data = decompressor.decompress(nxz_file.data())
                .map_err(|e| format!("展開エラー: {}", e))?;

            std::fs::write(&output_path, &decompressed_data)
                .map_err(|e| format!("ファイル書き込みエラー: {}", e))?;

            Ok(ExtractionResult {
                success: true,
                message: "展開が完了しました".to_string(),
                output_path: Some(output_path),
                extracted_size: Some(decompressed_data.len() as u64),
            })
        }
    })
    .await
    .map_err(|e| format!("タスク実行エラー: {}", e))?
}

/// ファイル情報を取得する
#[command]
async fn get_file_info(file_path: String) -> Result<FileInfo, String> {
    tokio::task::spawn_blocking(move || {
        if file_path.ends_with(".nxz.sec") {
            // セキュア形式の情報取得（パスワード不要で基本情報のみ）
            let file_size = std::fs::metadata(&file_path)
                .map(|m| m.len())
                .map_err(|e| format!("ファイル情報取得エラー: {}", e))?;

            Ok(FileInfo {
                filename: PathBuf::from(&file_path)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string(),
                original_size: 0, // 復号化が必要なため不明
                compressed_size: file_size,
                compression_ratio: 0.0,
                algorithm: "Unknown (encrypted)".to_string(),
                is_encrypted: true,
                encryption_algorithm: Some("Multiple layers".to_string()),
                created_at: "Unknown".to_string(),
            })
        } else {
            // 通常形式の情報取得
            let nxz_file = tokio::runtime::Handle::current().block_on(async {
                NxzFile::read_from_file(&file_path).await
            }).map_err(|e| format!("ファイル読み込みエラー: {}", e))?;

            let metadata = nxz_file.metadata();
            let file_size = std::fs::metadata(&file_path)
                .map(|m| m.len())
                .map_err(|e| format!("ファイル情報取得エラー: {}", e))?;

            let original_size = nxz_file.original_size();
            let compression_ratio = if original_size > 0 {
                (file_size as f64 / original_size as f64) * 100.0
            } else {
                0.0
            };

            let algorithm_name = match nxz_file.compression_algorithm() {
                CompressionAlgorithm::Zstd => "Zstd",
                CompressionAlgorithm::Lzma2 => "LZMA2",
                CompressionAlgorithm::Auto => "Auto",
            };

            Ok(FileInfo {
                filename: metadata.filename().unwrap_or("unknown".to_string()),
                original_size,
                compressed_size: file_size,
                compression_ratio,
                algorithm: algorithm_name.to_string(),
                is_encrypted: nxz_file.is_encrypted(),
                encryption_algorithm: if nxz_file.is_encrypted() {
                    nxz_file.encryption_algorithm().map(|algo| match algo {
                        EncryptionAlgorithm::AesGcm => "AES-256-GCM".to_string(),
                        EncryptionAlgorithm::XChaCha20Poly1305 => "XChaCha20-Poly1305".to_string(),
                    })
                } else {
                    None
                },
                created_at: metadata.created_at().to_rfc3339(),
            })
        }
    })
    .await
    .map_err(|e| format!("タスク実行エラー: {}", e))?
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            compress_file,
            extract_file,
            get_file_info
        ])
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
