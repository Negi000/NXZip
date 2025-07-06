use anyhow::Result;
use std::sync::{Arc, Mutex};
use tokio::task;
use tokio::fs;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::path::Path;

use crate::engine::{Compressor, CompressionAlgorithm};
use crate::crypto::{Encryptor, EncryptionAlgorithm};
use crate::formats::nxz::NxzFile;

/// マルチスレッド圧縮設定
#[derive(Debug, Clone)]
pub struct MultiThreadConfig {
    /// 並列スレッド数（0の場合はCPUコア数）
    pub thread_count: usize,
    /// チャンクサイズ（バイト）
    pub chunk_size: usize,
    /// 進捗表示の有効/無効
    pub show_progress: bool,
}

impl Default for MultiThreadConfig {
    fn default() -> Self {
        Self {
            thread_count: 0, // 自動検出
            chunk_size: 4 * 1024 * 1024, // 4MB
            show_progress: true,
        }
    }
}

/// マルチスレッド圧縮器
pub struct MultiThreadCompressor {
    config: MultiThreadConfig,
    compression_algo: CompressionAlgorithm,
    compression_level: u8,
    encryption_algo: Option<EncryptionAlgorithm>,
    password: Option<String>,
}

impl MultiThreadCompressor {
    pub fn new(
        config: MultiThreadConfig,
        compression_algo: CompressionAlgorithm,
        compression_level: u8,
    ) -> Self {
        Self {
            config,
            compression_algo,
            compression_level,
            encryption_algo: None,
            password: None,
        }
    }
    
    pub fn with_encryption(
        mut self,
        encryption_algo: EncryptionAlgorithm,
        password: String,
    ) -> Self {
        self.encryption_algo = Some(encryption_algo);
        self.password = Some(password);
        self
    }
    
    /// 暗号化設定を後から設定
    pub fn set_encryption(&mut self, encryption_algo: EncryptionAlgorithm, password: String) {
        self.encryption_algo = Some(encryption_algo);
        self.password = Some(password);
    }
    
    /// 大容量ファイルのマルチスレッド圧縮
    pub async fn compress_file(&self, input_path: &Path, output_path: &Path) -> Result<CompressionStats> {
        if !input_path.exists() {
            anyhow::bail!("入力ファイルが存在しません: {}", input_path.display());
        }
        
        let file_size = fs::metadata(input_path).await?.len();
        let thread_count = if self.config.thread_count == 0 {
            num_cpus::get()
        } else {
            self.config.thread_count
        };
        
        println!("🔧 マルチスレッド圧縮を開始します:");
        println!("  ファイル: {}", input_path.display());
        println!("  サイズ: {} bytes", file_size);
        println!("  スレッド数: {}", thread_count);
        println!("  チャンクサイズ: {} MB", self.config.chunk_size / (1024 * 1024));
        
        // ファイルをチャンクに分割して読み込み
        let chunks = self.split_file_into_chunks(input_path, file_size).await?;
        
        // マルチプログレスバー設定
        let multi_progress = Arc::new(MultiProgress::new());
        let main_pb = multi_progress.add(ProgressBar::new(chunks.len() as u64));
        main_pb.set_style(
            ProgressStyle::with_template(
                "🗜️  [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} chunks ({eta})"
            )?
            .progress_chars("#>-"),
        );
        
        // 並列圧縮実行
        let compressed_chunks = self.compress_chunks_parallel(chunks, thread_count, &multi_progress).await?;
        
        main_pb.finish_with_message("✅ 全チャンク圧縮完了!");
        
        // チャンク数を記録
        let chunks_count = compressed_chunks.len();
        
        // チャンクを結合してNXZファイル作成
        let final_data = self.combine_compressed_chunks(compressed_chunks)?;
        let is_encrypted = self.encryption_algo.is_some();
        
        let nxz_file = NxzFile::new(
            &final_data,
            file_size,
            self.compression_algo,
            is_encrypted,
            self.encryption_algo,
            self.compression_level,
        )?;
        
        nxz_file.write_to_file(output_path.to_str().unwrap()).await?;
        
        let output_size = fs::metadata(output_path).await?.len();
        let compression_ratio = ((file_size as f64 - output_size as f64) / file_size as f64) * 100.0;
        
        Ok(CompressionStats {
            original_size: file_size,
            compressed_size: output_size,
            compression_ratio,
            thread_count,
            chunks_processed: chunks_count,
        })
    }
    
    /// ファイルをチャンクに分割
    async fn split_file_into_chunks(&self, file_path: &Path, file_size: u64) -> Result<Vec<Vec<u8>>> {
        let mut chunks = Vec::new();
        let mut offset = 0u64;
        
        while offset < file_size {
            let chunk_size = std::cmp::min(self.config.chunk_size as u64, file_size - offset);
            let mut chunk = vec![0u8; chunk_size as usize];
            
            let file = fs::File::open(file_path).await?;
            use tokio::io::{AsyncReadExt, AsyncSeekExt};
            let mut file = file;
            file.seek(std::io::SeekFrom::Start(offset)).await?;
            file.read_exact(&mut chunk).await?;
            
            chunks.push(chunk);
            offset += chunk_size;
        }
        
        Ok(chunks)
    }
    
    /// チャンクを並列で圧縮
    async fn compress_chunks_parallel(
        &self,
        chunks: Vec<Vec<u8>>,
        thread_count: usize,
        multi_progress: &Arc<MultiProgress>,
    ) -> Result<Vec<CompressedChunk>> {
        let chunks = Arc::new(chunks);
        let chunk_count = chunks.len();
        let completed = Arc::new(Mutex::new(0));
        
        // ワーカータスクを起動
        let mut handles = Vec::new();
        let chunk_per_thread = (chunk_count + thread_count - 1) / thread_count;
        
        for thread_id in 0..thread_count {
            let chunks = Arc::clone(&chunks);
            let completed = Arc::clone(&completed);
            let multi_progress = Arc::clone(multi_progress);
            let compressor_config = (self.compression_algo, self.compression_level);
            let encryption_config = (self.encryption_algo, self.password.clone());
            
            let handle = task::spawn(async move {
                let start_idx = thread_id * chunk_per_thread;
                let end_idx = std::cmp::min(start_idx + chunk_per_thread, chunk_count);
                
                // 処理すべきチャンクが存在しない場合は空の結果を返す
                if start_idx >= chunk_count {
                    return Ok::<Vec<CompressedChunk>, anyhow::Error>(Vec::new());
                }
                
                let chunk_range = if end_idx > start_idx { end_idx - start_idx } else { 0 };
                let pb = multi_progress.add(ProgressBar::new(chunk_range as u64));
                pb.set_style(
                    ProgressStyle::with_template(
                        &format!("Thread {} [{{bar:20.green/blue}}] {{pos}}/{{len}}", thread_id)
                    ).unwrap()
                    .progress_chars("#>-"),
                );
                
                let mut thread_results = Vec::new();
                
                for chunk_idx in start_idx..end_idx {
                    // SPE変換
                    let spe_data = crate::engine::spe_stub::apply_spe_transform(&chunks[chunk_idx])?;
                    
                    // 圧縮
                    let compressor = Compressor::new(compressor_config.0, compressor_config.1);
                    let compressed = compressor.compress(&spe_data)?;
                    
                    // 暗号化（オプション）
                    let final_data = if let (Some(algo), Some(password)) = (encryption_config.0, &encryption_config.1) {
                        let encryptor = Encryptor::with_algorithm(password, algo)?;
                        encryptor.encrypt(&compressed)?
                    } else {
                        compressed
                    };
                    
                    thread_results.push(CompressedChunk {
                        index: chunk_idx,
                        data: final_data,
                        original_size: chunks[chunk_idx].len(),
                    });
                    
                    pb.inc(1);
                    
                    // 完了カウント更新
                    {
                        let mut count = completed.lock().unwrap();
                        *count += 1;
                    }
                }
                
                pb.finish_with_message("Thread 完了");
                Ok::<Vec<CompressedChunk>, anyhow::Error>(thread_results)
            });
            
            handles.push(handle);
        }
        
        // 全タスクの完了を待機
        let mut all_chunks = Vec::new();
        for handle in handles {
            let thread_chunks = handle.await??;
            all_chunks.extend(thread_chunks);
        }
        
        // インデックス順にソート
        all_chunks.sort_by_key(|chunk| chunk.index);
        
        Ok(all_chunks)
    }
    
    /// 圧縮されたチャンクを結合
    fn combine_compressed_chunks(&self, chunks: Vec<CompressedChunk>) -> Result<Vec<u8>> {
        let mut combined = Vec::new();
        
        // ヘッダ: チャンク数 + 各チャンクのサイズ情報
        combined.extend_from_slice(&(chunks.len() as u32).to_le_bytes());
        
        for chunk in &chunks {
            combined.extend_from_slice(&(chunk.data.len() as u32).to_le_bytes());
            combined.extend_from_slice(&(chunk.original_size as u32).to_le_bytes());
        }
        
        // データ部: 各チャンクのデータ
        for chunk in chunks {
            combined.extend_from_slice(&chunk.data);
        }
        
        Ok(combined)
    }
}

/// 圧縮されたチャンク
#[derive(Debug, Clone)]
struct CompressedChunk {
    index: usize,
    data: Vec<u8>,
    original_size: usize,
}

/// 圧縮統計情報
#[derive(Debug)]
pub struct CompressionStats {
    pub original_size: u64,
    pub compressed_size: u64,
    pub compression_ratio: f64,
    pub thread_count: usize,
    pub chunks_processed: usize,
}

impl CompressionStats {
    pub fn print_summary(&self) {
        println!();
        println!("📊 マルチスレッド圧縮結果:");
        println!("  元サイズ: {} bytes ({:.2} MB)", self.original_size, self.original_size as f64 / 1024.0 / 1024.0);
        println!("  圧縮後: {} bytes ({:.2} MB)", self.compressed_size, self.compressed_size as f64 / 1024.0 / 1024.0);
        println!("  圧縮率: {:.2}%", self.compression_ratio);
        println!("  使用スレッド数: {}", self.thread_count);
        println!("  処理チャンク数: {}", self.chunks_processed);
        
        let throughput = self.original_size as f64 / 1024.0 / 1024.0; // MB
        println!("  スループット: {:.2} MB", throughput);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;
    
    #[tokio::test]
    async fn test_multithread_compression() {
        // 大きめのテストデータ作成
        let test_data = "This is a test data for multithread compression. ".repeat(10000);
        
        let mut temp_input = NamedTempFile::new().unwrap();
        temp_input.write_all(test_data.as_bytes()).unwrap();
        let input_path = temp_input.path().to_str().unwrap();
        
        let temp_output = NamedTempFile::new().unwrap();
        let output_path = temp_output.path().to_str().unwrap();
        
        // マルチスレッド圧縮設定
        let config = MultiThreadConfig {
            thread_count: 2,
            chunk_size: 64 * 1024, // 64KB（テスト用に小さく）
            show_progress: false,
        };
        
        let compressor = MultiThreadCompressor::new(
            config,
            CompressionAlgorithm::Zstd,
            6,
        );
        
        // 圧縮実行
        let stats = compressor.compress_file(Path::new(input_path), Path::new(output_path)).await.unwrap();
        
        // 結果確認
        assert!(stats.compressed_size > 0);
        assert!(stats.compressed_size < stats.original_size);
        assert_eq!(stats.thread_count, 2);
        assert!(stats.chunks_processed > 1);
        
        // ファイルが実際に作成されていることを確認
        assert!(std::path::Path::new(output_path).exists());
    }
}
