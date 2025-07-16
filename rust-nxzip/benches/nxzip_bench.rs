use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use nxzip::engine::{Compressor, Decompressor, CompressionAlgorithm};
use nxzip::crypto::{Encryptor, Decryptor, EncryptionAlgorithm};
use nxzip::engine::spe_stub::{apply_spe_transform, reverse_spe_transform};
use nxzip::formats::nxz::NxzFile;
use nxzip::formats::nxz_sec::{NxzSecFile, KdfType};

fn generate_test_data(size: usize) -> Vec<u8> {
    // 様々なパターンのテストデータを生成
    let mut data = Vec::with_capacity(size);
    for i in 0..size {
        data.push((i % 256) as u8);
    }
    data
}

fn generate_text_data(size: usize) -> Vec<u8> {
    // テキストっぽいデータを生成（圧縮しやすい）
    let pattern = "The quick brown fox jumps over the lazy dog. ";
    let mut data = Vec::with_capacity(size);
    let pattern_bytes = pattern.as_bytes();
    
    while data.len() < size {
        let remaining = size - data.len();
        if remaining >= pattern_bytes.len() {
            data.extend_from_slice(pattern_bytes);
        } else {
            data.extend_from_slice(&pattern_bytes[..remaining]);
        }
    }
    data
}

fn generate_random_data(size: usize) -> Vec<u8> {
    // ランダムデータを生成（圧縮しにくい）
    use rand::RngCore;
    let mut data = vec![0u8; size];
    rand::thread_rng().fill_bytes(&mut data);
    data
}

fn bench_spe_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("SPE Transform");
    
    for size in [1024, 10240, 102400].iter() {
        let data = generate_test_data(*size);
        
        group.bench_with_input(
            BenchmarkId::new("apply", size),
            &data,
            |b, data| {
                b.iter(|| apply_spe_transform(black_box(data)).unwrap())
            },
        );
        
        let transformed = apply_spe_transform(&data).unwrap();
        group.bench_with_input(
            BenchmarkId::new("reverse", size),
            &transformed,
            |b, data| {
                b.iter(|| reverse_spe_transform(black_box(data)).unwrap())
            },
        );
    }
    
    group.finish();
}

fn bench_compression_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("Compression Algorithms");
    
    let algorithms = [
        CompressionAlgorithm::Zstd,
        CompressionAlgorithm::Lzma2,
    ];
    
    for algorithm in &algorithms {
        for size in [1024, 10240, 102400].iter() {
            // テキストデータ（圧縮しやすい）
            let text_data = generate_text_data(*size);
            let compressor = Compressor::new(*algorithm, 6);
            
            group.bench_with_input(
                BenchmarkId::new(format!("{:?}_text_compress", algorithm), size),
                &text_data,
                |b, data| {
                    b.iter(|| compressor.compress(black_box(data)).unwrap())
                },
            );
            
            let compressed = compressor.compress(&text_data).unwrap();
            let decompressor = Decompressor::new(*algorithm);
            
            group.bench_with_input(
                BenchmarkId::new(format!("{:?}_text_decompress", algorithm), size),
                &compressed,
                |b, data| {
                    b.iter(|| decompressor.decompress(black_box(data)).unwrap())
                },
            );
            
            // ランダムデータ（圧縮しにくい）
            let random_data = generate_random_data(*size);
            
            group.bench_with_input(
                BenchmarkId::new(format!("{:?}_random_compress", algorithm), size),
                &random_data,
                |b, data| {
                    b.iter(|| compressor.compress(black_box(data)).unwrap())
                },
            );
        }
    }
    
    group.finish();
}

fn bench_encryption_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("Encryption Algorithms");
    
    let algorithms = [
        EncryptionAlgorithm::AesGcm,
        EncryptionAlgorithm::XChaCha20Poly1305,
    ];
    
    let password = "benchmark_password_123";
    
    for algorithm in &algorithms {
        for size in [1024, 10240, 102400].iter() {
            let data = generate_test_data(*size);
            let encryptor = Encryptor::with_algorithm(password, *algorithm).unwrap();
            
            group.bench_with_input(
                BenchmarkId::new(format!("{:?}_encrypt", algorithm), size),
                &data,
                |b, data| {
                    b.iter(|| encryptor.encrypt(black_box(data)).unwrap())
                },
            );
            
            let encrypted = encryptor.encrypt(&data).unwrap();
            let decryptor = Decryptor::with_algorithm(password, *algorithm).unwrap();
            
            group.bench_with_input(
                BenchmarkId::new(format!("{:?}_decrypt", algorithm), size),
                &encrypted,
                |b, data| {
                    b.iter(|| decryptor.decrypt(black_box(data)).unwrap())
                },
            );
        }
    }
    
    group.finish();
}

fn bench_nxz_file_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("NXZ File Operations");
    
    for size in [1024, 10240, 102400].iter() {
        let data = generate_test_data(*size);
        
        // 基本NXZファイル作成
        group.bench_with_input(
            BenchmarkId::new("nxz_create", size),
            &data,
            |b, data| {
                b.iter(|| {
                    NxzFile::new(
                        black_box(data),
                        data.len() as u64,
                        CompressionAlgorithm::Zstd,
                        false,
                        None,
                        6,
                    ).unwrap()
                })
            },
        );
        
        // 暗号化NXZファイル作成
        group.bench_with_input(
            BenchmarkId::new("nxz_create_encrypted", size),
            &data,
            |b, data| {
                b.iter(|| {
                    NxzFile::new(
                        black_box(data),
                        data.len() as u64,
                        CompressionAlgorithm::Zstd,
                        true,
                        Some(EncryptionAlgorithm::AesGcm),
                        6,
                    ).unwrap()
                })
            },
        );
        
        // NXZファイルのバイナリ変換
        let nxz_file = NxzFile::new(
            &data,
            data.len() as u64,
            CompressionAlgorithm::Zstd,
            false,
            None,
            6,
        ).unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("nxz_to_bytes", size),
            &nxz_file,
            |b, nxz| {
                b.iter(|| black_box(nxz).to_bytes().unwrap())
            },
        );
    }
    
    group.finish();
}

fn bench_nxz_sec_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("NXZ Sec Operations");
    
    let password = "sec_benchmark_password";
    
    for size in [1024, 10240].iter() { // .nxz.secは少し重いので小さめのサイズ
        let data = generate_test_data(*size);
        let nxz_file = NxzFile::new(
            &data,
            data.len() as u64,
            CompressionAlgorithm::Zstd,
            false,
            None,
            6,
        ).unwrap();
        
        // PBKDF2
        group.bench_with_input(
            BenchmarkId::new("sec_create_pbkdf2", size),
            &nxz_file,
            |b, nxz| {
                b.iter(|| {
                    NxzSecFile::new(
                        black_box(nxz),
                        password,
                        EncryptionAlgorithm::AesGcm,
                        KdfType::Pbkdf2,
                    ).unwrap()
                })
            },
        );
        
        // Argon2（より重い）
        group.bench_with_input(
            BenchmarkId::new("sec_create_argon2", size),
            &nxz_file,
            |b, nxz| {
                b.iter(|| {
                    NxzSecFile::new(
                        black_box(nxz),
                        password,
                        EncryptionAlgorithm::AesGcm,
                        KdfType::Argon2,
                    ).unwrap()
                })
            },
        );
    }
    
    group.finish();
}

fn bench_end_to_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("End-to-End");
    
    // 完全なワークフロー: 元データ -> SPE -> 圧縮 -> 暗号化 -> 復号 -> 展開 -> SPE逆変換
    for size in [1024, 10240].iter() {
        let data = generate_text_data(*size);
        let password = "e2e_test_password";
        
        group.bench_with_input(
            BenchmarkId::new("complete_workflow", size),
            &data,
            |b, data| {
                b.iter(|| {
                    // SPE変換
                    let spe_data = apply_spe_transform(black_box(data)).unwrap();
                    
                    // 圧縮
                    let compressor = Compressor::new(CompressionAlgorithm::Zstd, 6);
                    let compressed = compressor.compress(&spe_data).unwrap();
                    
                    // 暗号化
                    let encryptor = Encryptor::with_algorithm(password, EncryptionAlgorithm::AesGcm).unwrap();
                    let encrypted = encryptor.encrypt(&compressed).unwrap();
                    
                    // 復号
                    let decryptor = Decryptor::with_algorithm(password, EncryptionAlgorithm::AesGcm).unwrap();
                    let decrypted = decryptor.decrypt(&encrypted).unwrap();
                    
                    // 展開
                    let decompressor = Decompressor::new(CompressionAlgorithm::Zstd);
                    let decompressed = decompressor.decompress(&decrypted).unwrap();
                    
                    // SPE逆変換
                    let original = reverse_spe_transform(&decompressed).unwrap();
                    
                    assert_eq!(&original, data);
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_spe_transform,
    bench_compression_algorithms,
    bench_encryption_algorithms,
    bench_nxz_file_operations,
    bench_nxz_sec_operations,
    bench_end_to_end
);
criterion_main!(benches);
