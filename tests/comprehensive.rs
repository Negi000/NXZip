use std::process::Command;
use tempfile::tempdir;
use std::fs;

/// テストヘルパー関数: NXZipバイナリを実行
fn run_nxzip(args: &[&str]) -> std::process::Output {
    Command::new("cargo")
        .arg("run")
        .arg("--")
        .args(args)
        .output()
        .expect("Failed to execute nxzip command")
}

/// テストヘルパー関数: コマンドの成功を確認
fn assert_command_success(output: &std::process::Output, command: &str) {
    if !output.status.success() {
        panic!(
            "{} command failed:\nSTDOUT: {}\nSTDERR: {}",
            command,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn test_basic_compress_extract() {
    let temp_dir = tempdir().unwrap();
    let temp_dir_path = temp_dir.path();
    
    // テストファイル作成
    let test_content = "Hello, NXZip! This is a test file for basic compression and extraction.";
    let input_file = temp_dir_path.join("test_input.txt");
    let output_file = temp_dir_path.join("test_output.nxz");
    let extracted_file = temp_dir_path.join("test_extracted.txt");
    
    fs::write(&input_file, test_content).unwrap();
    
    // 圧縮テスト
    let compress_output = run_nxzip(&[
        "compress",
        "--input", input_file.to_str().unwrap(),
        "--output", output_file.to_str().unwrap(),
    ]);
    assert_command_success(&compress_output, "compress");
    assert!(output_file.exists());
    
    // 展開テスト
    let extract_output = run_nxzip(&[
        "extract",
        "--input", output_file.to_str().unwrap(),
        "--output", extracted_file.to_str().unwrap(),
    ]);
    assert_command_success(&extract_output, "extract");
    
    // 内容確認
    let extracted_content = fs::read_to_string(&extracted_file).unwrap();
    assert_eq!(test_content, extracted_content);
}

#[test]
fn test_encrypted_compress_extract() {
    let temp_dir = tempdir().unwrap();
    let temp_dir_path = temp_dir.path();
    
    // テストファイル作成
    let test_content = "This is encrypted test data for NXZip.";
    let input_file = temp_dir_path.join("encrypted_input.txt");
    let output_file = temp_dir_path.join("encrypted_output.nxz");
    let extracted_file = temp_dir_path.join("encrypted_extracted.txt");
    let password = "test_password_123";
    
    fs::write(&input_file, test_content).unwrap();
    
    // 暗号化圧縮テスト
    let compress_output = run_nxzip(&[
        "compress",
        "--input", input_file.to_str().unwrap(),
        "--output", output_file.to_str().unwrap(),
        "--encrypt",
        "--password", password,
        "--encryption", "aes-gcm",
    ]);
    assert_command_success(&compress_output, "encrypt compress");
    assert!(output_file.exists());
    
    // 復号展開テスト
    let extract_output = run_nxzip(&[
        "extract",
        "--input", output_file.to_str().unwrap(),
        "--output", extracted_file.to_str().unwrap(),
        "--password", password,
    ]);
    assert_command_success(&extract_output, "decrypt extract");
    
    // 内容確認
    let extracted_content = fs::read_to_string(&extracted_file).unwrap();
    assert_eq!(test_content, extracted_content);
}

#[test]
fn test_xchacha20_encryption() {
    let temp_dir = tempdir().unwrap();
    let temp_dir_path = temp_dir.path();
    
    // テストファイル作成
    let test_content = "XChaCha20 encryption test data for NXZip.";
    let input_file = temp_dir_path.join("xchacha_input.txt");
    let output_file = temp_dir_path.join("xchacha_output.nxz");
    let extracted_file = temp_dir_path.join("xchacha_extracted.txt");
    let password = "xchacha_test_password";
    
    fs::write(&input_file, test_content).unwrap();
    
    // XChaCha20暗号化圧縮テスト
    let compress_output = run_nxzip(&[
        "compress",
        "--input", input_file.to_str().unwrap(),
        "--output", output_file.to_str().unwrap(),
        "--encrypt",
        "--password", password,
        "--encryption", "xchacha20",
    ]);
    assert_command_success(&compress_output, "xchacha20 compress");
    assert!(output_file.exists());
    
    // 復号展開テスト
    let extract_output = run_nxzip(&[
        "extract",
        "--input", output_file.to_str().unwrap(),
        "--output", extracted_file.to_str().unwrap(),
        "--password", password,
    ]);
    assert_command_success(&extract_output, "xchacha20 extract");
    
    // 内容確認
    let extracted_content = fs::read_to_string(&extracted_file).unwrap();
    assert_eq!(test_content, extracted_content);
}

#[test]
fn test_sec_compress_extract_pbkdf2() {
    let temp_dir = tempdir().unwrap();
    let temp_dir_path = temp_dir.path();
    
    // テストファイル作成
    let test_content = "Security enhanced .nxz.sec format test with PBKDF2.";
    let input_file = temp_dir_path.join("sec_input.txt");
    let output_file = temp_dir_path.join("sec_output.nxz.sec");
    let extracted_file = temp_dir_path.join("sec_extracted.txt");
    let password = "secure_password_123";
    
    fs::write(&input_file, test_content).unwrap();
    
    // セキュリティ強化圧縮テスト
    let compress_output = run_nxzip(&[
        "sec-compress",
        "--input", input_file.to_str().unwrap(),
        "--output", output_file.to_str().unwrap(),
        "--password", password,
        "--encryption", "aes-gcm",
        "--kdf", "pbkdf2",
    ]);
    assert_command_success(&compress_output, "sec-compress");
    assert!(output_file.exists());
    
    // セキュリティ強化展開テスト
    let extract_output = run_nxzip(&[
        "sec-extract",
        "--input", output_file.to_str().unwrap(),
        "--output", extracted_file.to_str().unwrap(),
        "--password", password,
    ]);
    assert_command_success(&extract_output, "sec-extract");
    
    // 内容確認
    let extracted_content = fs::read_to_string(&extracted_file).unwrap();
    assert_eq!(test_content, extracted_content);
}

#[test]
fn test_sec_compress_extract_argon2() {
    let temp_dir = tempdir().unwrap();
    let temp_dir_path = temp_dir.path();
    
    // テストファイル作成
    let test_content = "Security enhanced .nxz.sec format test with Argon2.";
    let input_file = temp_dir_path.join("argon2_input.txt");
    let output_file = temp_dir_path.join("argon2_output.nxz.sec");
    let extracted_file = temp_dir_path.join("argon2_extracted.txt");
    let password = "argon2_secure_password";
    
    fs::write(&input_file, test_content).unwrap();
    
    // Argon2セキュリティ強化圧縮テスト
    let compress_output = run_nxzip(&[
        "sec-compress",
        "--input", input_file.to_str().unwrap(),
        "--output", output_file.to_str().unwrap(),
        "--password", password,
        "--encryption", "xchacha20",
        "--kdf", "argon2",
    ]);
    assert_command_success(&compress_output, "sec-compress with argon2");
    assert!(output_file.exists());
    
    // Argon2セキュリティ強化展開テスト
    let extract_output = run_nxzip(&[
        "sec-extract",
        "--input", output_file.to_str().unwrap(),
        "--output", extracted_file.to_str().unwrap(),
        "--password", password,
    ]);
    assert_command_success(&extract_output, "sec-extract with argon2");
    
    // 内容確認
    let extracted_content = fs::read_to_string(&extracted_file).unwrap();
    assert_eq!(test_content, extracted_content);
}

#[test]
fn test_compression_algorithms() {
    let temp_dir = tempdir().unwrap();
    let temp_dir_path = temp_dir.path();
    
    // テストファイル作成（少し大きめのデータ）
    let test_content = "This is a larger test file for compression algorithm testing. ".repeat(100);
    let input_file = temp_dir_path.join("algo_input.txt");
    fs::write(&input_file, &test_content).unwrap();
    
    let algorithms = ["zstd", "lzma2", "auto"];
    
    for algorithm in &algorithms {
        let output_file = temp_dir_path.join(format!("algo_output_{}.nxz", algorithm));
        let extracted_file = temp_dir_path.join(format!("algo_extracted_{}.txt", algorithm));
        
        // アルゴリズム指定圧縮テスト
        let compress_output = run_nxzip(&[
            "compress",
            "--input", input_file.to_str().unwrap(),
            "--output", output_file.to_str().unwrap(),
            "--algorithm", algorithm,
        ]);
        assert_command_success(&compress_output, &format!("{} compress", algorithm));
        assert!(output_file.exists());
        
        // 展開テスト
        let extract_output = run_nxzip(&[
            "extract",
            "--input", output_file.to_str().unwrap(),
            "--output", extracted_file.to_str().unwrap(),
        ]);
        assert_command_success(&extract_output, &format!("{} extract", algorithm));
        
        // 内容確認
        let extracted_content = fs::read_to_string(&extracted_file).unwrap();
        assert_eq!(test_content, extracted_content);
    }
}

#[test]
fn test_info_command() {
    let temp_dir = tempdir().unwrap();
    let temp_dir_path = temp_dir.path();
    
    // テストファイル作成
    let test_content = "Information display test file for NXZip.";
    let input_file = temp_dir_path.join("info_input.txt");
    let output_file = temp_dir_path.join("info_output.nxz");
    let password = "info_test_password";
    
    fs::write(&input_file, test_content).unwrap();
    
    // 暗号化圧縮
    let compress_output = run_nxzip(&[
        "compress",
        "--input", input_file.to_str().unwrap(),
        "--output", output_file.to_str().unwrap(),
        "--encrypt",
        "--password", password,
        "--encryption", "xchacha20",
    ]);
    assert_command_success(&compress_output, "info test compress");
    
    // 情報表示テスト
    let info_output = run_nxzip(&[
        "info",
        "--input", output_file.to_str().unwrap(),
    ]);
    assert_command_success(&info_output, "info");
    
    // 出力内容確認
    let info_str = String::from_utf8_lossy(&info_output.stdout);
    assert!(info_str.contains("XChaCha20-Poly1305")); // 暗号化アルゴリズム表示
    assert!(info_str.contains("暗号化: 有効")); // 暗号化状態
    assert!(info_str.contains("SPE変換: 有効")); // SPE変換
}

#[test]
fn test_wrong_password() {
    let temp_dir = tempdir().unwrap();
    let temp_dir_path = temp_dir.path();
    
    // テストファイル作成
    let test_content = "Wrong password test file.";
    let input_file = temp_dir_path.join("wrong_pwd_input.txt");
    let output_file = temp_dir_path.join("wrong_pwd_output.nxz");
    let extracted_file = temp_dir_path.join("wrong_pwd_extracted.txt");
    let correct_password = "correct_password";
    let wrong_password = "wrong_password";
    
    fs::write(&input_file, test_content).unwrap();
    
    // 正しいパスワードで暗号化圧縮
    let compress_output = run_nxzip(&[
        "compress",
        "--input", input_file.to_str().unwrap(),
        "--output", output_file.to_str().unwrap(),
        "--encrypt",
        "--password", correct_password,
    ]);
    assert_command_success(&compress_output, "wrong password test compress");
    
    // 間違ったパスワードで展開試行（失敗すべき）
    let extract_output = run_nxzip(&[
        "extract",
        "--input", output_file.to_str().unwrap(),
        "--output", extracted_file.to_str().unwrap(),
        "--password", wrong_password,
    ]);
    assert!(!extract_output.status.success()); // 失敗することを確認
}

#[test]
fn test_file_not_found() {
    let temp_dir = tempdir().unwrap();
    let temp_dir_path = temp_dir.path();
    
    let non_existent_file = temp_dir_path.join("non_existent.txt");
    let output_file = temp_dir_path.join("output.nxz");
    
    // 存在しないファイルを圧縮試行（失敗すべき）
    let compress_output = run_nxzip(&[
        "compress",
        "--input", non_existent_file.to_str().unwrap(),
        "--output", output_file.to_str().unwrap(),
    ]);
    assert!(!compress_output.status.success()); // 失敗することを確認
}
