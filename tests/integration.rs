use std::process::Command;
use tempfile::{tempdir, NamedTempFile};
use std::fs;

#[test]
fn test_compress_and_extract() {
    let temp_dir = tempdir().unwrap();
    let temp_dir_path = temp_dir.path();
    
    // テストファイル作成
    let test_content = "Hello, NXZip! This is a test file for integration testing.";
    let input_file = temp_dir_path.join("test_input.txt");
    let output_file = temp_dir_path.join("test_output.nxz");
    let extracted_file = temp_dir_path.join("test_extracted.txt");
    
    fs::write(&input_file, test_content).unwrap();
    
    // 圧縮テスト
    let compress_output = Command::new("cargo")
        .args(&[
            "run", "--", "compress",
            "-i", input_file.to_str().unwrap(),
            "-o", output_file.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute compress command");
    
    if !compress_output.status.success() {
        panic!("Compress command failed: {}", String::from_utf8_lossy(&compress_output.stderr));
    }
    
    // NXZファイルが作成されたことを確認
    assert!(output_file.exists());
    
    // 展開テスト
    let extract_output = Command::new("cargo")
        .args(&[
            "run", "--", "extract",
            "-i", output_file.to_str().unwrap(),
            "-o", extracted_file.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute extract command");
    
    if !extract_output.status.success() {
        panic!("Extract command failed: {}", String::from_utf8_lossy(&extract_output.stderr));
    }
    
    // 展開されたファイルの内容確認
    let extracted_content = fs::read_to_string(&extracted_file).unwrap();
    assert_eq!(extracted_content, test_content);
}

#[test]
fn test_encrypt_compress_and_extract() {
    let temp_dir = tempdir().unwrap();
    let temp_dir_path = temp_dir.path();
    
    // テストファイル作成
    let test_content = "This is encrypted test content for NXZip.";
    let input_file = temp_dir_path.join("secret_input.txt");
    let output_file = temp_dir_path.join("secret_output.nxz.sec");
    let extracted_file = temp_dir_path.join("secret_extracted.txt");
    
    fs::write(&input_file, test_content).unwrap();
    
    let password = "testpassword123";
    
    // 暗号化圧縮テスト
    let compress_output = Command::new("cargo")
        .args(&[
            "run", "--", "compress",
            "-i", input_file.to_str().unwrap(),
            "-o", output_file.to_str().unwrap(),
            "--encrypt",
            "--password", password,
        ])
        .output()
        .expect("Failed to execute encrypt compress command");
    
    if !compress_output.status.success() {
        panic!("Encrypt compress command failed: {}", String::from_utf8_lossy(&compress_output.stderr));
    }
    
    // NXZ.secファイルが作成されたことを確認
    assert!(output_file.exists());
    
    // 復号展開テスト
    let extract_output = Command::new("cargo")
        .args(&[
            "run", "--", "extract",
            "-i", output_file.to_str().unwrap(),
            "-o", extracted_file.to_str().unwrap(),
            "--password", password,
        ])
        .output()
        .expect("Failed to execute decrypt extract command");
    
    if !extract_output.status.success() {
        panic!("Decrypt extract command failed: {}", String::from_utf8_lossy(&extract_output.stderr));
    }
    
    // 展開されたファイルの内容確認
    let extracted_content = fs::read_to_string(&extracted_file).unwrap();
    assert_eq!(extracted_content, test_content);
}
