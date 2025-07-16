// CLI引数解析の追加設定やバリデーション関数

use anyhow::Result;

/// パスワード強度の検証
pub fn validate_password(password: &str) -> Result<()> {
    if password.len() < 8 {
        anyhow::bail!("パスワードは8文字以上である必要があります");
    }
    
    let has_upper = password.chars().any(|c| c.is_uppercase());
    let has_lower = password.chars().any(|c| c.is_lowercase());
    let has_digit = password.chars().any(|c| c.is_numeric());
    
    if !has_upper || !has_lower || !has_digit {
        println!("⚠️  推奨: パスワードには大文字、小文字、数字を含めることを推奨します");
    }
    
    Ok(())
}

/// 圧縮レベルの検証
pub fn validate_compression_level(level: u8) -> Result<u8> {
    match level {
        1..=9 => Ok(level),
        _ => {
            println!("⚠️  圧縮レベルは1-9の範囲で指定してください。デフォルト値6を使用します。");
            Ok(6)
        }
    }
}
