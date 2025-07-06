use clap::{Parser, Subcommand};
use anyhow::Result;

mod cli;
mod engine;
mod crypto;
mod utils;
mod formats;

use cli::commands::{compress_command, extract_command, info_command, sec_compress_command, sec_extract_command};

#[derive(Parser)]
#[command(name = "nxzip")]
#[command(about = "NXZip - 次世代統合アーカイブシステム")]
#[command(version = "0.1.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// ファイルまたはディレクトリを圧縮
    Compress {
        /// 入力ファイルまたはディレクトリ
        #[arg(short, long)]
        input: String,
        
        /// 出力ファイル (.nxz または .nxz.sec)
        #[arg(short, long)]
        output: String,
        
        /// 暗号化を有効にする
        #[arg(long)]
        encrypt: bool,
        
        /// 暗号化パスワード
        #[arg(long)]
        password: Option<String>,
        
        /// 暗号化アルゴリズム (aes-gcm, xchacha20)
        #[arg(long, default_value = "aes-gcm")]
        encryption: String,
        
        /// 圧縮アルゴリズム (zstd, lzma2, auto)
        #[arg(long, default_value = "auto")]
        algorithm: String,
        
        /// 圧縮レベル (1-9)
        #[arg(long, default_value = "6")]
        level: u8,
    },
    
    /// セキュリティ強化型(.nxz.sec)でファイルを圧縮
    SecCompress {
        /// 入力ファイルまたはディレクトリ
        #[arg(short, long)]
        input: String,
        
        /// 出力ファイル (.nxz.sec)
        #[arg(short, long)]
        output: String,
        
        /// 暗号化パスワード
        #[arg(short, long)]
        password: String,
        
        /// 暗号化アルゴリズム (aes-gcm, xchacha20)
        #[arg(long, default_value = "aes-gcm")]
        encryption: String,
        
        /// 鍵導出方式 (pbkdf2, argon2)
        #[arg(long, default_value = "pbkdf2")]
        kdf: String,
        
        /// 圧縮アルゴリズム (zstd, lzma2, auto)
        #[arg(long, default_value = "auto")]
        algorithm: String,
        
        /// 圧縮レベル (1-9)
        #[arg(long, default_value = "6")]
        level: u8,
    },
    
    /// アーカイブを展開
    Extract {
        /// 入力アーカイブファイル (.nxz または .nxz.sec)
        #[arg(short, long)]
        input: String,
        
        /// 出力ディレクトリ (省略時は現在のディレクトリ)
        #[arg(short, long)]
        output: Option<String>,
        
        /// 復号化パスワード
        #[arg(long)]
        password: Option<String>,
        
        /// 復号化アルゴリズム (aes-gcm, xchacha20)
        #[arg(long, default_value = "aes-gcm")]
        encryption: String,
    },
    
    /// セキュリティ強化型(.nxz.sec)アーカイブを展開
    SecExtract {
        /// 入力アーカイブファイル (.nxz.sec)
        #[arg(short, long)]
        input: String,
        
        /// 出力ディレクトリ (省略時は現在のディレクトリ)
        #[arg(short, long)]
        output: Option<String>,
        
        /// 復号化パスワード
        #[arg(short, long)]
        password: String,
    },
    
    /// ファイル情報を表示
    Info {
        /// 入力アーカイブファイル
        #[arg(short, long)]
        input: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Compress {
            input,
            output,
            encrypt,
            password,
            encryption,
            algorithm,
            level,
        } => {
            compress_command(&input, &output, encrypt, password, &encryption, &algorithm, level).await?;
        }
        
        Commands::SecCompress {
            input,
            output,
            password,
            encryption,
            kdf,
            algorithm,
            level,
        } => {
            sec_compress_command(&input, &output, &password, &encryption, &kdf, &algorithm, level).await?;
        }
        
        Commands::Extract {
            input,
            output,
            password,
            encryption,
        } => {
            extract_command(&input, output, password, &encryption).await?;
        }
        
        Commands::SecExtract {
            input,
            output,
            password,
        } => {
            sec_extract_command(&input, output, &password).await?;
        }
        
        Commands::Info { input } => {
            info_command(&input).await?;
        }
    }
    
    Ok(())
}
