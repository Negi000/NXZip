use clap::{Parser, Subcommand};
use anyhow::Result;

mod cli;
mod engine;
mod crypto;
mod utils;
mod formats;

use cli::commands::{compress_command, extract_command};

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
            algorithm,
            level,
        } => {
            compress_command(&input, &output, encrypt, password, &algorithm, level).await?;
        }
        
        Commands::Extract {
            input,
            output,
            password,
        } => {
            extract_command(&input, output, password).await?;
        }
        
        Commands::Info { input } => {
            println!("ファイル情報表示機能は開発中です: {}", input);
        }
    }
    
    Ok(())
}
