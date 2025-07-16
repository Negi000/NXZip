# 🚀 NXZip クイックスタートガイド

NXZipの次世代GUIアプリケーションを利用したクイックスタートガイドです。

## 📦 必要なソフトウェア

### GUI版（推奨）
- **Node.js 18以降** - [公式サイト](https://nodejs.org/)からダウンロード
- **Rust 1.70以降** - [rustup](https://rustup.rs/)でインストール
- **Git** - [公式サイト](https://git-scm.com/)からダウンロード

### CLI版のみ使用する場合
- **Rust 1.70以降** - [rustup](https://rustup.rs/)でインストール
- **Git** - [公式サイト](https://git-scm.com/)からダウンロード

## 🖥️ GUI版インストール手順

### 1. リポジトリのクローン
```bash
git clone https://github.com/your-org/nxzip.git
cd nxzip
```

### 2. 依存関係のインストール
```bash
# Node.js依存関係
npm install
cd gui && npm install && cd ..

# Rust依存関係は自動でインストールされます
```

### 3. アプリケーションの起動
```bash
# 開発モード（デバッグ用）
npx tauri dev

# または、本番ビルド（高パフォーマンス）
npx tauri build
# 生成されたexeファイルはsrc-tauri/target/releaseにあります
```

### 4. アプリケーションの使用
1. GUI画面が起動します
2. 「圧縮」タブでファイルを選択
3. 圧縮設定を調整
4. 「圧縮開始」ボタンをクリック

## ⚡ CLI版インストール手順

### 1. リポジトリのクローン
```bash
git clone https://github.com/your-org/nxzip.git
cd nxzip
```

### 2. ビルド
```bash
# リリースビルド
cargo build --release

# またはデバッグビルド（開発用）
cargo build
```

### 3. 実行ファイルの使用
```bash
# リリースビルドを使用
./target/release/nxzip --help

# デバッグビルドを使用
./target/debug/nxzip --help
```

## 📖 基本的な使用例

### GUI版
1. **ファイル圧縮**
   - ファイルをドラッグ&ドロップまたは「ファイル選択」
   - アルゴリズム（Auto/Zstd/LZMA2）を選択
   - 暗号化が必要な場合はパスワードを設定
   - 「圧縮開始」ボタンをクリック

2. **ファイル展開**
   - .nxzまたは.nxz.secファイルを選択
   - 暗号化されている場合はパスワードを入力
   - 出力先を指定
   - 「展開開始」ボタンをクリック

### CLI版
```bash
# 基本的な圧縮
nxzip compress --input data.txt --output data.nxz

# 暗号化付き圧縮
nxzip compress --input secret.txt --output secret.nxz --encrypt --password "my_password"

# セキュリティ強化型圧縮
nxzip sec-compress --input confidential.doc --output confidential.nxz.sec --password "secure_key"

# 展開
nxzip extract --input data.nxz --output extracted.txt

# ファイル情報表示
nxzip info --input data.nxz
```

## 🛠 トラブルシューティング

### よくある問題

**Q: `npx tauri dev` でエラーが発生する**
- A: Node.js、Rustが正しくインストールされているか確認
- A: `npm install`と`cd gui && npm install`を実行済みか確認

**Q: Rustのビルドが失敗する**
- A: Rust 1.70以降がインストールされているか確認: `rustc --version`
- A: `cargo clean`でキャッシュをクリアして再実行

**Q: GUI画面が表示されない**
- A: ファイアウォールがポート1420をブロックしていないか確認
- A: 他のアプリケーションがポートを使用していないか確認

**Q: 圧縮・展開が失敗する**
- A: ファイルの読み書き権限があるか確認
- A: 出力先ディレクトリが存在するか確認
- A: パスワードが正しいか確認（暗号化ファイルの場合）

### ログの確認
```bash
# 詳細なログでビルド
RUST_LOG=debug npx tauri dev

# CLIでのデバッグ情報
RUST_LOG=debug ./target/debug/nxzip compress --input test.txt --output test.nxz
```

## 🎯 パフォーマンス最適化

### 推奨設定
- **高速圧縮**: Algorithm=Zstd, Level=3
- **高圧縮率**: Algorithm=LZMA2, Level=9
- **バランス**: Algorithm=Auto, Level=6
- **セキュリティ**: Encryption=XChaCha20, KDF=Argon2

### 大容量ファイル
- 8GB以上のファイルはCLI版を推奨
- 十分なRAM（圧縮するファイルサイズの2倍以上）を確保
- SSDストレージを使用して高速化

## 📞 サポート

- 🐛 **バグ報告**: [GitHub Issues](https://github.com/your-org/nxzip/issues)
- 💬 **質問・議論**: [GitHub Discussions](https://github.com/your-org/nxzip/discussions)
- 📚 **ドキュメント**: [README.md](README.md)
- 📧 **連絡先**: nxzip-support@example.com

## 🔗 関連リンク

- [メインREADME](README.md) - 詳細な技術仕様
- [API ドキュメント](https://docs.rs/nxzip) - ライブラリAPI
- [リリースノート](https://github.com/your-org/nxzip/releases) - 更新履歴
- [コントリビューションガイド](CONTRIBUTING.md) - 開発者向け

---

**NXZip** - 次世代統合アーカイブシステム 🚀
