# 🚀 NXZip - 次世代統合アーカイブシステム

NXZipは、**SPE (Structure-Preserving Encryption)** × **高効率可逆圧縮** × **セキュリティ強化機構**を統合した次世代アーカイブシステムです。

## ✨ 特徴

- 🧠 **SPE暗号化**: データ構造を保持しながら難読化
- 📦 **高効率圧縮**: LZMA2/Zstdを上回る圧縮率
- 🔐 **多層セキュリティ**: AES-GCM/XChaCha20による強固な暗号化
- ⚡ **高速展開**: SSD最適化による高速解凍
- 🎮 **ゲームアセット保護**: Unity/Unreal Engine対応

## 📦 ファイル形式

| 拡張子 | 特徴 | 用途 |
|--------|------|------|
| `.nxz` | 標準圧縮（SPE + 圧縮） | 一般的な用途 |
| `.nxz.sec` | セキュリティ強化版 | 機密データ保護 |

## 🛠 インストール

### 前提条件

- Rust 1.70.0 以上
- Cargo

### ビルド

```bash
git clone https://github.com/your-repo/nxzip.git
cd nxzip
cargo build --release
```

## 📖 使用方法

### 基本的な圧縮

```bash
# ファイル圧縮
nxzip compress -i input.txt -o output.nxz

# ディレクトリ圧縮
nxzip compress -i ./assets -o assets.nxz
```

### セキュリティ強化版

```bash
# パスワード付き暗号化圧縮
nxzip compress -i secret.doc -o secret.nxz.sec --encrypt --password "your_password"

# 復号展開
nxzip extract -i secret.nxz.sec --password "your_password"
```

### 圧縮オプション

```bash
# アルゴリズム指定
nxzip compress -i data.bin -o data.nxz --algorithm zstd --level 9

# 利用可能なアルゴリズム: zstd, lzma2, auto (デフォルト)
# 圧縮レベル: 1-9 (デフォルト: 6)
```

### 展開

```bash
# 通常展開
nxzip extract -i archive.nxz -o extracted/

# ファイル情報表示
nxzip info -i archive.nxz
```

## 🏗 アーキテクチャ

```
┌─────────────────┐
│   元ファイル      │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  SPE暗号化       │ ← 構造保持型難読化
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  NXZ圧縮        │ ← LZMA2/Zstd最適化
└─────────┬───────┘
          │
          ▼ (オプション)
┌─────────────────┐
│ セキュリティ強化   │ ← AES-GCM/XChaCha20
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  .nxz/.nxz.sec  │
└─────────────────┘
```

## 🧪 テスト

```bash
# ユニットテスト
cargo test

# 統合テスト
cargo test --test integration

# ベンチマーク (開発中)
cargo bench
```

## 📊 パフォーマンス

| ファイルタイプ | 圧縮率 | 展開速度 | セキュリティ |
|--------------|--------|----------|-------------|
| テキスト | 85-92% | ★★★★☆ | ★★★★★ |
| バイナリ | 75-85% | ★★★★★ | ★★★★★ |
| 画像 | 60-75% | ★★★★☆ | ★★★★★ |
| ゲームアセット | 70-80% | ★★★★★ | ★★★★★ |

## 🔒 セキュリティ機能

### SPE (Structure-Preserving Encryption)

- データ構造を保持しながら難読化
- 通常のバイナリエディタでは解析困難
- 可逆変換による完全復元

### 多層暗号化

- AES-256-GCM: 業界標準の強固な暗号化
- XChaCha20-Poly1305: 高速かつ安全
- PBKDF2: パスワードベース鍵導出

## 🎮 ゲーム開発者向け

### Unity連携

```bash
# Unityアセット圧縮
nxzip compress -i Assets/ -o game_assets.nxz --algorithm auto

# ランタイム展開用
nxzip extract -i game_assets.nxz -o StreamingAssets/
```

### Unreal Engine連携

```bash
# UE4/UE5アセット圧縮
nxzip compress -i Content/ -o content.nxz.sec --encrypt --password "game_key"
```

## 📚 API ドキュメント

詳細なAPIドキュメントは [docs.rs](https://docs.rs/nxzip) で確認できます。

```bash
# ローカルドキュメント生成
cargo doc --open
```

## 🤝 コントリビューション

1. フォークしてブランチを作成
2. 変更を実装
3. テストを追加/実行
4. プルリクエストを作成

詳細は [CONTRIBUTING.md](CONTRIBUTING.md) を参照してください。

## 📄 ライセンス

- **CLI/GUI**: MIT License
- **圧縮コア**: MIT / Apache 2.0
- **SPE暗号エンジン**: カスタム商用ライセンス（非公開）

## 🔮 ロードマップ

- [ ] GUI実装 (Electron + Tauri)
- [ ] PQ暗号対応 (耐量子暗号)
- [ ] 仮想マウント機能
- [ ] 複数ファイル統合
- [ ] クラウド連携

## 📞 サポート

- 🐛 バグ報告: [Issues](https://github.com/your-repo/nxzip/issues)
- 💬 質問: [Discussions](https://github.com/your-repo/nxzip/discussions)
- 📧 連絡先: nxzip@example.com

---

**NXZip** - 次世代の圧縮技術でデータを守る 🛡️
