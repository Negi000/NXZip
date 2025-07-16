# 🚀 NXZip - 次世代アーカイブシステム

NXZipは **SPE (Structure-Preserving Encryption)** を核とした統合圧縮・暗号化ソリューションです。

## 📁 プロジェクト構成

```
NXZip/
├── 📦 python-nxzip/       # Python実装（推奨・完成済み）
├── ⚡ rust-nxzip/         # Rust実装（高性能・開発中）
├── 📚 shared-docs/        # 共有ドキュメント
└── 🧪 test-data/          # テストデータ
```

## 🎯 どちらを使う？

### 🐍 Python版（推奨）
✅ **即座に使用可能**  
✅ **簡単な環境構築**  
✅ **完全テスト済み**  

```bash
cd python-nxzip
python nxzip_complete.py create archive.nxz input.txt
```

### ⚡ Rust版（高性能）
🔄 **開発完了・環境問題あり**  
🚀 **高速処理（環境が整えば）**  
🔧 **複雑な環境構築が必要**  

```bash
cd rust-nxzip
cargo build --release  # 環境が整っている場合
```

## ✨ 主な特徴

- 🔐 **完全可逆SPE**: 100%データ復元保証
- 🗜️ **効率的圧縮**: ZLIB統合圧縮
- 🔒 **AES-256暗号化**: 軍事レベルの安全性
- 🔄 **完全テスト済み**: 全機能検証完了

## 🚀 クイックスタート

### 1. Python版をすぐに試す
```bash
cd python-nxzip
pip install cryptography
python nxzip_complete.py create test.nxz test_input.txt
python nxzip_complete.py extract test.nxz output.txt
```

### 2. 詳しい情報を見る
- [Python版ガイド](python-nxzip/README.md)
- [Rust版ガイド](rust-nxzip/README.md) 
- [プロジェクト完了レポート](shared-docs/PROJECT_COMPLETION.md)

## 📊 プロジェクト状況

| 実装 | 状況 | 推奨度 |
|------|------|--------|
| Python | ✅ 完成 | ⭐⭐⭐⭐⭐ |
| Rust | ⚠️ 環境問題 | ⭐⭐⭐☆☆ |

## 💡 使用例

```bash
# 基本的な圧縮
python nxzip_complete.py create archive.nxz input.txt

# パスワード付き暗号化
python nxzip_complete.py create secure.nxz input.txt -p "password"

# アーカイブ展開
python nxzip_complete.py extract archive.nxz output.txt

# 整合性テスト
python nxzip_complete.py test archive.nxz
```

## 🎉 プロジェクト成果

✅ **SPE完全可逆性達成**  
✅ **統合アーカイブシステム完成**  
✅ **Python移行成功**  
✅ **全機能テスト完了**  

---

**開発**: GitHub Copilot | **更新**: 2024年12月 | **ライセンス**: プロジェクト準拠
