# NXZip - Next Generation Archive System

## 最適化エンジン一覧

### 🚀 汎用超高速圧縮エンジン
- **nxzip_ultra_fast_binary_collapse.py**
  - 処理速度: 22.5 MB/s
  - 汎用圧縮率: 10%
  - 対象: 全ファイル形式
  - 特徴: 超高速処理、実用性重視

### 🖼️ 画像専用最適化エンジン
- **nxzip_smart_image_compressor.py**
  - JPEG圧縮率: 8.4%
  - PNG最適化: 制約認識
  - 特徴: フォーマット特化最適化

### 🎯 AV1インスパイア高品質エンジン
- **nexus_cablc_enhanced.py**
  - ブロック分割最適化
  - 複数予測モード
  - 特徴: 高品質圧縮

### 🔧 ユーティリティ
- **analyze_formats.py**: フォーマット解析
- **final_repository_cleanup.py**: リポジトリ整理

## 使用方法

```bash
# 汎用高速圧縮
python nxzip_ultra_fast_binary_collapse.py <ファイル>

# 画像専用圧縮
python nxzip_smart_image_compressor.py <画像ファイル>

# 高品質圧縮
python nexus_cablc_enhanced.py <ファイル>
```

## 性能比較

| エンジン | 速度 | JPEG圧縮 | PNG圧縮 | WAV圧縮 | 特徴 |
|---------|------|----------|---------|---------|------|
| Ultra Fast | 22.5 MB/s | 10.0% | -0.0% | 100%+ | 超高速 |
| Smart Image | 2.3 MB/s | 8.4% | 制約認識 | - | 画像特化 |
| CABLC Enhanced | 可変 | 可変 | 可変 | 可変 | 高品質 |

## アーキテクチャ

```
NXZip/
├── bin/                    # 最適化エンジン
│   ├── nxzip_ultra_fast_binary_collapse.py
│   ├── nxzip_smart_image_compressor.py
│   ├── nexus_cablc_enhanced.py
│   └── analyze_formats.py
└── README.md              # この説明
```
