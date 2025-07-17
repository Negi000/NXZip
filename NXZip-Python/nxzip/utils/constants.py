#!/usr/bin/env python3
"""
NXZip Constants
システム全体で使用される定数定義
"""

# 暗号化アルゴリズム定数
class EncryptionAlgorithm:
    AES_GCM = "aes-gcm"
    XCHACHA20_POLY1305 = "xchacha20-poly1305"

# 圧縮アルゴリズム定数
class CompressionAlgorithm:
    ZLIB = "zlib"
    LZMA2 = "lzma2"
    ZSTD = "zstd"
    AUTO = "auto"

# KDF (鍵導出) アルゴリズム定数
class KDFAlgorithm:
    PBKDF2 = "pbkdf2"
    SCRYPT = "scrypt"
    ARGON2 = "argon2"

# ファイルフォーマット定数
class FileFormat:
    MAGIC_V2 = b'NXZ\x02'  # NXZ v2.0
    HEADER_SIZE_V2 = 160   # v2.0 ヘッダーサイズ

# セキュリティ定数
class SecurityConstants:
    PBKDF2_ITERATIONS = 100000
    SCRYPT_N = 2**16  # 65536
    SCRYPT_R = 8
    SCRYPT_P = 1
    ARGON2_TIME_COST = 3     # Number of iterations
    ARGON2_MEMORY_COST = 64  # Memory usage in KiB
    ARGON2_PARALLELISM = 1   # Number of parallel threads
    SALT_SIZE = 16
    NONCE_SIZE_AES = 12
    NONCE_SIZE_XCHACHA = 12
    TAG_SIZE = 16
