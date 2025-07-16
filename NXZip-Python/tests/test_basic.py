#!/usr/bin/env python3
"""
NXZip v2.0 Basic Tests
基本機能のテストスイート
"""

import pytest
import tempfile
import os
from nxzip import SuperNXZipFile, NXZipError, verify_installation


class TestSPECore:
    """SPEコアのテスト"""
    
    def test_spe_reversibility(self):
        """SPE変換の可逆性テスト"""
        from nxzip.engine.spe_core import SPECore
        
        spe = SPECore()
        test_data = b"Hello, NXZip SPE Test!"
        
        # 変換と逆変換
        transformed = spe.apply_transform(test_data)
        restored = spe.reverse_transform(transformed)
        
        assert restored == test_data
        assert transformed != test_data  # 変換効果の確認
    
    def test_spe_different_sizes(self):
        """異なるサイズでのSPEテスト"""
        from nxzip.engine.spe_core import SPECore
        
        spe = SPECore()
        
        # 様々なサイズでテスト
        for size in [1, 15, 16, 17, 32, 100, 1000]:
            test_data = b"x" * size
            transformed = spe.apply_transform(test_data)
            restored = spe.reverse_transform(transformed)
            assert restored == test_data


class TestCompression:
    """圧縮機能のテスト"""
    
    def test_compression_algorithms(self):
        """圧縮アルゴリズムのテスト"""
        from nxzip.engine.compressor import SuperCompressor
        from nxzip.utils.constants import CompressionAlgorithm
        
        test_data = b"This is a test string for compression." * 100
        
        for algo in [CompressionAlgorithm.ZLIB, CompressionAlgorithm.LZMA2]:
            compressor = SuperCompressor(algo)
            compressed, used_algo = compressor.compress(test_data)
            decompressed = compressor.decompress(compressed, used_algo)
            
            assert decompressed == test_data
            assert len(compressed) < len(test_data)  # 圧縮効果確認


class TestEncryption:
    """暗号化機能のテスト"""
    
    def test_aes_gcm_encryption(self):
        """AES-GCM暗号化のテスト"""
        from nxzip.crypto.encrypt import SuperCrypto
        from nxzip.utils.constants import EncryptionAlgorithm
        
        crypto = SuperCrypto(EncryptionAlgorithm.AES_GCM)
        test_data = b"Secret message for encryption test"
        password = "test_password"
        
        encrypted_data, metadata = crypto.encrypt(test_data, password)
        decrypted_data = crypto.decrypt(encrypted_data, metadata, password)
        
        assert decrypted_data == test_data
        assert encrypted_data != test_data
    
    def test_xchacha20_encryption(self):
        """XChaCha20-Poly1305暗号化のテスト"""
        from nxzip.crypto.encrypt import SuperCrypto
        from nxzip.utils.constants import EncryptionAlgorithm
        
        crypto = SuperCrypto(EncryptionAlgorithm.XCHACHA20_POLY1305)
        test_data = b"Secret message for XChaCha20 test"
        password = "test_password"
        
        encrypted_data, metadata = crypto.encrypt(test_data, password)
        decrypted_data = crypto.decrypt(encrypted_data, metadata, password)
        
        assert decrypted_data == test_data
        assert encrypted_data != test_data


class TestNXZipFile:
    """NXZipFile統合テスト"""
    
    def test_basic_archive(self):
        """基本的なアーカイブ作成・展開テスト"""
        nxzip = SuperNXZipFile()
        test_data = b"Hello, NXZip v2.0!"
        
        # アーカイブ作成
        archive = nxzip.create_archive(test_data)
        
        # 展開
        restored = nxzip.extract_archive(archive)
        
        assert restored == test_data
    
    def test_encrypted_archive(self):
        """暗号化アーカイブのテスト"""
        nxzip = SuperNXZipFile()
        test_data = b"Encrypted test data for NXZip"
        password = "secret123"
        
        # 暗号化アーカイブ作成
        archive = nxzip.create_archive(test_data, password=password)
        
        # 正しいパスワードで展開
        restored = nxzip.extract_archive(archive, password=password)
        assert restored == test_data
        
        # 間違ったパスワードでエラー確認
        with pytest.raises(Exception):
            nxzip.extract_archive(archive, password="wrong_password")
    
    def test_archive_info(self):
        """アーカイブ情報取得のテスト"""
        nxzip = SuperNXZipFile()
        test_data = b"Info test data" * 100
        
        archive = nxzip.create_archive(test_data)
        info = nxzip.get_info(archive)
        
        assert info['version'] == 'NXZ v2.0'
        assert info['original_size'] == len(test_data)
        assert info['compression_ratio'] > 0  # 圧縮効果確認
        assert not info['is_encrypted']  # 暗号化なし
    
    def test_large_data(self):
        """大容量データのテスト"""
        nxzip = SuperNXZipFile()
        
        # 1MBのテストデータ
        test_data = b"Large data test " * 65536  # 1MB
        
        archive = nxzip.create_archive(test_data)
        restored = nxzip.extract_archive(archive)
        
        assert restored == test_data
        assert len(archive) < len(test_data)  # 圧縮効果確認


class TestInstallation:
    """インストール検証テスト"""
    
    def test_package_integrity(self):
        """パッケージ整合性のテスト"""
        assert verify_installation()
    
    def test_spe_integrity(self):
        """SPEコア整合性のテスト"""
        from nxzip.engine.spe_core import verify_spe_integrity
        assert verify_spe_integrity()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
