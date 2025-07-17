#!/usr/bin/env python3
"""
🌍 NXZip Internationalization (i18n) Module
多言語対応システム - 日本語優先設計

Created by Japanese developer for Japanese users first.
"""

import os
import json
from typing import Dict, Any

class Messages:
    """多言語メッセージ管理クラス"""
    
    def __init__(self, language: str = "ja"):
        self.language = language
        self.messages = self._load_messages()
    
    def _load_messages(self) -> Dict[str, Any]:
        """言語ファイルから メッセージを読み込み"""
        return {
            "ja": {
                # バナー・タイトル
                "banner_title": "NXZip Proven - 次世代アーカイブシステム",
                "banner_subtitle": "═══════════════════════════════════════════════════════════",
                "banner_performance": "🏆 世界クラス性能: 99.95%圧縮率",
                "banner_security": "🔒 エンタープライズセキュリティ: 6段階SPE + 軍事グレード暗号化",
                "banner_speed": "⚡ 超高速処理: 11.37 MB/s処理速度",
                
                # メニュー・セクション
                "language_selection": "言語選択",
                "encryption_method": "暗号化方式",
                "password_setup": "パスワード設定",
                "key_derivation": "鍵導出関数",
                "compression_settings": "圧縮設定",
                "config_summary": "設定確認",
                
                # 暗号化オプション
                "encryption_none": "暗号化なし",
                "encryption_none_desc": "暗号化なし - 最高速度",
                "encryption_spe": "SPE暗号化",
                "encryption_spe_desc": "6段階構造保持暗号化",
                "encryption_aes": "AES-GCM",
                "encryption_aes_desc": "業界標準 AES-256-GCM",
                "encryption_xchacha": "XChaCha20",
                "encryption_xchacha_desc": "次世代 XChaCha20-Poly1305",
                
                # KDF オプション
                "kdf_pbkdf2": "PBKDF2",
                "kdf_pbkdf2_desc": "標準鍵導出 (高速)",
                "kdf_scrypt": "Scrypt",
                "kdf_scrypt_desc": "メモリハード鍵導出 (安全)",
                "kdf_argon2": "Argon2",
                "kdf_argon2_desc": "最新標準 (最高セキュリティ)",
                
                # 圧縮レベル
                "compression_fast": "高速 (レベル 3)",
                "compression_fast_desc": "高速圧縮、大きめファイル",
                "compression_balanced": "バランス (レベル 6)",
                "compression_balanced_desc": "速度とサイズの良いバランス",
                "compression_maximum": "最大 (レベル 9)",
                "compression_maximum_desc": "最高圧縮、低速",
                
                # プロンプト・メッセージ
                "select_language": "🌍 言語を選択してください (1-2)",
                "select_encryption": "🔒 暗号化方式を選択 (1-4)",
                "select_kdf": "🔑 鍵導出方式を選択 (1-3)",
                "select_compression": "📦 圧縮レベルを選択 (1-3)",
                "enter_password": "🔐 パスワードを入力 (非表示)",
                "confirm_password": "🔐 パスワードを再入力",
                "password_strength": "🔐 パスワード強度",
                "proceed_question": "✅ この設定で実行しますか？ (Y/n)",
                
                # 状態メッセージ
                "recommended": "推奨",
                "password_weak": "⚠️  弱い (短すぎます)",
                "password_moderate": "📊 普通",
                "password_good": "✅ 良い",
                "password_strong": "💪 強い (特殊文字含む)",
                "operation_cancelled": "❌ ユーザーによりキャンセルされました",
                "invalid_choice": "❌ 無効な選択です。以下から選んでください",
                "passwords_not_match": "❌ パスワードが一致しません。再試行してください。",
                "weak_password_warning": "⚠️  弱いパスワードが検出されました。続行しますか？ (y/N)",
                "no_password_continue": "⚠️  パスワードが入力されていません。暗号化なしで続行しますか？ (y/N)",
                
                # パスワード入力強化
                "password_requirements_title": "🔐 パスワード要件ガイド",
                "password_min_length": "• 最小8文字以上を推奨",
                "password_uppercase": "• 大文字を含む (A-Z)",
                "password_lowercase": "• 小文字を含む (a-z)", 
                "password_numbers": "• 数字を含む (0-9)",
                "password_symbols": "• 記号を含む (!@#$%^&*)",
                "password_no_common": "• 一般的な単語は避ける",
                "password_unique": "• 他のサイトと異なるパスワード",
                "password_length_counter": "文字数",
                "password_typing_indicator": "入力中...",
                "password_current_strength": "現在の強度",
                "password_realtime_feedback": "リアルタイム評価",
                "password_input_mask": "●",
                "password_excellent": "🟣 優秀 (完璧なセキュリティ)",
                "password_backspace_hint": "※ Backspaceで削除, Enterで完了",
                
                # Compact strength indicators
                "strength_weak": "🔴弱",
                "strength_moderate": "🟡普",
                "strength_good": "🟢良",
                "strength_strong": "🔵強",
                "strength_excellent": "🟣優",
                
                # 実行メッセージ
                "welcome_interactive": "🎨 NXZip インタラクティブモードへようこそ！",
                "creating_archive": "🚀 {}を証明済みアルゴリズムで作成中",
                "nexus_performance": "📦 NEXUS エンジン: 99.98% テキスト, 99.84% 画像, 99.77% 音声",
                "spe_encryption_enabled": "🔒 6段階Enterprise SPE暗号化が有効",
                "no_encryption_max_speed": "📂 暗号化なし (最高速度)",
                "password_protection_enabled": "🔐 パスワード保護: 有効 (ハッシュ: {})",
                
                # 設定サマリー
                "summary_encryption": "🔒 暗号化",
                "summary_password": "🔐 パスワード",
                "summary_kdf": "🔑 鍵導出",
                "summary_compression": "📦 圧縮",
                
                # コマンドヘルプ
                "help_description": "🚀 NXZip Proven - 認証済み高性能アルゴリズムによるアーカイブツール",
                "help_examples": "📚 使用例:",
                "help_basic_archive": "🔸 基本アーカイブ:",
                "help_interactive_mode": "🔸 インタラクティブモード (推奨):",
                "help_advanced_encryption": "🔸 高度暗号化:",
                "help_extract_archive": "🔸 アーカイブ展開:",
                "help_performance_test": "🔸 性能テスト:",
                "help_encryption_options": "🔐 暗号化オプション:",
                "help_key_derivation": "🔑 鍵導出:",
            },
            
            "en": {
                # Banner & Title
                "banner_title": "NXZip Proven - Next-Generation Archive System",
                "banner_subtitle": "═══════════════════════════════════════════════════════════",
                "banner_performance": "🏆 World-Class Performance: 99.95% Compression",
                "banner_security": "🔒 Enterprise Security: 6-Stage SPE + Military-Grade Crypto",
                "banner_speed": "⚡ Lightning Fast: 11.37 MB/s Processing Speed",
                
                # Menu & Sections
                "language_selection": "Language Selection",
                "encryption_method": "Encryption Method",
                "password_setup": "Password Setup",
                "key_derivation": "Key Derivation Function",
                "compression_settings": "Compression Settings",
                "config_summary": "Configuration Summary",
                
                # Encryption Options
                "encryption_none": "None",
                "encryption_none_desc": "No encryption - Maximum speed",
                "encryption_spe": "SPE",
                "encryption_spe_desc": "6-Stage Structure-Preserving Encryption",
                "encryption_aes": "AES-GCM",
                "encryption_aes_desc": "Industry standard AES-256-GCM",
                "encryption_xchacha": "XChaCha20",
                "encryption_xchacha_desc": "Next-generation XChaCha20-Poly1305",
                
                # KDF Options
                "kdf_pbkdf2": "PBKDF2",
                "kdf_pbkdf2_desc": "Standard key derivation (fast)",
                "kdf_scrypt": "Scrypt",
                "kdf_scrypt_desc": "Memory-hard key derivation (secure)",
                "kdf_argon2": "Argon2",
                "kdf_argon2_desc": "Latest standard (most secure)",
                
                # Compression Levels
                "compression_fast": "Fast (Level 3)",
                "compression_fast_desc": "Faster compression, larger files",
                "compression_balanced": "Balanced (Level 6)",
                "compression_balanced_desc": "Good balance of speed and size",
                "compression_maximum": "Maximum (Level 9)",
                "compression_maximum_desc": "Best compression, slower",
                
                # Prompts & Messages
                "select_language": "🌍 Select language (1-2)",
                "select_encryption": "🔒 Select encryption method (1-4)",
                "select_kdf": "🔑 Select KDF method (1-3)",
                "select_compression": "📦 Select compression level (1-3)",
                "enter_password": "🔐 Enter password (hidden)",
                "confirm_password": "🔐 Confirm password",
                "password_strength": "🔐 Password strength",
                "proceed_question": "✅ Proceed with these settings? (Y/n)",
                
                # Status Messages
                "recommended": "Recommended",
                "password_weak": "⚠️  WEAK (too short)",
                "password_moderate": "📊 MODERATE",
                "password_good": "✅ GOOD",
                "password_strong": "💪 STRONG (with special characters)",
                "operation_cancelled": "❌ Operation cancelled by user",
                "invalid_choice": "❌ Invalid choice. Please select from",
                "passwords_not_match": "❌ Passwords don't match. Please try again.",
                "weak_password_warning": "⚠️  Weak password detected. Continue anyway? (y/N)",
                "no_password_continue": "⚠️  No password entered. Continue without encryption? (y/N)",
                
                # Password input enhancements
                "password_requirements_title": "🔐 Password Requirements Guide",
                "password_min_length": "• Minimum 8+ characters recommended",
                "password_uppercase": "• Include uppercase letters (A-Z)",
                "password_lowercase": "• Include lowercase letters (a-z)",
                "password_numbers": "• Include numbers (0-9)",
                "password_symbols": "• Include symbols (!@#$%^&*)",
                "password_no_common": "• Avoid common words",
                "password_unique": "• Use unique password",
                "password_length_counter": "Length",
                "password_typing_indicator": "Typing...",
                "password_current_strength": "Current strength",
                "password_realtime_feedback": "Real-time evaluation",
                "password_input_mask": "●",
                "password_excellent": "🟣 EXCELLENT (Perfect security)",
                "password_backspace_hint": "※ Backspace to delete, Enter to confirm",
                
                # Compact strength indicators
                "strength_weak": "🔴WEAK",
                "strength_moderate": "🟡MOD",
                "strength_good": "🟢GOOD",
                "strength_strong": "🔵STR",
                "strength_excellent": "🟣EXC",
                
                # Execution Messages
                "welcome_interactive": "🎨 Welcome to NXZip Interactive Mode!",
                "creating_archive": "🚀 Creating {} with proven algorithms",
                "nexus_performance": "📦 NEXUS Engine: 99.98% Text, 99.84% Images, 99.77% Audio",
                "spe_encryption_enabled": "🔒 6-Stage Enterprise SPE encryption enabled",
                "no_encryption_max_speed": "📂 No encryption (maximum speed)",
                "password_protection_enabled": "🔐 Password protection: Enabled (Hash: {})",
                
                # Configuration Summary
                "summary_encryption": "🔒 Encryption",
                "summary_password": "🔐 Password",
                "summary_kdf": "🔑 KDF",
                "summary_compression": "📦 Compression",
                
                # Command Help
                "help_description": "🚀 NXZip Proven - Archive Tool with Certified High-Performance Algorithms",
                "help_examples": "📚 USAGE EXAMPLES:",
                "help_basic_archive": "🔸 Basic Archive:",
                "help_interactive_mode": "🔸 Interactive Mode (Recommended):",
                "help_advanced_encryption": "🔸 Advanced Encryption:",
                "help_extract_archive": "🔸 Extract Archive:",
                "help_performance_test": "🔸 Performance Test:",
                "help_encryption_options": "🔐 ENCRYPTION OPTIONS:",
                "help_key_derivation": "🔑 KEY DERIVATION:",
            }
        }
    
    def get(self, key: str, *args) -> str:
        """メッセージを取得（フォーマット対応）"""
        message = self.messages.get(self.language, {}).get(key, key)
        if args:
            try:
                return message.format(*args)
            except:
                return message
        return message
    
    def set_language(self, language: str):
        """言語を変更"""
        if language in self.messages:
            self.language = language
    
    def get_available_languages(self) -> Dict[str, str]:
        """利用可能言語リストを取得"""
        return {
            "ja": "🇯🇵 日本語 (Japanese)",
            "en": "🇺🇸 English"
        }

# グローバルメッセージインスタンス
msg = Messages("ja")  # 日本語をデフォルトに設定
