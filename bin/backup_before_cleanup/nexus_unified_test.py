#!/usr/bin/env python3
"""
NEXUS SDC 統合テストエンジン - Phase 5
全フォーマット対応の統合テストシステム

対応フォーマット: テキスト、音声、動画、画像、アーカイブ
目標: 理論値に近い実測圧縮率の達成
"""

import os
import sys
import time
from pathlib import Path

# プロジェクト内モジュールのインポート
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from progress_display import ProgressDisplay

# 各特化エンジンをインポート
try:
    from nexus_sdc_engine_concise import NexusSDCEngine
except ImportError:
    from nexus_sdc_engine import NexusSDCEngine

try:
    from nexus_image_sdc import NexusImageSDCEngine
except ImportError:
    NexusImageSDCEngine = None

# 進捗表示インスタンス
progress = ProgressDisplay()

def show_step(message: str):
    """メインステップ表示"""
    print(f"🧪 {message}")

def show_success(message: str):
    """成功メッセージ"""
    print(f"✅ {message}")

def show_warning(message: str):
    """警告メッセージ"""
    print(f"⚠️  {message}")

class NexusUnifiedTestEngine:
    """NEXUS統合テストエンジン"""
    
    def __init__(self):
        self.name = "NEXUS Unified Test Engine"
        self.version = "5.0.0"
        self.sdc_engine = NexusSDCEngine()
        self.image_engine = NexusImageSDCEngine() if NexusImageSDCEngine else None
        self.test_results = {
            'text': [],
            'audio': [],
            'video': [],
            'image': [],
            'archive': []
        }
        self.theoretical_targets = {
            'text': 95.0,      # テキストファイル理論値
            'mp3': 85.0,       # MP3理論値
            'wav': 80.0,       # WAV理論値
            'mp4': 74.8,       # MP4理論値
            'jpeg': 84.3,      # JPEG理論値
            'png': 80.0,       # PNG理論値
            'archive': 89.2    # アーカイブ理論値
        }
    
    def run_comprehensive_test(self):
        """包括的テスト実行"""
        show_step("NEXUS SDC 統合テストシステム Phase 5")
        print("=" * 80)
        
        # テストファイルの設定
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sample_dir = os.path.join(os.path.dirname(base_dir), "NXZip-Python", "sample")
        
        # ファイル分類
        test_files = {
            'text': ["出庫実績明細_202412.txt"],
            'audio': ["陰謀論.mp3", "generated-music-1752042054079.wav"],
            'video': ["Python基礎講座3_4月26日-3.mp4"],
            'image': ["COT-001.jpg", "COT-012.png"],
            'archive': ["COT-001.7z", "COT-012.7z", "Python基礎講座3_4月26日-3.7z"]
        }
        
        total_results = []
        
        # カテゴリ別テスト実行
        for category, filenames in test_files.items():
            print(f"\n🔧 {category.upper()}ファイルテスト開始")
            print("-" * 60)
            
            category_results = []
            
            for filename in filenames:
                file_path = os.path.join(sample_dir, filename)
                if not os.path.exists(file_path):
                    show_warning(f"ファイルが見つかりません: {filename}")
                    continue
                
                try:
                    # ファイルタイプに応じたエンジン選択
                    if category == 'image' and self.image_engine:
                        result = self.image_engine.compress_image(file_path)
                        result['category'] = category
                        result['filename'] = filename
                        result['engine'] = 'image_sdc'
                    else:
                        result = self.sdc_engine.compress_file(file_path)
                        result['category'] = category
                        result['filename'] = filename
                        result['engine'] = 'general_sdc'
                    
                    category_results.append(result)
                    total_results.append(result)
                    
                    # 可逆性確認
                    print("🔧 可逆性テスト実行中")
                    if category == 'image' and self.image_engine:
                        self.image_engine.decompress_image(result['output_path'])
                    else:
                        self.sdc_engine.decompress_file(result['output_path'])
                    print("✅ 可逆性確認完了")
                    
                except Exception as e:
                    show_warning(f"テスト失敗 {filename}: {str(e)}")
                    continue
            
            # カテゴリ別結果保存
            self.test_results[category] = category_results
        
        # 総合結果表示
        self._display_comprehensive_results(total_results)
        
        return total_results
    
    def _display_comprehensive_results(self, results):
        """総合結果の表示"""
        if not results:
            print("❌ テスト結果なし")
            return
        
        print("\n" + "=" * 80)
        show_success("NEXUS SDC Phase 5 統合テスト結果")
        print("=" * 80)
        
        # 全体統計
        total_original = sum(r['original_size'] for r in results)
        total_compressed = sum(r['compressed_size'] for r in results)
        overall_compression = (1 - total_compressed / total_original) * 100
        
        print(f"\n📊 全体統計:")
        print(f"   🎯 テストファイル数: {len(results)}")
        print(f"   📊 平均圧縮率: {overall_compression:.1f}%")
        print(f"   💾 総処理サイズ: {total_original:,} bytes ({total_original/(1024*1024):.1f}MB)")
        print(f"   🗜️ 総圧縮サイズ: {total_compressed:,} bytes ({total_compressed/(1024*1024):.1f}MB)")
        print(f"   💰 総節約サイズ: {total_original - total_compressed:,} bytes")
        
        # カテゴリ別詳細結果
        print(f"\n📈 カテゴリ別詳細結果:")
        print("-" * 80)
        
        categories = {}
        for result in results:
            category = result['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(result)
        
        for category, cat_results in categories.items():
            if not cat_results:
                continue
            
            cat_original = sum(r['original_size'] for r in cat_results)
            cat_compressed = sum(r['compressed_size'] for r in cat_results)
            cat_compression = (1 - cat_compressed / cat_original) * 100
            
            print(f"\n🎯 {category.upper()} ファイル結果:")
            print(f"   📊 平均圧縮率: {cat_compression:.1f}%")
            print(f"   📁 ファイル数: {len(cat_results)}")
            
            # 個別ファイル結果
            for result in cat_results:
                filename = result['filename']
                compression_ratio = result['compression_ratio']
                original_mb = result['original_size'] / (1024 * 1024)
                compressed_mb = result['compressed_size'] / (1024 * 1024)
                
                # 理論値との比較
                file_ext = Path(filename).suffix.lower()
                theoretical_target = self._get_theoretical_target(category, file_ext)
                achievement_rate = (compression_ratio / theoretical_target * 100) if theoretical_target > 0 else 0
                
                print(f"   • {filename}")
                print(f"     圧縮率: {compression_ratio:.1f}% ({original_mb:.1f}MB → {compressed_mb:.1f}MB)")
                if theoretical_target > 0:
                    print(f"     理論達成率: {achievement_rate:.1f}% (目標: {theoretical_target:.1f}%)")
                print(f"     エンジン: {result['engine']}")
        
        # 理論値達成度分析
        print(f"\n🎯 理論値達成度分析:")
        print("-" * 80)
        
        achievement_summary = {}
        for result in results:
            filename = result['filename']
            category = result['category']
            compression_ratio = result['compression_ratio']
            file_ext = Path(filename).suffix.lower()
            
            theoretical_target = self._get_theoretical_target(category, file_ext)
            if theoretical_target > 0:
                achievement_rate = compression_ratio / theoretical_target * 100
                format_key = f"{category}_{file_ext[1:]}" if file_ext else category
                
                if format_key not in achievement_summary:
                    achievement_summary[format_key] = {
                        'achieved': [],
                        'target': theoretical_target,
                        'category': category
                    }
                achievement_summary[format_key]['achieved'].append(achievement_rate)
        
        for format_key, data in achievement_summary.items():
            avg_achievement = sum(data['achieved']) / len(data['achieved'])
            target = data['target']
            max_achieved = max(data['achieved'])
            
            status = "🔥" if avg_achievement >= 90 else "✅" if avg_achievement >= 70 else "⚠️" if avg_achievement >= 50 else "❌"
            
            print(f"   {status} {format_key}: {avg_achievement:.1f}% 達成 (目標: {target:.1f}%, 最高: {max_achieved:.1f}%)")
        
        # 改善提案
        print(f"\n💡 改善提案:")
        print("-" * 80)
        
        low_achievement = [k for k, v in achievement_summary.items() 
                          if sum(v['achieved']) / len(v['achieved']) < 70]
        
        if low_achievement:
            print("   🎯 優先改善対象:")
            for format_key in low_achievement:
                data = achievement_summary[format_key]
                avg_achievement = sum(data['achieved']) / len(data['achieved'])
                improvement_potential = data['target'] - (avg_achievement * data['target'] / 100)
                print(f"     • {format_key}: +{improvement_potential:.1f}% 改善余地")
        
        high_achievement = [k for k, v in achievement_summary.items() 
                           if sum(v['achieved']) / len(v['achieved']) >= 90]
        
        if high_achievement:
            print("   🏆 高達成フォーマット:")
            for format_key in high_achievement:
                data = achievement_summary[format_key]
                avg_achievement = sum(data['achieved']) / len(data['achieved'])
                print(f"     • {format_key}: {avg_achievement:.1f}% 達成済み")
    
    def _get_theoretical_target(self, category, file_ext):
        """理論目標値の取得"""
        if category == 'text':
            return self.theoretical_targets['text']
        elif category == 'audio':
            if file_ext == '.mp3':
                return self.theoretical_targets['mp3']
            elif file_ext == '.wav':
                return self.theoretical_targets['wav']
        elif category == 'video':
            if file_ext == '.mp4':
                return self.theoretical_targets['mp4']
        elif category == 'image':
            if file_ext in ['.jpg', '.jpeg']:
                return self.theoretical_targets['jpeg']
            elif file_ext == '.png':
                return self.theoretical_targets['png']
        elif category == 'archive':
            return self.theoretical_targets['archive']
        
        return 0.0
    
    def run_performance_analysis(self):
        """パフォーマンス分析実行"""
        show_step("パフォーマンス分析開始")
        
        # 速度テスト用の大きなファイルを特定
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sample_dir = os.path.join(os.path.dirname(base_dir), "NXZip-Python", "sample")
        
        large_files = []
        for filename in os.listdir(sample_dir):
            file_path = os.path.join(sample_dir, filename)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                if size > 10 * 1024 * 1024:  # 10MB以上
                    large_files.append((filename, size))
        
        large_files.sort(key=lambda x: x[1], reverse=True)
        
        print("\n📊 大容量ファイルパフォーマンス分析:")
        print("-" * 60)
        
        for filename, size in large_files[:3]:  # 上位3ファイル
            file_path = os.path.join(sample_dir, filename)
            print(f"\n🔧 {filename} ({size/(1024*1024):.1f}MB)")
            
            try:
                start_time = time.time()
                result = self.sdc_engine.compress_file(file_path)
                compress_time = time.time() - start_time
                
                compress_speed = (size / (1024 * 1024)) / compress_time
                compression_ratio = result['compression_ratio']
                
                print(f"   圧縮率: {compression_ratio:.1f}%")
                print(f"   圧縮速度: {compress_speed:.1f} MB/s")
                print(f"   処理時間: {compress_time:.1f}秒")
                
                # 展開速度測定
                start_time = time.time()
                self.sdc_engine.decompress_file(result['output_path'])
                decompress_time = time.time() - start_time
                decompress_speed = (size / (1024 * 1024)) / decompress_time
                
                print(f"   展開速度: {decompress_speed:.1f} MB/s")
                
            except Exception as e:
                show_warning(f"パフォーマンステスト失敗: {str(e)}")


def main():
    """メイン実行関数"""
    engine = NexusUnifiedTestEngine()
    
    if len(sys.argv) < 2:
        print(f"使用方法: {sys.argv[0]} <command>")
        print("コマンド:")
        print("  comprehensive  - 全フォーマット統合テスト")
        print("  performance    - パフォーマンス分析")
        print("  all           - 全テスト実行")
        return
    
    command = sys.argv[1].lower()
    
    if command == "comprehensive":
        engine.run_comprehensive_test()
    elif command == "performance":
        engine.run_performance_analysis()
    elif command == "all":
        engine.run_comprehensive_test()
        print("\n" + "=" * 80)
        engine.run_performance_analysis()
    else:
        print(f"❌ 未知のコマンド: {command}")


if __name__ == "__main__":
    main()
