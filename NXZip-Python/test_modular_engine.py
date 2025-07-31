#!/usr/bin/env python3
"""
TMC v9.1 モジュラーエンジンのテストスクリプト
"""

try:
    from nxzip.engine.nexus_tmc_v91_modular import NEXUSTMCEngineV91
    print('✅ TMC v9.1 モジュラーエンジンのインポート成功')

    # エンジンの初期化テスト
    engine = NEXUSTMCEngineV91()
    print(f'✅ エンジン初期化成功: {engine.max_workers}ワーカー')

    # 簡単な圧縮テスト
    test_data = b'Hello, World! ' * 100
    print(f'📝 テストデータサイズ: {len(test_data)} bytes')
    
    compressed, info = engine.compress_sync(test_data)
    
    compression_ratio = info.get('compression_ratio', 0)
    engine_version = info.get('engine_version', 'Unknown')
    
    print(f'✅ 圧縮テスト成功: {compression_ratio:.1f}% 圧縮')
    print(f'📊 エンジンバージョン: {engine_version}')
    print(f'📦 圧縮後サイズ: {len(compressed)} bytes')
    
    # 統計情報表示
    stats = engine.get_stats()
    print(f'📈 処理統計: {stats["files_processed"]}ファイル処理済み')
    
    print('\n🎉 TMC v9.1 モジュラーエンジン動作確認完了！')

except ImportError as e:
    print(f'❌ インポートエラー: {e}')
    print('💡 分離されたモジュールが正しく配置されているか確認してください')

except Exception as e:
    print(f'❌ 実行エラー: {e}')
    import traceback
    traceback.print_exc()
