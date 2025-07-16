#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

from nxzip.engine.spe_core import SPECore

spe = SPECore()
test_data = b'NXZip SPE Core Test Vector 2024'
print('🔍 テストデータ:', test_data)

try:
    transformed = spe.apply_transform(test_data)
    print('🔄 変換後サイズ:', len(transformed))
    restored = spe.reverse_transform(transformed)
    print('🔄 復元後サイズ:', len(restored))
    print('✅ 可逆性:', restored == test_data)
    print('✅ 変換効果:', transformed != test_data)
    if restored == test_data and transformed != test_data:
        print('🎉 SPEコア: 正常動作')
    else:
        print('❌ SPEコア: 問題あり')
except Exception as e:
    print('❌ エラー:', e)
    import traceback
    traceback.print_exc()
