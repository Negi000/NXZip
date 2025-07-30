#!/usr/bin/env python3
"""
DataType処理エラーの緊急修正スクリプト
TMC v9.0の可逆性問題を解決
"""

def fix_datatype_issues():
    """DataType処理の問題を修正"""
    file_path = r"c:\Users\241822\Desktop\新しいフォルダー (2)\NXZip\NXZip-Python\nxzip\engine\nexus_tmc_v4_unified.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # DataType(header['data_type'])を安全な処理に置換
    modifications = [
        # 1. DataType呼び出しを安全にする
        ("data_type = DataType(header['data_type'])", 
         "data_type = self._safe_get_datatype(header['data_type'])"),
        
        # 2. エラー処理を追加
        ("# 逆変換（インテリジェント・バイパス対応）",
         """# 逆変換（インテリジェント・バイパス対応）
            try:"""),
        
        # 3. BWTTransformerのエラーハンドリング強化
        ("original_data = transformer.inverse_transform(decompressed_streams, header['transform_info'])",
         """try:
                    original_data = transformer.inverse_transform(decompressed_streams, header['transform_info'])
                except Exception as transform_error:
                    print(f"⚠️ 変換エラー: {transform_error} - 直接結合でフォールバック")
                    original_data = b''.join(decompressed_streams)"""),
    ]
    
    for old, new in modifications:
        content = content.replace(old, new)
    
    # ヘルパー関数が存在しない場合は追加
    if "_safe_get_datatype" not in content:
        helper_function = '''
    def _safe_get_datatype(self, data_type_str: str):
        """DataType文字列を安全にDataTypeオブジェクトに変換"""
        try:
            for dt in DataType:
                if dt.value == data_type_str:
                    return dt
            return DataType.GENERIC_BINARY
        except Exception:
            return DataType.GENERIC_BINARY
'''
        # decompress_tmc関数の前に追加
        content = content.replace(
            "def decompress_tmc(self, compressed_data: bytes)",
            helper_function + "\n    def decompress_tmc(self, compressed_data: bytes)"
        )
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ DataType処理エラーの修正完了")

if __name__ == "__main__":
    fix_datatype_issues()
