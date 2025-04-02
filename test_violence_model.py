import argparse
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from app.utils.violence_detector import ViolenceDetector
from app.utils.violence_detector_v2 import EnhancedViolenceDetector

def test_model(model_path, test_file=None):
    """测试微调模型效果"""
    # 初始化检测器
    basic_detector = ViolenceDetector()
    enhanced_detector = EnhancedViolenceDetector(
        basic_detector=basic_detector,
        model_path=model_path,
        db=None
    )
    
    # 如果提供了测试文件，批量测试
    if test_file:
        df = pd.read_csv(test_file)
        results = []
        
        for i, row in df.iterrows():
            text = row['comment']
            
            # 使用两种检测器
            basic_result = basic_detector.detect(text)
            enhanced_result = enhanced_detector.detect(text)
            
            results.append({
                'comment': text,
                'true_score': row.get('violence_score', None),
                'basic_score': basic_result['violence_score'],
                'enhanced_score': enhanced_result['violence_score'],
                'embedding_score': enhanced_result['embedding_score'],
                'basic_type': basic_result['violence_type'],
                'enhanced_type': enhanced_result['violence_type'],
            })
        
        # 保存结果比较
        result_df = pd.DataFrame(results)
        output_file = 'model_comparison_results.csv'
        result_df.to_csv(output_file, index=False)
        print(f"比较结果已保存至 {output_file}")
        
        # 计算统计信息
        if 'true_score' in result_df.columns:
            basic_mse = ((result_df['true_score'] - result_df['basic_score']) ** 2).mean()
            enhanced_mse = ((result_df['true_score'] - result_df['enhanced_score']) ** 2).mean()
            
            print(f"基础检测器 MSE: {basic_mse:.4f}")
            print(f"增强检测器 MSE: {enhanced_mse:.4f}")
    
    # 交互式测试
    while True:
        text = input("\n请输入要测试的文本 (输入q退出): ")
        if text.lower() == 'q':
            break
        
        basic_result = basic_detector.detect(text)
        enhanced_result = enhanced_detector.detect(text)
        
        print("\n基础检测器结果:")
        print(f"暴力分数: {basic_result['violence_score']:.4f}")
        print(f"暴力类型: {basic_result['violence_type']}")
        print(f"匹配词: {basic_result.get('matched_words', [])}")
        
        print("\n增强检测器结果:")
        print(f"暴力分数: {enhanced_result['violence_score']:.4f}")
        print(f"暴力类型: {enhanced_result['violence_type']}")
        print(f"嵌入分数: {enhanced_result['embedding_score']:.4f}")
        print(f"基础分数: {enhanced_result['basic_score']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='测试微调模型效果')
    parser.add_argument('--model', required=True, help='微调模型路径')
    parser.add_argument('--test', help='测试数据文件（可选）')
    
    args = parser.parse_args()
    
    test_model(args.model, args.test)