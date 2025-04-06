import pandas as pd
import argparse
import numpy as np

def balance_risk_categories(input_file, output_file, high_risk_ratio=1.5):
    """平衡数据集中的风险类别"""
    print(f"读取数据集: {input_file}")
    df = pd.read_csv(input_file)
    
    # 将数据分为三类
    low_risk = df[df['violence_score'] < 0.3]
    mid_risk = df[(df['violence_score'] >= 0.3) & (df['violence_score'] <= 0.7)]
    high_risk = df[df['violence_score'] > 0.7]
    
    print(f"原始数据分布 - 低风险: {len(low_risk)}, 中风险: {len(mid_risk)}, 高风险: {len(high_risk)}")
    
    # 计算要保留的样本数
    min_samples = min(len(low_risk), len(high_risk))
    mid_samples = int(min_samples * 1.0)  # 保持中风险样本数与最小类别相同
    high_samples = int(min_samples * high_risk_ratio)  # 可以增加高风险样本比例
    
    # 确保不超过实际样本数
    mid_samples = min(mid_samples, len(mid_risk))
    high_samples = min(high_samples, len(high_risk))
    
    # 采样
    balanced_low = low_risk.sample(min_samples)
    balanced_mid = mid_risk.sample(mid_samples)
    balanced_high = high_risk.sample(high_samples)
    
    # 合并
    balanced_df = pd.concat([balanced_low, balanced_mid, balanced_high])
    print(f"平衡后数据分布 - 低风险: {len(balanced_low)}, 中风险: {len(balanced_mid)}, 高风险: {len(balanced_high)}")
    
    # 随机打乱
    balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)
    
    # 保存
    balanced_df.to_csv(output_file, index=False)
    print(f"平衡后的数据已保存至: {output_file}")
    
    return balanced_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='平衡数据集中的风险类别')
    parser.add_argument('--input', required=True, help='输入CSV文件')
    parser.add_argument('--output', required=True, help='输出CSV文件')
    parser.add_argument('--high_ratio', type=float, default=1.5, help='高风险样本比例')
    
    args = parser.parse_args()
    balance_risk_categories(args.input, args.output, args.high_ratio)