import pandas as pd
import numpy as np

# 步骤1: 读取并处理第一个CSV（将分数*0.7或0.8）
def process_first_csv(file_path, scale_factor=0.8):
    df1 = pd.read_csv(file_path)
    print(f"原始数据集1分布 - 最小值: {df1['violence_score'].min()}, 最大值: {df1['violence_score'].max()}, 均值: {df1['violence_score'].mean()}")
    
    # 缩放暴力分数
    df1['original_score'] = df1['violence_score'].copy()  # 保存原始分数
    df1['violence_score'] = df1['violence_score'] * scale_factor
    
    # 确保最大值不超过0.8
    df1.loc[df1['violence_score'] > 0.8, 'violence_score'] = 0.81130982
    # 确保最小值不低于0.0
    df1.loc[df1['violence_score'] < 0.1, 'violence_score'] = 0.10581201
    
    print(f"处理后数据集1分布 - 最小值: {df1['violence_score'].min()}, 最大值: {df1['violence_score'].max()}, 均值: {df1['violence_score'].mean()}")
    return df1

# 步骤2: 处理高暴力分数的CSV（分数控制在0.8-1之间）
def process_high_violence_csv(file_path):
    df2 = pd.read_csv(file_path)
    print(f"原始高暴力数据集分布 - 最小值: {df2['violence_score'].min()}, 最大值: {df2['violence_score'].max()}, 均值: {df2['violence_score'].mean()}")
    
    # 保存原始分数
    df2['original_score'] = df2['violence_score'].copy()
    
    # 线性映射到0.8-1.0范围
    min_val = df2['violence_score'].min()
    max_val = df2['violence_score'].max()
    
    # 映射公式: new_val = 0.8 + 0.2 * (val - min_val) / (max_val - min_val)
    df2['violence_score'] = 0.6 + 0.4 * (df2['violence_score'] - min_val) / (max_val - min_val)
    
    print(f"处理后高暴力数据集分布 - 最小值: {df2['violence_score'].min()}, 最大值: {df2['violence_score'].max()}, 均值: {df2['violence_score'].mean()}")
    return df2

# 步骤3: 处理低暴力分数的CSV（分数控制在0.0-0.1之间）
def process_low_violence_csv(file_path):
    df3 = pd.read_csv(file_path)
    print(f"原始低暴力数据集分布 - 最小值: {df3['violence_score'].min()}, 最大值: {df3['violence_score'].max()}, 均值: {df3['violence_score'].mean()}")
    
    # 保存原始分数
    df3['original_score'] = df3['violence_score'].copy()
    
    # 线性映射到0.0-0.1范围
    min_val = df3['violence_score'].min()
    max_val = df3['violence_score'].max()
    
    # 映射公式: new_val = 0.1 * (val - min_val) / (max_val - min_val)
    df3['violence_score'] = 0.1 * (df3['violence_score'] - min_val) / (max_val - min_val)
    
    print(f"处理后低暴力数据集分布 - 最小值: {df3['violence_score'].min()}, 最大值: {df3['violence_score'].max()}, 均值: {df3['violence_score'].mean()}")
    return df3

# 步骤4: 合并所有数据集
def merge_datasets(df1, df2, df3):
    # 合并三个数据集
    merged_df = pd.concat([df1, df2, df3], ignore_index=True)
    
    # 随机打乱
    merged_df = merged_df.sample(frac=1).reset_index(drop=True)
    
    # 创建新的风险类别（5类）
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['极低风险', '低风险', '中风险', '高风险', '极高风险']
    merged_df['risk_level'] = pd.cut(merged_df['violence_score'], bins=bins, labels=labels)
    
    print("\n合并后数据分布:")
    print(merged_df['risk_level'].value_counts())
    print(f"合并后数据集 - 最小值: {merged_df['violence_score'].min()}, 最大值: {merged_df['violence_score'].max()}, 均值: {merged_df['violence_score'].mean()}")
    
    return merged_df

# 主函数
def main():
    # 文件路径
    file1 = '/Volumes/base/violence_embedding/labeled_data_20250401_190931.csv'
    file2 = '/Users/kawarox/Downloads/high-violence-comments.csv'  # 请替换为你的高暴力数据文件路径
    file3 = '/Users/kawarox/Downloads/low-violence-comments.csv'   # 请替换为你的低暴力数据文件路径
    output_file = '/Volumes/base/violence_embedding/merged_violence_data.csv'
    
    # 处理数据
    df1 = process_first_csv(file1, scale_factor=0.7)
    df2 = process_high_violence_csv(file2)
    df3 = process_low_violence_csv(file3)
    
    # 合并数据
    merged_df = merge_datasets(df1, df2, df3)
    
    # 保存合并后的数据
    merged_df.to_csv(output_file, index=False)
    print(f"合并后的数据已保存至: {output_file}")

if __name__ == "__main__":
    main()