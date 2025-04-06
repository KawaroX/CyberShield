import pandas as pd
from sklearn.utils import resample

# 读取合并后的数据
df = pd.read_csv('/Volumes/base/violence_embedding/merged_violence_data.csv')

# 检查风险级别分布
if 'risk_level' in df.columns:
    print(df['risk_level'].value_counts())
else:
    # 如果还没有创建风险级别，先创建
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['极低风险', '低风险', '中风险', '高风险', '极高风险']
    df['risk_level'] = pd.cut(df['violence_score'], bins=bins, labels=labels)
    print(df['risk_level'].value_counts())

# 分离各个风险级别
extreme_low = df[df['violence_score'] < 0.2]
low = df[(df['violence_score'] >= 0.2) & (df['violence_score'] < 0.4)]
medium = df[(df['violence_score'] >= 0.4) & (df['violence_score'] < 0.6)]
high = df[(df['violence_score'] >= 0.6) & (df['violence_score'] < 0.8)]
extreme_high = df[df['violence_score'] >= 0.8]

# 打印各级别样本数
print(f"极低风险样本数: {len(extreme_low)}")
print(f"低风险样本数: {len(low)}")
print(f"中风险样本数: {len(medium)}")
print(f"高风险样本数: {len(high)}")
print(f"极高风险样本数: {len(extreme_high)}")

# 确定目标样本数 (可以取各类别中的最大值，或其他策略)
target_samples = max(len(extreme_low), len(low), len(medium), len(high), len(extreme_high))
# 或者可以设置一个固定值
# target_samples = 500

# 对每个类别进行过采样或欠采样
balanced_extreme_low = resample(extreme_low, 
                              replace=len(extreme_low) < target_samples,  # 如果样本少则允许重复
                              n_samples=target_samples,
                              random_state=42)

balanced_low = resample(low, 
                      replace=len(low) < target_samples,
                      n_samples=target_samples,
                      random_state=42)

balanced_medium = resample(medium, 
                         replace=len(medium) < target_samples,
                         n_samples=target_samples,
                         random_state=42)

balanced_high = resample(high, 
                       replace=len(high) < target_samples,
                       n_samples=target_samples,
                       random_state=42)

balanced_extreme_high = resample(extreme_high, 
                               replace=len(extreme_high) < target_samples,
                               n_samples=target_samples,
                               random_state=42)

# 合并平衡后的数据
balanced_df = pd.concat([
    balanced_extreme_low, 
    balanced_low, 
    balanced_medium, 
    balanced_high, 
    balanced_extreme_high
])

# 随机打乱
balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)

# 保存平衡后的数据
balanced_df.to_csv('/Volumes/base/violence_embedding/balanced_5class_data_1.csv', index=False)

print(f"平衡后的数据已保存，共 {len(balanced_df)} 条记录")
print(balanced_df['risk_level'].value_counts())