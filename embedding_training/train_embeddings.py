import argparse
import pandas as pd
import torch
import os
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.readers import InputExample
from sentence_transformers import losses
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def train_model(dataset_file, model_name, output_dir, epochs=5, batch_size=16, 
                learning_rate=2e-5, max_seq_length=128, test_size=0.1):
    """微调嵌入模型"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据
    print(f"读取数据集 {dataset_file}")
    df = pd.read_csv(dataset_file)
    print(f"共 {len(df)} 条数据")
    
    # 数据预处理
    # 去除异常值
    df = df[(df['violence_score'] >= 0) & (df['violence_score'] <= 1)]
    # 移除重复项
    df = df.drop_duplicates(subset=['comment'])
    
    # 动态调整分箱数量
    n_bins = min(5, len(df) // 10)  # 确保每个分箱至少有10个样本
    if n_bins < 2:
        # 当数据量太少时关闭分层
        train_df, val_df = train_test_split(df, test_size=test_size, random_state=42)
    else:
        try:
            # 尝试使用分位数分箱
            df['score_bin'] = pd.qcut(df['violence_score'], q=n_bins, duplicates='drop')
            train_df, val_df = train_test_split(df, test_size=test_size, random_state=42, 
                                                stratify=df['score_bin'])
        except ValueError:
            # 分箱失败时回退到随机分割
            train_df, val_df = train_test_split(df, test_size=test_size, random_state=42)
    
    print(f"训练集: {len(train_df)} 条, 验证集: {len(val_df)} 条")
    
    # 准备训练数据
    train_examples = []
    for _, row in train_df.iterrows():
        train_examples.append(InputExample(texts=[row['comment']], label=row['violence_score']))
    
    # 准备验证数据
    val_examples = []
    for _, row in val_df.iterrows():
        val_examples.append(InputExample(texts=[row['comment']], label=row['violence_score']))
    
    # 初始化模型
    print(f"加载基础模型 {model_name}")
    model = SentenceTransformer(model_name)
    # 替换原来的设备设置方式
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)  # 使用标准的PyTorch设备迁移方式
    model.max_seq_length = max_seq_length
    
    # 自定义合并函数
    def custom_collate_fn(batch):
        texts = [example.texts[0] for example in batch]
        labels = [example.label for example in batch]
        labels = torch.tensor(labels, dtype=torch.float32)
        return texts, labels
    
    # 创建数据加载器
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_examples, shuffle=False, batch_size=batch_size, collate_fn=custom_collate_fn)
    
    # 自定义损失函数
    class CustomMSELoss(losses.MSELoss):
        def forward(self, sentence_features, labels):
            reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
            # 对嵌入向量取平均值，使其成为一个单一值
            embeddings = torch.mean(reps[0], dim=1)
            
            # 添加权重 - 给高风险和低风险样本更高的权重
            weights = torch.ones_like(labels).to(labels.device)
            weights[labels < 0.3] = 2.0  # 低风险样本权重
            weights[labels > 0.7] = 2.0  # 高风险样本权重
        
            # 计算加权MSE
            squared_error = (embeddings - labels) ** 2
            weighted_squared_error = squared_error * weights
            return torch.mean(weighted_squared_error)
    
    # 使用自定义损失函数
    train_loss = CustomMSELoss(model=model)
    
    # 记录当前时间
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(output_dir, f"bge-violence-{start_time}")
    
    # 添加评估器
    from sentence_transformers.evaluation import SequentialEvaluator, EmbeddingSimilarityEvaluator, SentenceEvaluator
    class RegressionEvaluator(SentenceEvaluator):
        """
        评估回归任务，例如暴力评分预测
        """
        def __init__(self, dataloader, name=''):
            self.dataloader = dataloader
            self.name = name
            
        def __call__(self, model, output_path=None, epoch=None, steps=None):
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            
            model.eval()
            embeddings = []
            labels = []
            
            with torch.no_grad():
                for batch in self.dataloader:
                    features, label_ids = batch
                    tokenized_features = model.tokenize(features)
                    # 使用模型当前的设备信息
                    tokenized_features = {k: v.to(model.device) for k, v in tokenized_features.items()}
                    emb = model(tokenized_features)['sentence_embedding']
                    
                    # 对嵌入向量取平均值，使其成为一个单一值
                    emb = torch.mean(emb, dim=1)
                    
                    # 将嵌入和标签转移到CPU
                    embeddings.extend(emb.cpu().numpy())
                    labels.extend(label_ids.cpu().numpy())
                    
            embeddings = np.array(embeddings)
            labels = np.array(labels)
            
            # 计算均方误差和平均绝对误差
            # 这里我们简单地使用均方误差作为评估指标
            mse = mean_squared_error(labels, embeddings)
            mae = mean_absolute_error(labels, embeddings)
            
            print(f"Epoch {epoch}: MSE = {mse:.4f}, MAE = {mae:.4f}")
            
            return mse  # 返回 MSE 作为主要评估指标

    # 使用我们定制的评估器
    evaluator = RegressionEvaluator(val_dataloader, name='violence-validation')
    
    # 训练模型
    print(f"开始训练，epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        evaluation_steps=500,  # 更频繁的评估
        warmup_steps=int(len(train_dataloader) * 0.2),  # 增加warmup步数
        output_path=model_save_path,
        optimizer_params={'lr': learning_rate, 'weight_decay': 0.01},  # 添加权重衰减
        save_best_model=True,
        use_amp=True  # 使用混合精度训练加速
    )
    
    print(f"模型训练完成，保存至 {model_save_path}")
    
    # 保存模型信息
    with open(os.path.join(model_save_path, "training_info.txt"), "w") as f:
        f.write(f"Base model: {model_name}\n")
        f.write(f"Training dataset: {dataset_file}\n")
        f.write(f"Training examples: {len(train_examples)}\n")
        f.write(f"Validation examples: {len(val_examples)}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Max sequence length: {max_seq_length}\n")
        f.write(f"Training date: {start_time}\n")
    
    # 保存分数分布信息
    score_bins = [0, 0.3, 0.7, 1.0]
    labels = ['低风险', '中风险', '高风险']
    df['risk_level'] = pd.cut(df['violence_score'], bins=score_bins, labels=labels)
    distribution = df['risk_level'].value_counts()
    
    with open(os.path.join(model_save_path, "data_distribution.txt"), "w") as f:
        f.write("数据分布:\n")
        f.write(str(distribution))
        f.write("\n\n暴力类型分布:\n")
        f.write(str(df['violence_type'].value_counts()))
    
    return model_save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='微调嵌入模型')
    parser.add_argument('--dataset', required=True, help='标注数据文件')
    parser.add_argument('--model', default='BAAI/bge-m3', help='基础模型')
    parser.add_argument('--output', required=True, help='输出模型目录')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=2e-5, help='学习率')
    parser.add_argument('--seq_len', type=int, default=128, help='最大序列长度')
    
    args = parser.parse_args()
    
    train_model(
        args.dataset, 
        args.model, 
        args.output,
        args.epochs,
        args.batch,
        args.lr,
        args.seq_len
    )