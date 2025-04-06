# bert_classification/train_classifier.py

import argparse
import pandas as pd
import torch
import os
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
from tqdm import tqdm

class ViolenceDataset(Dataset):
    """暴力评分数据集"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = float(self.labels[idx])
        
        # 将文本转换为模型输入格式
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 去除批次维度
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding['labels'] = torch.tensor(label, dtype=torch.float32)
        
        return encoding

def balance_dataset(df, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                   labels=['极低风险', '低风险', '中风险', '高风险', '极高风险']):
    """平衡数据集，处理极端类别"""
    
    print("执行数据平衡...")
    
    # 检查并处理NaN值
    print(f"NaN值检查 - comment列: {df['comment'].isna().sum()}, violence_score列: {df['violence_score'].isna().sum()}")
    
    # 移除包含NaN的行
    df = df.dropna(subset=['comment', 'violence_score'])
    print(f"移除NaN后剩余 {len(df)} 条数据")
    
    # 分箱处理
    df['risk_level'] = pd.cut(df['violence_score'], bins=bins, labels=labels)
    
    # 检查risk_level列是否有NaN值
    if df['risk_level'].isna().sum() > 0:
        df = df.dropna(subset=['risk_level'])
        print(f"移除risk_level为NaN后剩余 {len(df)} 条数据")
    
    # 分离各个风险级别
    categories = {}
    for label in labels:
        categories[label] = df[df['risk_level'] == label]
        print(f"{label}样本数: {len(categories[label])}")
    
    # 计算目标样本数
    category_sizes = [len(cat_df) for cat_df in categories.values()]
    target_samples = min(
        10,  # 最大上限
        int(np.percentile(category_sizes, 75))  # 75%分位数
    )
    print(f"目标每类样本数: {target_samples}")
    
    # 平衡各类别
    from sklearn.utils import resample
    balanced_categories = []
    
    for label, cat_df in categories.items():
        if len(cat_df) < target_samples:
            # 过采样
            resampled = resample(cat_df, replace=True, n_samples=target_samples, random_state=42)
            balanced_categories.append(resampled)
            print(f"{label}: 过采样 {len(cat_df)} -> {len(resampled)}")
        elif len(cat_df) > target_samples:
            # 欠采样
            resampled = cat_df.sample(target_samples, random_state=42)
            balanced_categories.append(resampled)
            print(f"{label}: 欠采样 {len(cat_df)} -> {len(resampled)}")
        else:
            # 保持不变
            balanced_categories.append(cat_df)
            print(f"{label}: 保持不变 {len(cat_df)}")
    
    # 合并平衡后的数据
    balanced_df = pd.concat(balanced_categories, ignore_index=True)
    # 随机打乱
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"平衡后数据集: {len(balanced_df)} 条")
    
    return balanced_df

def train_model(dataset_file, model_name, output_dir, epochs=5, batch_size=16, 
                learning_rate=3e-5, max_seq_length=128, test_size=0.1, balance_data=True):
    """训练MacBERT分类器模型"""
    
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
    
    # 数据平衡
    if balance_data:
        df = balance_dataset(df)
    
    # 拆分训练集和验证集
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=42)
    print(f"训练集: {len(train_df)} 条, 验证集: {len(val_df)} 条")
    
    # 初始化tokenizer和模型
    print(f"加载模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 创建回归模型 - 输出单个值代表暴力分数
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,   # 回归任务，输出暴力分数
        problem_type="regression"
    )
    
    # 将模型移至可用设备
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    
    # 创建数据集
    train_dataset = ViolenceDataset(
        train_df['comment'].tolist(),
        train_df['violence_score'].tolist(),
        tokenizer,
        max_length=max_seq_length
    )
    
    val_dataset = ViolenceDataset(
        val_df['comment'].tolist(),
        val_df['violence_score'].tolist(),
        tokenizer,
        max_length=max_seq_length
    )
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size
    )
    
    # 定义优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # 使用MSE损失进行回归
    loss_fn = torch.nn.MSELoss()
    
    # 定义学习率调度器
    total_steps = len(train_dataloader) * epochs
    warmup_steps = int(total_steps * 0.1)  # 10% 的步骤用于预热
    
    from transformers import get_scheduler
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 训练循环
    best_val_loss = float('inf')
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(output_dir, f"macbert-violence-{start_time}")
    best_model_path = os.path.join(output_dir, f"macbert-violence-{start_time}-best")
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(best_model_path, exist_ok=True)
    
    print(f"开始训练，epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}")
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for batch in progress_bar:
            # 将输入移至设备
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 清除梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(**{k: v for k, v in batch.items() if k != 'labels'})
            
            # 计算损失
            loss = loss_fn(outputs.logits.squeeze(-1), batch['labels'])
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            scheduler.step()
            
            # 累加损失
            train_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_dataloader)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        
        with torch.no_grad():
            for batch in progress_bar:
                # 将输入移至设备
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # 前向传播
                outputs = model(**{k: v for k, v in batch.items() if k != 'labels'})
                
                # 计算损失
                loss = loss_fn(outputs.logits.squeeze(-1), batch['labels'])
                
                # 累加损失
                val_loss += loss.item()
                
                # 更新进度条
                progress_bar.set_postfix({'loss': loss.item()})
        
        avg_val_loss = val_loss / len(val_dataloader)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            print(f"Saved best model with val_loss: {best_val_loss:.4f}")
    
    # 保存最终模型
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    # 保存训练信息
    with open(os.path.join(model_save_path, "training_info.txt"), "w") as f:
        f.write(f"Base model: {model_name}\n")
        f.write(f"Training dataset: {dataset_file}\n")
        f.write(f"Training examples: {len(train_df)}\n")
        f.write(f"Validation examples: {len(val_df)}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Max sequence length: {max_seq_length}\n")
        f.write(f"Training date: {start_time}\n")
        f.write(f"Best validation loss: {best_val_loss:.4f}\n")
    
    # 保存数据分布信息
    score_bins = [0, 0.3, 0.7, 1.0]
    labels = ['低风险', '中风险', '高风险']
    df['risk_level'] = pd.cut(df['violence_score'], bins=score_bins, labels=labels)
    distribution = df['risk_level'].value_counts()
    
    with open(os.path.join(model_save_path, "data_distribution.txt"), "w") as f:
        f.write("数据分布:\n")
        f.write(str(distribution))
        f.write("\n\n暴力类型分布:\n")
        f.write(str(df['violence_type'].value_counts()))
    
    print(f"训练完成，模型保存至: {model_save_path}")
    print(f"最佳模型保存至: {best_model_path}")
    
    return model_save_path, best_model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练MacBERT文本分类器')
    parser.add_argument('--dataset', required=True, help='标注数据文件')
    parser.add_argument('--model', default='hfl/chinese-macbert-base', help='基础模型')
    parser.add_argument('--output', required=True, help='输出模型目录')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=2e-5, help='学习率')
    parser.add_argument('--seq_len', type=int, default=256, help='最大序列长度')
    parser.add_argument('--balance', action='store_true', help='是否平衡数据集')
    
    args = parser.parse_args()
    
    train_model(
        args.dataset, 
        args.model, 
        args.output,
        args.epochs,
        args.batch,
        args.lr,
        args.seq_len,
        balance_data=args.balance
    )