# violence_trainer.py
import pandas as pd
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
import argparse
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
from sentence_transformers import InputExample

class ViolenceDataset(Dataset):
    """暴力评分数据集"""
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        return self.texts[idx], float(self.labels[idx])

class ViolenceTrainer:
    """暴力评分模型训练器"""
    def __init__(self, model_name, output_dir, epochs=5, batch_size=16, lr=2e-5):
        self.model_name = model_name
        self.output_dir = output_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        
        # 检测可用设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        # 加载模型
        print(f"加载基础模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)  # 将模型移动到设备上
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成时间戳
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_save_path = os.path.join(output_dir, f"violence-model-{self.timestamp}")

    def load_data(self, data_file):
        """加载并处理数据"""
        print(f"读取数据文件: {data_file}")
        df = pd.read_csv(data_file)
        print(f"数据集共 {len(df)} 条记录")
        
        # 分割训练集和验证集
        train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
        print(f"训练集: {len(train_df)} 条, 验证集: {len(val_df)} 条")
        
        # 创建数据集
        train_dataset = ViolenceDataset(train_df['comment'].tolist(), train_df['violence_score'].tolist())
        val_dataset = ViolenceDataset(val_df['comment'].tolist(), val_df['violence_score'].tolist())
        
        return train_dataset, val_dataset

    def train(self, train_dataset, val_dataset=None):
        """训练模型"""
        # 创建数据加载器
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size)
        
        # 优化器
        params = list(self.model.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.lr)
        
        # 损失函数
        loss_fn = torch.nn.MSELoss()
        
        # 训练循环
        print(f"开始训练: {self.epochs} 轮, 每批 {self.batch_size} 条样本")
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            # 训练
            self.model.train()
            total_loss = 0
            train_progress = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")
            
            for batch_idx, (texts, labels) in enumerate(train_progress):
                # 将标签转换为tensor并移动到设备上
                labels = torch.tensor(labels, dtype=torch.float32).to(self.device)
                
                # 计算嵌入向量
                with torch.set_grad_enabled(True):
                    # 前向传播
                    # 修改：将文本转换为适合模型输入的格式
                    input_examples = [InputExample(texts=[text]) for text in texts]
                    model_input = self.model.tokenize([example.texts[0] for example in input_examples])
                    for key in model_input:
                        model_input[key] = model_input[key].to(self.device)
                    model_output = self.model(model_input)
                    if isinstance(model_output, tuple):
                        # 如果是元组，假设第一个元素是嵌入向量
                        embeddings = model_output[0].to(self.device)
                    elif isinstance(model_output, dict):
                        embeddings = model_output['sentence_embedding'].to(self.device)
                    else:
                        raise ValueError(f"Unexpected model output type: {type(model_output)}")
                    
                    # 对嵌入进行处理，使其适合预测单一值
                    # 方法1: 取嵌入向量的平均值
                    pred = torch.mean(embeddings, dim=1)
                    
                    # 计算损失
                    loss = loss_fn(pred, labels)
                    
                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # 更新进度条
                    total_loss += loss.item()
                    train_progress.set_postfix({'loss': total_loss / (batch_idx + 1)})
            
            # 验证
            if val_dataset:
                val_loss = self.evaluate(val_dataset)
                print(f"Epoch {epoch+1}/{self.epochs} - 训练损失: {total_loss / len(train_dataloader):.4f}, 验证损失: {val_loss:.4f}")
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(f"{self.model_save_path}-best")
                    print(f"保存最佳模型, 验证损失: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{self.epochs} - 训练损失: {total_loss / len(train_dataloader):.4f}")
        
        # 保存最终模型
        self.save_model(self.model_save_path)
        print(f"训练完成, 模型保存至: {self.model_save_path}")
        
        return self.model_save_path

    def evaluate(self, val_dataset):
        """评估模型"""
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size)
        loss_fn = torch.nn.MSELoss()
    
        self.model.eval()
        total_loss = 0
    
        with torch.no_grad():
            for texts, labels in val_dataloader:
                # 确保标签数据类型为 float32
                if isinstance(labels, torch.Tensor):
                    if labels.dtype != torch.float32:
                        labels = labels.to(torch.float32)
                    labels = labels.clone().detach().to(self.device)
                else:
                    labels = torch.tensor(labels, dtype=torch.float32).to(self.device)
    
                # 计算嵌入向量
                input_examples = [InputExample(texts=[text]) for text in texts]
                model_input = self.model.tokenize([example.texts[0] for example in input_examples])
                for key in model_input:
                    model_input[key] = model_input[key].to(self.device)
                model_output = self.model(model_input)
                if isinstance(model_output, tuple):
                    # 如果是元组，假设第一个元素是嵌入向量
                    embeddings = model_output[0].to(self.device)
                elif isinstance(model_output, dict):
                    embeddings = model_output['sentence_embedding'].to(self.device)
                else:
                    raise ValueError(f"Unexpected model output type: {type(model_output)}")
    
                # 取平均值作为预测
                pred = torch.mean(embeddings, dim=1)
    
                # 计算损失
                loss = loss_fn(pred, labels)
                total_loss += loss.item()
    
        return total_loss / len(val_dataloader)

    def save_model(self, path):
        """保存模型"""
        self.model.save(path)
        
        # 保存训练信息
        info = {
            "base_model": self.model_name,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.lr,
            "timestamp": self.timestamp,
            "description": "暴力评分模型，输入文本输出0-1之间的暴力分数"
        }
        
        with open(os.path.join(path, "training_info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description="训练暴力评分模型")
    parser.add_argument("--dataset", required=True, help="标注数据CSV文件")
    parser.add_argument("--model", default="BAAI/bge-small-zh-v1.5", help="基础模型名称")
    parser.add_argument("--output", required=True, help="模型输出目录")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    parser.add_argument("--lr", type=float, default=2e-5, help="学习率")
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = ViolenceTrainer(
        model_name=args.model,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr
    )
    
    # 加载数据
    train_dataset, val_dataset = trainer.load_data(args.dataset)
    
    # 训练模型
    model_path = trainer.train(train_dataset, val_dataset)
    
    print(f"训练完成! 模型已保存至: {model_path}")

if __name__ == "__main__":
    main()