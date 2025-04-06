# bert_classification/evaluate_classifier.py

import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def evaluate_model(model_path, test_data, output_dir):
    """评估MacBERT分类器模型性能"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型和分词器
    print(f"加载模型: {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 将模型移至设备
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()  # 设置为评估模式
    
    # 读取测试数据
    print(f"读取测试数据: {test_data}")
    df = pd.read_csv(test_data)
    
    # 生成预测
    print("生成预测...")
    predicted_scores = []
    comments = df['comment'].tolist()
    true_scores = df['violence_score'].values
    
    # 批处理预测
    batch_size = 32
    for i in tqdm(range(0, len(comments), batch_size)):
        batch = comments[i:i+batch_size]
        
        # 使用tokenizer处理文本
        inputs = tokenizer(batch, truncation=True, padding=True, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 进行预测
        with torch.no_grad():
            outputs = model(**inputs)
            
        # 获取预测分数
        batch_scores = outputs.logits.squeeze(-1).cpu().numpy()
        predicted_scores.extend(batch_scores)
    
    predicted_scores = np.array(predicted_scores)
    
    # 确保预测分数在0-1范围内
    predicted_scores = np.clip(predicted_scores, 0, 1)
    
    # 计算评估指标
    mse = mean_squared_error(true_scores, predicted_scores)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_scores, predicted_scores)
    corr = np.corrcoef(true_scores, predicted_scores)[0, 1]
    
    # 将分数分为5个风险类别评估
    def score_to_category(score):
        if score < 0.15:
            return 0  # 极低风险
        elif score < 0.30:
            return 1  # 低风险
        elif score < 0.5:
            return 2  # 中风险
        elif score < 0.85:
            return 3  # 高风险
        else:
            return 4  # 极高风险
    
    true_categories = [score_to_category(s) for s in true_scores]
    pred_categories = [score_to_category(s) for s in predicted_scores]
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_categories, pred_categories, average='weighted'
    )
    
    # 记录结果
    results = {
        "Model": model_path,
        "Test Data": test_data,
        "Test Examples": len(df),
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "Correlation": corr,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }
    
    # 保存评估结果
    results_file = os.path.join(output_dir, "evaluation_results.txt")
    with open(results_file, "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    print("评估结果已保存至:", results_file)
    
    # 绘制预测分数与真实分数的散点图
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.family'] = 'SimHei'  # 使用中文字体
    plt.scatter(true_scores, predicted_scores, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('真实暴力分数')
    plt.ylabel('预测暴力分数')
    plt.title(f'暴力分数预测 (相关系数: {corr:.3f})')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加统计信息
    stats_text = f"MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nCorr: {corr:.4f}"
    plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # 保存图表
    chart_file = os.path.join(output_dir, "prediction_chart.png")
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("评估图表已保存至:", chart_file)
    
    # 生成混淆矩阵
    category_labels = ['极低风险', '低风险', '中风险', '高风险', '极高风险']
    cm = confusion_matrix(true_categories, pred_categories)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=category_labels,
                yticklabels=category_labels)
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.title('混淆矩阵')
    
    # 保存混淆矩阵
    cm_file = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("混淆矩阵已保存至:", cm_file)
    
    # 返回评估结果
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='评估MacBERT文本分类器')
    parser.add_argument('--model', required=True, help='模型路径')
    parser.add_argument('--test', required=True, help='测试数据文件')
    parser.add_argument('--output', required=True, help='评估结果输出目录')
    
    args = parser.parse_args()
    
    evaluate_model(args.model, args.test, args.output)