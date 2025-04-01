# embedding_training/evaluate_model.py
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_recall_fscore_support
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def evaluate_model(model_path, test_data, output_dir):
    """评估微调后的模型性能"""
    # 加载模型
    print(f"加载模型: {model_path}")
    model = SentenceTransformer(model_path)
    
    # 读取测试数据
    print(f"读取测试数据: {test_data}")
    df = pd.read_csv(test_data)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成嵌入向量
    print("生成评论嵌入向量...")
    embeddings = []
    comments = df['comment'].tolist()
    
    # 批量处理以提高效率
    batch_size = 32
    for i in tqdm(range(0, len(comments), batch_size)):
        batch = comments[i:i+batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.extend(batch_embeddings)
    
    embeddings = np.array(embeddings)
    
    # 创建参考点（0分和1分的锚点）
    zero_anchor = model.encode(["这是一个完全正常的评论，没有任何暴力内容。"])
    one_anchor = model.encode(["这是极度暴力的辱骂和威胁内容，应该立即删除。"])
    
    # 计算与锚点的相似度
    zero_similarities = cosine_similarity(embeddings, zero_anchor).flatten()
    one_similarities = cosine_similarity(embeddings, one_anchor).flatten()
    
    # 根据相似度计算预测分数
    # 将与1分锚点的相似度归一化为预测分数
    predicted_scores = one_similarities - zero_similarities
    predicted_scores = (predicted_scores - min(predicted_scores)) / (max(predicted_scores) - min(predicted_scores))
    
    # 计算评估指标
    true_scores = df['violence_score'].values
    mse = mean_squared_error(true_scores, predicted_scores)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_scores, predicted_scores)
    corr = np.corrcoef(true_scores, predicted_scores)[0, 1]
    
    # 将分数分为高、中、低风险类别评估
    def score_to_category(score):
        if score < 0.3:
            return 0  # 低风险
        elif score < 0.7:
            return 1  # 中风险
        else:
            return 2  # 高风险
    
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
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(true_categories, pred_categories)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['低风险', '中风险', '高风险'],
                yticklabels=['低风险', '中风险', '高风险'])
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
    parser = argparse.ArgumentParser(description='评估微调的嵌入模型')
    parser.add_argument('--model', required=True, help='模型路径')
    parser.add_argument('--test', required=True, help='测试数据文件')
    parser.add_argument('--output', required=True, help='评估结果输出目录')
    
    args = parser.parse_args()
    
    evaluate_model(args.model, args.test, args.output)