import os
import requests
import json
import pandas as pd
import time
import argparse
from tqdm import tqdm
import random
import numpy as np
import sys

class ViolenceLabelGenerator:
    """使用本地LLM生成网络暴力标注数据"""
    
    def __init__(self, model_url="http://localhost:11434/api/generate", 
                 model_name="gemma3", batch_size=10, checkpoint_interval=50,
                 uncertainty_threshold=0.2):
        self.model_url = model_url
        self.model_name = model_name
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval
        self.uncertainty_threshold = uncertainty_threshold
        self.headers = {"Content-Type": "application/json"}
        
    def generate_prompt(self, text):
        """生成用于判断网络暴力的提示词"""
        return f"""请判断以下评论是否包含网络暴力内容。网络暴力包括侮辱、诽谤、歧视、威胁、恶意煽动等有害内容。
        
评论内容: "{text}"

请分析这条评论，并以JSON格式提供以下信息:
1. violence_score：暴力程度评分(0到1的浮点数，0表示无暴力，1表示极度暴力)
2. violence_type：暴力类型("harassment"/"威胁"/"歧视"/"煽动"/"无")
3. reason：简要解释评分的原因
4. confidence：您对这个判断的信心(0到1的浮点数，0表示完全不确定，1表示非常确定)

仅返回JSON格式数据，不要有多余文本。格式如下：
{{"violence_score": 0.7, "violence_type": "harassment", "reason": "包含侮辱性语言", "confidence": 0.8}}"""

    def process_comment(self, comment, retries=3):
        """使用LLM处理单条评论，返回暴力评分和类型"""
        prompt = self.generate_prompt(comment)
        
        for attempt in range(retries):
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.1  # 低温度以获得更一致的结果
                }
                
                response = requests.post(self.model_url, 
                                        headers=self.headers, 
                                        data=json.dumps(payload))
                
                if response.status_code == 200:
                    result = response.json()
                    generated_text = result.get("response", "")
                    
                    # 提取JSON部分
                    try:
                        # 尝试直接解析返回的文本
                        analysis = json.loads(generated_text)
                        
                        # 添加需要审核标志
                        confidence = analysis.get("confidence", 0.7)  # 默认置信度
                        needs_review = confidence < self.uncertainty_threshold
                        
                        return {
                            "violence_score": float(analysis.get("violence_score", 0)),
                            "violence_type": analysis.get("violence_type", "无"),
                            "reason": analysis.get("reason", ""),
                            "confidence": float(confidence),
                            "needs_review": needs_review
                        }
                    except json.JSONDecodeError:
                        # 如果无法解析，尝试查找JSON字符串
                        import re
                        json_match = re.search(r'(\{.*\})', generated_text, re.DOTALL)
                        if json_match:
                            try:
                                analysis = json.loads(json_match.group(1))
                                
                                # 添加需要审核标志
                                confidence = analysis.get("confidence", 0.7)  # 默认置信度
                                needs_review = confidence < self.uncertainty_threshold
                                
                                return {
                                    "violence_score": float(analysis.get("violence_score", 0)),
                                    "violence_type": analysis.get("violence_type", "无"),
                                    "reason": analysis.get("reason", ""),
                                    "confidence": float(confidence),
                                    "needs_review": needs_review
                                }
                            except Exception as inner_e:
                                print(f"JSON提取错误: {inner_e}")
                        
                        # 如果是最后一次尝试，返回错误信息
                        if attempt == retries - 1:
                            return {
                                "violence_score": 0,
                                "violence_type": "解析错误",
                                "reason": "LLM返回格式无法解析",
                                "confidence": 0.0,
                                "needs_review": True,
                                "raw_response": generated_text
                            }
                else:
                    print(f"API错误: {response.status_code}")
                    time.sleep(2)  # 遇到错误时等待时间
                    
                    # 如果是最后一次尝试，返回错误信息
                    if attempt == retries - 1:
                        return {
                            "violence_score": 0,
                            "violence_type": "API错误",
                            "reason": f"状态码: {response.status_code}",
                            "confidence": 0.0,
                            "needs_review": True
                        }
                    
            except Exception as e:
                print(f"处理评论时出错: {e}")
                time.sleep(2)  # 遇到错误时等待时间
                
                # 如果是最后一次尝试，返回错误信息
                if attempt == retries - 1:
                    return {
                        "violence_score": 0,
                        "violence_type": "处理错误",
                        "reason": str(e),
                        "confidence": 0.0,
                        "needs_review": True
                    }
    
    def generate_dataset(self, input_file, output_file, start_idx=0, max_items=None):
        """从评论文件生成标注数据集"""
        # 读取评论
        df = pd.read_csv(input_file)
        print(f"读取到 {len(df)} 条评论")
        
        # 检查是否继续之前的工作
        if os.path.exists(output_file) and start_idx > 0:
            try:
                existing_df = pd.read_csv(output_file)
                print(f"找到已处理的 {len(existing_df)} 条评论")
            except Exception as e:
                print(f"无法读取已有文件: {e}")
                existing_df = pd.DataFrame(columns=['comment', 'violence_score', 'violence_type', 'reason', 'confidence', 'needs_review'])
        else:
            existing_df = pd.DataFrame(columns=['comment', 'violence_score', 'violence_type', 'reason', 'confidence', 'needs_review'])
        
        # 设置处理范围
        end_idx = len(df) if max_items is None else min(start_idx + max_items, len(df))
        print(f"将处理 {start_idx} 到 {end_idx} 范围内的评论")
        
        # 准备进度条
        progress_bar = tqdm(total=end_idx-start_idx, file=sys.stdout)
        
        # 处理每条评论
        results = []
        for idx in range(start_idx, end_idx):
            comment = df.iloc[idx]['comment']
            
            # 检查是否已处理过
            existing = existing_df[existing_df['comment'] == comment]
            if not existing.empty:
                results.append({
                    'comment': comment,
                    'violence_score': existing.iloc[0]['violence_score'],
                    'violence_type': existing.iloc[0]['violence_type'],
                    'reason': existing.iloc[0]['reason'],
                    'confidence': existing.iloc[0].get('confidence', 0.7),
                    'needs_review': existing.iloc[0].get('needs_review', True)
                })
                progress_bar.update(1)
                continue
            
            # 调用LLM处理
            result = self.process_comment(comment)
            
            # 添加评论文本
            result['comment'] = comment
            
            # 添加原始字段
            for col in df.columns:
                if col != 'comment' and col in df.iloc[idx]:
                    result[col] = df.iloc[idx][col]
            
            results.append(result)
            
            # 更新进度条
            progress_bar.update(1)
            
            # 保存检查点
            if len(results) % self.checkpoint_interval == 0:
                checkpoint_df = pd.DataFrame(results)
                # 合并已有结果和新结果
                combined_df = pd.concat([existing_df, checkpoint_df], ignore_index=True)
                # 去重
                combined_df = combined_df.drop_duplicates(subset=['comment'], keep='last')
                combined_df.to_csv(output_file, index=False, encoding='utf-8')
                print(f"\n保存检查点：已处理 {len(results)} 条评论")
            
            # 随机等待，避免过度请求
            time.sleep(random.uniform(0.1, 0.3))
        
        progress_bar.close()
        
        # 合并已有结果和新结果
        final_df = pd.DataFrame(results)
        combined_df = pd.concat([existing_df, final_df], ignore_index=True)
        # 去重
        combined_df = combined_df.drop_duplicates(subset=['comment'], keep='last')
        combined_df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"处理完成！共 {len(combined_df)} 条评论")
        
        # 显示统计信息
        print("\n数据分布:")
        # 根据暴力分数分布
        score_bins = [0, 0.3, 0.7, 1.0]
        labels = ['低风险', '中风险', '高风险']
        combined_df['risk_level'] = pd.cut(combined_df['violence_score'], 
                                           bins=score_bins, 
                                           labels=labels)
        print(combined_df['risk_level'].value_counts())
        
        # 暴力类型分布
        print("\n暴力类型分布:")
        print(combined_df['violence_type'].value_counts())
        
        # 需要审核的数量
        if 'needs_review' in combined_df.columns:
            print("\n需要审核的评论:")
            print(combined_df['needs_review'].value_counts())
        
        return combined_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='生成网络暴力标注数据集')
    parser.add_argument('--input', required=True, help='输入评论CSV文件')
    parser.add_argument('--output', required=True, help='输出标注数据集CSV文件')
    parser.add_argument('--model', default='gemma3', help='使用的模型名称')
    parser.add_argument('--start', type=int, default=0, help='起始索引（用于继续处理）')
    parser.add_argument('--max', type=int, help='最大处理条数（可选）')
    parser.add_argument('--batch', type=int, default=50, help='检查点批次大小')
    parser.add_argument('--uncertainty', type=float, default=0.3, help='需要审核的不确定性阈值')
    parser.add_argument('--url', default='http://localhost:11434/api/generate', 
                       help='Ollama API URL')
    
    args = parser.parse_args()
    
    generator = ViolenceLabelGenerator(
        model_url=args.url,
        model_name=args.model,
        batch_size=1,  # 单条处理以提高稳定性
        checkpoint_interval=args.batch,
        uncertainty_threshold=args.uncertainty
    )
    
    generator.generate_dataset(args.input, args.output, args.start, args.max)