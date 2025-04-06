# embedding_training/process_comments.py
import json
import os
import random
import pandas as pd
import argparse
from tqdm import tqdm
import sys
import emoji
import re

def remove_emojis(text):
    """移除表情符号"""
    if not text:
        return text
    return emoji.replace_emoji(text, replace='')

def clean_text(text):
    """清理文本，移除表情符号、@用户和多余空格"""
    if not isinstance(text, str):
        return ""
    
    # 移除表情符号
    text = remove_emojis(text)
    
    # 移除@用户
    text = re.sub(r'@\S+', '', text)
    
    # 移除方括号内容，如[傻眼]
    text = re.sub(r'\[.*?\]', '', text)
    
    # 移除多余空格和换行
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def process_csv_files(input_files, output_file, sample_size=None, resume=False):
    """处理CSV评论文件，提取非空评论并可选择抽样"""
    
    # 检查是否继续之前的处理
    if resume and os.path.exists(output_file):
        print(f"继续处理，从已有文件 {output_file} 开始")
        try:
            processed_df = pd.read_csv(output_file)
            # 创建已处理评论的集合，用于去重
            processed_set = set(processed_df['comment'].tolist())
            print(f"已找到 {len(processed_set)} 条已处理评论")
        except Exception as e:
            print(f"无法加载已处理文件: {e}")
            processed_set = set()
    else:
        processed_set = set()
    
    # 处理所有CSV文件
    all_valid_comments = []
    
    for input_file in input_files:
        file_name = os.path.basename(input_file)
        print(f"\n处理文件: {file_name}")
        
        try:
            # 读取CSV文件
            df = pd.read_csv(input_file)
            total_rows = len(df)
            print(f"文件包含 {total_rows} 行")
            
            # 检查列名
            required_cols = ['content']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"文件缺少必要的列: {col}")
            
            # 确定使用哪个列作为清理后的内容
            content_col = 'content'
            cleaned_content_col = None
            
            if 'cleaned_content' in df.columns:
                cleaned_content_col = 'cleaned_content'
            
            # 处理每一行
            valid_comments = []
            progress_bar = tqdm(total=total_rows, file=sys.stdout)
            
            for i, row in df.iterrows():
                progress_bar.update(1)
                
                # 获取原始内容
                raw_content = row[content_col]
                
                # 获取或生成清理后的内容
                if cleaned_content_col and pd.notna(row[cleaned_content_col]) and row[cleaned_content_col]:
                    cleaned_content = str(row[cleaned_content_col])
                else:
                    cleaned_content = clean_text(raw_content)
                
                # 如果清理后的内容非空且未处理过
                if cleaned_content and cleaned_content not in processed_set:
                    # 创建评论字典
                    comment_dict = {
                        'comment': cleaned_content,
                        'raw_content': raw_content,
                        'source_file': file_name
                    }
                    
                    # 添加其他有用的字段
                    for col in df.columns:
                        if col not in ['content', 'cleaned_content'] and pd.notna(row[col]):
                            comment_dict[col] = row[col]
                    
                    valid_comments.append(comment_dict)
                    processed_set.add(cleaned_content)
                
                # 定期保存中间结果
                if len(valid_comments) % 5000 == 0 and len(valid_comments) > 0:
                    print(f"\n处理中... 当前已找到 {len(valid_comments)} 条有效评论")
            
            progress_bar.close()
            all_valid_comments.extend(valid_comments)
            print(f"从 {file_name} 中提取了 {len(valid_comments)} 条有效评论")
            
            # 定期保存中间结果
            if len(all_valid_comments) % 10000 == 0:
                save_intermediate_results(all_valid_comments, output_file)
                
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {e}")
    
    print(f"\n所有文件处理完成，共找到 {len(all_valid_comments)} 条有效评论")
    
    # 如果需要抽样
    if sample_size and sample_size < len(all_valid_comments):
        print(f"从 {len(all_valid_comments)} 条评论中抽取 {sample_size} 条样本")
        all_valid_comments = random.sample(all_valid_comments, sample_size)
    
    # 保存结果
    save_final_results(all_valid_comments, output_file)
    
    return all_valid_comments

def save_intermediate_results(comments, output_file):
    """保存中间结果"""
    temp_file = f"{output_file}.temp"
    df = pd.DataFrame(comments)
    df.to_csv(temp_file, index=False, encoding='utf-8')
    print(f"已保存中间结果至 {temp_file}")

def save_final_results(comments, output_file):
    """保存最终结果"""
    df = pd.DataFrame(comments)
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"处理完成，共 {len(comments)} 条评论，保存至 {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='处理评论数据')
    parser.add_argument('--input', nargs='+', help='输入CSV文件列表')
    parser.add_argument('--dir', help='CSV文件目录')
    parser.add_argument('--output', required=True, help='输出CSV文件')
    parser.add_argument('--sample', type=int, help='抽样大小（可选）')
    parser.add_argument('--resume', action='store_true', help='继续之前的处理')
    
    args = parser.parse_args()
    
    # 如果没有指定输入文件，使用文件选择器
    if not args.input:
        from select_files import select_files
        input_files = select_files(args.dir)
    else:
        input_files = args.input
    
    if input_files:
        process_csv_files(input_files, args.output, args.sample, args.resume)
    else:
        print("未选择任何文件，退出程序")