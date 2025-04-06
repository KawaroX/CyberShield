import pandas as pd
import random
import argparse
import os
import sys

def review_dataset(dataset_file, output_file, review_all=False, sample_size=50, resume=False):
    """审核生成的标注数据，可以选择只审核不确定的评论"""
    # 读取数据
    df = pd.read_csv(dataset_file)
    print(f"数据集共有 {len(df)} 条记录")
    
    # 确定审核范围
    if not review_all and 'needs_review' in df.columns:
        to_review_df = df[df['needs_review'] == True].copy()
        print(f"需要审核的记录: {len(to_review_df)} 条")
    else:
        to_review_df = df.copy()
        print(f"将审核所有记录: {len(to_review_df)} 条")
    
    # 如果继续上次的工作
    if resume and os.path.exists(output_file):
        reviewed_df = pd.read_csv(output_file)
        print(f"已审核 {len(reviewed_df)} 条记录")
        # 找出已审核的评论
        reviewed_comments = set(reviewed_df['comment'].tolist())
    else:
        if not os.path.exists(output_file):
            # 创建新的输出文件
            columns = df.columns.tolist()
            if 'needs_review' not in columns:
                columns.append('needs_review')
            empty_df = pd.DataFrame(columns=columns)
            empty_df.to_csv(output_file, index=False)
            print(f"创建新的输出文件: {output_file}")
        reviewed_df = pd.DataFrame(columns=df.columns)
        reviewed_comments = set()
    
    # 筛选未审核的记录
    unreviewed_df = to_review_df[~to_review_df['comment'].isin(reviewed_comments)]
    print(f"还有 {len(unreviewed_df)} 条记录未审核")
    
    if len(unreviewed_df) == 0:
        print("所有需要审核的记录已审核完毕！")
        # 合并已审核和不需要审核的记录
        if not review_all and 'needs_review' in df.columns:
            # 读取最新的输出文件
            if os.path.exists(output_file):
                reviewed_df = pd.read_csv(output_file)
            not_to_review_df = df[df['needs_review'] == False]
            final_df = pd.concat([reviewed_df, not_to_review_df], ignore_index=True)
            # 去重
            final_df = final_df.drop_duplicates(subset=['comment'], keep='last')
            final_df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"已合并所有记录并保存至 {output_file}")
        return
    
    # 抽样未审核的记录
    if sample_size and sample_size < len(unreviewed_df):
        sample_indices = random.sample(range(len(unreviewed_df)), sample_size)
        samples = unreviewed_df.iloc[sample_indices].copy()
        print(f"从 {len(unreviewed_df)} 条未审核记录中抽取 {sample_size} 条进行审核")
    else:
        samples = unreviewed_df.copy()
        print(f"将审核所有 {len(unreviewed_df)} 条未审核记录")
    
    # 逐条审核
    corrected = []
    skipped = []
    approved = []
    
    for i, row in samples.iterrows():
        print("\n" + "="*50)
        print(f"评论: {row['comment']}")
        print(f"暴力分数: {row['violence_score']}")
        print(f"暴力类型: {row['violence_type']}")
        print(f"理由: {row['reason']}")
        if 'confidence' in row:
            print(f"置信度: {row['confidence']}")
        if 'raw_content' in row and row['raw_content'] != row['comment']:
            print(f"原始内容: {row['raw_content']}")
        
        # 显示已审核进度
        total_approved = len(approved)
        total_corrected = len(corrected)
        total_skipped = len(skipped)
        total_processed = total_approved + total_corrected + total_skipped
        print(f"\n进度: {total_processed}/{len(samples)} " 
              f"(通过: {total_approved}, 修正: {total_corrected}, 跳过: {total_skipped})")
        
        while True:
            choice = input("\n这个标注正确吗? (y/n/s - 跳过): ").lower()
            if choice in ['y', 'n', 's']:
                break
        
        if choice == 'y':
            # 保留原标注，但标记为已审核
            row_dict = row.to_dict()
            row_dict['needs_review'] = False
            reviewed_df = pd.concat([reviewed_df, pd.DataFrame([row_dict])], ignore_index=True)
            approved.append(i)
        elif choice == 'n':
            # 修正标注
            try:
                current_score = row['violence_score']
                new_score = float(input(f"正确的暴力分数 (0-1) [当前: {current_score}]: ") or current_score)
                
                current_type = row['violence_type']
                violence_types = ['harassment', '威胁', '歧视', '煽动', '无']
                print("暴力类型选项:")
                for j, vt in enumerate(violence_types):
                    marker = "*" if vt == current_type else " "
                    print(f"{j+1}: {marker}{vt}")
                    
                type_input = input(f"选择暴力类型 (1-5) [当前: {current_type}]: ")
                new_type = violence_types[int(type_input)-1] if type_input else current_type
                
                current_reason = row['reason']
                new_reason = input(f"修改理由 [当前: {current_reason}]: ") or current_reason
                if "[人工修正]" not in new_reason:
                    new_reason = new_reason + " [人工修正]"
                    
                # 更新数据
                row_dict = row.to_dict()
                row_dict['violence_score'] = new_score
                row_dict['violence_type'] = new_type
                row_dict['reason'] = new_reason
                row_dict['needs_review'] = False
                # 如果有置信度字段，修改为1.0
                if 'confidence' in row_dict:
                    row_dict['confidence'] = 1.0  # 人工审核后的置信度为1
                
                reviewed_df = pd.concat([reviewed_df, pd.DataFrame([row_dict])], ignore_index=True)
                corrected.append(i)
                
                print("\n已更新标注:")
                print(f"暴力分数: {new_score}")
                print(f"暴力类型: {new_type}")
                print(f"理由: {new_reason}")
                
            except Exception as e:
                print(f"修改时出错: {e}")
                skipped.append(i)
        else:
            # 跳过
            skipped.append(i)
        
        # 每审核10条保存一次
        if (len(approved) + len(corrected)) % 10 == 0 and (len(approved) + len(corrected)) > 0:
            # 读取最新的输出文件
            if os.path.exists(output_file):
                existing_df = pd.read_csv(output_file)
                # 合并已有结果和新结果
                combined_df = pd.concat([existing_df, reviewed_df], ignore_index=True)
                # 去重
                combined_df = combined_df.drop_duplicates(subset=['comment'], keep='last')
                combined_df.to_csv(output_file, index=False, encoding='utf-8')
            else:
                reviewed_df.to_csv(output_file, index=False, encoding='utf-8')
                
            print(f"已保存当前进度，审核了 {len(approved) + len(corrected)} 条记录")
    
    # 读取最新的输出文件
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        # 合并已有结果和新结果
        combined_df = pd.concat([existing_df, reviewed_df], ignore_index=True)
        # 去重
        combined_df = combined_df.drop_duplicates(subset=['comment'], keep='last')
    else:
        combined_df = reviewed_df.copy()
    
    # 如果选择不审核所有记录，添加不需要审核的记录
    if not review_all and 'needs_review' in df.columns:
        not_to_review_df = df[~df['comment'].isin(combined_df['comment'].tolist()) & (df['needs_review'] == False)]
        combined_df = pd.concat([combined_df, not_to_review_df], ignore_index=True)
    
    # 保存最终结果
    combined_df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"\n审核完成，共审核 {len(samples)} 条评论")
    print(f"- 通过: {len(approved)} 条")
    print(f"- 修正: {len(corrected)} 条")
    print(f"- 跳过: {len(skipped)} 条")
    print(f"审核结果已保存至 {output_file}")
    
    # 显示审核后的数据分布
    if os.path.exists(output_file):
        final_df = pd.read_csv(output_file)
        print("\n审核后的数据分布:")
        # 根据暴力分数分布
        score_bins = [0, 0.3, 0.7, 1.0]
        labels = ['低风险', '中风险', '高风险']
        
        try:
            final_df['risk_level'] = pd.cut(final_df['violence_score'], 
                                             bins=score_bins, 
                                             labels=labels)
            print(final_df['risk_level'].value_counts())
            
            # 暴力类型分布
            print("\n暴力类型分布:")
            print(final_df['violence_type'].value_counts())
        except Exception as e:
            print(f"计算数据分布时出错: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='审核生成的标注数据')
    parser.add_argument('--dataset', required=True, help='标注数据文件')
    parser.add_argument('--output', required=True, help='审核后的输出文件')
    parser.add_argument('--review_all', action='store_true', help='审核所有记录而非仅需要审核的')
    parser.add_argument('--sample', type=int, default=50, help='抽样大小')
    parser.add_argument('--resume', action='store_true', help='继续之前的审核')
    
    args = parser.parse_args()
    
    review_dataset(args.dataset, args.output, args.review_all, args.sample, args.resume)