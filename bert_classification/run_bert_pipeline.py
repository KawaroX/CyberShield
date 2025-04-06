# embedding_training/run_pipeline.py
import argparse
import os
import subprocess
import time
from datetime import datetime
import pandas as pd
from select_files import select_files


def run_command(cmd, desc):
    """运行命令并打印输出"""
    print(f"\n{'='*50}")
    print(f"开始: {desc}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*50}\n")
    
    start_time = time.time()
    
    # 使用subprocess.Popen直接将输出传递到终端，确保进度条可见
    process = subprocess.Popen(cmd)
    process.wait()
    return_code = process.returncode
    
    end_time = time.time()
    
    print(f"\n{'='*50}")
    print(f"完成: {desc}")
    print(f"耗时: {end_time - start_time:.2f} 秒")
    print(f"退出码: {return_code}")
    print(f"{'='*50}\n")
    
    return return_code == 0

def main():
    parser = argparse.ArgumentParser(description='运行嵌入模型微调流程')
    parser.add_argument('--input', nargs='+', help='输入CSV文件列表')
    parser.add_argument('--dir', default='/Users/kawarox/dev/get_cmts', help='CSV文件目录')
    parser.add_argument('--output_dir', default='/Volumes/base/bert_output', help='输出目录')
    parser.add_argument('--model', default='hfl/chinese-macbert-base', help='基础模型')
    parser.add_argument('--sample_size', type=int, help='抽样大小（可选）')
    parser.add_argument('--ollama_model', default='gemma3', help='Ollama模型')
    parser.add_argument('--skip_processing', action='store_true', help='跳过数据预处理')
    parser.add_argument('--skip_labeling', action='store_true', help='跳过标注生成')
    parser.add_argument('--skip_review', action='store_true', help='跳过数据审核')
    parser.add_argument('--skip_training', action='store_true', help='跳过模型训练')
    parser.add_argument('--review_all', action='store_true', help='审核所有记录而非仅需要审核的')
    parser.add_argument('--epoch', type=int, default=5, help='训练轮数')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    parser.add_argument('--uncertainty', type=float, default=0.3, help='需要审核的不确定性阈值')
    parser.add_argument('--skip_balancing', action='store_true', help='跳过数据平衡')
    parser.add_argument('--high_ratio', type=float, default=1.5, help='高风险样本比例')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 如果没有指定输入文件，使用文件选择器
    if not args.input:
        input_files = select_files(args.dir)
        if not input_files:
            print("未选择任何文件，退出程序")
            return
    else:
        input_files = args.input
    
    # 设置文件路径
    processed_data = os.path.join(args.output_dir, f"processed_comments_{timestamp}.csv")
    labeled_data = os.path.join(args.output_dir, f"labeled_data_{timestamp}.csv")
    reviewed_data = os.path.join(args.output_dir, f"reviewed_data_{timestamp}.csv")
    model_output = os.path.join(args.output_dir, "models")
    evaluation_output = os.path.join(args.output_dir, f"evaluation_{timestamp}")
    
    os.makedirs(model_output, exist_ok=True)
    os.makedirs(evaluation_output, exist_ok=True)
    
    # 保存配置信息
    with open(os.path.join(args.output_dir, f"config_{timestamp}.txt"), "w") as f:
        f.write(f"输入文件: {', '.join(input_files)}\n")
        for arg, value in vars(args).items():
            if arg != 'input':
                f.write(f"{arg}: {value}\n")
    
    # 步骤1: 数据预处理
    if not args.skip_processing:
        sample_arg = []
        if args.sample_size:
            sample_arg = ["--sample", str(args.sample_size)]
            
        cmd = ["python", "process_comments.py"]
        for file in input_files:
            cmd.extend(["--input", file])
        cmd.extend(["--output", processed_data] + sample_arg)
        
        success = run_command(cmd, "数据预处理")
        if not success:
            print("数据预处理失败，退出。")
            return
    else:
        # 如果跳过处理，使用第一个输入文件作为处理后数据
        processed_data = input_files[0]
        print(f"跳过数据预处理，使用输入文件 {processed_data} 作为处理后数据。")
    
    # 步骤2: 生成标注数据
    if not args.skip_labeling:
        success = run_command(
            ["python", "generate_labels.py", 
             "--input", processed_data, 
             "--output", labeled_data,
             "--model", args.ollama_model,
             "--uncertainty", str(args.uncertainty)],
            "生成标注数据"
        )
        if not success:
            print("生成标注数据失败，退出。")
            return
    else:
        labeled_data = processed_data
        print("跳过标注生成，使用处理后数据作为标注数据。")
    
    # 步骤3: 数据审核
    if not args.skip_review:
        review_args = ["--dataset", labeled_data, "--output", reviewed_data]
        if args.review_all:
            review_args.append("--review_all")
            
        success = run_command(
            ["python", "review_dataset.py"] + review_args,
            "数据审核"
        )
        if not success:
            print("数据审核失败，退出。")
            return
    else:
        reviewed_data = labeled_data
        print("跳过数据审核，使用标注数据作为审核后数据。")

    # 步骤3.5: 数据平衡
    if not args.skip_balancing:
        print("\n执行数据平衡...")
        balanced_data = os.path.join(args.output_dir, f"balanced_data_{timestamp}.csv")
        
        # 调用平衡函数
        success = run_command(
            ["python", "balance_dataset.py", 
            "--input", reviewed_data, 
            "--output", balanced_data,
            "--high_ratio", str(args.high_ratio)],
            "数据平衡"
        )
        
        if not success:
            print("数据平衡失败，使用原始数据继续。")
            balanced_data = reviewed_data
    else:
        balanced_data = reviewed_data
        print("跳过数据平衡，使用审核后数据。")
    
    # 步骤4: 模型训练
    if not args.skip_training:
        success = run_command(
            ["python", "bert_classification/train_classifier.py", 
             "--dataset", reviewed_data, 
             "--model", args.model, 
             "--output", model_output,
             "--epochs", str(args.epoch),
             "--batch", str(args.batch),
             "--lr", str(args.lr),
             "--balance"] if args.balance else [],
            "模型训练"
        )
        if not success:
            print("模型训练失败，退出。")
            return
        
        # 找到最新训练的模型
        model_dirs = [os.path.join(model_output, d) for d in os.listdir(model_output) 
                      if d.startswith("macbert-violence-")]
        
        if model_dirs:
            latest_model = max(model_dirs, key=os.path.getmtime)
            
            # 步骤5: 模型评估
            success = run_command(
                ["python", "evaluate_model.py", 
                 "--model", latest_model, 
                 "--test", reviewed_data, 
                 "--output", evaluation_output],
                "模型评估"
            )
            if not success:
                print("模型评估失败，但训练已完成。")
        else:
            print("找不到训练好的模型，跳过评估。")
    
    print("\n流程完成！")
    print(f"处理后数据: {processed_data}")
    print(f"标注数据: {labeled_data}")
    print(f"审核后数据: {reviewed_data}")
    print(f"模型输出目录: {model_output}")
    print(f"评估结果: {evaluation_output}")

if __name__ == "__main__":
    main()