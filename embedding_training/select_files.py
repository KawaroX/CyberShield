import os
import argparse
import pandas as pd

def list_csv_files(directory):
    """列出目录中所有的CSV文件"""
    csv_files = []
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            csv_files.append(os.path.join(directory, file))
    return csv_files

def display_files(files):
    """显示文件列表供用户选择"""
    print("\n可用的CSV文件:")
    for i, file in enumerate(files):
        # 尝试读取文件的行数
        try:
            df = pd.read_csv(file)
            rows = len(df)
            print(f"[{i}] {os.path.basename(file)} (包含 {rows} 条评论)")
        except Exception as e:
            print(f"[{i}] {os.path.basename(file)} (无法读取: {e})")
    
    return files

def select_files(directory=None, file=None):
    """
    允许用户选择要处理的文件
    
    参数:
    - directory: 包含CSV文件的目录
    - file: 直接指定的文件
    
    返回:
    - 选择的文件路径列表
    """
    if file:
        if os.path.exists(file):
            return [file]
        else:
            print(f"错误: 文件 {file} 不存在")
    
    if not directory:
        directory = input("请输入CSV文件的目录路径: ")
    
    if not os.path.exists(directory):
        print(f"错误: 目录 {directory} 不存在")
        return []
    
    csv_files = list_csv_files(directory)
    if not csv_files:
        print(f"在 {directory} 中没有找到CSV文件")
        return []
    
    files = display_files(csv_files)
    
    # 提示用户选择
    while True:
        choice = input("\n请选择要处理的文件 (输入序号，多个序号用逗号分隔，输入 'all' 选择所有文件): ")
        if choice.lower() == 'all':
            return files
        
        try:
            indices = [int(idx.strip()) for idx in choice.split(',')]
            selected_files = [files[idx] for idx in indices if 0 <= idx < len(files)]
            if selected_files:
                return selected_files
            else:
                print("无效的选择，请重试")
        except (ValueError, IndexError):
            print("无效的输入，请输入有效的序号")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='选择CSV文件')
    parser.add_argument('--dir', help='CSV文件的目录')
    parser.add_argument('--file', help='直接指定的CSV文件')
    
    args = parser.parse_args()
    
    selected = select_files(args.dir, args.file)
    if selected:
        print("\n选择的文件:")
        for file in selected:
            print(f"- {file}")