#!/usr/bin/env python3
import os
import subprocess
import argparse
import sys

def run_command(cmd):
    """运行命令"""
    print(f"运行: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0
def main():
    parser = argparse.ArgumentParser(description='完整实验流程')
    parser.add_argument('--input', type=str, required=True, help='输入图像目录')
    parser.add_argument('--output', type=str, default='./results', help='输出目录')
    parser.add_argument('--type', type=str, default='images', help='输入类型')
    parser.add_argument('--method', type=str, choices=['nerf', 'gaussian', 'both'], 
                       default='both', help='训练方法')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NeRF vs 3D Gaussian Splatting 对比实验")
    print("=" * 60)
    
    # 检查输入
    if not os.path.exists(args.input):
        print(f"错误: 输入路径不存在: {args.input}")
        return 1
    
    os.makedirs(args.output, exist_ok=True)
    data_dir = os.path.join(args.output, 'data')
    
    # 步骤1: 数据预处理
    print("\n步骤 1: 数据预处理")
    print("=" * 30)
    
    prep_cmd = [
        'python', 'scripts/prepare_data.py',
        '--input', args.input,
        '--output', data_dir,
        '--type', args.type,
        '--max_size', '800'
    ]
    
    if not run_command(prep_cmd):
        print("数据预处理失败")
        return 1
    
    print("数据预处理完成")
    
    # 步骤2: 训练模型
    print("\n步骤 2: 训练模型")
    print("=" * 30)
    
    train_cmd = [
        'python', 'main.py',
        '--task', args.method,
        '--data_dir', data_dir,
        '--output_dir', args.output
    ]
    
    if not run_command(train_cmd):
        print("模型训练失败")
        return 1
    
    print("训练完成")
    
    # 显示结果
    print(f"结果保存在: {args.output}")
    
    return 0

if __name__ == "__main__":
    exit(main())
