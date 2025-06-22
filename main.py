import os
import sys
import argparse
import yaml
import torch
import time

def main():
    parser = argparse.ArgumentParser(description='NeRF vs 3D Gaussian Splatting')
    parser.add_argument('--task', type=str, choices=['nerf', 'gaussian', 'both'], 
                       default='both', help='选择任务')
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录')
    parser.add_argument('--output_dir', type=str, default='./results', help='输出目录')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    results = {}
    
    if args.task in ['nerf', 'both']:
        print("=== 训练NeRF模型 ===")
        try:
            from models.tensorf import TensoRF
            from data_processing.dataset import NeRFDataset
            from utils.evaluation import evaluate_model
            from utils.rendering import render_video
            
            # 加载配置
            with open('configs/nerf_config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            # 加载数据
            train_dataset = NeRFDataset(args.data_dir, 'train')
            test_dataset = NeRFDataset(args.data_dir, 'test')
            
            print(f"训练集大小: {len(train_dataset)}")
            print(f"测试集大小: {len(test_dataset)}")
            
            # 创建模型
            aabb = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], device=device)
            model = TensoRF(
                aabb=aabb,
                gridSize=config['tensorf']['gridSize'],
                device=device,
                density_n_comp=config['tensorf']['density_n_comp'],
                app_n_comp=config['tensorf']['app_n_comp']
            ).to(device)
            
            print("模型创建成功，开始训练...")
            
            # 训练
            start_time = time.time()

            config['tensorf']['log_dir'] = os.path.join(args.output_dir, 'logs/nerf')
            os.makedirs(config['tensorf']['log_dir'], exist_ok=True)
            
            from scripts.train_nerf import train_tensorf
            trained_model = train_tensorf(model, train_dataset, config['tensorf'], test_dataset)

            training_time = time.time() - start_time
            
            # 评估
            print("开始评估...")
            eval_results = evaluate_model(trained_model, test_dataset, device)
            
            # 渲染视频
            nerf_output_dir = os.path.join(args.output_dir, 'nerf')
            os.makedirs(nerf_output_dir, exist_ok=True)
            
            print("渲染视频...")
            video_path = os.path.join(nerf_output_dir, 'nerf_video.mp4')
            render_video(trained_model, video_path, device)
            
            # 保存模型
            torch.save(trained_model.state_dict(), 
                      os.path.join(nerf_output_dir, 'nerf_model.pth'))
            
            results['NeRF'] = {
                **eval_results,
                'training_time': training_time
            }
            
            print(f"NeRF完成 - PSNR: {eval_results.get('PSNR', 0):.2f}, 训练时间: {training_time:.1f}s")
            
        except Exception as e:
            print(f"NeRF训练失败: {e}")
            import traceback
            traceback.print_exc()
            results['NeRF'] = {'PSNR': 0, 'SSIM': 0, 'training_time': 0, 'error': str(e)}
    
    if args.task in ['gaussian', 'both']:
        print("=== 训练Gaussian模型 ===")
        try:
            from scripts.train_gaussian import SimpleGaussianModel, train_gaussian
            from data_processing.dataset import NeRFDataset
            from utils.evaluation import evaluate_model
            from utils.rendering import render_video
            
            # 加载配置
            with open('configs/gaussian_config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            # 加载数据
            train_dataset = NeRFDataset(args.data_dir, 'train')
            test_dataset = NeRFDataset(args.data_dir, 'test')
            
            # 创建模型
            model = SimpleGaussianModel(config['model']['num_points'])
            
            print("开始训练Gaussian模型...")
            
            # 训练
            start_time = time.time()
            
            config['training']['log_dir'] = os.path.join(args.output_dir, 'logs/gaussian')
            os.makedirs(config['training']['log_dir'], exist_ok=True)
            
            from scripts.train_gaussian import train_gaussian
            trained_model = train_gaussian(model, train_dataset, config['training'], test_dataset)
            
            training_time = time.time() - start_time
            
            # 评估
            print("开始评估...")
            eval_results = evaluate_model(trained_model, test_dataset, device)
            
            # 渲染视频
            gaussian_output_dir = os.path.join(args.output_dir, 'gaussian')
            os.makedirs(gaussian_output_dir, exist_ok=True)
            
            print("渲染视频...")
            video_path = os.path.join(gaussian_output_dir, 'gaussian_video.mp4')
            render_video(trained_model, video_path, device)
            
            # 保存模型
            torch.save(trained_model.state_dict(), 
                      os.path.join(gaussian_output_dir, 'gaussian_model.pth'))
            
            results['Gaussian'] = {
                **eval_results,
                'training_time': training_time
            }
            
            print(f"Gaussian完成 - PSNR: {eval_results.get('PSNR', 0):.2f}, 训练时间: {training_time:.1f}s")
            
        except Exception as e:
            print(f"Gaussian训练失败: {e}")
            import traceback
            traceback.print_exc()
            results['Gaussian'] = {'PSNR': 0, 'SSIM': 0, 'training_time': 0, 'error': str(e)}
    
    print(f"实验完成！结果保存在: {args.output_dir}")
    
    # 显示结果摘要
    print("\n=== 结果摘要 ===")
    for method, result in results.items():
        status = "✅" if result.get('PSNR', 0) > 0 else "❌"
        print(f"{status} {method}: PSNR={result.get('PSNR', 0):.2f}, 时间={result.get('training_time', 0):.1f}s")

if __name__ == "__main__":
    main()
