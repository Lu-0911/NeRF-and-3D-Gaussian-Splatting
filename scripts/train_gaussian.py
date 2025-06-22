import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import yaml
import argparse
import sys

class SimpleGaussianModel(nn.Module):
    """简化的Gaussian模型"""
    def __init__(self, num_points=1000):
        super().__init__()
        self.num_points = num_points
        
        # 高斯参数
        self.xyz = nn.Parameter(torch.randn(num_points, 3) * 0.3)
        self.colors = nn.Parameter(torch.randn(num_points, 3) * 0.1)
        self.opacity = nn.Parameter(torch.ones(num_points) * 0.5)

    def forward(self, rays_o, rays_d):
        """前向传播 - 返回RGB和depth"""
        # 确保输入维度正确
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.view(-1, 3)
        
        batch_size = rays_o.shape[0]
        device = rays_o.device
        
        # 射线采样
        t = torch.linspace(2.0, 4.0, 32, device=device)
        pts = rays_o[:, None, :] + rays_d[:, None, :] * t[None, :, None]
        
        # 计算距离和权重
        distances = torch.cdist(pts.view(-1, 3), self.xyz)
        distances = distances.view(batch_size, 32, self.num_points)
        weights = torch.exp(-distances * 5.0) * torch.sigmoid(self.opacity)[None, None, :]
        
        # 渲染颜色
        colors = torch.sigmoid(self.colors)
        weighted_colors = torch.einsum('ijk,kl->ijl', weights, colors)
        rgb = torch.mean(weighted_colors, dim=1)
        
        # 计算深度
        depth = torch.mean(t.expand(batch_size, -1), dim=1, keepdim=True)
        
        return rgb, depth

def safe_get_batch_data(batch, device):
    """安全地提取batch数据，处理各种可能的格式"""
    try:
        # 提取数据
        rays_o = batch['rays_o']
        rays_d = batch['rays_d'] 
        target_rgb = batch['rgb']
        
        # 处理squeeze
        if rays_o.dim() > 2:
            rays_o = rays_o.squeeze(0)
        if rays_d.dim() > 2:
            rays_d = rays_d.squeeze(0)
        if target_rgb.dim() > 2:
            target_rgb = target_rgb.squeeze(0)
            
        # 转换为tensor并移到设备
        if not isinstance(rays_o, torch.Tensor):
            rays_o = torch.tensor(rays_o, dtype=torch.float32)
        if not isinstance(rays_d, torch.Tensor):
            rays_d = torch.tensor(rays_d, dtype=torch.float32)
        if not isinstance(target_rgb, torch.Tensor):
            target_rgb = torch.tensor(target_rgb, dtype=torch.float32)
            
        rays_o = rays_o.to(device)
        rays_d = rays_d.to(device)
        target_rgb = target_rgb.to(device)
        
        # 确保正确维度
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.view(-1, 3)
        target_rgb = target_rgb.view(-1, 3)
        
        return rays_o, rays_d, target_rgb
        
    except Exception as e:
        print(f"数据提取失败: {e}")
        raise

def train_gaussian(model, dataset, config, test_dataset=None):
    """训练Gaussian模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=config.get('lr', 0.01))
    
    # 训练参数
    num_epochs = min(config.get('num_epochs', 10), 10)
    max_batches = 2000000   # 每个epoch最大批次数
    max_rays = 512   # 每批次最大射线数
    
    print(f"开始训练Gaussian模型: {num_epochs} epochs, 数据集大小: {len(dataset)}")
    
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        valid_batches = 0
        
        # 创建dataloader
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        pbar = tqdm(enumerate(dataloader), total=min(len(dataloader), max_batches), 
                   desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in pbar:
            if batch_idx >= max_batches:
                break
                
            try:
                # 安全地获取数据
                rays_o, rays_d, target_rgb = safe_get_batch_data(batch, device)
                
                # 随机采样减少计算量
                if rays_o.shape[0] > max_rays:
                    indices = torch.randperm(rays_o.shape[0])[:max_rays]
                    rays_o = rays_o[indices]
                    rays_d = rays_d[indices]
                    target_rgb = target_rgb[indices]
                
                # 前向传播
                pred_rgb, pred_depth = model(rays_o, rays_d)
                
                # 确保维度匹配
                if pred_rgb.shape[0] != target_rgb.shape[0]:
                    min_size = min(pred_rgb.shape[0], target_rgb.shape[0])
                    pred_rgb = pred_rgb[:min_size]
                    target_rgb = target_rgb[:min_size]
                
                # 计算损失
                loss = F.mse_loss(pred_rgb, target_rgb)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                # 记录
                epoch_loss += loss.item()
                valid_batches += 1
                
                # 更新进度条
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
                
            except Exception as e:
                print(f"批次 {batch_idx} 失败: {e}")
                continue
        
        # 计算平均损失和PSNR
        if valid_batches > 0:
            avg_loss = epoch_loss / valid_batches
            psnr = -10 * torch.log10(torch.tensor(avg_loss)) if avg_loss > 0 else 0
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, PSNR={psnr:.2f}dB, "
                  f"Valid batches={valid_batches}/{max_batches}")
        else:
            print(f"Epoch {epoch+1}: 没有有效的训练批次")
    
    print("Gaussian训练完成")
    return model

def evaluate_gaussian_test(model, test_dataloader, device, max_batches=5):
    """测试评估"""
    if not test_dataloader:
        return 0.0, 0.0
        
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            if i >= max_batches:
                break
                
            try:
                rays_o, rays_d, target_rgb = safe_get_batch_data(batch, device)
                
                # 减少采样
                if rays_o.shape[0] > 256:
                    indices = torch.randperm(rays_o.shape[0])[:256]
                    rays_o = rays_o[indices]
                    rays_d = rays_d[indices]
                    target_rgb = target_rgb[indices]
                
                pred_rgb, _ = model(rays_o, rays_d)
                
                # 确保维度匹配
                if pred_rgb.shape[0] != target_rgb.shape[0]:
                    min_size = min(pred_rgb.shape[0], target_rgb.shape[0])
                    pred_rgb = pred_rgb[:min_size]
                    target_rgb = target_rgb[:min_size]
                
                loss = F.mse_loss(pred_rgb, target_rgb)
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                print(f"测试批次 {i} 失败: {e}")
                continue
    
    model.train()
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    psnr = -10 * torch.log10(torch.tensor(avg_loss)) if avg_loss > 0 else 0
    
    return avg_loss, psnr.item() if isinstance(psnr, torch.Tensor) else psnr

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练Gaussian模型')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录')
    parser.add_argument('--output_dir', type=str, default='./output', help='输出目录')
    args = parser.parse_args()

    # 加载配置
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"配置文件加载失败: {e}")
        return

    # 导入数据集
    try:
        sys.path.append('.')
        from data_processing.dataset import NeRFDataset
        
        print("加载数据集...")
        train_dataset = NeRFDataset(args.data_dir, 'train')
        print(f"训练数据集大小: {len(train_dataset)}")
        
        # 可选的测试数据集
        test_dataset = None
        try:
            test_dataset = NeRFDataset(args.data_dir, 'test')
            print(f"测试数据集大小: {len(test_dataset)}")
        except:
            print("未找到测试数据集，跳过")
            
    except Exception as e:
        print(f"数据集加载失败: {e}")
        return

    try:
        # 创建模型
        num_points = config.get('model', {}).get('num_points', 1000)
        model = SimpleGaussianModel(num_points)
        print(f"创建模型，高斯点数: {num_points}")
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")

        # 训练模型
        trained_model = train_gaussian(model, train_dataset, config.get('training', {}), test_dataset)

        # 保存模型
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, 'gaussian_model.pth')
        torch.save(trained_model.state_dict(), save_path)
        print(f"模型保存至: {save_path}")

    except Exception as e:
        print(f"训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
