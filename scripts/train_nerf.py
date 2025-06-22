import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import argparse
import os
import torch.utils.tensorboard as tb


def train_tensorf(model, dataset, config, test_dataset=None):
    """训练TensoRF模型"""
    device = next(model.parameters()).device
    
    log_dir = config.get('log_dir', './logs/tensorf')
    writer = tb.SummaryWriter(log_dir)
    
    # 确保学习率是浮点数类型
    lr_init = float(config.get('lr_init', 0.02))
    lr_basis = float(config.get('lr_basis', 0.001))
    
    optimizer = optim.Adam([
        {'params': model.density_grid, 'lr': lr_init},
        {'params': model.color_features, 'lr': lr_init},
        {'params': model.density_net.parameters(), 'lr': lr_basis},
        {'params': model.color_net.parameters(), 'lr': lr_basis}
    ])
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    num_epochs = min(config.get('num_epochs', 50), 50)
    
    # 准备测试数据加载器
    test_dataloader = None
    if test_dataset:
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model.train()
    print(f"开始训练，总轮数: {num_epochs}")
    
    global_step = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        batch_count = 0
        max_batches_per_epoch = 2000000
        
        # 训练阶段
        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            if i >= max_batches_per_epoch:
                break
                
            rays_o = batch['rays_o'].squeeze(0).to(device)
            rays_d = batch['rays_d'].squeeze(0).to(device)
            target_rgb = batch['rgb'].squeeze(0).to(device)
            
            N_rand = 512
            if rays_o.shape[0] > N_rand:
                select_inds = torch.randperm(rays_o.shape[0])[:N_rand]
                rays_o = rays_o[select_inds]
                rays_d = rays_d[select_inds]
                target_rgb = target_rgb[select_inds]
            
            rgb_pred, _ = model(rays_o, rays_d)
            loss = F.mse_loss(rgb_pred, target_rgb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            # 记录训练损失到Tensorboard
            writer.add_scalar('Loss/Train', loss.item(), global_step)
            
            global_step += 1
        
        # 算并记录epoch级别的指标
        avg_train_loss = epoch_loss / batch_count if batch_count > 0 else 0
        train_psnr = -10. * torch.log10(torch.tensor(avg_train_loss)) if avg_train_loss > 0 else 0
        
        writer.add_scalar('Loss/Train_Epoch', avg_train_loss, epoch)
        writer.add_scalar('PSNR/Train', train_psnr.item(), epoch)
        
        # 测试阶段评估
        if test_dataloader and epoch % 5 == 0:  # 每5个epoch评估一次
            test_loss, test_psnr = evaluate_on_test(model, test_dataloader, device)
            writer.add_scalar('Loss/Test', test_loss, epoch)
            writer.add_scalar('PSNR/Test', test_psnr, epoch)
            
            print(f"Epoch {epoch}: Train Loss={avg_train_loss:.6f}, Train PSNR={train_psnr:.2f}, "
                  f"Test Loss={test_loss:.6f}, Test PSNR={test_psnr:.2f}")
        elif epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss={avg_train_loss:.6f}, Train PSNR={train_psnr:.2f}")
    
    writer.close()
    return model

def evaluate_on_test(model, test_dataloader, device):
    """在测试集上评估模型"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            if i >= 20:  # 限制测试批次数量
                break
                
            rays_o = batch['rays_o'].squeeze(0).to(device)
            rays_d = batch['rays_d'].squeeze(0).to(device)
            target_rgb = batch['rgb'].squeeze(0).to(device)
            
            # 随机采样以减少计算量
            N_rand = 1024
            if rays_o.shape[0] > N_rand:
                select_inds = torch.randperm(rays_o.shape[0])[:N_rand]
                rays_o = rays_o[select_inds]
                rays_d = rays_d[select_inds]
                target_rgb = target_rgb[select_inds]
            
            rgb_pred, _ = model(rays_o, rays_d)
            loss = F.mse_loss(rgb_pred, target_rgb)
            
            total_loss += loss.item()
            num_batches += 1
    
    model.train()
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    psnr = -10. * torch.log10(torch.tensor(avg_loss)) if avg_loss > 0 else 0
    
    return avg_loss, psnr.item()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./logs')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 导入必要的模块
    import sys
    sys.path.append('.')
    from models.tensorf import TensoRF
    from data_processing.dataset import NeRFDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    train_dataset = NeRFDataset(args.data_dir, 'train')

    # 创建模型
    aabb = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], device=device)
    model = TensoRF(
        aabb=aabb,
        gridSize=config['tensorf']['gridSize'],
        device=device,
        density_n_comp=config['tensorf']['density_n_comp'],
        app_n_comp=config['tensorf']['app_n_comp']
    ).to(device)

    # 训练
    trained_model = train_tensorf(model, train_dataset, config['tensorf'])

    # 保存模型
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(trained_model.state_dict(), os.path.join(args.output_dir, 'nerf_model.pth'))
    print("训练完成!")

if __name__ == "__main__":
    main()
