import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
def evaluate_model(model, dataset, device):
    """评估模型性能"""
    model.eval()
    total_psnr = 0
    total_ssim = 0
    num_images = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataset):
                
            rays_o = batch['rays_o'].squeeze(0).to(device)
            rays_d = batch['rays_d'].squeeze(0).to(device)
            target_rgb = batch['rgb'].squeeze(0).to(device)
            
            # 分块处理以避免内存问题
            chunk_size = 1024
            rgb_pred_list = []
            
            for j in range(0, rays_o.shape[0], chunk_size):
                rays_o_chunk = rays_o[j:j+chunk_size]
                rays_d_chunk = rays_d[j:j+chunk_size]
                
                # 检查模型返回值数量
                outputs = model(rays_o_chunk, rays_d_chunk)
                if isinstance(outputs, tuple):
                    rgb_chunk = outputs[0]  # 如果返回tuple，取第一个值
                else:
                    rgb_chunk = outputs     # 如果只返回一个值
                    
                rgb_pred_list.append(rgb_chunk)
            
            rgb_pred = torch.cat(rgb_pred_list, dim=0)
            
            # 计算PSNR
            mse = torch.mean((rgb_pred - target_rgb) ** 2)
            psnr = -10. * torch.log10(mse) if mse > 0 else 0
            total_psnr += psnr
            
            num_images += 1
    
    avg_psnr = total_psnr / num_images if num_images > 0 else 0
    
    return {
        'PSNR': float(avg_psnr),
        'SSIM': float(ssim(rgb_pred.cpu().numpy(), target_rgb.cpu().numpy(), multichannel=True))
    }
