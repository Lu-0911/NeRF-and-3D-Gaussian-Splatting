import torch
import numpy as np
import imageio

def render_video(model, output_path, device='cuda', num_frames=120):
    """渲染环绕视频 - 修复版本"""
    model.eval()
    frames = []
    # 创建圆形相机轨迹
    angles = np.linspace(0, 2*np.pi, num_frames)
    radius = 3.0
    with torch.no_grad():
        for angle in angles:
            # 相机位置
            cam_pos = np.array([radius * np.cos(angle), radius * np.sin(angle), 0])
            
            # 朝向中心
            look_at = np.array([0, 0, 0])
            forward = look_at - cam_pos
            forward = forward / np.linalg.norm(forward)
            
            up = np.array([0, 0, 1])
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            
            # 生成射线
            H, W = 400, 400
            focal = W * 0.7
            
            i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
            dirs = np.stack([
                (i - W*0.5) / focal,
                -(j - H*0.5) / focal,
                -np.ones_like(i)
            ], -1)
            
            # 转换到世界坐标
            rays_d = np.sum(dirs[..., np.newaxis, :] * np.array([right, up, -forward]), -1)
            rays_o = np.broadcast_to(cam_pos, rays_d.shape)
            
            rays_o = torch.from_numpy(rays_o.reshape(-1, 3).copy()).float().to(device)
            rays_d = torch.from_numpy(rays_d.reshape(-1, 3).copy()).float().to(device)
            
            # 渲染
            rgb_chunks = []
            chunk_size = 1024
            
            for k in range(0, rays_o.shape[0], chunk_size):
                rays_o_chunk = rays_o[k:k+chunk_size]
                rays_d_chunk = rays_d[k:k+chunk_size]
                
                # 处理模型输出
                outputs = model(rays_o_chunk, rays_d_chunk)
                if isinstance(outputs, tuple):
                    rgb_chunk = outputs[0]  # 取RGB，忽略depth
                else:
                    rgb_chunk = outputs     # 兼容只返回RGB的情况
                
                rgb_chunks.append(rgb_chunk.cpu())
            
            rgb = torch.cat(rgb_chunks, 0)
            rgb = rgb.reshape(H, W, 3)
            rgb = torch.clamp(rgb, 0, 1)
            
            frame = (rgb.numpy() * 255).astype(np.uint8)
            frames.append(frame)
    
    # 保存视频
    imageio.mimwrite(output_path, frames, fps=30)
    print(f"视频已保存: {output_path}")
