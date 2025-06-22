import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TensoRF(nn.Module):
    """TensoRF模型"""
    
    def __init__(self, aabb, gridSize, device, density_n_comp=16, app_n_comp=48):
        super(TensoRF, self).__init__()
        
        self.aabb = aabb
        self.device = device
        self.gridSize = gridSize
        
        # 简化的直接3D网格表示，避免复杂的张量分解
        self.density_grid = nn.Parameter(torch.randn(gridSize, gridSize, gridSize) * 0.1)
        self.color_features = nn.Parameter(torch.randn(gridSize, gridSize, gridSize, 32) * 0.1)
        
        # 网络层
        self.density_net = nn.Sequential(
            nn.Linear(4, 64),  # 位置(3) + 网格特征(1)
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.color_net = nn.Sequential(
            nn.Linear(35, 128),  # 位置(3) + 网格特征(32)
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )
    
    def normalize_coord(self, xyz):
        """归一化坐标到[0, gridSize-1]"""
        normalized = (xyz - self.aabb[0]) / (self.aabb[1] - self.aabb[0])
        return normalized * (self.gridSize - 1)
    
    def trilinear_interpolation(self, grid, coords):
        """三线性插值采样"""
        # 限制坐标范围
        coords = torch.clamp(coords, 0, self.gridSize - 1)
        
        # 获取整数和小数部分
        coords_floor = torch.floor(coords).long()
        coords_ceil = torch.ceil(coords).long()
        coords_frac = coords - coords_floor.float()
        
        # 限制索引范围
        coords_floor = torch.clamp(coords_floor, 0, self.gridSize - 1)
        coords_ceil = torch.clamp(coords_ceil, 0, self.gridSize - 1)
        
        # 8个角的值
        x0, y0, z0 = coords_floor[:, 0], coords_floor[:, 1], coords_floor[:, 2]
        x1, y1, z1 = coords_ceil[:, 0], coords_ceil[:, 1], coords_ceil[:, 2]
        
        xf, yf, zf = coords_frac[:, 0], coords_frac[:, 1], coords_frac[:, 2]
        
        # 如果是4D网格（有特征维度）
        if len(grid.shape) == 4:
            # 获取8个角的特征值
            c000 = grid[x0, y0, z0]  # [N, features]
            c001 = grid[x0, y0, z1]
            c010 = grid[x0, y1, z0]
            c011 = grid[x0, y1, z1]
            c100 = grid[x1, y0, z0]
            c101 = grid[x1, y0, z1]
            c110 = grid[x1, y1, z0]
            c111 = grid[x1, y1, z1]
        else:
            # 3D网格
            c000 = grid[x0, y0, z0].unsqueeze(-1)  # [N, 1]
            c001 = grid[x0, y0, z1].unsqueeze(-1)
            c010 = grid[x0, y1, z0].unsqueeze(-1)
            c011 = grid[x0, y1, z1].unsqueeze(-1)
            c100 = grid[x1, y0, z0].unsqueeze(-1)
            c101 = grid[x1, y0, z1].unsqueeze(-1)
            c110 = grid[x1, y1, z0].unsqueeze(-1)
            c111 = grid[x1, y1, z1].unsqueeze(-1)
        
        # 三线性插值
        xf = xf.unsqueeze(-1)
        yf = yf.unsqueeze(-1)
        zf = zf.unsqueeze(-1)
        
        # 沿x轴插值
        c00 = c000 * (1 - xf) + c100 * xf
        c01 = c001 * (1 - xf) + c101 * xf
        c10 = c010 * (1 - xf) + c110 * xf
        c11 = c011 * (1 - xf) + c111 * xf
        
        # 沿y轴插值
        c0 = c00 * (1 - yf) + c10 * yf
        c1 = c01 * (1 - yf) + c11 * yf
        
        # 沿z轴插值
        result = c0 * (1 - zf) + c1 * zf
        
        return result
    
    def forward(self, rays_o, rays_d, is_train=True):
        """前向传播"""
        # 射线采样
        N_samples = 64
        near, far = 2.0, 6.0
        
        t_vals = torch.linspace(0., 1., steps=N_samples, device=self.device)
        z_vals = near * (1. - t_vals) + far * (t_vals)
        z_vals = z_vals.expand([rays_o.shape[0], N_samples])
        
        if is_train:
            # 添加随机扰动
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            t_rand = torch.rand(z_vals.shape, device=self.device)
            z_vals = lower + (upper - lower) * t_rand
        
        # 生成采样点
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        pts_flat = pts.reshape(-1, 3)
        
        # 归一化坐标到网格空间
        grid_coords = self.normalize_coord(pts_flat)
        
        # 从网格中采样特征
        density_feat = self.trilinear_interpolation(self.density_grid, grid_coords)  # [N, 1]
        color_feat = self.trilinear_interpolation(self.color_features, grid_coords)   # [N, 32]
        
        # 计算密度
        density_input = torch.cat([pts_flat, density_feat], dim=-1)  # [N, 4]
        sigma = F.softplus(self.density_net(density_input)).squeeze(-1)  # [N]
        
        # 计算颜色
        color_input = torch.cat([pts_flat, color_feat], dim=-1)  # [N, 35]
        rgb_flat = self.color_net(color_input)  # [N, 3]
        
        # 重塑形状
        sigma = sigma.reshape(*pts.shape[:-1])  # [batch, N_samples]
        rgb = rgb_flat.reshape(*pts.shape)      # [batch, N_samples, 3]
        
        # 体积渲染
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], -1)
        
        alpha = 1. - torch.exp(-sigma * dists)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones_like(alpha[..., :1]), 1. - alpha + 1e-10], -1), -1
        )[..., :-1]
        
        # 最终颜色和深度
        rgb_map = torch.sum(weights[..., None] * rgb, -2)
        depth_map = torch.sum(weights * z_vals, -1)
        
        return rgb_map, depth_map
