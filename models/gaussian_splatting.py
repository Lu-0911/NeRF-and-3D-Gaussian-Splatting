import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GaussianModel(nn.Module):
    """3D Gaussian Splatting模型"""
    
    def __init__(self, num_points=100000):
        super().__init__()
        
        # 高斯点的基本属性
        self._xyz = nn.Parameter(torch.randn(num_points, 3) * 0.1)
        self._features_dc = nn.Parameter(torch.randn(num_points, 1, 3))
        self._features_rest = nn.Parameter(torch.randn(num_points, 15, 3))
        self._scaling = nn.Parameter(torch.randn(num_points, 3))
        self._rotation = nn.Parameter(torch.randn(num_points, 4))
        self._opacity = nn.Parameter(torch.randn(num_points, 1))
        
        # 激活函数
        self.scaling_activation = torch.exp
        self.opacity_activation = torch.sigmoid
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return F.normalize(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

class GaussianRenderer:
    """3D Gaussian Splatting渲染器"""
    
    def __init__(self, gaussians):
        self.gaussians = gaussians
        
    def render(self, rays_o, rays_d):
        """简化的渲染过程"""
        # 获取高斯参数
        means3D = self.gaussians.get_xyz
        opacity = self.gaussians.get_opacity
        features = self.gaussians.get_features
        
        # 简化的颜色计算（使用DC分量）
        colors = torch.sigmoid(features[:, 0, :])  # 只使用DC分量
        
        # 简化的投影和混合
        batch_size = rays_o.shape[0]
        
        # 计算每个高斯点到射线的距离
        ray_centers = rays_o + rays_d * 3.0  # 假设物体在距离3处
        
        # 对每条射线，找到最近的几个高斯点
        distances = torch.cdist(ray_centers, means3D)  # [batch_size, num_points]
        
        # 选择最近的K个点
        K = 10
        _, indices = torch.topk(distances, K, dim=1, largest=False)
        
        # 简化的alpha混合渲染
        rgb_output = torch.zeros(batch_size, 3, device=rays_o.device)
        
        for i in range(batch_size):
            selected_indices = indices[i]
            selected_colors = colors[selected_indices]
            selected_opacities = opacity[selected_indices]
            selected_distances = distances[i, selected_indices]
            
            # 基于距离的权重
            weights = torch.exp(-selected_distances * 0.1)
            weights = weights * selected_opacities.squeeze()
            weights = weights / (weights.sum() + 1e-8)
            
            # 加权颜色
            rgb_output[i] = torch.sum(weights.unsqueeze(-1) * selected_colors, dim=0)
        
        return rgb_output

def quaternion_to_rotation_matrix(quaternions):
    """四元数转旋转矩阵"""
    q = F.normalize(quaternions, dim=1)
    
    r, i, j, k = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    R = torch.zeros(q.shape[0], 3, 3, device=q.device)
    
    R[:, 0, 0] = 1 - 2 * (j*j + k*k)
    R[:, 0, 1] = 2 * (i*j - k*r)
    R[:, 0, 2] = 2 * (i*k + j*r)
    R[:, 1, 0] = 2 * (i*j + k*r)
    R[:, 1, 1] = 1 - 2 * (i*i + k*k)
    R[:, 1, 2] = 2 * (j*k - i*r)
    R[:, 2, 0] = 2 * (i*k - j*r)
    R[:, 2, 1] = 2 * (j*k + i*r)
    R[:, 2, 2] = 1 - 2 * (i*i + j*j)
    
    return R

class SimpleGaussianSplatting(nn.Module):
    """3D Gaussian Splatting实现"""
    
    def __init__(self, num_points=10000):
        super().__init__()
        self.gaussians = GaussianModel(num_points)
        self.renderer = GaussianRenderer(self.gaussians)
    
    def forward(self, rays_o, rays_d):
        """前向传播"""
        return self.renderer.render(rays_o, rays_d)
    
    def get_scaling(self):
        return self.gaussians.get_scaling
    
    def get_xyz(self):
        return self.gaussians.get_xyz
    
    def get_opacity(self):
        return self.gaussians.get_opacity
    
    def get_features(self):
        return self.gaussians.get_features
    
    def parameters(self):
        """返回所有可训练参数"""
        return self.gaussians.parameters()
    
    def train(self, mode=True):
        """设置训练模式"""
        self.gaussians.train(mode)
        return self
    
    def eval(self):
        """设置评估模式"""
        self.gaussians.eval()
        return self
    
    def to(self, device):
        """移动到指定设备"""
        self.gaussians = self.gaussians.to(device)
        return self
    
    def state_dict(self):
        """返回状态字典"""
        return self.gaussians.state_dict()
    
    def load_state_dict(self, state_dict):
        """加载状态字典"""
        return self.gaussians.load_state_dict(state_dict)

# 为了保持兼容性，创建一个别名
class SimpleGaussianModel(SimpleGaussianSplatting):
    """Gaussian Splatting模型（别名）"""
    pass

