import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Embedder:
    """位置编码器"""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d
                
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, i=0):
    """获取位置编码器"""
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x: embedder_obj.embed(x)
    return embed, embedder_obj.out_dim

class NeRF(nn.Module):
    """原版NeRF网络"""
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, 
                 skips=[4], use_viewdirs=False):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        # 位置编码的MLP层
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + 
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) 
             for i in range(D-1)]
        )
        
        # 视角方向的处理
        if use_viewdirs:
            self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        
        # 通过MLP处理位置信息
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            # 分离密度和特征
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            
            # 结合视角方向
            h = torch.cat([feature, input_views], -1)
            
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False):
    """将网络原始输出转换为RGB和深度"""
    
    def raw2alpha(raw, dists, act_fn=F.relu):
        return 1. - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.tensor([1e10], device=dists.device).expand(dists[..., :1].shape)], -1)

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape, device=raw.device) * raw_noise_std

    alpha = raw2alpha(raw[..., 3] + noise, dists)
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    
    rgb_map = torch.sum(weights[..., None] * rgb, -2)
    depth_map = torch.sum(weights * z_vals, -1)
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, depth_map, acc_map, weights

class NeRFModel:
    """完整的NeRF模型包装器"""
    def __init__(self, multires=10, multires_views=4, use_viewdirs=True):
        # 创建位置编码器
        self.embed_fn, self.input_ch = get_embedder(multires, 0)
        
        input_ch_views = 0
        self.embeddirs_fn = None
        if use_viewdirs:
            self.embeddirs_fn, input_ch_views = get_embedder(multires_views, 0)
        
        # 创建网络
        self.network_coarse = NeRF(
            D=8, W=256,
            input_ch=self.input_ch, output_ch=5,
            skips=[4], input_ch_views=input_ch_views,
            use_viewdirs=use_viewdirs
        )
        
        self.network_fine = NeRF(
            D=8, W=256,
            input_ch=self.input_ch, output_ch=5,
            skips=[4], input_ch_views=input_ch_views,
            use_viewdirs=use_viewdirs
        )
    
    def forward(self, rays_o, rays_d):
        """简化的前向传播"""
        # 射线采样
        N_samples = 64
        near, far = 2.0, 6.0
        
        t_vals = torch.linspace(0., 1., steps=N_samples, device=rays_o.device)
        z_vals = near * (1. - t_vals) + far * (t_vals)
        z_vals = z_vals.expand([rays_o.shape[0], N_samples])
        
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        pts_flat = pts.reshape(-1, 3)
        
        # 位置编码
        pts_embedded = self.embed_fn(pts_flat)
        
        # 方向编码（简化处理）
        viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        viewdirs_embedded = self.embeddirs_fn(viewdirs) if self.embeddirs_fn else torch.zeros_like(pts_embedded[:, :0])
        viewdirs_embedded = viewdirs_embedded[:, None].expand(pts.shape[0], N_samples, -1).reshape(-1, viewdirs_embedded.shape[-1])
        
        # 网络推理
        inputs = torch.cat([pts_embedded, viewdirs_embedded], -1)
        raw = self.network_coarse(inputs)
        raw = raw.reshape(*pts.shape[:-1], -1)
        
        # 体积渲染
        rgb_map, depth_map, acc_map, weights = raw2outputs(raw, z_vals, rays_d)
        
        return rgb_map, depth_map
