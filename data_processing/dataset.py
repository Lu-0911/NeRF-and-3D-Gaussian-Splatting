import torch
import numpy as np
import os
import json
from torch.utils.data import Dataset
import cv2
from PIL import Image

class NeRFDataset(Dataset):
    def __init__(self, data_dir, split='train', white_bkgd=True):
        self.data_dir = data_dir
        self.split = split
        self.white_bkgd = white_bkgd
        
        self.read_meta()
        
    def read_meta(self):
        with open(os.path.join(self.data_dir, f'transforms_{self.split}.json'), 'r') as f:
            self.meta = json.load(f)
        
        self.focal = 0.5 * 800 / np.tan(0.5 * self.meta['camera_angle_x'])
        
        self.image_paths = []
        self.poses = []
        
        for frame in self.meta['frames']:
            img_path = os.path.join(self.data_dir, 'images', frame['file_path'] + '.jpg')
            if not os.path.exists(img_path):
                img_path = os.path.join(self.data_dir, 'images', frame['file_path'] + '.png')
            
            if os.path.exists(img_path):
                self.image_paths.append(img_path)
                self.poses.append(np.array(frame['transform_matrix']))
        
        self.poses = np.array(self.poses).astype(np.float32)
        
        # 读取图像
        self.images = []
        for img_path in self.image_paths:
            img = Image.open(img_path)
            img = np.array(img) / 255.0
            
            if img.shape[-1] == 4:  # RGBA
                img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:]) * self.white_bkgd
            
            self.images.append(img)
        
        self.images = np.array(self.images).astype(np.float32)
        self.H, self.W = self.images.shape[1:3]
        
        # 生成射线
        self.generate_rays()
    
    def generate_rays(self):
        """生成所有射线"""
        i, j = np.meshgrid(np.arange(self.W), np.arange(self.H), indexing='xy')
        dirs = np.stack([(i-self.W*0.5)/self.focal, -(j-self.H*0.5)/self.focal, -np.ones_like(i)], -1)
        
        self.rays_o = []
        self.rays_d = []
        self.rgb = []
        
        for pose, img in zip(self.poses, self.images):
            rays_d = np.sum(dirs[..., np.newaxis, :] * pose[:3,:3], -1)
            rays_o = np.broadcast_to(pose[:3,-1], np.shape(rays_d))
            
            self.rays_o.append(rays_o.reshape(-1, 3))
            self.rays_d.append(rays_d.reshape(-1, 3))
            self.rgb.append(img.reshape(-1, 3))
        
        self.rays_o = np.concatenate(self.rays_o, 0)
        self.rays_d = np.concatenate(self.rays_d, 0)
        self.rgb = np.concatenate(self.rgb, 0)
        
    def __len__(self):
        return len(self.rays_o)
    
    def __getitem__(self, idx):
        return {
            'rays_o': torch.FloatTensor(self.rays_o[idx]),
            'rays_d': torch.FloatTensor(self.rays_d[idx]),
            'rgb': torch.FloatTensor(self.rgb[idx])
        }
