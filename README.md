# NeRF vs 3D Gaussian Splatting 对比实验

基于NeRF变体和3D Gaussian Splatting的物体重建与新视图合成对比。

## 快速开始

### 1. 环境安装


#### 自动安装
```bash
chmod +x install.sh
./install.sh
source venv/bin/activate
```

#### 手动安装
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖包
pip install torch torchvision numpy opencv-python imageio matplotlib pyyaml scikit-image tqdm pillow
```

### 2. 数据准备
将您的多角度图像放入一个文件夹，例如 `./my_images/`

### 3. 一键运行
```bash
python run.py --input ./my_images/ --output ./results --method both
```

### 4. 分步运行

#### 数据预处理
```bash
python scripts/prepare_data.py --input ./my_images/ --output ./data/my_object
```

#### 训练NeRF
```bash
python main.py --task nerf --data_dir ./data/my_object --output_dir ./results
```

#### 训练Gaussian Splatting
```bash
python main.py --task gaussian --data_dir ./data/my_object --output_dir ./results
```

#### 完整对比
```bash
python main.py --task both --data_dir ./data/my_object --output_dir ./results
```

#### 配置调整

编辑 `configs/nerf_config.yaml` 和 `configs/gaussian_config.yaml` 来调整训练参数。

### 5. 查看结果
- 训练好的模型: `./results/nerf/` 和 `./results/gaussian/`
- 渲染视频: `nerf_video.mp4` 和 `gaussian_video.mp4`


### 6. 其他

#### 数据要求

- 图像数量: 100-200张
- 角度覆盖: 360度全方位
- 图像重叠: 60-80%
- 图像质量: 清晰，光照稳定

#### 项目结构

```
project/
├── configs/              # 配置文件
├── data_processing/      # 数据处理
├── models/              # 模型实现
├── utils/               # 工具函数
├── scripts/             # 训练脚本
├── main.py             # 主程序
└── run.py   # 一键运行
```


#### 技术细节

- **NeRF**: 使用TensoRF加速版本，基于张量分解
- **3D Gaussian**: 简化实现，专注于对比效果
- **评估指标**: PSNR、SSIM、训练时间
```

