#!/bin/bash

echo "Installing NeRF vs 3D Gaussian Splatting Comparison Environment"
echo "=============================================================="

# 检查Python版本
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
echo "Python version: $python_version"

if ! python3 -c 'import sys; exit(not (sys.version_info >= (3, 8)))' 2>/dev/null; then
    echo "Error: Python 3.8 or higher is required"
    exit 1
fi

# 创建虚拟环境
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 升级pip
echo "Upgrading pip..."
pip install --upgrade pip

# 安装PyTorch (CUDA版本)
echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "No CUDA detected, installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# 安装其他依赖
echo "Installing other dependencies..."
pip install numpy opencv-python pillow imageio imageio-ffmpeg
pip install matplotlib seaborn scikit-image
pip install tqdm pyyaml
pip install lpips

# 检查COLMAP安装
echo "Checking COLMAP installation..."
if ! command -v colmap &> /dev/null; then
    echo "Warning: COLMAP not found. Please install COLMAP manually."
    echo "Ubuntu/Debian: sudo apt-get install colmap"
    echo "Or build from source: https://github.com/colmap/colmap"
else
    echo "COLMAP found: $(colmap --version)"
fi

echo ""
echo "Installation completed!"
echo "To activate the environment, run: source venv/bin/activate"
echo "To test the installation, run: python -c 'import torch; print(torch.__version__)'"
