import os
import json
import numpy as np
import cv2
import argparse

def create_simple_transforms(images_dir, output_dir):
    """创建简单的transforms文件"""
    
    image_files = sorted([f for f in os.listdir(images_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if not image_files:
        print("未找到图像文件")
        return False
    
    # 读取第一张图像获取分辨率
    first_img = cv2.imread(os.path.join(images_dir, image_files[0]))
    h, w = first_img.shape[:2]
    
    # 估计相机参数
    focal = w * 0.7
    
    frames = []
    n_images = len(image_files)
    
    # 创建简单的圆形轨迹
    for i, img_file in enumerate(image_files):
        theta = 2 * np.pi * i / n_images
        
        # 相机位置
        radius = 3.0
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = 0.0
        
        # 朝向中心
        look_at = np.array([0, 0, 0])
        cam_pos = np.array([x, y, z])
        
        forward = look_at - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        transform_matrix = [
            [right[0], up[0], -forward[0], cam_pos[0]],
            [right[1], up[1], -forward[1], cam_pos[1]],
            [right[2], up[2], -forward[2], cam_pos[2]],
            [0, 0, 0, 1]
        ]
        
        frames.append({
            "file_path": os.path.splitext(img_file)[0],
            "transform_matrix": transform_matrix
        })
    
    # 创建transforms数据
    transforms_data = {
        "camera_angle_x": 2 * np.arctan(w / (2 * focal)),
        "fl_x": focal,
        "fl_y": focal,
        "cx": w / 2,
        "cy": h / 2,
        "w": w,
        "h": h,
        "frames": frames
    }
    
    # 数据集划分
    train_frames = [frames[i] for i in range(len(frames)) if i % 8 != 0]
    test_frames = [frames[i] for i in range(len(frames)) if i % 8 == 0]
    
    # 保存文件
    for split, split_frames in [('train', train_frames), ('test', test_frames), ('val', test_frames)]:
        split_data = transforms_data.copy()
        split_data['frames'] = split_frames
        
        with open(os.path.join(output_dir, f'transforms_{split}.json'), 'w') as f:
            json.dump(split_data, f, indent=2)
    
    print("数据划分完成")
    
    return True

def resize_images(input_dir, output_dir, max_size=800):
    """调整图像大小"""
    os.makedirs(output_dir, exist_ok=True)
    
    count = 0
    for img_file in sorted(os.listdir(input_dir)):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, img_file)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            h, w = img.shape[:2]
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            output_name = f'im_{count+1:04d}.jpg'
            output_path = os.path.join(output_dir, output_name)
            cv2.imwrite(output_path, img)
            count += 1
    
    print(f"处理了 {count} 张图像")
    return count > 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--type', type=str, default='images')
    parser.add_argument('--max_size', type=int, default=800)
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    images_dir = os.path.join(args.output, 'images')
    
    # 处理图像
    if not resize_images(args.input, images_dir, args.max_size):
        print("图像处理失败")
        return 1
    
    # 创建transforms文件
    if not create_simple_transforms(images_dir, args.output):
        print("创建transforms失败")
        return 1
    
    print("数据预处理完成")
    return 0

if __name__ == "__main__":
    exit(main())
