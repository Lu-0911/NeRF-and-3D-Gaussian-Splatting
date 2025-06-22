import matplotlib.pyplot as plt
import numpy as np

def plot_comparison_results(results_dict, save_path=None):
    """绘制比较结果"""
    if not results_dict:
        print("没有结果可显示")
        return
        
    methods = list(results_dict.keys())
    psnr_values = [results_dict[method].get('PSNR', 0) for method in methods]
    ssim_values = [results_dict[method].get('SSIM', 0) for method in methods]
    
    # 过滤掉无效值
    valid_methods = []
    valid_psnr = []
    valid_ssim = []
    
    for i, method in enumerate(methods):
        if psnr_values[i] > 0 or ssim_values[i] > 0:
            valid_methods.append(method)
            valid_psnr.append(psnr_values[i])
            valid_ssim.append(ssim_values[i])
    
    if not valid_methods:
        print("没有有效的结果数据")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # PSNR比较
    bars1 = ax1.bar(valid_methods, valid_psnr, color=['blue', 'green', 'red'][:len(valid_methods)])
    ax1.set_title('PSNR Comparison')
    ax1.set_ylabel('PSNR (dB)')
    
    # 设置合理的y轴范围
    if max(valid_psnr) > 0:
        ax1.set_ylim(0, max(valid_psnr) * 1.2)
    else:
        ax1.set_ylim(0, 1)
    
    # 在柱状图上显示数值
    for bar, value in zip(bars1, valid_psnr):
        if value > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(valid_psnr)*0.01,
                    f'{value:.2f}', ha='center', va='bottom')
    
    # SSIM比较
    bars2 = ax2.bar(valid_methods, valid_ssim, color=['blue', 'green', 'red'][:len(valid_methods)])
    ax2.set_title('SSIM Comparison')
    ax2.set_ylabel('SSIM')
    ax2.set_ylim(0, 1)
    
    # 在柱状图上显示数值
    for bar, value in zip(bars2, valid_ssim):
        if value > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"比较图表已保存: {save_path}")
    
    # 不显示图表，只保存
    plt.close()

def plot_training_curves(losses, save_path=None):
    """绘制训练曲线"""
    if not losses:
        return
        
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()
