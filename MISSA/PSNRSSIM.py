import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os

def calculate_psnr_and_ssim(hr_path, sr_path):
    psnr_values = []
    ssim_values = []

    # 获取HR和SR文件夹中的所有文件名
    hr_files = [f for f in os.listdir(hr_path) if os.path.isfile(os.path.join(hr_path, f))]
    sr_files = [f for f in os.listdir(sr_path) if os.path.isfile(os.path.join(sr_path, f))]

    # 确保文件名是一一对应的
    for hr_file, sr_file in zip(hr_files, sr_files):
        # 构建完整的文件路径
        hr_image_path = os.path.join(hr_path, hr_file)
        sr_image_path = os.path.join(sr_path, sr_file)

        # 读取HR和SR图像
        hr_image = cv2.imread(hr_image_path, cv2.IMREAD_GRAYSCALE)
        sr_image = cv2.imread(sr_image_path, cv2.IMREAD_GRAYSCALE)

        # 确保图像被成功加载
        if hr_image is None or sr_image is None:
            print(f"Error loading image: {hr_file} or {sr_file}")
            continue

        # 计算PSNR
        psnr = peak_signal_noise_ratio(hr_image, sr_image)
        psnr_values.append(psnr)

        # 计算SSIM
        ssim = structural_similarity(hr_image, sr_image, data_range=hr_image.max() - hr_image.min())
        ssim_values.append(ssim)

    # 计算PSNR和SSIM的平均值
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    return avg_psnr, avg_ssim

# HR和SR图像的文件夹路径
hr_folder_path = r'D:\dataset\wafer100\HR'
sr_folder_path = r'D:\Experiment model\Model 1\MRCAN\MRCAN_BIX4_wafer\results-wafer100'

# 计算并打印平均PSNR和SSIM
avg_psnr, avg_ssim = calculate_psnr_and_ssim(hr_folder_path, sr_folder_path)
print(f'Average PSNR: {avg_psnr}')
print(f'Average SSIM: {avg_ssim}')
