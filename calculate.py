import os
import cv2
import math
import sys
from skimage import io, color
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
from scipy.spatial.distance import cosine


def calculate_sam(original_image_path, output_image_path):
    original_image = cv2.imread(original_image_path)
    output_image = cv2.imread(output_image_path)
    # 将图像展平成一维数组
    original_flat = original_image.flatten()
    output_flat = output_image.flatten()
    # 计算余弦距离
    cosine_distance = cosine(original_flat, output_flat)
    # 计算余弦距离对应的角度（SMA）
    sam = np.arccos(1 - cosine_distance)    # 0.719240758392923     # 0.20817536298441316
    # 将弧度转换为度
    # sam_degrees = np.degrees(sam)       # 41.20945990970303     # 11.927569697610815
    return sam


def calculate_rmse(original_image_path, output_image_path):
    # 打开图像文件
    original_image = Image.open(original_image_path)
    output_image = Image.open(output_image_path)
    # 将图像转换为NumPy数组
    array1 = np.array(original_image)
    array2 = np.array(output_image)
    # 计算均方根误差
    rmse = np.sqrt(np.mean((array1 - array2) ** 2))    # 9.556921368457909      # 10.081827624005134
    return rmse


def calculate_ssim(original_image_path, output_image_path):
    # 读取原始图像和输出图像
    original_image = io.imread(original_image_path)
    output_image = io.imread(output_image_path)

    # 转换为灰度图像（SAM通常在灰度空间中计算）
    # original_gray = color.rgb2gray(original_image)
    # output_gray = color.rgb2gray(output_image)

    # 计算SSIM
    ssim_index = ssim(original_image, output_image, multichannel=True)
    return ssim_index


def calculate_psnr(original_image_path, output_image_path):
    original_image = cv2.imread(original_image_path)
    output_image = cv2.imread(output_image_path)
    mse = np.mean((original_image/1.0 - output_image/1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)

source_folder = './dataset/building_change_detection/2012/test/images'
result_folder = './test_results/images'
files = os.listdir(result_folder)
Total_SAM = 0
Total_RMSE = 0
Total_SSIM = 0
Total_PSNR = 0
n = len(files)
print(n)    # 690
for image in files:
    input_path = os.path.join(source_folder, image)
    output_path = os.path.join(result_folder, image)
    # SAM指数
    Total_SAM += calculate_sam(input_path, output_path)
    # RMSE指数
    Total_RMSE += calculate_rmse(input_path, output_path)
    # SSIM指数
    Total_SSIM += calculate_ssim(input_path, output_path)
    # PSNR指数
    Total_PSNR += calculate_psnr(input_path, output_path)


print(f"Average_SAM:{Total_SAM/n}")
print(f"Average_RMSE:{Total_RMSE/n}")
print(f"Average_SSIM:{Total_SSIM/n}")
print(f"Average_PSNR:{Total_PSNR/n}")

