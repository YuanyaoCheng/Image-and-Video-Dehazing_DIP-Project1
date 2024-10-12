import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


# 1. PSNR计算函数
def calculate_psnr(original, defogged):
    mse = np.mean((original - defogged) ** 2)
    if mse == 0:
        return float('inf')  # 如果MSE为0，意味着图像完全相同
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


# 2. SSIM计算函数
def calculate_ssim(original, defogged):
    score, _ = ssim(original, defogged, full=True, channel_axis=-1)  # 使用multichannel=True确保处理多通道彩色图像
    return score


def read_image(image_path):
    # 读取图像并将其转换为RGB格式（cv2默认读取为BGR格式）
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


if __name__ == '__main__':
    original_folder_path = "D:/study/Senior Coureses/Digital Image Processing/2024 projects/reside2/gtpic/"  # 原始图像文件夹路径
    defogged_folder_path = "D:/study/Senior Coureses/Digital Image Processing/2024 projects/reside2/dcppic/"  # 去雾后图像文件夹路径
    imgs_path = os.listdir(original_folder_path)

    psnr_values = []
    ssim_values = []

    results_file = "D:/study/Senior Coureses/Digital Image Processing/2024 projects/output/results.txt"  # 存储结果的文本文件路径

    with open(results_file, 'w') as file:
        for img_name in imgs_path:
            original_img_path = os.path.join(original_folder_path, img_name)
            new_name=img_name
            defogged_img_path = os.path.join(defogged_folder_path, new_name)

            # 确保两个文件夹中都有同名图片
            if not os.path.exists(defogged_img_path):
                print(f"Defogged output_image not found for: {img_name}, skipping.")
                continue

            print(f"Processing {img_name}...")

            # 读取图片
            original_img = read_image(original_img_path)
            defogged_img = read_image(defogged_img_path)

            # 计算PSNR和SSIM
            psnr = calculate_psnr(original_img, defogged_img)
            ssim_value = calculate_ssim(original_img, defogged_img)
            psnr_values.append(psnr)
            ssim_values.append(ssim_value)

            # 写入每张图片的PSNR和SSIM值到文件
            file.write(f"Image: {img_name}\nPSNR: {psnr:.2f}\nSSIM: {ssim_value:.4f}\n\n")
            print(f"Image: {img_name} | PSNR: {psnr:.2f} | SSIM: {ssim_value:.4f}")

        # 计算平均PSNR和SSIM
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)

        # 将平均值写入文件
        file.write(f"\nAverage PSNR: {avg_psnr:.2f}\nAverage SSIM: {avg_ssim:.4f}\n")

    # 打印平均PSNR和SSIM
    print(f"\nAverage PSNR: {avg_psnr:.2f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
