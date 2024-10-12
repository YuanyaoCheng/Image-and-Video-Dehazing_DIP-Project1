
import os
import cv2
import torch
import torch.optim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from collections import OrderedDict
from skimage.metrics import structural_similarity as ssim  # 用于SSIM计算

import dehazeformer

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(0.9)  # 你可以根据需要调整该值

def single(save_dir):
    state_dict = torch.load(save_dir)['state_dict']
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    return new_state_dict

def calculate_psnr(original, defogged):
    mse = np.mean((original - defogged) ** 2)
    if mse == 0:
        return float('inf')  # 如果MSE为0，意味着图像完全相同
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(original, defogged):
    score, _ = ssim(original, defogged, full=True, multichannel=True)  # 使用multichannel=True确保处理多通道彩色图像
    return score

def dehaze_image1(image_name, new_dir):
    with torch.no_grad():

        data_hazy = Image.open(image_name)
        data_hazy = np.array(data_hazy) / 255.0
        original_img = data_hazy.copy()

        data_hazy = torch.from_numpy(data_hazy).float()
        data_hazy = data_hazy.permute(2, 0, 1)
        data_hazy = data_hazy.unsqueeze(0).to(device)

        dehaze_net = torch.load('/root/saved_models/dehaze_net_epoch_164.pth', map_location=device)
        dehaze_net = dehaze_net.cuda()  # 确保模型放在GPU上

        clean_image = dehaze_net(data_hazy).detach().cpu().numpy().squeeze()  # 在GPU上处理，并在返回时转移到CPU

        clean_image = np.swapaxes(clean_image, 0, 1)
        clean_image = np.swapaxes(clean_image, 1, 2)

        # plot the original and dehazing images
        plt.subplot(1, 2, 1)
        plt.imshow(original_img)
        plt.axis('off')
        plt.title('Original Image')

        plt.subplot(1, 2, 2)
        plt.imshow(clean_image)
        plt.axis('off')
        plt.title('Dehaze Image')
        plt.show()

        # store the output images.jpg
        clean_image = (clean_image * 255).astype(np.uint8)
        clean_image_pil = Image.fromarray(clean_image)
        clean_image_pil.save(new_dir, 'JPEG')

        return  original_img * 255, clean_image

if __name__ == '__main__':
    folder_path = "your datapath"
    imgs_path = os.listdir(folder_path)
    print(imgs_path)

    psnr_values = []
    ssim_values = []

    results_file = "DM/output_image/results.txt"

    with open(results_file, 'w') as file:
        for img_path in imgs_path:
            abs_path = folder_path + "/" + img_path
            new_path = "/root/" + img_path
            print("Old Image:" + abs_path + "\tTo" + "New Image:" + new_path)

            hazy_img, dehazed_img = dehaze_image1(abs_path, new_path)

            # PSNR SSIM
            psnr = calculate_psnr(hazy_img, dehazed_img)
            ssim_value = calculate_ssim(hazy_img, dehazed_img)
            psnr_values.append(psnr)
            ssim_values.append(ssim_value)

            file.write(f"Image: {img_path}\nPSNR: {psnr:.2f}\nSSIM: {ssim_value:.4f}\n\n")
            print(f"Image: {img_path} | PSNR: {psnr:.2f} | SSIM: {ssim_value:.4f}")

        # average PSNR and SSIM
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)

        file.write(f"\nAverage PSNR: {avg_psnr:.2f}\nAverage SSIM: {avg_ssim:.4f}\n")

    print(f"\nAverage PSNR: {avg_psnr:.2f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
