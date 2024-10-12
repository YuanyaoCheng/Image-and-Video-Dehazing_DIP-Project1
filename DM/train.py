import os
import numpy as np
import torch
import torch.optim
import torch.nn as nn
from PIL import Image

from AODNet import AODNet
import dehazeformer

from data import MyDataset
from utils import weights_init
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from pytorch_msssim import ssim

from skimage.metrics import peak_signal_noise_ratio as psnr_metric

torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(0.9)

# 创建保存文件夹
if not os.path.exists("indicator"):
    os.makedirs("indicator")

def save_to_txt(data, filename):
    with open(filename, 'w') as f:
        for value in data:
            f.write(f"{value}\n")

def plot_metric(data, title, ylabel, filename):
    plt.figure()
    plt.plot(data)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.savefig(f"indicator/{filename}")
    plt.close()

def train(orig_images_path, hazy_images_path, batch_size, epochs):
    # devide into the train and val dataset 7:3
    dataset = MyDataset(orig_images_path, hazy_images_path, mode='train')
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoader for train and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)


    # dehaze_net = dehazeformer.dehazeformer_b().cuda()
    dehaze_net = torch.load("/root/saved_models/dehaze_net_epoch_100.pth")
    # dehaze_net = AODNet().cuda()
    dehaze_net.apply(weights_init)
    criterion_mse = nn.MSELoss().cuda()
    criterion_l1 = nn.L1Loss().cuda()
    optimizer = torch.optim.Adam(dehaze_net.parameters(), lr=1e-3, weight_decay=1e-4)


    train_losses = []
    val_psnrs = []
    val_ssims = []

    for epoch in range(epochs):
        dehaze_net.train()
        epoch_loss = 0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for iteration, (img_orig, img_haze) in enumerate(train_loader_tqdm):
            img_orig = img_orig.cuda()
            img_haze = img_haze.cuda()
            clean_image = dehaze_net(img_haze)

            # loss = criterion_l1(clean_image, img_orig)
            # loss = criterion_mse(clean_image, img_orig)
            loss = 1 - ssim(clean_image, img_orig, data_range=1, size_average=True)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dehaze_net.parameters(), 0.1)
            optimizer.step()

            epoch_loss += loss.item()
            train_loader_tqdm.set_postfix(loss=epoch_loss / (iteration + 1))

        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)

        # save the model
        torch.save(dehaze_net, f'./saved_models/dehaze_net_epoch_{epoch + 1}.pth')

        # validation
        val_psnr, val_ssim = validate(dehaze_net, val_loader, epoch)
        val_psnrs.append(val_psnr)
        val_ssims.append(val_ssim)

        # save the loss and other indicators
        save_to_txt(train_losses, "indicator/train_loss.txt")
        save_to_txt(val_psnrs, "indicator/val_psnr.txt")
        save_to_txt(val_ssims, "indicator/val_ssim.txt")

        # plot the loss and other indicators
        plot_metric(train_losses, "Training Loss", "Loss", "train_loss.png")
        plot_metric(val_psnrs, "Validation PSNR", "PSNR", "val_psnr.png")
        plot_metric(val_ssims, "Validation SSIM", "SSIM", "val_ssim.png")

def validate(dehaze_net, val_loader, epoch):
    dehaze_net.eval()
    psnr_total = 0
    ssim_total = 0
    sample_images = []

    with torch.no_grad():
        val_loader_tqdm = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}")
        for iteration, (img_orig, img_haze) in enumerate(val_loader_tqdm):
            img_orig = img_orig.cuda()
            img_haze = img_haze.cuda()
            clean_image = dehaze_net(img_haze)

            img_orig_np = img_orig.cpu().numpy()
            clean_image_np = clean_image.cpu().numpy()
            psnr = psnr_metric(img_orig_np, clean_image_np, data_range=1.0)
            ssim_val = ssim(clean_image, img_orig, data_range=1, size_average=True).item()

            psnr_total += psnr
            ssim_total += ssim_val

            if iteration < 5:
                sample_images.append((img_haze.cpu(), clean_image.cpu(), img_orig.cpu()))

            val_loader_tqdm.set_postfix(psnr=psnr, ssim=ssim_val)

    avg_psnr = psnr_total / len(val_loader)
    avg_ssim = ssim_total / len(val_loader)

    visualize_images(sample_images, epoch)
    return avg_psnr, avg_ssim

def visualize_images(sample_images, epoch):
## This function is to visualize the output of each epoch

    fig, axs = plt.subplots(3, 3, figsize=(15, 3 * len(sample_images)))

    for i, (img_haze, clean_image, img_orig) in enumerate(sample_images):
        if i >= 3:
            break
        img_haze_np = (img_haze[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        clean_image_np = (clean_image[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_orig_np = (img_orig[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # Convert numpy arrays to PIL Images
        img_haze_pil = Image.fromarray(img_haze_np)
        clean_image_pil = Image.fromarray(clean_image_np)
        img_orig_pil = Image.fromarray(img_orig_np)

        # Resize images to 640x480
        img_haze_resized = img_haze_pil.resize((640, 480))
        clean_image_resized = clean_image_pil.resize((640, 480))
        img_orig_resized = img_orig_pil.resize((640, 480))

        # Plotting the resized images
        axs[i, 0].imshow(img_haze_resized, cmap='gray')
        axs[i, 0].set_title('Hazy Image')
        axs[i, 0].axis('off')

        axs[i, 1].imshow(clean_image_resized, cmap='gray')
        axs[i, 1].set_title('Dehazed Image')
        axs[i, 1].axis('off')

        axs[i, 2].imshow(img_orig_resized, cmap='gray')
        axs[i, 2].set_title('Original Image')
        axs[i, 2].axis('off')


    plt.tight_layout()
    plt.savefig(f'./visualizations/epoch_{epoch + 1}_visualization.png')
    plt.show()

if __name__ == '__main__':
    orig_images_path = "Your dataset path"
    hazy_images_path = "Your dataset path"
    batch_size = 4
    epochs = 200
    NUM_WORKERS = 16
    train(orig_images_path, hazy_images_path, batch_size, epochs)
