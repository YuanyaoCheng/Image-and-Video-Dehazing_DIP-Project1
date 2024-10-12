import os

import cv2
import math
import numpy as np

def homomorphic_filter(src, d0=10, rl=0.5, rh=2.0, c=4, h=2.0, l=0.5):
    gray = src.copy()
    #if len(src.shape) > 2:
    #    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)  # 转换成灰度图像

    gray = np.log(1e-5 + gray)  # 取对数

    rows, cols = gray.shape
    gray_fft = np.fft.fft2(gray)  # FFT傅里叶变换
    gray_fftshift = np.fft.fftshift(gray_fft)  # FFT中心化

    M, N = np.meshgrid(np.arange(-cols // 2, cols // 2), np.arange(-rows // 2, rows // 2))
    D = np.sqrt(M ** 2 + N ** 2)  # 计算距离
    Z = (rh - rl) * (1 - np.exp(-c * (D ** 2 / d0 ** 2))) + rl  # H(u,v)传输函数

    dst_fftshift = Z * gray_fftshift
    dst_fftshift = (h - l) * dst_fftshift + l
    dst_ifftshift = np.fft.ifftshift(dst_fftshift)

    dst_ifft = np.fft.ifft2(dst_ifftshift)  # IFFT逆傅里叶变换
    dst = np.real(dst_ifft)  # IFFT取实部

    dst = np.exp(dst) - 1  # 还原
    dst = np.uint8(np.clip(dst, 0, 255))
    return dst


folder_path='D:/study/Senior Coureses/Digital Image Processing/2024 projects/reside3/hazy/'
imgs=os.listdir(folder_path)

for imgname in imgs:
    img = cv2.imread(folder_path+imgname)
    (B, G, R) = cv2.split(img)  # 提取R、G、B分量
    # R、G、B的合并
    merged = cv2.merge([B, G, R])  # 合并R、G、B分量

    img_new = cv2.merge([homomorphic_filter(B), homomorphic_filter(G), homomorphic_filter(R)])

    new_path='D:/study/Senior Coureses/Digital Image Processing/2024 projects/reside3/trained/new_'+imgname
    cv2.imwrite(new_path, img_new)
    print(imgname)


key = cv2.waitKey(0)
cv2.destroyAllWindows()