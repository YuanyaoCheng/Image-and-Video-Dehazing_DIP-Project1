import cv2
import numpy as np
import math
import sys
import os
import time
time_start = time.perf_counter()  # 记录开始时间



def DarkChannel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark


def AtmLight(im, dark):
    [h, w] = im.shape[:2]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark.reshape(imsz);
    imvec = im.reshape(imsz, 3);

    indices = darkvec.argsort();
    indices = indices[imsz - numpx::]

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A


def TransmissionEstimate(im, A, sz):
    omega = 0.95;
    im3 = np.empty(im.shape, im.dtype);

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]

    transmission = 1 - omega * DarkChannel(im3, sz);
    return transmission


def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r));
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r));
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r));
    cov_Ip = mean_Ip - mean_I * mean_p;

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r));
    var_I = mean_II - mean_I * mean_I;

    a = cov_Ip / (var_I + eps);
    b = mean_p - a * mean_I;

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r));
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r));

    q = mean_a * im + mean_b;
    return q;


def TransmissionRefine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray) / 255;
    r = 60;
    eps = 0.0001;
    t = Guidedfilter(gray, et, r, eps);

    return t;


def Recover(im, t, A, tx=0.1):
    res = np.empty(im.shape, im.dtype);
    t = cv2.max(t, tx);

    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]

    return res


def homomorphic_filter(src, d0=10, rl=0.5, rh=2.0, c=4, h=2.0, l=0.5):
    gray = src.copy()
    # if len(src.shape) > 2:
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


video = cv2.VideoCapture('D:/study/Senior Coureses/Digital Image Processing/2024 projects/airplane.mp4')
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(video.get(cv2.CAP_PROP_FPS))
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_size = (frame_width, frame_height)
out = cv2.VideoWriter('D:/study/Senior Coureses/Digital Image Processing/2024 projects/output/output_video_fast.mp4', fourcc, fps, frame_size)


I_accum=0
A_mean=0.0

for i in range(frame_count):
    ret, frame = video.read()
    if not ret:
        break

    # cv2.imwrite("D:/study/Senior Coureses/Digital Image Processing/2024 projects/output/output_pic1.jpg",frame)
    # img=cv2.imread("D:/study/Senior Coureses/Digital Image Processing/2024 projects/output/output_pic1.jpg")
    img = frame
    I = img.astype('float64') / 255
    dark = DarkChannel(I, 15)
    A = AtmLight(I, dark)
    A_mean=(A_mean*i+A)/(i+1)
    te = TransmissionEstimate(I, A_mean, 15)
    t = TransmissionRefine(img, te)
    J = Recover(I, t, A_mean, 0.1)
    new_frame = J * 255
    cv2.imwrite("D:/study/Senior Coureses/Digital Image Processing/2024 projects/output/output_pic2.jpg", new_frame)
    new_frame = cv2.imread("D:/study/Senior Coureses/Digital Image Processing/2024 projects/output/output_pic2.jpg")
    out.write(new_frame)
    print(f"进度: {i+1} 总帧数: {frame_count}")

video.release()
out.release()
cv2.destroyAllWindows()

time_end = time.perf_counter()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print(time_sum)






