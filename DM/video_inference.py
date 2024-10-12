import cv2
import numpy as np
import torch
import os

from tqdm import tqdm

torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(0.9)  # 你可以根据需要调整该值
# 定义 dehaze_image1 函数
def dehaze_image1(batch_frames):
    with torch.no_grad():
        batch_size = len(batch_frames)
        frames = np.stack(batch_frames, axis=0) / 255.0
        data_hazy = torch.from_numpy(frames).float()
        data_hazy = data_hazy.permute(0, 3, 1, 2)
        data_hazy = data_hazy.to(device)

        # 加载模型，并放入GPU中
        dehaze_net = torch.load('/root/saved_models/dehaze_net_epoch_11.pth', map_location=device)
        dehaze_net = dehaze_net.cuda()

        # 去雾操作
        clean_images = dehaze_net(data_hazy).detach().cpu().numpy()  # 转移到CPU
        clean_images = np.clip(clean_images, 0, 1) * 255  # 转换回0-255的范围
        clean_images = clean_images.transpose(0, 2, 3, 1).astype(
            np.uint8)  # 转换为原来的维度顺序 (batch_size, height, width, channels)

        return clean_images


# 视频逐帧处理和保存函数
def process_video(input_video_path, output_video_path, dehaze_func, batch_size=16):

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return

    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_num = 0
    frame_batch = []


    with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break


            frame_batch.append(frame)
            frame_num += 1


            if len(frame_batch) == batch_size or frame_num == total_frames:

                dehazed_batch = dehaze_func(frame_batch)


                for dehazed_frame in dehazed_batch:
                    out.write(dehazed_frame)


                pbar.update(len(frame_batch))


                frame_batch = []


    cap.release()
    out.release()


    cap.release()
    out.release()

    print(f"Videos have been dehazed, the result is in {output_video_path}")


# 主要流程
if __name__ == "__main__":
    # 设定设备
    input_video_folder_path = "put the video folder path here"
    output_video_folder_path = "put the video output folder path here"

    for video_name in os.listdir(input_video_folder_path):
        print(video_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 输入输出视频路径
        input_video_path = input_video_folder_path + "/" + video_name
        output_video_path = output_video_folder_path + "/" + video_name

        # 创建输出文件夹（如果不存在）
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

        # 处理视频，使用批处理模式
        process_video(input_video_path, output_video_path, dehaze_image1, batch_size=16)

