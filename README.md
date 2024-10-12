# DIP Project 1 -- Image and Video Dehazing 


------------

**This project is for digital and signal processing course. **The task is to find method to dehaze the original images in foler /IEI2019 and videos in folder /IEV2022.

In this project, we used both traditional method and deep-learning based method.

**Traditional methods: **Dark Channel Prior, Retinex, Homorphic Filtering
**Deep learning method: **AODNet, Dehazeformer

Thanks to Song Cheng, Zhang Huaijin, Zhou Xingyu, Cheng Yuanyao
For any details, you can contact: [yuanyao_cheng[]std.uestc.edu.cn]

------------
## 1. Deep Learning Method
The code is put in folder **/DM**.  
### Inference
For simple inference, you should first create a folder, and put your images into the folder.
And before you run the **Inference.py** ,you should change the following code:
```python
    folder_path = "change into your datapath"
```
To see the outcomes directly, we have put some results in the following folders:
/DM/output_image: The dehazing images
/DM/visualizations: The output of some images in every epoch
/DM/indicators: The loss value in train and PSNR&SSIM value in validation.

### train
You should put your dataset as follows:
┬─ DM
│   ├─** dataset**
│   │   ├─ **gt**
│   │   │   └─ 0001_GT.png
│   │   │   ├─ 0002_GT.png
│   │   │   ├─ .........
│   │   └─** hazy**
│   │   │   └─ 0001_hazy.png
│   │   │   ├─ 0002_hazy.png
│   │   │   ├─ .........
│   └─ train.py
│   └─ .......
└─ TM

There are also some useful datasets, you can find them as follows:
[D-HAZY: A dataset to evaluate quantitatively dehazing algorithms](http://https://ieeexplore.ieee.org/document/7532754 "D-HAZY: A dataset to evaluate quantitatively dehazing algorithms") 
[RESIDE: A Benchmark for Single Image Dehazing](https://sites.google.com/view/reside-dehaze-datasets/reside-standard "RESIDE: A Benchmark for Single Image Dehazing")

You can choose the loss function (L1loss, MSE, SSIM) in train.py:
```python
# loss = criterion_l1(clean_image, img_orig)
# loss = criterion_mse(clean_image, img_orig)
loss = 1 - ssim(clean_image, img_orig, data_range=1, size_average=True)
```
Basic settings are in the end of the train.py:
```python
    hazy_images_path = "Your dataset path"
    batch_size = 4
    epochs = 200
    NUM_WORKERS = 16
```

## 2. Traditional Method
This method is in /TM. It consists of five files, detailed as follows:


- **DCP.py**: This file contains the algorithm for image dehazing using the Dark Channel Prior (DCP) method.
- **homomorphic_filter.py**: This file includes the algorithm for homomorphic filtering to process images.
- **evaluate.py**: This file contains methods for evaluating the quality of dehazed images.
- **File video dehazing.py**: This file holds the algorithm for video dehazing using the Dark Channel Prior method.
- **File video dehazing iterating.py**: This file presents an algorithm for video dehazing using an iterative approach with the Dark Channel Prior.
