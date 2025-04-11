import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import glob
import pandas as pd
import torch
import cv2
from torch.utils.data.dataset import Dataset
import numpy as np
from skimage.io import imsave, imread
import matplotlib.pyplot as plt


class MyDataset(Dataset):
    def __init__(self, csv_path, num_primary_color, mode=None):
        self.path = csv_path
        if mode == "train":
            self.img_paths = np.array(pd.read_csv(csv_path, header=None)).reshape(-1)[:-26771]
        if mode == "val":
            self.img_paths = np.array(pd.read_csv(csv_path, header=None)).reshape(-1)[-50:]
        if mode == "test":
            self.img_paths = np.array(pd.read_csv(csv_path, header=None)).reshape(-1)[-10:]
        self.num_primary_color = num_primary_color

    def __getitem__(self, index):
        im = imread(self.img_paths[index])
        # (256, 256, 3)
        im = im.transpose((2, 0, 1))    # (3, 256, 256)
        # 归一化
        target_img = im/255
        # print(f"index:{index}")
        target_img = torch.from_numpy(target_img.astype(np.float32))
        # torch.Size([3, 256, 256])
        return target_img

    def __len__(self):
        return len(self.img_paths)

    def make_primary_color_layer(self, palette_values):
        primary_color = palette_values / 255
        return primary_color

# if __name__ == '__main__':
#     csv_path = 'sample.csv'
#     num_primary_color = 7
#     train_dataset = MyDataset2(csv_path, num_primary_color, mode='train')
#     img, color = train_dataset[5]
#     print(color)
#     # print(img)
#     img = img.numpy()
#     img = img.transpose(1, 2, 0)
#     plt.imshow(img)
#     plt.show()
