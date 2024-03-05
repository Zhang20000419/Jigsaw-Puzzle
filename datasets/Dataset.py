import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import pandas as pd


class JigsawDataset2(Dataset):

    def __init__(self, image_dir, csv_path):
        # image_dir是一个文件夹路径，存放了所有的图片文件
        # 读取文件夹中的所有图片文件名，并保存到一个列表中
        # 打印当前路径
        self.image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
        self.data_csv = pd.read_csv(csv_path)
        self.data_csv.drop(['Unnamed: 0'], axis=1, inplace=True)
        self.labels = self.data_csv.iloc[:, 1]

    def __len__(self):
        # 返回数据集的长度，即图片文件的数量
        return len(self.image_files)

    def __getitem__(self, index):
        # 根据索引返回一张图片和对应的排列
        # 读取图片文件，并转换为PIL.Image对象
        img = Image.open(self.image_files[index]).convert('RGB')
        # 将图片转换
        image = torch.from_numpy(np.array(img)).float()
        # 将图片分割成2*2的切片
        tiles = [image[x:x + 100, y:y + 100] for x in range(0, image.shape[0], 100) for y in
                 range(0, image.shape[1], 100)]
        # 通道，长，宽
        tiles = torch.tensor(np.array(tiles)).permute(0, 3, 1, 2)
        # label
        label = np.array([float(self.labels[index][0]), float(self.labels[index][2]), float(self.labels[index][4]),
                          float(self.labels[index][6])])
        label = torch.from_numpy(label).long()
        return tiles, label

