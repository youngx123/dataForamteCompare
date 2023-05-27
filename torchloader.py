# -*- coding: utf-8 -*-
# @Author : xyoung
# @Time : 21:00  2023-05-17
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import cv2
import time

HEIGHT = 320
WIDTH = 320
class torchImageLoader(Dataset):
    def __init__(self, datafolder):
        super(torchImageLoader, self).__init__()
        self.datafolder = datafolder
        self.labelDict = {
            "cat": 0,
            "dog": 1
        }
        self.imgList = os.listdir(self.datafolder)
    
    def __len__(self):
        # print(len(self.imgList))
        return len(self.imgList)
    
    def __getitem__(self, item):
        filename = self.imgList[item]
        name = filename[:3]
        label = self.labelDict[name]
        
        data = cv2.imread(os.path.join(self.datafolder, filename))
        data = cv2.resize(data, (HEIGHT, WIDTH))
        data = data[...,[2,0,1]]
        data = torch.from_numpy(data)
        label = torch.from_numpy(np.array(label))
        
        return data, label


if __name__ == '__main__':
    datafolder = "./train"
    dataset = torchImageLoader(datafolder)
    
    trainLoader = DataLoader(dataset, num_workers=2,batch_size=512, shuffle=True)
    
    startTime = time.time()
    for i in range(3):
        for idx, data in enumerate(trainLoader):
            print(idx)
    endTime = time.time()
    print("per batch time :  %.4f s" % ((endTime-startTime)))