# -*- coding: utf-8 -*-
# @Author : xyoung
# @Time : 22:59  2023-05-17
import numpy as np
import tfrecord
from tfrecord.torch.dataset import TFRecordDataset
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import cv2
from PIL import Image
import io
import time

HEIGHT = 320
WIDTH = 320

description = {
    'image': "byte",
    'label': "int",
    'index': "int",
    "filename": "byte"
}

def saveTfcord(datadir, savedir, number=20000):
    imagesList = os.listdir(datadir)
    num_lenth = len(imagesList)
    print(num_lenth)
    number = num_lenth
    file_nums = round(num_lenth / number)

    count = 0
    for i in range(file_nums):
        record_file = os.path.join(savedir, str(i).zfill(5) + ".tfrecord")
        writer = tfrecord.TFRecordWriter(record_file)

        startIdx = i * number
        endIdx = (i + 1) * number

        fileList = imagesList[startIdx:endIdx]

        for idx, filename in tqdm(enumerate(fileList)):
            img = cv2.imread(os.path.join(datadir, filename))
            img2 = cv2.resize(img, (HEIGHT, WIDTH))

            bytes_io = io.BytesIO()
            img2 = Image.fromarray(img2)
            img2.save(bytes_io, format='JPEG')
            encoded_jpg = bytes_io.getvalue()

            feature = {
                "image": (encoded_jpg, "byte"),
                "label": (1, "int"),
                "index": (count, "int"),
                "filename": (str.encode(filename), "byte")
            }

            writer.write(feature)
            count += 1
        writer.close()
    return record_file


def decode_image(features):
    # get BGR image from bytes
    features[0] = cv2.imdecode(np.frombuffer(features["image"], np.int8), -1)
    # features["image"] = features["image"]
    features[1] = features["label"]
    features[2] = features["filename"]
    return features


if __name__ == '__main__':

    savedir = "./trcord-torch"
    if not os.path.exists(savedir): os.makedirs(savedir, exist_ok=True)
    dataset = "./train"
    imageList = os.listdir(dataset)
    saveTfcord(dataset, savedir)

    index_path = None
    batch_size = 512
    recordsList = os.listdir(savedir)
    tfrecord_path = [os.path.join(savedir, file) for file in recordsList]

    dataset = TFRecordDataset(tfrecord_path[0], index_path, description, transform=decode_image)
    trainLoader = DataLoader(dataset, num_workers=1, batch_size=batch_size)

    startTime = time.time()
    for i in range(3):
        for idx, record in enumerate(trainLoader):
            print(i, idx, record["image"].shape)

    endTime = time.time()
    print("per batch time :  %.4f s" % ((endTime - startTime)))
