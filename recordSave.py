# -*- coding: utf-8 -*-
# @Author : xyoung
# @Time : 22:59  2023-05-17
import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import glob
import cv2
import io
from PIL import Image

HEIGHT = 320
WIDTH = 320


def saveTfcord(images, savedir, number=10000):

    num_lenth = len(images)
    file_nums = round(num_lenth / number)
    for i in range(file_nums):
        record_file = os.path.join(savedir, str(i).zfill(4) + ".tfrecord")
        writer = tf.io.TFRecordWriter(record_file)

        startIdx = i * number
        endIdx = (i + 1) * number

        fileList = images[startIdx:endIdx]

        for idx, filename in tqdm(enumerate(fileList)):
            img = cv2.imread(os.path.join(dataset, filename))
            img2 = cv2.resize(img, (HEIGHT, WIDTH))
            bytes_io = io.BytesIO()

            img2 = Image.fromarray(img2)
            img2.save(bytes_io, format='JPEG')
            encoded_jpg = bytes_io.getvalue()

            # imgbite = bytes(img)

            # image = open(os.path.join(dataset, filename), 'rb').read()

            # image2 = img2.to_sring()
            feature = {
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
                # 'float': tf.train.Feature(float_list=tf.train.FloatList(value=[1.0, 2.0])),
                'name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str.encode(filename)]))
            }

            features = tf.train.Features(feature=feature)
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())
        writer.close()


def map_func(example):
    # feature 的属性解析表
    feature_map = {
        'image': tf.io.FixedLenFeature((), tf.string),
        'label': tf.io.FixedLenFeature((), tf.int64),
        # 'float': tf.FixedLenFeature((), tf.float32),
        'name': tf.io.FixedLenFeature((), tf.string)
    }
    parsed_example = tf.parse_single_example(example, features=feature_map)

    image = tf.io.decode_jpeg(parsed_example["image"])
    label = parsed_example["label"]
    # parsedfloat = parsed_example["float"]
    filename = parsed_example["name"]

    return image, label, filename


def cordDecode(savedir):
    recordsList = os.listdir(savedir)
    recordsList = [os.path.join(savedir, file) for file in recordsList]
    dataset = tf.data.TFRecordDataset(recordsList)
    dataset = dataset.map(map_func=map_func)
    iterator = dataset.make_one_shot_iterator()
    element = iterator.get_next()

    count = 0
    with tf.Session() as sess:
        while (1):
            try:
                image, label, filename = sess.run(element)
                count += 1
                print(count, image.shape, label, filename)
            except tf.errors.OutOfRangeError:
                print("end")
                break


if __name__ == '__main__':
    savedir = "./trcord"
    if not os.path.exists(savedir): os.makedirs(savedir, exist_ok=True)
    dataset = "./train"
    imageList = os.listdir(dataset)
    # saveTfcord(imageList, savedir)
    cordDecode(savedir)
