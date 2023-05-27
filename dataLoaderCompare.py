
# -*- coding: utf-8 -*-
# @Author : xyoung
# @Time : 21:10  2023-05-18

from torchloader import torchImageLoader
from tfrecordLoader import saveTfcord, description, decode_image
from lmdbRead import savelmdb, lmdbLoader
import os
import torch
import tfrecord
from tfrecord.torch.dataset import TFRecordDataset
import time
from torch.utils.data import DataLoader

if __name__ == '__main__':
    datadir = "./train"
    savedir = "compare_result"
    if not os.path.exists(savedir): os.makedirs(savedir, exist_ok=True)
    record_filename = saveTfcord(datadir, savedir)
    lmdb_filename = savelmdb(datadir, savedir)

    # record_filename = "./compare_result/00000.tfrecord"
    # lmdb_filename = "./compare_result/trian_lmdb.lmdb"
    print("record name ", record_filename)

    print("lmdb name ", lmdb_filename)
    batch_size = 512
    dataset_torch = torchImageLoader(datadir)
    datasetLoader_torch = torch.utils.data.DataLoader(dataset_torch, num_workers=2,
                                                      batch_size=batch_size, shuffle=True)

    dataset_record = TFRecordDataset(record_filename, None, description, transform=decode_image)
    datasetLoader_record = torch.utils.data.DataLoader(dataset_record, num_workers=1, batch_size=batch_size)

    dataset_lmdb = lmdbLoader(lmdb_filename)
    datasetLoader_lmdb = torch.utils.data.DataLoader(dataset_lmdb, num_workers=0, batch_size=batch_size)

    epoch = 2
    names = ["record", "lmdb", "torch"]
    loaderlist = [datasetLoader_record, datasetLoader_lmdb, datasetLoader_torch]
    for name, dataloader in zip(names, loaderlist):
        startTime = time.time()
        for i in range(epoch):
            print(i, " epoch")
            for idx, data in enumerate(dataloader):
                # if(idx%5==0):
                    # print(i, " epoch", idx, data[1])
                pass
        endTime = time.time()
        print("%s use total time :  %.4f s" % (name, (endTime - startTime)))
