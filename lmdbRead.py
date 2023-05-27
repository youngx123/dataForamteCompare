# -*- coding: utf-8 -*-
# @Author : xyoung
# @Time : 21:00  2023-05-18

import os
from PIL import Image
import pickle
import numpy as np
import lmdb
import cv2
import io
from torch.utils.data import Dataset
from tqdm import tqdm
import six
import torch
import pyarrow as pa
import msgpack

HEIGHT = 320
WIDTH = 320


def dumps_data(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pickle.dumps(obj)
    # return pa.serialize(obj).to_buffer()

def loads_data(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pickle.loads(buf)

class lmdbLoader(Dataset):
    def __init__(self, lmdb_path):
        super().__init__()
        self.db_path = lmdb_path
        self.env = lmdb.open(lmdb_path, subdir=os.path.isdir(lmdb_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_data(txn.get(b'__len__'))
            self.keys = loads_data(txn.get(b'__keys__'))

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

    def __getitem__(self, item):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[item])

        unpacked = loads_data(byteflow)

        imgbuf = unpacked[0]
        # buf = six.BytesIO()
        # buf.write(imgbuf)
        # buf.seek(0)
        # img = Image.open(buf).convert('RGB')

        label = unpacked[1]

        return np.array(imgbuf), label


def savelmdb(datadir, savedir, write_frequency=5000):
    fileNames = os.listdir(datadir)
    print("Loading dataset from %s" % datadir)

    lmdb_path = os.path.join(savedir, "trian_lmdb.lmdb")
    isdir = os.path.isdir(lmdb_path)
    # create lmdb environment

    img_file = os.path.join(datadir, fileNames[0])
    img = cv2.imread(img_file)
    img2 = cv2.resize(img, (HEIGHT, WIDTH))
    _, img_byte = cv2.imencode(
        '.png', img2, [cv2.IMWRITE_PNG_COMPRESSION, 80])
    data_size_per_img = img_byte.nbytes
    print('Data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len(fileNames)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=9511627776,readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)
    for idx, file in tqdm(enumerate(fileNames)):
        img_file = os.path.join(datadir, file)
        img = cv2.imread(img_file)
        img2 = cv2.resize(img, (HEIGHT, WIDTH))

        # bytes_io = io.BytesIO()

        # img2 = Image.fromarray(img2)
        # img2.save(bytes_io, format='JPEG')
        # encoded_jpg = bytes_io.getvalue()

        label = 1 #[1.0, 2.0]
        txn.put(u'{}'.format(idx).encode('ascii'), dumps_data((img2, label)))
        if idx % write_frequency == 0:
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_data(keys))
        txn.put(b'__len__', dumps_data(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()
    return lmdb_path


def readLMDB(lmdb_path):
    env = lmdb.open(lmdb_path, subdir=os.path.isdir(lmdb_path),
                    readonly=True, lock=False,
                    readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        length = pickle.loads(txn.get(b'__len__'))
        keys = pickle.loads(txn.get(b'__keys__'))

    with env.begin(write=False) as txn:
        byteflow = txn.get(keys[0])

    unpacked = pickle.loads(byteflow)


if __name__ == '__main__':
    savedir = "./lmdb-torch"
    if not os.path.exists(savedir): os.makedirs(savedir, exist_ok=True)
    dataset = "./train"
    imageList = os.listdir(dataset)
    # savelmdb(dataset, savedir)

    # readLMDB("./lmdb-torch/trian-lmdb.lmdb")

    dataset_lmdb = lmdbLoader("./lmdb-torch/trian_lmdb.lmdb")
    datasetLoader_lmdb = torch.utils.data.DataLoader(dataset_lmdb, num_workers=0,
                                                     batch_size=1)
    for data in (datasetLoader_lmdb):
        # data = ge(datasetLoader_lmdb)
        print(data[0].shape, data[1])
