- [tfrecord](#tfrecord)
  - [tfrecord 创建流程](#tfrecord-创建流程)
  - [tfrecord 解析流程](#tfrecord-解析流程)
- [LMDB](#lmdb)
  - [LMDB的流程](#lmdb的流程)

```python
选择合适的数据格式提前保存，训练时加快数据读取。LMDB、tfrecord-pytorch、实时读取加载不同数据时间为。

使用猫狗数据集中的训练集2.5w张数据，测试两个epoch耗时。

record use total time :  80.9821 s

lmdb use total time :  33.8593 s

torch use total time :  122.5067 s
```

## tfrecord 
`tfrec` 格式的文件存储形式会很合理的帮我们存储数据，核心就是`tfrec`内部使用`Protocol Buffer`的二进制数据编码方案，
这个方案可以极大的压缩存储空间。

`tfrec`可以是**多个tf.train.Example**文件组成的序列 (每一个`example`是一个样本)

每一个`tf.train.Example`又是由若干个`tf.train.Features`字典组成。这个`Features`可以理解为**这个样本的一些信息**，如果
是图片样本，那么肯定有一个Features是图片像素值数据，一个Features是图片的标签值.

`tf.train.Feature` 只接收3种格式的数据, **BytesList ,Int64List ,FloatList**, 且必须以 **list** 的形式传进入。**value= [...]** 

### tfrecord 创建流程
1. 首先创建一个`writer`，也就是`TFrecord`生成器 

2. 创建存储类型`tf_feature`
   
3. 将 `tf_feature` 转换成 `tf_example` 以及进行序列化

4. 使用`writer` 保存序列化的样本

```python
import tensorflow as tf
import glob
# 先记录一下要保存的tfrec文件的名字
tfrecord_file = './train.tfrec'
# 获取指定目录的所有以jpeg结尾的文件list
images = glob.glob('./*.jpeg')
with tf.io.TFRecordWriter(tfrecord_file) as writer: # 创建writer
   for filename in images:
      image = open(filename, 'rb').read()  # 读取数据集图片到内存，image 为一个 Byte 类型的字符串
      
      # 建立 tf.train.Feature 字典
      feature = {  
          # 图片是一个 Bytes 对象
         'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])), 
         'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
         'float':tf.train.Feature(float_list=tf.train.FloatList(value=[1.0,2.0])),
         'name':tf.train.Feature(bytes_list=tf.train.BytesList(value=[str.encode(filename)]))
      }
      # 创建 tf_feature
      features=tf.train.Features(feature=feature)
      # tf.train.Example 进行序列化
      example = tf.train.Example(features=features)  
      # 保存example
      writer.write(example.SerializeToString())  
```


### tfrecord 解析流程
解析基本就是写入时的逆过程，所以会需要写入时对图像进行的操作。

1. 创建解析函数
2. 解析样本
2. 转变特征

```python
import tensorflow as tf
 
dataset = tf.data.TFRecordDataset('./train.tfrec')
 
def decode(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'float': tf.io.FixedLenFeature([1, 2], tf.float32),
        'name': tf.io.FixedLenFeature([], tf.string)
    }
    feature_dict = tf.io.parse_single_example(example, feature_description)
    feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'])  # 解码 JEPG 图片
    return feature_dict
 
dataset = dataset.map(decode).batch(4)
for i in dataset.take(1):
    print(i['image'].shape)
    print(i['label'].shape)
    print(i['float'].shape)
    print(bytes.decode(i['name'][0].numpy()))
```


## LMDB
Caffe使用LMDB来存放训练/测试用的数据集，以及使用网络提取出的feature(为了方便，以下还是统称数据集)。
数据集的结构很简单，就是大量的矩阵/向量数据平铺开来。数据之间没有什么关联，数据内没有复杂的对象结构，
就是向量和矩阵。既然数据并不复杂，**Caffe就选择了LMDB这个简单的数据库来存放数据。**

`LMDB`的全称是`Lightning Memory-Mapped Database`，闪电般的内存映射数据库。它文件结构简单，一个文件夹，

里面一个数据文件，一个锁文件。**数据随意复制，随意传输**。它的**访问简单，不需要运行单独的数据库管理进程**，

只要**在访问数据的代码里引用LMDB库，访问时给文件路径即可**。


让系统**访问大量小文件的开销很大**，而LMDB使用内存映射的方式访问文件，**使得文件内寻址的开销非常小**，
使用指针运算就能实现。数据库单文件**还能减少数据集复制/传输过程的开销**。

### LMDB的流程

1. env = lmdb.open()：创建 lmdb 环境
   
2. txn = env.begin()：建立事务
   
3. txn.put(key, value)：进行插入和修改
   
4. txn.delete(key)：进行删除
   
5. txn.get(key)：进行查询
   
6. txn.cursor()：进行遍历
   
7. txn.commit()：提交更改
8. 
put 和 delete 后一定注意要 commit ，不然根本没有存进去

每一次 commit 后，需要再定义一次 txn=env.begin(write=True







>tfrecord
>
>https://zhuanlan.zhihu.com/p/114982658

>LMDB
>
>https://blog.csdn.net/P_LarT/article/details/103208405
>
>https://www.cnblogs.com/zhangxianrong/p/14919706.html
>
> git clone https://github.com/xunge/pytorch_lmdb_imagenet

>pkl
>https://zhuanlan.zhihu.com/p/445935216

>DALI
>https://zhuanlan.zhihu.com/p/518240063
