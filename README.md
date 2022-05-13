#### Dataset

* 原始数据集可以通过canvas链接直接下载，包含了时间等更多的信息

* 使用https://github.com/CMACH508/RPCL-pix2seq/blob/main/seq2png.py 提取的png图片数据集（经过了压缩处理）可以通过https://jbox.sjtu.edu.cn/l/X1XSjP 进行下载

* 如何使用：参考https://github.com/CMACH508/RPCL-pix2seq/blob/87d31fb072a0295707993f32953cd3dbfb82042c/utils.py#L182

  

#### Environment

```
python=3.8
torch
torchvision
tqdm
```



#### Update

* 添加了data loader和Resnet分类模型（5.12）

* 使用pre-trained模型，对输入的图片进行归一化，准确率80.5%（5.13）
