## CS420_ML_Project

#### Branch
* 本项目一共包含三个branch，其中main分支包含了全部四个模型（CNN,RNN,Resnet,Resnet+RNN)的代码（默认调用Resnet，如果需要使用其他模型，可能需要修改main.py与dataloader.py)
* Resnet+RNN分支包含了Resnet+RNN模型，切换到这个分支下，可以直接运行Resnet+RNN这个模型
* Sketch-R2CNN分支下包含了实验性的R2CNN模型代码，可以切换到这个分支可以直接运行R2CNN模型

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

