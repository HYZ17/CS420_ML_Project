## CS420_ML_Project

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
* 添加一个新的分支，同时使用了resnet和rnn，效果明显提升，截至目前准确度83.8%（5.14）
* 修复bug：模型参数保存不完整（5.14）

#### To Do
* 在已有的基础上，写一个只使用了LSTM，输入序列信息的模型，这样就可以对比CNN，LSTM，CNN+LSTM三个的性能
* 调试模型的参数，充分发挥每一个模型的性能
* 运行代码，搜集三个模型对应的最优checkpoint，搜集它们的log；使用搜集的log进行画图（正则表达式提取数据），将三个模型的loss，acc画在一起，方便report中进行分析
* 把三个模型整合到main branch中，方便老师查看（完全可行，使用一个data loader，定义多个model即可）
* 报告撰写（三个人一起写）
