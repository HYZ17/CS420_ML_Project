from torch.utils.data import Dataset
import os
import numpy as np


class MyDataset(Dataset):
    def __init__(self, data_dir:str, categories:list, mode:str):
        self.data_dir=data_dir
        self.categories=categories

        #设置数据集读取的模式，可取的值为'train','valid','test'
        self.mode=mode

        self.data=None
        self.label=None

        #保证序号唯一
        self.categories.sort()
        for index,category in enumerate(self.categories):
            path=os.path.join(data_dir,category+"_png.npz")
            png=np.load(path, encoding='latin1', allow_pickle=True)
            if self.data is None:
                self.data=np.expand_dims(png[mode],axis=1)
                self.label=index*np.ones(png[mode].shape[0],dtype=np.int)
            else:
                self.data=np.concatenate([self.data,np.expand_dims(png[mode],axis=1)])
                self.label=np.concatenate([self.label,index*np.ones(png[mode].shape[0],dtype=np.int)])

    def __getitem__(self, item):
        return self.data[item],self.label[item]

    def __len__(self):
        return self.data.shape[0]









