import torch
from torch.utils.data import Dataset
import os
import numpy as np


class MyDataset(Dataset):
    def __init__(self, png_dir:str, seq_dir:str,categories:list, mode:str,max_len=100):
        self.png_dir=png_dir
        self.categories=categories
        self.seq_dir=seq_dir
        #设置数据集读取的模式，可取的值为'train','valid','test'
        self.mode=mode

        self.data=None
        self.label=None
        self.seq=None
        self.seq_len=None
        self.max_len=max_len
        #保证序号唯一
        self.categories.sort()
        for index,category in enumerate(self.categories):
            png_path=os.path.join(png_dir,category+"_png.npz")
            seq_path=os.path.join(seq_dir,"sketchrnn_"+category+".npz")
            seq=np.load(seq_path, encoding='latin1', allow_pickle=True)
            png=np.load(png_path, encoding='latin1', allow_pickle=True)

            seq_list=[]
            len_list=[]
            for item in seq[mode]:
                length=item.shape[0]
                if length>=self.max_len:
                    seq_list.append(item[:self.max_len,:])
                    len_list.append(self.max_len)
                else:
                    seq_list.append(np.pad(item,((0, self.max_len-length),(0, 0)),"constant"))
                    len_list.append(length)

            seq_list=np.array(seq_list,dtype=np.int16)
            len_list=np.array(len_list,dtype=np.int16)

            if self.data is None:
                self.data=np.expand_dims(png[mode],axis=1)
                self.label=index*np.ones(png[mode].shape[0],dtype=np.int)
                self.seq=seq_list
                self.seq_len=len_list
            else:
                self.data=np.concatenate([self.data,np.expand_dims(png[mode],axis=1)])
                self.label=np.concatenate([self.label,index*np.ones(png[mode].shape[0],dtype=np.int)])
                self.seq=np.concatenate([self.seq,seq_list],axis=0)
                self.seq_len=np.concatenate([self.seq_len,len_list])

    def __getitem__(self, item):
        return self.data[item],self.seq[item],self.seq_len[item],self.label[item]

    def __len__(self):
        return self.data.shape[0]








