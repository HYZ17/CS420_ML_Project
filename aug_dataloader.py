from torch.utils.data import Dataset
import os
import numpy as np
import imgaug.augmenters as iaa
import imgaug as ia
ia.seed(1)
seq1 = iaa.Sequential([
    #iaa.Fliplr(0.5), # horizontal flips
    #iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    #iaa.Sometimes(
    #    0.5,
    #    iaa.GaussianBlur(sigma=(0, 0.5))
    #),
    # Strengthen or weaken the contrast in each image.
    #iaa.LinearContrast((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    #iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    #iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True) # apply augmenters in random order
seq2 = iaa.Sequential([
    iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.LinearContrast((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.9, 1.1), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
], random_order=True) # apply augmenters in random order
class MyDataset(Dataset):
    def __init__(self,  png_dir:str, seq_dir:str=None, categories:list=None, aug:bool=False, mode:str=None, max_len=100):
        self.png_dir = png_dir
        self.categories = categories
        self.seq_dir = seq_dir
        # 设置数据集读取的模式，可取的值为'train','valid','test'
        self.mode = mode

        self.data = None
        self.label = None
        self.seq = None
        self.seq_len = None
        self.max_len = max_len
        # 保证序号唯一
        self.categories.sort()
        for index, category in enumerate(self.categories):
            png_path = os.path.join(png_dir, category + "_png.npz")
            seq_path = os.path.join(seq_dir, "sketchrnn_" + category + ".npz")
            seq = np.load(seq_path, encoding='latin1', allow_pickle=True)
            png = np.load(png_path, encoding='latin1', allow_pickle=True)

            seq_list = []
            len_list = []
            for item in seq[mode]:
                length = item.shape[0]
                if length >= self.max_len:
                    seq_list.append(item[:self.max_len, :])
                    len_list.append(self.max_len)
                else:
                    seq_list.append(np.pad(item, ((0, self.max_len - length), (0, 0)), "constant"))
                    len_list.append(length)

            seq_list = np.array(seq_list, dtype=np.int16)
            len_list = np.array(len_list, dtype=np.int16)

            if self.data is None:
                self.data = np.expand_dims(png[mode], axis=1)
                self.label = index * np.ones(png[mode].shape[0], dtype=np.int)
                self.seq = seq_list
                self.seq_len = len_list
            else:
                self.data = np.concatenate([self.data, np.expand_dims(png[mode], axis=1)])
                self.label = np.concatenate([self.label, index * np.ones(png[mode].shape[0], dtype=np.int)])
                self.seq = np.concatenate([self.seq, seq_list], axis=0)
                self.seq_len = np.concatenate([self.seq_len, len_list])
            if aug:
                if mode != 'train':
                    images = np.transpose(self.data, (0, 2, 3, 1))
                    images_aug = np.array([seq1(image=images[i]) for i in range(images.shape[0])])
                    self.data = np.transpose(images_aug, (0, 3, 1, 2))
                else:
                    images = np.transpose(self.data, (0, 2, 3, 1))
                    images_aug = np.array([seq2(image=images[i]) for i in range(images.shape[0]) for _ in range(3)])
                    images_new = np.transpose(images_aug, (0, 3, 1, 2))
                    self.data = np.concatenate((self.data, images_new), axis=0)
                    self.seq = np.concatenate((self.seq,self.seq,self.seq,self.seq), axis=0)
                    self.seq_len = np.concatenate((self.seq_len,self.seq_len,self.seq_len,self.seq_len), axis=0)
                    self.label = np.concatenate((self.label,self.label,self.label,self.label), axis=0)
    def __getitem__(self, item):
        return self.data[item], self.seq[item], self.seq_len[item], self.label[item]

    def __len__(self):
        return self.data.shape[0]

