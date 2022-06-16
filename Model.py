
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch

from torchvision import models
resnet_dict = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
}
#from .basemodel import BaseModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import os.path
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

_project_folder_ = os.path.realpath(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
if _project_folder_ not in sys.path:
    sys.path.insert(0, _project_folder_)
'''
from neuralline.rasterize import RasterIntensityFunc


class SeqEncoder(nn.Module):
    """
    Encode vectorized sketch and extract per-point features
    """

    def __init__(self,
                 input_size=3,
                 hidden_size=512,
                 num_layers=2,
                 out_channels=1,
                 batch_first=True,
                 bidirect=True,
                 dropout=0,
                 requires_grad=True):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.batch_first = batch_first
        self.bidirect = bidirect
        self.proj_last_hidden = False

        # RNN for processing stroke-based sketches
        # http://ryankresse.com/dealing-with-pad-tokens-in-sequence-models-loss-masking-and-pytorchs-packed-sequence/
        # https://github.com/pytorch/pytorch/issues/1788
        # https://pytorch.org/docs/stable/nn.html#lstm

        # Alternative: nn.GRU
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=batch_first,
                           bidirectional=bidirect,
                           dropout=dropout)

        # FC layers
        num_directs = 2 if bidirect else 1
        self.attend_fc = nn.Linear(hidden_size * num_directs, out_channels)

        # Whether apply fc to the last hidden
        if self.proj_last_hidden:
            self.last_hidden_size = hidden_size
            self.last_hidden_fc = nn.Linear(num_directs * num_layers * hidden_size, self.last_hidden_size)
        else:
            self.last_hidden_size = num_directs * num_layers * hidden_size
            self.last_hidden_fc = None

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, points, lengths):
        """
        :param points:
        :param lengths:
        :return:
        """
        # https://blog.nelsonliu.me/2018/01/24/extracting-last-timestep-outputs-from-pytorch-rnns/
        # https://zhuanlan.zhihu.com/p/28472545
        batch_size = points.shape[0]
        num_points = points.shape[1]  # [batch_size, num_points, 3]
        point_dim = points.shape[2]

        if point_dim != self.input_size:
            # Slice
            points = points[:, :, :self.input_size]

        # Pack
        points_packed = pack_padded_sequence(points, lengths, batch_first=self.batch_first)

        # Recurrence
        # hiddens_packed.data: [Variable, hidden_size * num_directs]
        hiddens_packed, (last_hidden, _) = self.rnn(points_packed)  # For nn.LSTM
        # hiddens_packed, last_hidden = self.rnn(points_packed) # For nn.GRU

        # sigmoid is better than relu
        intensities_act = torch.sigmoid(self.attend_fc(hiddens_packed.data))

        # Unpack
        intensities_packed = PackedSequence(intensities_act, hiddens_packed.batch_sizes)
        intensities, _ = pad_packed_sequence(intensities_packed, batch_first=self.batch_first, total_length=num_points)

        # Last-step hidden state
        last_hidden = last_hidden.view(batch_size, -1)

        if self.proj_last_hidden:
            last_hidden = F.relu(self.last_hidden_fc(last_hidden))

        return intensities, last_hidden


class SketchR2CNN(BaseModel):
    """
    Single-branch RNN-Rasterization-CNN model
    """

    def __init__(self,
                 cnn_fn,
                 rnn_input_size,
                 rnn_dropout,
                 img_size,
                 thickness,
                 num_categories,
                 intensity_channels=1,
                 train_cnn=True,
                 device=None):
        super().__init__()

        self.img_size = img_size
        self.thickness = thickness
        self.intensity_channels = intensity_channels
        self.eps = 1e-4
        self.device = device

        nets = list()
        names = list()
        train_flags = list()

        # RNN for analyzing stroke-based sketches
        self.rnn = SeqEncoder(rnn_input_size, out_channels=intensity_channels, dropout=rnn_dropout)

        # CNN for 2D analysis
        self.cnn = cnn_fn(pretrained=False, requires_grad=train_cnn, in_channels=intensity_channels)

        # Last layer for classification
        num_fc_in_features = self.cnn.num_out_features
        self.fc = nn.Linear(num_fc_in_features, num_categories)

        nets.extend([self.rnn, self.cnn, self.fc])
        names.extend(['rnn', 'conv', 'fc'])
        train_flags.extend([True, train_cnn, True])

        self.register_nets(nets, names, train_flags)
        self.to(device)

    def __call__(self, points, points_offset, lengths):
        # Use RNN to compute point-wise attention
        # points:       [B, N, 3]
        # intensities:  [B, N, F]
        intensities, _ = self.rnn(points_offset, lengths)

        # Rasterize and forward through CNN
        images = RasterIntensityFunc.apply(points, intensities, self.img_size, self.thickness, self.eps, self.device)
        if images.size(1) == 1:
            images = images.repeat(1, 3, 1, 1)
        cnnfeat = self.cnn(images)
        logits = self.fc(cnnfeat)

        return logits, intensities, images


'''
class resnetRnn(nn.Module):
    def __init__(self,resnet_type="resnet18",num_classes=25):
        super(resnetRnn,self).__init__()
        self.base_network=resnet_dict[resnet_type](pretrained=True)
        self.base_network.conv1=nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_network.fc=nn.Linear(512,256)

        #resnet输出的特征长度为256，看上面的那一行，输出是256；
        #lstm的参数解释：input_size=3，因为输入的每一个特征是[x,y,state],因此是3，这个不能改
        #hidden_size,你可以理解为输出的特征的维度，可以调大一点，观察效果
        #num_layers：lstm层的数量，可以调大一些，观察效果
        #bidirectional，是否使用双向lstm，这里使用了
        self.lstm=nn.LSTM(
            input_size=3,
            hidden_size=64,
            num_layers=2,
            bidirectional=True
        )

        #fc1的input_dim=lstm.hidden_size*2+self.base_network.fc.output_dim
        #例如：384=64*2+256
        #fc1的输出维度可以调整
        self.fc1=nn.Linear(384,128)
        self.relu=nn.ReLU()

        #fc2的输入维度和fc1的输出相对应，输出一定是25，因为有25个类
        self.fc2=nn.Linear(128,25)

    def forward(self,input_png,input_seq,input_seq_len):
        predicted_label=self.base_network(input_png)
        packed_seq=pack_padded_sequence(input_seq,input_seq_len,batch_first=True,enforce_sorted=False)
        lstm_hidden,(h_last,c_last)=self.lstm(packed_seq)
        padded_hidden,_=pad_packed_sequence(lstm_hidden,batch_first=True)
        lstm_feature=torch.stack([padded_hidden[i][input_seq_len[i]-1, :] for i in range(0, len(input_seq_len))], dim=0)
        # print(lstm_feature.shape)
        # print(predicted_label.shape)
        total_feature=torch.cat([predicted_label,lstm_feature],dim=1)
        x=self.fc1(total_feature)
        x=self.relu(x)
        x=self.fc2(x)
        x = F.softmax(x, dim=-1)
        return x
class rnn(nn.Module):
    def __init__(self, num_classes=25):
        super(rnn,self).__init__()
        #self.base_network=resnet_dict[resnet_type](pretrained=True)
        #self.base_network.conv1=nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #self.base_network.fc=nn.Linear(512,256)

        #resnet输出的特征长度为256，看上面的那一行，输出是256；
        #lstm的参数解释：input_size=3，因为输入的每一个特征是[x,y,state],因此是3，这个不能改
        #hidden_size,你可以理解为输出的特征的维度，可以调大一点，观察效果
        #num_layers：lstm层的数量，可以调大一些，观察效果
        #bidirectional，是否使用双向lstm，这里使用了
        self.lstm=nn.LSTM(
            input_size=3,
            hidden_size=64,
            num_layers=2,
            bidirectional=True
        )

        #fc1的input_dim=lstm.hidden_size*2+self.base_network.fc.output_dim
        #例如：384=64*2+256
        #fc1的输出维度可以调整
        self.fc1=nn.Linear(128, 256)
        self.relu=nn.ReLU()

        #fc2的输入维度和fc1的输出相对应，输出一定是25，因为有25个类
        self.fc2=nn.Linear(256, 25)

    def forward(self,input_seq,input_seq_len):
        #predicted_label=self.base_network(input_png)
        packed_seq=pack_padded_sequence(input_seq,input_seq_len,batch_first=True,enforce_sorted=False)
        lstm_hidden,(h_last,c_last)=self.lstm(packed_seq)
        padded_hidden,_=pad_packed_sequence(lstm_hidden,batch_first=True)
        lstm_feature=torch.stack([padded_hidden[i][input_seq_len[i]-1, :] for i in range(0, len(input_seq_len))], dim=0)
        # print(lstm_feature.shape)
        # print(predicted_label.shape)
        #total_feature=torch.cat([predicted_label,lstm_feature],dim=1)
        x=self.fc1(lstm_feature)
        x=self.relu(x)
        x=self.fc2(x)
        x = F.softmax(x, dim=-1)
        return x
class resnet(nn.Module):
    def __init__(self,resnet_type="resnet18",num_classes=25):
        super(resnet,self).__init__()
        self.base_network=resnet_dict[resnet_type](pretrained=True)
        self.base_network.conv1=nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_network.fc=nn.Linear(512,num_classes)

    def forward(self,input):
        predicted_label=self.base_network(input)
        x = F.softmax(predicted_label, dim=-1)
        return predicted_label
class Classifier(nn.Module):
    def __init__(self, num_classes=25):
        super(Classifier, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Dropout(p=0.2),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(3200, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )


    def forward(self,input):
        x = self.layer1(input)
        x = x.flatten(start_dim=1)
        print(x.shape)
        prediction = self.layer2(x)
        return prediction

def save_parameters(model,save_path):
    torch.save(model.state_dict(),save_path)


def load_parameters(model,load_path,device):
    tmp=torch.load(load_path,map_location=device)
    model.load_state_dict(tmp)
