import torch.nn as nn
from torchvision import models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import torch
import torch.nn.functional as F
resnet_dict = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
}


class My_model(nn.Module):
    def __init__(self,resnet_type="resnet18",num_classes=25):
        super(My_model,self).__init__()
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
        return x


def save_parameters(model,save_path):
    torch.save(model.state_dict(),save_path)


def load_parameters(model,load_path,device):
    tmp=torch.load(load_path,map_location=device)
    model.load_state_dict(tmp)