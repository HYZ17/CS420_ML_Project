import torch.nn as nn
from torchvision import models
import torch
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
        self.base_network.fc=nn.Linear(512,num_classes)

    def forward(self,input):
        predicted_label=self.base_network(input)
        return predicted_label

def save_parameters(model,save_path):
    torch.save(model.state_dict(),save_path)


def load_parameters(model,load_path,device):
    tmp=torch.load(load_path,map_location=device)
    model.load_state_dict(tmp)