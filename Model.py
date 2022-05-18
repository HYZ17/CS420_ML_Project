
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch


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
