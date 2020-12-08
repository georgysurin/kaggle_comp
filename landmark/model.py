import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F


class MyResNet(nn.Module):
    def __init__(self):
        super(MyResNet, self).__init__()
        resnet = models.resnet50(pretrained=True)
        resnet = list(resnet.children())[:-2]
        # self.bn4 = nn.BatchNorm2d(2048)
        self.fc1 = nn.Linear(2048, 512)
        self.dropout = nn.Dropout()
        # self.bn5 = nn.BatchNorm1d(512)
        self.resnet = nn.Sequential(*resnet)

    def forward(self, x):
        x = self.resnet(x)
        x = F.avg_pool2d(x.clamp(min=1e-6).pow(3), (x.size(-2), x.size(-1))).pow(1. / 3)
        # x = self.bn4(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        # x = x / torch.norm(x, dim=1).reshape(-1, 1)

        return x
