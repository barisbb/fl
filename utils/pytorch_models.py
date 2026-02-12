import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        init.kaiming_normal_(m.weight.data)


def fc_init_weights(m):
    if type(m) == nn.Linear:
        init.kaiming_normal_(m.weight.data)


class ResNet50(nn.Module):
    def __init__(self, num_cls=19, channels=10, FC_dim=512, pretrained=True):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)

        self.conv1 = nn.Conv2d(
            channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.encoder = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
        )
        self.FC = nn.Linear(FC_dim, num_cls)

        if not pretrained:
            self.apply(weights_init_kaiming)
            self.apply(fc_init_weights)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        logits = self.FC(x)

        return logits

class ResNet50(nn.Module):
    def __init__(self, name, num_cls=19, channels=10, FC_dim=2048, pretrained=True):
        super(ResNet50, self).__init__()
        self.name = name
        self.len = 0
        self.loss = 0
        resnet = models.resnet50(pretrained=pretrained)
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.FC = nn.Linear(FC_dim, num_cls)
        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        logits = self.FC(x)
        return logits
