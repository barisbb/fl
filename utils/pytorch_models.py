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


class Modulation(nn.Module):
    """
    Feature-wise affine modulation: y = gamma * x + beta
    gamma,beta are 1xC (broadcast over H,W).
    """
    def __init__(self, channels: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        return x * self.gamma + self.beta


class BottleneckWith3Mods(nn.Module):
    """
    Same 3 mods, but moved BEFORE BN (more stable with FedBN).
    Order becomes:
      conv1 -> mod1 -> bn1 -> relu
      conv2 -> mod2 -> bn2 -> relu
      conv3 -> mod3 -> bn3
      + residual -> relu
    """
    def __init__(self, bottleneck_block: nn.Module):
        super().__init__()
        self.block = bottleneck_block

        c1 = self.block.bn1.num_features
        c2 = self.block.bn2.num_features
        c3 = self.block.bn3.num_features

        self.mod1 = Modulation(c1)
        self.mod2 = Modulation(c2)
        self.mod3 = Modulation(c3)

    def forward(self, x):
        identity = x

        out = self.block.conv1(x)
        out = self.mod1(out)          # <<< moved here (before bn1)
        out = self.block.bn1(out)
        out = self.block.relu(out)

        out = self.block.conv2(out)
        out = self.mod2(out)          # <<< moved here (before bn2)
        out = self.block.bn2(out)
        out = self.block.relu(out)

        out = self.block.conv3(out)
        out = self.mod3(out)          # <<< moved here (before bn3)
        out = self.block.bn3(out)

        if self.block.downsample is not None:
            identity = self.block.downsample(x)

        out = out + identity
        out = self.block.relu(out)
        return out


class ResNet50(nn.Module):
    def __init__(self, name, num_cls=19, channels=10, FC_dim=2048, pretrained=True):
        super(ResNet50, self).__init__()
        self.name = name
        self.len = 0
        self.loss = 0
        resnet = models.resnet50(pretrained=pretrained)

        # keep your same replacements
        resnet.layer2[0] = BottleneckWith3Mods(resnet.layer2[0])
        resnet.layer3[0] = BottleneckWith3Mods(resnet.layer3[0])

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
