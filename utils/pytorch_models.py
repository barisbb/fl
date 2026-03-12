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
    """
    Plain ResNet50 backbone for FedAWA version.
    No modulation layers.
    No personalization-specific blocks.
    """
    def __init__(self, name, num_cls=19, channels=10, FC_dim=2048, pretrained=True):
        super(ResNet50, self).__init__()
        self.name = name
        self.len = 0
        self.loss = 0

        # torchvision compatibility
        try:
            if pretrained:
                resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            else:
                resnet = models.resnet50(weights=None)
        except Exception:
            resnet = models.resnet50(pretrained=pretrained)

        # replace first conv for 10-channel input
        self.conv1 = nn.Conv2d(
            channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )

        # optional: initialize new conv1
        init.kaiming_normal_(self.conv1.weight.data)

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
        self.FC.apply(fc_init_weights)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        logits = self.FC(x)
        return logits
