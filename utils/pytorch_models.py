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
        if m.bias is not None:
            init.zeros_(m.bias.data)


class ResNet50(nn.Module):
    """
    Standard ResNet50 for FedBN.

    FedBN behavior is implemented in the training code:
    - BN parameters and BN running stats stay local to each client
    - all non-BN parameters are aggregated globally
    """
    def __init__(self, name, num_cls=10, channels=3, FC_dim=2048, pretrained=True):
        super(ResNet50, self).__init__()
        self.name = name
        self.len = 0
        self.loss = 0

        # For newer torchvision versions, you may want:
        # weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(pretrained=pretrained)

        if channels != 3:
            self.conv1 = nn.Conv2d(
                channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        else:
            self.conv1 = resnet.conv1

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

        if channels != 3:
            weights_init_kaiming(self.conv1)

        fc_init_weights(self.FC)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        logits = self.FC(x)
        return logits
