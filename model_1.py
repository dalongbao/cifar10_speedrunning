import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class ResLayerConfig:
    num_channels: int
    num_layers: int = 2
    stride: int = 1


@dataclass
class ResNetConfig: # defaults to ResNet18
    num_blocks: int = 18
    num_classes: int = 10
    blocks: List[ResLayerConfig] = [
        ResLayerConfig(64, 2, 1),
        ResLayerConfig(128, 2, 2),
        ResLayerConfig(256, 2, 2),
        ResLayerConfig(512, 2, 2)
    ]


@dataclass
class TrainingConfig:
    trainloader: DataLoader
    testloader: DataLoader
    epochs: int = 150 # taken from keller jordan
    lr: float = 0.2
    wd: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    t_max: int = 20000
    momentum: float = 0.9


class ResBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(out_dim)
        )

        self.res = nn.Sequential()
        if stride != 1 or in_dim != out_dim:
            self.res = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_dim)
            )

    def forward(self, x):
        return F.relu(self.res(x) + self.block(x))


class ResNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config.num_classes
        self.num_blocks = config.num_blocks
        self.in_dim = config.blocks[0].num_channels
        self.out_dim = config.blocks[-1].num_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.in_dim, kernel_size=3, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(self.in_dim), 
            nn.ReLU()
        )
        self.resblocks = self._build_config(config)
        self.fc = nn.Linear(self.out_dim, self.num_classes)

    def _build_config(self, config):
        layers = []
        for layer_config in config.blocks:
            for _layer in range(layer_config.num_layers):
                layers.append(ResBlock(self.in_dim, layer_config.num_channels, layer_config.stride))
                self.in_dim = layer_config.num_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.resblocks(x)
        x = F.avg_pool2d(x, 4)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
