import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
from typing import Tuple, List

import math
import numpy as np

@dataclass
class ResLayerConfig():
    num_channels: int
    num_layers: int = 2
    stride: int = 1

@dataclass
class ResNetConfig:
    num_blocks: int = 18
    num_classes: int = 10
    blocks: List[ResLayerConfig] = [
        ResLayerConfig(64, 2, 1),
        ResLayerConfig(128, 2, 2),
        ResLayerConfig(256, 2, 2),
        ResLayerConfig(512, 2, 2)
    ]

@dataclass
class TrainingConfig():
    trainloader
    testloader
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
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
            nn.BatchNorm2d(out_dim)
        )

        self.res = nn.Sequential()
        if stride != 1 or in_dim != out_dim:
            self.res = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, padding=1),
                nn.BatchNorm2d(out_dim)
            )

    def forward(self, x):
        return F.relu(self.res(x) + self.block(x))

class ResNet34(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config.num_classes
        self.num_blocks = config.num_blocks
        self.in_dim = config.blocks[0].num_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.in_dim, kernel_size=3, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(self.in_dim), 
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.resmodel = self._build_config(config)

        self.block1 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64)
        )
        self.block2 = nn.Sequential(
            ResBlock(64, 128, 2),
            ResBlock(128, 128),
            ResBlock(128, 128)
        )
        self.block3 = nn.Sequential(
            ResBlock(128, 256, 2),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256)
        )
        self.block4 = nn.Sequential(
            ResBlock(256, 512, 2),
            ResBlock(512, 512),
            ResBlock(512, 512)
        )
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def self._build_model(self, config):
        blocks = []
        for layer_config in config.blocks:
            layers = []
            for layer in config.num_layers:
                layers.append(ResBlock(self.in_dim, config.num_channels, config.stride))
                self.in_dim = config.num_channels
            blocks.append(nn.Sequential(*layers))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return F.softmax(x, dim=1)
