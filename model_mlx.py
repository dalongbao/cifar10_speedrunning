import mlx
import mlx.nn as nn
import mlx.core as mx
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

from utils_mlx import triangular_lr_scheduler

@dataclass
class ResLayerConfig:
    num_channels: int
    num_layers: int = 2
    stride: int = 1


@dataclass
class ResNetConfig: # defaults to ResNet18
    num_classes: int = 10
    blocks: List[ResLayerConfig] = field(default_factory=lambda: [
        ResLayerConfig(64, 2, 1), # num_channels, num_layers, stride
        ResLayerConfig(128, 2, 2),
        ResLayerConfig(256, 2, 2),
        ResLayerConfig(512, 2, 2)
    ])


@dataclass
class TrainingConfig:
    batch_size: int = 512
    epochs: int = 150 # taken from keller jordan
    lr: float = 0.2
    wd: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    t_max: int = 20000
    momentum: float = 0.9
    scheduler: Optional[dict] = field(init=False)

    def __post_init__(self):
        self.scheduler = self._get_scheduler_config("triangular", (0.2, 0, 1, 20000, 0.4))

    def _get_scheduler_config(self, scheduler: str, config: Tuple[float, float, float, int, float]):
        if scheduler == "triangular":
            assert len(config) == 5
            return self._get_triangular_scheduler(config)
        elif scheduler == "linear":
            assert len(config) == 3
            return self._get_linear_scheduler(config)
        elif scheduler == "cosine":
            assert len(config) == 3  # (init_lr, decay_steps, end_lr)
            return self._get_cosine_scheduler(config)
        elif scheduler == "exponential":
            assert len(config) == 2  # (init_lr, decay_rate)
            return self._get_exponential_scheduler(config)
        else:
            raise ValueError(f"Scheduler {scheduler} not supported.")

    def _get_triangular_scheduler(self, config: Tuple[float, float, float, int, float]):
        """
        Creates a triangular learning rate scheduler.
        """
        return triangular_lr_scheduler(config[0], config[1], config[2], config[3], config[4])


    def _get_linear_scheduler(self, config: Tuple[float, float, int]):
        """
        Creates a linear learning rate scheduler.
        """
        init_lr = config[0]
        end_lr = config[1]
        steps = config[2]
        return optim.linear_schedule(init_lr, end_lr, steps)

    def _get_cosine_scheduler(self, config: Tuple[float, int, float]):
        """
        Creates a cosine decay scheduler.
        """
        init_lr = config[0]
        decay_steps = config[1]
        end_lr = config[2]
        return optim.cosine_decay(init_lr, decay_steps, end_lr)

    def _get_exponential_scheduler(self, config: Tuple[float, float]):
        """
        Creates an exponential decay scheduler.
        """
        init_lr = config[0]
        decay_rate = config[1]
        return optim.exponential_decay(init_lr, decay_rate)


class ResBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm(out_dim)
        )
        self.relu = nn.ReLU()

        self.res = nn.Sequential()
        if stride != 1 or in_dim != out_dim:
            self.res = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm(out_dim)
            )

    def __call__(self, x):
        return self.relu(self.res(x) + self.block(x))


class ResNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config.num_classes
        self.in_dim = config.blocks[0].num_channels
        self.out_dim = config.blocks[-1].num_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.in_dim, kernel_size=3, stride=1, padding=3, bias=False),
            nn.BatchNorm(self.in_dim), 
            nn.ReLU()
        )
        self.resblocks = self._build_config(config)
        self.fc = nn.Linear(self.out_dim, self.num_classes)
        self.pool = nn.AvgPool2d((1, 1))

    def _build_config(self, config):
        layers = []
        for layer_config in config.blocks:
            for _layer in range(layer_config.num_layers):
                layers.append(ResBlock(self.in_dim, layer_config.num_channels, layer_config.stride))
                self.in_dim = layer_config.num_channels
        return nn.Sequential(*layers)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.resblocks(x)
        x = self.pool(x, (1, 1))
        x = mx.flatten(x, 1)
        x = self.fc(x)
        return x
