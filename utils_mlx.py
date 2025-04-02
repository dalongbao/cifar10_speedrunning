import mlx
import mlx.core as mx
import mlx.optimizers as optim
import mlx.nn as nn
from mlx.data.datasets import load_cifar10

import time
import sys
import os
import numpy as np

def triangular_lr_scheduler(
    init: float, end: float, peak: float, steps: int, peak_ratio: float
):
    if steps < 1:
        raise ValueError(f"steps must be greater than 0, but got {steps}.")

    peak_step = int(peak_ratio * steps)

    def schedule(step):
        step = mx.minimum(step, steps)
        if step < peak_step:
            return init + (step / peak_step) * (peak - init)
        return peak + (step - peak_step) / (steps - peak_step) * (end - peak)

    return schedule

def one_hot(x, num_classes):
    batch_size = x.shape[0]
    arr = mx.zeros((batch_size, num_classes))
    for i in range(batch_size):
        arr[i, x[i]] = 1
    return arr


def loss_fn(model, X, y):
    pred = model(X)
    loss = mx.mean(nn.losses.cross_entropy(pred, y))
    acc = mx.mean(mx.argmax(pred, axis=1) == y)
    return loss, acc


def eval_fn(model, X, y):
    return mx.mean(mx.argmax(model(X), axis=1) == y)


def get_cifar10(batch_size, root=None): # yoinked from CIFAR10 example
    tr = load_cifar10(root=root)

    mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

    def normalize(x):
        x = x.astype("float32") / 255.0
        return (x - mean) / std

    group = mx.distributed.init()

    tr_iter = (
        tr.shuffle()
        .partition_if(group.size() > 1, group.size(), group.rank())
        .image_random_h_flip("image", prob=0.5)
        .pad("image", 0, 4, 4, 0.0)
        .pad("image", 1, 4, 4, 0.0)
        .image_random_crop("image", 32, 32)
        .key_transform("image", normalize)
        .batch(batch_size)
    )

    test = load_cifar10(root=root, train=False)
    test_iter = (
        test
        .partition_if(group.size() > 1, group.size(), group.rank())
        .key_transform("image", normalize)
        .batch(batch_size)
    )

    return tr_iter, test_iter


def eval(model, testloader):
    total_loss = 0
    total_acc = 0
    total_samples = 1

    for batch_count, batch in enumerate(testloader):
        X = mx.array(batch["image"])
        y = mx.array(batch["label"])
        pred = model(X)

        loss = nn.losses.cross_entropy(pred, y)
        total_loss += loss.item() * y.shape(0)
        total_acc += (pred.argmax(1) == y).float().sum().item()
        total_samples += y.shape(0)
        print(batch)
        print(total_samples, total_loss, total_acc)

    return total_loss / total_samples, total_acc / total_samples



