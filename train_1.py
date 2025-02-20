import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import time
import math
import numpy as np
import tqdm

from model_1 import ResNet34, ResNetConfig, TrainingConfig
from utils import get_data, eval

def train(config):
    model = ResNet34().to("cuda")
    model = torch.compile(model)
    model.eval()

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, betas=config.betas)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.t_max)

    train_loss, train_acc, test_acc = [], [], [torch.nan]

    start = time.time()
    it = tqdm(range(config.epochs))
    for epoch in it:
        model.train()

        for X, y in train_dataloader:
            pred = model(X)
            loss = F.cross_entropy(pred, y)
            train_loss.append(loss.mean().item())
            train_acc.append((pred.detach().argmax(1) == y).float().mean().item())
            loss.backward()
            optimizer.zero_grad()
            optimizer.step()
            scheduler.step()
            it.set_description('Training loss=%.4f acc=%.4f' % (train_loss[-1], train_acc[-1]))

        test_acc.append(eval(model, test_loader))
        print('acc: %.4f' % test_acc[-1])

    duration = time.time() - start 
    log = dict(train_loss=train_loss, train_acc=train_acc, test_acc=test_acc, time=duration)
    return model, log

if __name__ == "__main__":
    trainloader, testloader = get_data()

    config = TrainingConfig(
        trainloader = trainloader,
        testloader = testloader,
        epochs = 150,
        lr = 0.2
    )

    accs = []
    for _ in range(5):
        model, log = train(config)
        acc = log['test_acc'][-1]
        accs.append(acc)

    print(accs)
