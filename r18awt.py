import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

import os
import time
import uuid
from tqdm import tqdm

from model import ResNet, ResNetConfig, TrainingConfig
from utils import get_data, get_device, eval


def train(config):
    device = config.device

    model_config = ResNetConfig()
    model = ResNet(model_config).to(device)
    model = torch.compile(model)


    total_train_steps = config.epochs * config.trainloader.batch_size
    lr_schedule = np.interp(np.arange(1+total_train_steps),
                            [0, int(0.1 * total_train_steps), total_train_steps],
                            [0.2, 1, 0]) # Triangular learning rate schedule

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, betas=config.betas)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)

    train_loss, train_acc, test_acc = [], [], [torch.nan]

    start = time.time()
    it = tqdm(range(config.epochs))
    for epoch in it:
        model.train()

        for X, y in config.trainloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = F.cross_entropy(pred, y, reduction='none')
            train_loss.append(loss.mean().item())
            train_acc.append((pred.detach().argmax(1) == y).float().mean().item())

            optimizer.zero_grad(set_to_none=True)
            loss.sum().backward()
            optimizer.step()
            scheduler.step()

            it.set_description('Training loss=%.4f acc=%.4f' % (train_loss[-1], train_acc[-1]))

    test_loss, epoch_test_acc = eval(model, config.testloader)
    test_acc.append(epoch_test_acc)
    print('acc: %.4f' % test_acc[-1])

    duration = time.time() - start 
    log = dict(train_loss=train_loss, train_acc=train_acc, test_acc=test_acc, duration=duration)
    return model, log


if __name__ == "__main__":
    trainloader, testloader = get_data()
    device = get_device()

    print(f"using device {device}")

    config = TrainingConfig(
        trainloader=trainloader,
        testloader=testloader,
        epochs=150,
        lr=1e-4,
        device=device
    )

    accs, logs = [], []
    for _ in range(5):
        model, log = train(config)
        acc = log['test_acc'][-1]
        accs.append(acc)
        logs.append(log)

    for log in logs:
        print(log[duration])

    # loaders aren't serializable
    config_dict = config.__dict__.copy()
    config_dict.pop('trainloader', None)
    config_dict.pop('testloader', None)
    print(config_dict)

    with open("lazysave.json", "w") as f:
        f.writelines(config_dict)
        f.writelines(logs)

    # yoinked directly from keller jordan
    # log_dir = os.path.join('logs', str(uuid.uuid4()))
    # os.makedirs(log_dir, exist_ok=True)
    # log_path = os.path.join(log_dir, 'log.pt')
    # print(os.path.abspath(log_path))
    # torch.save(log, os.path.join(log_dir, 'log.pt'))
