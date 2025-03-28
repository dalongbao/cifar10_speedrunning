import mlx
import mlx.core as mx
import mlx.optimizers as optim

import numpy as np

import os
import time
import uuid
from tqdm import tqdm

from model_mlx import ResNet, ResNetConfig, TrainingConfig
from utils_mlx import triangular_lr_scheduler, get_cifar10, loss_fn, eval

def train(config):
    model_config = ResNetConfig()
    model = ResNet(model_config)
    model = mx.compile(model)  # Compile the model here

    trainloader, testloader = get_cifar10(config.batch_size)

    total_train_steps = config.epochs * config.batch_size
    optimizer = optim.AdamW(learning_rate=config.lr, betas=config.betas)
    scheduler = config.scheduler

    train_loss, test_acc = [], [0]

    def step(X, y):
        (pred, loss), grads = nn.value_and_grad_fn(model, loss_fn)(model, X, y)
        optimizer.update(model, grads)
        return loss

    start = time.time()
    it = tqdm(range(config.epochs))
    for epoch in it:
        for batch_counter, batch in enumerate(trainloader): 
            X = batch["image"]
            y = batch["label"]
            loss = step(X, y)
            train_loss.append(loss.mean().item())
            it.set_description('Training loss=%.4f' % (train_loss[-1]))

    test_loss, epoch_test_acc = eval(model, testloader)
    test_acc.append(epoch_test_acc)
    print('acc: %.4f' % test_acc[-1])

    duration = time.time() - start
    log = dict(train_loss=train_loss, test_acc=test_acc, duration=duration)
    return model, log

if __name__ == "__main__":
    config = TrainingConfig(
            batch_size = 512,
            epochs = 150
    )
    config.scheduler = config._get_scheduler_config("triangular", (0.2, 0, 1, 20000, 0.4))

    accs, logs = [], []
    for _ in range(5):
        model, log = train(config)
        acc = log['test_acc'][-1]
        accs.append(acc)
        logs.append(log)

    for log in logs:
        print(log["duration"])

    config_dict = str(config.__dict__.copy())
    logs_str = [str(log) for log in logs]  

    with open("lazysave.json", "w") as f:
        f.write(config_dict + "\n")
        f.write("\n".join(logs_str))
