import argparse
import os
import sys
import time

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from model import MyAwesomeModel


def train(data_path, lr):
    print("Training day and night")
    print(lr)

    # train_set = CorruptMnist(train=True)

    model = MyAwesomeModel()
    # train_set, _ = mnist()
    # print(train_set.data.shape)

    train_set = torch.load(os.path.join(data_path, "train.pt"))

    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    gamma = 0.5
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50, 70, 90], gamma=gamma)

    epochs = 100
    loss_to_compare = np.inf
    early_stop_counter = 0
    running_losses = []
    for e in range(epochs):
        model.train()
        running_loss = 0
        # print(len(train_set.targets))
        images = train_set["images"]
        labels = train_set["labels"]

        optimizer.zero_grad()
        log_ps = model(images.float())
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
        if running_loss < loss_to_compare:
            early_stop_counter = 0
            loss_to_compare = running_loss
        else:
            early_stop_counter += 1
            if early_stop_counter > 10:
                print(
                    f"Early stopping at epoch {e}, triggered by loss {running_loss}\
                        vs compared_loss {loss_to_compare}"
                )
                break

        running_losses.append(running_loss / len(train_set))
        print(f"Epoch {e} --- train loss: {loss/len(train_set)}")
    model_folder = os.path.join("models", time.strftime("%Y%m%d_%H%M%S"))
    os.mkdir(model_folder)
    torch.save(model.state_dict(), os.path.join(model_folder, f"checkpoint_{lr}_{gamma}.pth"))
    torch.save(model, os.path.join(model_folder, f"checkpoint_{lr}_{gamma}_full.pth"))

    plt.figure(figsize=(8, 4))
    plt.plot(running_losses, label="training loss")
    # plt.plot(test_losses, label='test')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train(data_path="data/processed", lr=1e-2)
