import argparse
import os
import sys
import time

import click

# from model import MyAwesomeModel
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm


def evaluate(data_path, model_checkpoint):

    print("Evaluating until hitting the ceiling")

    print(model_checkpoint)
    # _, test_set = mnist()
    test_set = torch.load(os.path.join(data_path, "test.pt"))

    # print(len(test_set[0]))
    # print(len(test_set[1]))

    # .shape)
    model = torch.load(model_checkpoint)
    # model = MyAwesomeModel()

    with torch.no_grad():
        model.eval()
    # model.load_state_dict(torch.load(model_checkpoint))

    criterion = torch.nn.NLLLoss()

    test_losses = []
    test_loss = 0
    # images,labels = test_set
    images = test_set["images"]
    labels = test_set["labels"]
    # for images, labels in test_set:
    log_ps = model(images.float())
    loss = criterion(log_ps, labels)
    test_loss += loss.item()
    test_losses.append(loss.item())

    ps = torch.exp(log_ps)
    top_p, top_class = ps.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    accuracy = torch.mean(equals.type(torch.FloatTensor))
    # print(test_loss)
    print(f"Accuracy: {accuracy.item()*100}%")


if __name__ == "__main__":
    evaluate(
        data_path="data/processed",
        model_checkpoint="models/20230109_143851/checkpoint_0.01_0.5_full.pth",
    )
