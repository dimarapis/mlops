import torch
import click
from data import CorruptMnist
from model import MyAwesomeModel
import matplotlib.pyplot as plt
import tqdm
import os
import numpy as np
import argparse.parser


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-2, help="learning rate to use for training")
def train(lr):
    print("Training day and night")
    print(lr)

    train_set = CorruptMnist(train=True)

    model = MyAwesomeModel()
    # train_set, _ = mnist()
    print(train_set.data.shape)
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50, 70, 90], gamma=0.1)

    epochs = 100
    loss_to_compare = np.inf
    early_stop_counter = 0
    running_losses = []
    for e in range(epochs):
        model.train()
        running_loss = 0
        images = train_set.data
        labels = train_set.targets

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
            torch.save(model.state_dict(), "models/day1_best.pth")
        else:
            early_stop_counter += 1
            if early_stop_counter > 10:
                print(
                    f"Early stopping at epoch {e}, triggered by loss {running_loss} \
                        vs compared_loss {loss_to_compare}"
                )
                break

        running_losses.append(running_loss / len(train_set))
        print(f"Epoch {e} --- train loss: {loss/len(train_set)}")

    # torch.save(model.state_dict(), 'checkpoint.pth')

    plt.figure(figsize=(8, 4))
    plt.plot(running_losses, label="training loss")
    plt.legend()
    plt.show()


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    assert model_checkpoint is not None, "Please provide a model checkpoint"
    assert os.path.exists(
        model_checkpoint
    ), "File does not exist \
        please train and produce pth file first"

    test_set = CorruptMnist(train=False)

    model = MyAwesomeModel()

    with torch.no_grad():
        model.eval()
    model.load_state_dict(torch.load(model_checkpoint))

    criterion = torch.nn.NLLLoss()

    test_losses = []
    test_loss = 0
    images = test_set.data
    labels = test_set.targets
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


cli.add_command(train)
cli.add_command(evaluate)

if __name__ == "__main__":
    cli()
