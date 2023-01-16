import argparse
import os
import sys
import time

import click

# sys.path is a list of absolute path strings
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from sklearn.manifold import TSNE

# from sklearn.manifold._t_sne import TSNE


def visualize(data_path, model_checkpoint, saving_path):

    print("Visualizing")

    print(model_checkpoint)
    test_set = torch.load(os.path.join(data_path, "test.pt"))
    #print(test_set['images'].shape)
    #images = torch.load(test_set['images'])
    images = test_set['images'].float()#.squeeze().float()
    model = torch.load(model_checkpoint)
    #print(images.shape)
    #model = torch.load(model_checkpoint)
    #print(model)
    #model = torch.load(model_checkpoint)

    features = model(images)
    print(features.shape)
    #current_outputs = outputs.cpu().detach().numpy()
    #features = np.concatenate((outputs, current_outputs))

    
    #features = model.fc4
    #print(features.shape)

    #tsne = TSNE(n_components=2).fit_transform(features)
    tsne = TSNE(n_components=2, verbose=1, random_state=123)

    z = tsne.fit_transform(features.cpu().detach().numpy())

    df = pd.DataFrame()
    df["y"] = y_train
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 10),
                    data=df).set(title="MNIST data T-SNE projection")

if __name__ == "__main__":
    visualize(
        data_path="data/processed",
        model_checkpoint="models/20230109_143851/checkpoint_0.01_0.5_full.pth",
        saving_path="reports/figures/",
    )
