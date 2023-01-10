import argparse
import sys

import torch
import click
import time
import sys
# sys.path is a list of absolute path strings
import matplotlib.pyplot as plt
import tqdm
import os
import numpy as np
from sklearn.manifold import TSNE  
#from sklearn.manifold._t_sne import TSNE

def visualize(data_path, model_checkpoint, saving_path):
    
    print("Visualizing")
    
    print(model_checkpoint)
    test_set = torch.load(os.path.join(data_path,'test.pt'))

    model = torch.load(model_checkpoint)
    
    features = model.fc4.weight
    print(features.shape)
    
    	
    tsne = TSNE(n_components=2).fit_transform(features)

if __name__ == "__main__":
    visualize(data_path='data/processed', model_checkpoint='models/20230109_143851/checkpoint_0.01_0.5_full.pth', saving_path='reports/figures/')
