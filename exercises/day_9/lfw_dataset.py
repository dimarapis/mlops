"""
LFW dataloading
"""
import argparse
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
import os 

import matplotlib.pyplot as plt

import torchvision.transforms.functional as F
from torchvision.utils import make_grid

plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform) -> None:
        # TODO: fill out with what you need
        self.img_dir = path_to_folder
        self.transform = transform
        self.img_files = self.list_files(self.img_dir)
        
    def count_files(self, directory):
        count = 0
        for root, dirs, files in os.walk(directory):
            count += len(files)
        return count
    
    def list_files(self, directory):
        list = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                name_len = len(file.split('_'))
                if name_len == 3:
                    list.append(os.path.join(file.split('_')[0]+'_'+file.split('_')[1],file))
                elif name_len == 4:
                    list.append(os.path.join(file.split('_')[0]+'_'+file.split('_')[1]+'_'+file.split('_')[2],file))

        return list
    
    def __len__(self):
        return self.count_files(self.img_dir)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        # TODO: fill out
        img_path = os.path.join(self.img_dir, self.img_files[index])
        image= Image.open(img_path)
        return self.transform(image)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_to_folder', default='data/processed/lfw', type=str)
    parser.add_argument('-batch_size', default=4, type=int)
    #parser.add_argument('-num_workers', default=32, type=int)
    parser.add_argument('-visualize_batch', action='store_true')
    parser.add_argument('-get_timing', action='store_true')
    parser.add_argument('-batches_to_check', default=100, type=int)
    
    args = parser.parse_args()
    
    lfw_trans = transforms.Compose([
        transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)
    #print(len(dataset))
    # Define dataloader
    x,y,yerr = [],[],[]
    for num_workers in range(16+1):#[1, 2, 4, 8, 10, 12, 16]:
        
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=num_workers
        )
        
        if args.visualize_batch:
            batch = next(iter(dataloader))
            grid = make_grid(batch)
            show(grid)
            
            
            
        if args.get_timing:
            # lets do some repetitions
            res = [ ]
            for _ in range(5):
                start = time.time()
                for batch_idx, batch in enumerate(dataloader):
                    if batch_idx > args.batches_to_check:
                        break
                end = time.time()

                res.append(end - start)
                
            res = np.array(res)
            x.append(num_workers)
            y.append(np.mean(res))
            yerr.append(np.std(res))
            print(f'Timing for num_workers {num_workers}: {np.mean(res)}+-{np.std(res)}')
            
    plt.errorbar(x,y,yerr=yerr,fmt='o')
    plt.savefig('reports/figures/num_workers_more_aug.png')