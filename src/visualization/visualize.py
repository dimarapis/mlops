import torch
from sklearn.manifold import TSNE
import src.models.model as mamodel
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def visualize(data_path, model_checkpoint, saving_path):

    test_set = torch.load(data_path)
    images = test_set['images'].float()
    if model_checkpoint is None:
        model = mamodel.MyAwesomeModel()
    else:
        model = torch.load(model_checkpoint)
    #model = torch.load(model_checkpoint)
    features = model(images)
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(features.cpu().detach().numpy())
    
    df = pd.DataFrame()
    df["y"] = test_set['labels'].tolist()
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 10),
                    data=df).set(title="MNIST data T-SNE projection")
    
    plt.savefig(saving_path)    
    
if __name__ == "__main__":
    visualize(
        data_path="data/processed/test.pt",
        model_checkpoint="models/day2_best.pth",
        saving_path="reports/figures/visualize.png",
    )
    visualize(
        data_path="data/processed/test.pt",
        model_checkpoint=None,
        saving_path="reports/figures/visualize_untrained.png",
    )
    
