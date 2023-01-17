# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import click
import numpy as np
import torch
import wget
from dotenv import find_dotenv, load_dotenv
from torch.utils.data import Dataset


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # corrupted_train dataset load
    train_data = []
    for i in range(5):
        train_data.append(np.load(os.path.join(input_filepath, f"train_{i}.npz"), allow_pickle=True))
    train_images = torch.tensor(np.concatenate([t["images"] for t in train_data])).reshape(-1, 1, 28, 28)
    train_targets = torch.tensor(np.concatenate([t["labels"] for t in train_data]))

    # correpted_test_dataset load
    test_data = np.load(os.path.join(input_filepath, "test.npz"), allow_pickle=True)
    test_image = torch.tensor(test_data["images"]).reshape(-1, 1, 28, 28)
    test_targets = torch.tensor(test_data["labels"])

    # Normalize data
    train_ = {
        "images": (train_images - train_images.mean()) / train_images.std(),
        "labels": train_targets,
    }
    test_ = {
        "images": (test_image - test_image.mean()) / test_image.std(),
        "labels": test_targets,
    }

    # Save as tensors
    torch.save(train_, os.path.join(output_filepath, "train.pt"))
    torch.save(test_, os.path.join(output_filepath, "test.pt"))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
