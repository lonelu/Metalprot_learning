"""
Author: Jonathan Zhang <jon.zhang@ucsf.edu>

This file contains functions for loading and splitting data for model training and validation.
"""

#imports
import os
import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class ImageSet(Dataset):
    "Custom dataset class for CNN image data."
    def __init__(self, df: pd.DataFrame, encodings: bool):
        self.labels = np.vstack([array for array in df['labels']])
        self.observations = np.stack([channel for channel in df['channels']], axis=0) if encodings else np.stack([channel[0:4] for channel in df['channels']], axis=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        observation = self.observations[index]
        label = self.labels[index]
        return observation, label 

class DistanceData(Dataset):
    "Custom dataset class for FCN distance data"
    def __init__(self, df: pd.DataFrame, encodings: bool):
        self.labels = np.vstack([array for array in df['labels']])
        observations = np.hstack([np.vstack([array for array in df['distance_matrices']]), np.vstack([array for array in df['encodings']])]) if encodings else np.vstack([array for array in df['distance_matrices']])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        observation = self.observations[index]
        label = self.labels[index]
        return observation, label 

def split_data(features_file: str, path2output: str, partitions: tuple, seed: int, write_json: bool):
    """
    Splits data into training and test sets. Returns a tuple of train, test, and validation dataframes, in that order.
    """
    data = pd.read_pickle(features_file)
    barcodes = np.unique(np.array(data['barcode']))
    np.random.seed(seed)
    np.random.shuffle(barcodes)

    train_len, test_len = round(len(barcodes) * partitions[0]), round(len(barcodes) * partitions[1])
    train_barcodes, test_barcodes, val_barcodes = barcodes[0: train_len], barcodes[train_len: train_len + test_len], barcodes[train_len + test_len: len(data)]

    if write_json:
        with open(os.path.join(path2output, 'barcodes.json'), 'w') as f:
            json.dump({'train': train_barcodes.tolist(), 'test': test_barcodes.tolist(), 'val': val_barcodes.tolist()}, f)
        
    return data[data['barcode'].isin(train_barcodes)], data[data['barcode'].isin(test_barcodes)], data[data['barcode'].isin(val_barcodes)]

def split_data_kfolds(features_file: str, k: int, seed: int):
    """
    Splits data into k separate folds.
    """
    data = pd.read_pickle(features_file)
    barcodes = np.unique(np.array(data['barcode']))
    np.random.seed(seed)
    np.random.shuffle(barcodes)

    x, y = divmod(len(barcodes), k)
    _folds = list((barcodes[i*x+min(i, y):(i+1)*x+min(i+1, y)] for i in range(k)))
    folds = {}
    for number, fold in enumerate(_folds):
        folds[str(number)] = data[data['barcode'].isin(fold)]
    return folds