import os
import torch
from torchtext.vocab import GloVe
import numpy as np
from utils import *

def load_data(path):
    """
    Load data set into a list. For each story, the data should look like: (first few words, remainder of story)

    Parameters:
        Path: Path of the data set
    """
    if not os.path.exists(path):
        raise Exception("File path {} does not exist".format(path))
    data_set = []
    with open(path, "r") as file:
        x, t = [], []
        for line in file:
            stripped = line.strip()
            words = stripped.split()

            if stripped == '':
                if x and t:
                    data_set.append((x, t[:300]))
                    x, t = [], []
            else:
                # Check if it is a new story
                if not x:
                    x = words[:5] # First five words 
                    t = words # Remainder of line
                else:
                    t += words
        data_set.append((x, t[:300]))
    return data_set


def split_data(data_set, train_split, val_split):
    """
    Split data into separate training, validation, and test sets.
    
    Parameters:
        data_set: Data in list format
        train_split: Percentage to split into training (0.7 for 70%)
        val_split: Percentage to split into validation
        test_split: Percentage to split into testing
    """
    total = len(data_set)
    train_num = int(total * train_split)
    val_num = train_num + int(total*val_split)

    return data_set[:train_num], data_set[train_num:val_num], data_set[val_num:]


def embed_data(data):
    indices = []

    for token in data:
        if token in glove.stoi:
            indices.append(glove.stoi[token])
        else:
            indices.append(glove.stoi.get('<pad>', 0)) 
        ind_tensor = torch.tensor(indices, dtype=torch.long)

    if len(indices) < 300:
        # Pad if shorter
        ind_tensor = torch.nn.functional.pad(ind_tensor, (0, 300 - len(indices)), value=glove.stoi.get('<pad>', 0))
    else:
        # Truncate if longer
        ind_tensor = ind_tensor[:300]
    return ind_tensor


global glove 
glove = GloVe(name="6B",dim=300)

data = load_data('data/merged_data.txt')
train_data, val_data, test_data = split_data(data, 0.7, 0.15)



