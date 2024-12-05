import os
import torch
from torchtext.vocab import GloVe
import numpy as np
from utils import *
import torch.nn as nn

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

def get_glove_representation(list_of_words):
  list_of_embedding = []
  for word in list_of_words:
    list_of_embedding.append(glove[word])
  return list_of_embedding

def embed_data_alt(data):
  """
  Alternative approach to embed a string into a tensor of indices from GloVe vocab.
  Instead of padding, we use a certain number of words that is closest to each word of the data 
  Issues so far:
  - Takes a long time to embed (took 1m 17s to complete 15 items of the training set)
  - Sometimes the length of the prompt was less than 300 (which could impact the length of the story)
  Parameters:
    data: A list of strings
  """
  indices = []

  for token in data:
      if token in glove.stoi:
          indices.append(glove.stoi[token])
      else:
          indices.append(glove.stoi.get('<pad>', 0))
      ind_tensor = torch.tensor(indices, dtype=torch.long)

  remaining_words_to_add = 300 - len(data)
  remaining_words_to_add_per_word = remaining_words_to_add // len(data)

  list_of_embedding = get_glove_representation(data)

  if len(indices) < 300:
    # Pad if shorter
    for embedding in list_of_embedding:
        # Inspired by the euclidean distance calculation seen at lecture
        distance = torch.norm(glove.vectors - embedding, dim=1)
        lst_of_words_found = sorted(enumerate(distance.numpy()), key=lambda x: x[1])[:remaining_words_to_add_per_word]

        lst_of_word_indices = torch.tensor([word_indices for (word_indices, distance) in lst_of_words_found], dtype=torch.long)
        ind_tensor =  torch.cat((ind_tensor, lst_of_word_indices), -1)
  else:
      # Truncate if longer
      ind_tensor = ind_tensor[:300]
  return ind_tensor.to("cpu")

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

def embed_data_tuples(data):
    """
    Embeds training/val/test data
    """
    embedded = []
    for x,t in data:
        x_embed = embed_data(x)
        t_embed = embed_data(t)
        embedded.append((x_embed, t_embed))
    return embedded

def fetch_word_representation_of_story(model_output):
  # if we run a singular prompt, the input model shape will be set to (300)
  # after applying the model, our output shape will likely be set to (300, embedding_szie)
  indices = torch.argmax(model_output, axis=1) # Same result if we did softmax before applying argmax
  story = ""
  for index in indices:
    story += glove.itos[index]
    story += " "
  return story

global glove 
glove = GloVe(name="6B",dim=300)

data = load_data('data/merged_data.txt')
train_data, val_data, test_data = split_data(data, 0.7, 0.15)

