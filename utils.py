import os
import torch
from torchtext.vocab import GloVe
import numpy as np
import torch.nn as nn
from torchtext.data.utils import get_tokenizer

global glove 
global device
glove = GloVe(name="6B",dim=100)
device = torch.device("cuda" if torch.cuda.is_available else "cpu")
tokenizer = get_tokenizer("basic_english")


def load_data(path):
    """
    Load data set into a list. CHANGING: For each story, the data should look like: (first few words, remainder of story)

    Parameters:
        Path: Path of the data set
    """
    if not os.path.exists(path):
        raise Exception("File path {} does not exist".format(path))
    data_set = []
    # Load stories into a list
    with open(path, "r", encoding="utf-8") as file:
        # , t = [], []
        story = []
        for line in file:
            stripped = line.strip()
            words = tokenizer(stripped)
    
            if stripped == '':
                if story:
                    data_set.append(story[:300])
                    #x, t = [], []
                    story = []
            else:
                # Check if it is a new story
                if not story:
                    story = words
                else:
                    story += words
        data_set.append(story)

    
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

def embed_data(data, default=len(glove)-1):
    result = []
    for text, label in data:
        indices = []
        for w in text:
            if w in glove.stoi:
                indices.append(glove.stoi[w])
            else:
                # this is a bit of a hack, but we will repurpose *last* word
                # (least common word) appearing in the GloVe vocabluary as our
                # '<pad>' token
                indices.append(default)
        result.append((indices, label),)
    return result

# def embed_data(data):
#     indices = []

#     for token in data:
#         if token in glove.stoi:
#             indices.append(glove.stoi[token])
#         else:
#             indices.append(glove.stoi.get('<pad>', 0)) 
#         ind_tensor = torch.tensor(indices, dtype=torch.long)

#     if len(indices) < 300:
#         # Pad if shorter
#         ind_tensor = torch.nn.functional.pad(ind_tensor, (0, 300 - len(indices)), value=glove.stoi.get('<pad>', 0))
#     else:
#         # Truncate if longer
#         ind_tensor = ind_tensor[:300]
#     return ind_tensor

def embed_data_tuples(data):
    """
    Embeds training/val/test data
    """
    embedded = []
    for story in data:
        temp = embed_data(story)
        embedded.append(temp)
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

def fetch_input():
    prompt = ""
    # TODO: come back to reading level after finishing the model
    reading_level = 0
    prompt_set = False
    # reading_level_set = False
    while not prompt_set:
        prompt = input("Story prompt: ")
        prompt = prompt.lower().split(sep=" ")
        prompt = [words for words in prompt if words != ""]
        if len(prompt) <= 0:
            print("You must add a prompt with at least one word. Please try again")
        else:
            prompt_set = True

    # while not reading_level_set:
    #     try:
    #         reading_level = int(input("Reading difficulty level (1-3): "))
    #         if not (1 <= reading_level <= 3):
    #             print("You must set a reading level between 1 to 3. Please try again")
    #         else:
    #             reading_level_set = True
    #     except ValueError:
    #         print("You must set a reading level between 1 to 3. Please try again")
    return prompt, reading_level

def create_training_sequences(data, seq_length):
    formatted = []
    for story in data:
        sequences = []
        for i in range(0, len(story)-seq_length):
            seq_in = story[i:i+seq_length]
            seq_out = story[i+seq_length]
            sequences.append((seq_in, seq_out))
        formatted.append(sequences)
    return formatted

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
data = load_data('data/stories2.txt') # using samller txt file for testing purposes

# convert data to sequences in order to predict next character
seq_data = create_training_sequences(data, 5)
# embed data
seq_data_embed = embed_data_tuples(seq_data)
# split to train, val, test sets ->> MAY NEED TO CHANGE VAL AND TEST SETS
train_data, val_data, test_data = split_data(seq_data_embed, 0.7, 0.15)

