import os
import torch
from torchtext.vocab import GloVe
import numpy as np
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
import string


global glove 
global device
glove = GloVe(name="6B",dim=100)
device = torch.device("cuda" if torch.cuda.is_available else "cpu")
tokenizer = get_tokenizer("basic_english")


def remove_punctuation(word):
    """
    Removes punctuation characters from a string

    Paramaters:
        word: string
    """
    translator = str.maketrans('', '', string.punctuation)
    return word.translate(translator)


def load_data(path):
    """
    Load data set into a list.

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
            stripped = line.strip().lower()
            words = [remove_punctuation(word) for word in stripped.split()]#tokenizer(stripped)
    
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
    """
    Gets the glove representation of a list of word
    
    Parameters:
        list_of_words: List of strings to get the glove representation of
    """
    list_of_embedding = []
    for word in list_of_words:
        list_of_embedding.append(glove[word])
    return list_of_embedding


def embed_data(data, default=len(glove)-1):
    """
    Embed the data of a story into a tensor of indices from GloVe vocab.

    Parameters:
        data: A list of tuples. The first element of the tuple is a list of strings. The second is a string
              The length of the first tuple element is constant throughout the list.
        default: The padding token. Set as the last element in the glove embedding.
    """
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
        if label in glove.stoi:
            label_glove = glove.stoi[label]
        else:
            label_glove = default

        if len(indices) < 5:
            indices += 5-len(indices)*default
        result.append((indices, label_glove))
    return result



def embed_data_stories(data):
    """
    Embeds all the story data. Taken si from load_data.

    Parameters:
        data: A list of a list of tuples (check embed data)
    """
    embedded = []
    for story in data:
        temp = embed_data(story)
        embedded.append(temp)
    return embedded

def fetch_word_representation_of_story(model_output):
    """
    Gets the english word representation of story given the model output.

    Parameters:
        model_output: The output of the model
    """
    story = ""
    for index in model_output:
        story += glove.itos[index]
        story += " "
    return story


def fetch_input():
    """
    Asks the user for a story prompt and reading level
    """
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
    """
    Converts the data into sequential format, for the purpose of training. 
    The output is a list of stories. The format of each story is as follows:
        [(word_1, word_2, ..., word_x), word_(x+1), (word_2, ..., word_(x+1)), word_(x+2),....]
        
        (word_1, word_2, ..., word_x) is the first x words to appear in the story. word_(x+1) is the next
        word after this sequence.

    Parameters:
        data: The data to convert
        seq_legnth: The length of the sequence
    """
    formatted = []
    for story in data:
        sequences = []
        for i in range(0, len(story)-seq_length):
            seq_in = story[i:i+seq_length]
            seq_out = story[i+seq_length]
            sequences.append((seq_in, seq_out))
        formatted.append(sequences)
    return formatted

