import torch.nn as nn
from utils import *

# Inspired from lab 10
class BidirectionalRNNGenerator(nn.Module):
  def __init__(self, hidden_size = 300):
    super(BidirectionalRNNGenerator, self).__init__()
    self.vocab_size, self.embedding_size = glove.vectors.shape
    self.hidden_size = hidden_size # to match our limit of words in the story
    self.embedding = nn.Embedding.from_pretrained(glove.vectors)
    self.embedding.requires_grad = False
    self.bidirectionalrnn = nn.RNN(self.embedding_size, self.hidden_size, batch_first=True, bidirectional=True)
    self.fully_connected = nn.Linear(2*hidden_size, self.vocab_size)


  def forward(self, X, hidden):
    # Find the word embedding for the prompt/start of sentence
    embedded_prompt = self.embedding(X)
    out, hidden = self.bidirectionalrnn(embedded_prompt, hidden) # size of the output will be (batch_size, sequence_length, 2*hidden_size)
    out = out.contiguous().reshape(-1, self.hidden_size*2)
    # Note the sequence_length is our time step, so to have 300 words in our sequence, our input sequence_length must have 300 words (will need
    # to deal with padding to ensure that case is possible)

    # Our z value should be the logit of the embedded words, therefore the size of the output should be (batch_size, sequence_length, embedding_size)
    z = self.fully_connected(out)
    return z, hidden


  # Similar to the lab we do not want to update the parameters defined in the GloVe embedding vector since it is
  # pretrained
  def parameters(self):
    return (parameter for parameter in super(BidirectionalRNNGenerator, self).parameters() if parameter.requires_grad)

  
  def init_hidden(self, batch_size):
    return torch.zeros(2, batch_size, self.hidden_size).to(device)