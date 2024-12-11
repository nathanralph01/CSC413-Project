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
    embedded_prompt = self.embedding(X) # resulting shape: (batch_size, sequence_length, 100)
    out, hidden = self.bidirectionalrnn(embedded_prompt, hidden) # output shape: (batch_size, sequence_length, 2*300) and hidden shape:  (2, batch_size, 300)
    out = out.contiguous().reshape(-1, self.hidden_size*2) # reshaped to: (batch_size * sequence_length, 2*300)

    z = self.fully_connected(out) # resulting shape: (batch_size * sequence_length, 400000)
    return z, hidden


  # Similar to the lab we do not want to update the parameters defined in the GloVe embedding vector since it is
  # pretrained
  def parameters(self):
    return (parameter for parameter in super(BidirectionalRNNGenerator, self).parameters() if parameter.requires_grad)


  def init_hidden(self, batch_size):
    return torch.zeros(2, batch_size, self.hidden_size).to(device)