import torch
import torch.nn as nn
from torch.autograd import Variable

class LM(nn.module):
    def __init__(self, vocab_size, embed_size, layers=4, p=0.5):
        super(LM, self).__init__()
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM()

