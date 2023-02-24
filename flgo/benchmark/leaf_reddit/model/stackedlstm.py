import os

import torch
import torch.nn as nn
import ujson

from flgo.utils.fmodule import FModule

class Model(FModule):

    def __init__(self, embedding_dim=300, vocab_size=10000, hidden_size=256):
        super(Model, self).__init__()
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, input_seq):
        embeds = self.embeddings(input_seq.T)
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(embeds)
        encoding = torch.cat((lstm_out[0], lstm_out[-1]), dim=1)
        output = self.fc(encoding)
        return output
