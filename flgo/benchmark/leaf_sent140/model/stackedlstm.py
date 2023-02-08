import os

import torch
import torch.nn as nn
import ujson

from utils.fmodule import FModule

class Model(FModule):

    def __init__(self, embedding_dim=300, vocab_size=400001, hidden_size=256, output_dim=2):
        super(Model, self).__init__()
        with open(os.path.join('./benchmark/RAW_DATA/SENTIMENT140', 'raw_data', 'embs.json'), 'r') as inf:
            embs = ujson.load(inf)
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.embeddings.weight.data.copy_(torch.Tensor(embs['emba']))
        self.embeddings.weight.requires_grad = False
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_dim)

    def forward(self, input_seq):
        embeds = self.embeddings(input_seq.T)
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(embeds)
        encoding = torch.cat((lstm_out[0], lstm_out[-1]), dim=1)
        output = self.fc(encoding)
        return output
