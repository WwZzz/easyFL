import os
import torchtext
import flgo.benchmark
import torch.nn
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer, ngrams_iterator

path = os.path.join(flgo.benchmark.path, 'RAW_DATA','IMDB')
train_data = torchtext.datasets.IMDB(root=path, split='train')
test_data = torchtext.datasets.IMDB(root=path, split='test')
ngrams = 2
tokenizer = get_tokenizer('basic_english')
def yield_tokens(data_iter, ngrams):
    for _, text in data_iter:
        yield ngrams_iterator(tokenizer(text), ngrams)

vocab = build_vocab_from_iterator(yield_tokens(train_data, ngrams), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

class TextClassificationModel(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = torch.nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = torch.nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

def get_model():
    return TextClassificationModel(vocab_size=len(vocab), embed_dim=64, num_class=2)