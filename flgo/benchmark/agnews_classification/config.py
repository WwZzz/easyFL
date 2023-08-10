import os
import torchtext
import flgo.benchmark
import torch.nn
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer, ngrams_iterator

path = os.path.join(flgo.benchmark.data_root,'AG_NEWS')
train_data = torchtext.datasets.AG_NEWS(root=path, split='train')
test_data = torchtext.datasets.AG_NEWS(root=path, split='test')
ngrams = 2
tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter, ngrams):
    for _, text in data_iter:
        yield ngrams_iterator(tokenizer(text), ngrams)

vocab = build_vocab_from_iterator(yield_tokens(train_data, ngrams), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

def text_pipeline(x):
    return vocab(list(ngrams_iterator(tokenizer(x), ngrams)))

def label_pipeline(x):
    return int(x) - 1

def apply_transform(x):
    return text_pipeline(x[1]), label_pipeline(x[0])

train_data = train_data.map(apply_transform)
test_data = test_data.map(apply_transform)

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

    def forward(self, text):
        offsets = [0]
        for t in text:
            offsets.append(t.size(0))
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text = torch.cat(text)
        offsets = offsets.to(text.device)
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

def get_model():
    return TextClassificationModel(vocab_size=len(vocab), embed_dim=64, num_class=4)