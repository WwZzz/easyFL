import torchtext.transforms as T
from torch.hub import load_state_dict_from_url
import torchdata.datapipes.iter as dpiter
from torchtext.models import RobertaClassificationHead, XLMR_BASE_ENCODER
from tqdm import tqdm
import torchtext.functional as F
import torch
from torch.utils.data import DataLoader
from flgo.utils.fmodule import FModule
from torchtext.data.functional import to_map_style_dataset

padding_idx = 1
bos_idx = 0
eos_idx = 2
max_seq_len = 256
xlmr_vocab_path = r"https://download.pytorch.org/models/text/xlmr.vocab.pt"
xlmr_spm_model_path = r"https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"

text_transform = T.Sequential(
    T.SentencePieceTokenizer(xlmr_spm_model_path),
    T.VocabTransform(load_state_dict_from_url(xlmr_vocab_path)),
    T.Truncate(max_seq_len - 2),
    T.AddToken(token=bos_idx, begin=True),
    T.AddToken(token=eos_idx, begin=False),
)
num_classes = 2
input_dim = 768

def collect_fn(batch):
    # features = [d['token_ids'] for d in batch]
    input = F.to_tensor([d['token_ids'] for d in batch], padding_value=padding_idx)
    labels = torch.tensor([d["target"] for d in batch])
    return input, labels

def apply_transform(x):
    return text_transform(x[0]), x[1]

def feat2dict(x):
    return {'token_ids':x[0], 'target':x[1]}

def init_dataset(object):
    datanames = object.get_data_names()
    for key in datanames:
        data = object.get_data(key)
        if data is None: continue
        dp = dpiter.IterableWrapper(data)
        dp = dp.map(apply_transform)
        dp = dp.map(feat2dict)
        object.set_data(to_map_style_dataset(dp), key)
        object.calculator.set_collect_fn(collect_fn)

class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.model = XLMR_BASE_ENCODER.get_model(head=RobertaClassificationHead(num_classes=num_classes, input_dim=input_dim))

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

def init_local_module(object):
    return

def init_global_module(object):
    if 'Server' in object.get_classname():
        model = Model().to(object.device)
        object.set_model(model)