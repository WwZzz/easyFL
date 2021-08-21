from torch.utils.data import Dataset
import torch

class XYDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs = torch.tensor(xs)
        self.ys = torch.tensor(ys)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, item):
        return self.xs[item], self.ys[item]
