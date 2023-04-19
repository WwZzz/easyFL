from torch.utils.data import Dataset
import torch
import os
import numpy as np

from flgo.benchmark.toolkits import download_from_url, extract_one_from_zip


class UCRArchiveDataset(Dataset):
    def __init__(self, root, dataset_name, train=True):
        self.root = root
        self.train = train
        self.raw_folder = os.path.join(self.root, 'raw_data')
        self.processed_folder = os.path.join(self.root, 'UCRArchive_2018')
        self.dataset_name = dataset_name
        self.file = '{}_TRAIN.tsv'.format(dataset_name) if self.train else '{}_TEST.tsv'.format(dataset_name)
        self.url = "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip"
        if not os.path.exists(os.path.join(self.processed_folder, dataset_name, self.file)):
            self.download()
        with open(os.path.join(self.processed_folder, dataset_name, self.file), 'r') as f:
            data = np.loadtxt(f)
        self.x = torch.from_numpy(data[:, 1:]).to(torch.float)
        self.y = torch.from_numpy(data[:, 0]).to(torch.long) - 1

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

    def download(self):
        os.makedirs(self.raw_folder, exist_ok=True)
        src_path = os.path.join(self.raw_folder, 'UCRArchive_2018.zip')
        if not os.path.exists(src_path):
            download_from_url(self.url, src_path)
        extract_one_from_zip(src_path, self.root, file_name=os.path.join('UCRArchive_2018', self.dataset_name, self.file), pwd=b'someone')
