import math
import shutil
import urllib.request
import random
import zipfile
import flgo.benchmark
import os.path
import ujson
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import MNIST, utils
from PIL import Image
import os.path
import torch
import torchvision
from tqdm import tqdm

from flgo.benchmark.toolkits.cv.horizontal.image_classification import BuiltinClassGenerator, BuiltinClassPipe, GeneralCalculator

def download_from_url(url= None, filepath = '.'):
    """Download dataset from url to filepath."""
    if url:urllib.request.urlretrieve(url, filepath)
    return filepath

def extract_from_zip(src_path, target_path):
    """Unzip the .zip file (src_path) to target_path"""
    f = zipfile.ZipFile(src_path)
    f.extractall(target_path)
    targets = f.namelist()
    f.close()
    return [os.path.join(target_path, tar) for tar in targets]

class FEMNIST(MNIST):
    """
    This dataset is derived from the Leaf repository
    (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
    dataset, grouping examples by writer. Details about Leaf were published in
    "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.
    """

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        if not os.path.exists(os.path.join(self.processed_folder, data_file)):
            self.download_and_process()
        self.data, self.targets, self.user_index = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='F')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def download_and_process(self):
        """
        Download the raw data and process it and save it in Torch format
        Modified from https://github.com/alibaba/FederatedScope/blob/master/federatedscope/cv/dataset/leaf_cv.py
        """
        if not os.path.exists(os.path.join(self.raw_folder, 'all_data.json')):
            """Download the FEMNIST data if it doesn't exist in processed_folder already."""

            os.makedirs(self.raw_folder, exist_ok=True)
            os.makedirs(self.processed_folder, exist_ok=True)

            # Download to `self.raw_dir`.
            url = 'https://federatedscope.oss-cn-beijing.aliyuncs.com'
            name = 'femnist_all_data.zip'
            src_path = download_from_url(f'{url}/{name}', os.path.join(self.raw_folder, 'tmp'))
            tar_paths = extract_from_zip(src_path, self.raw_folder)
            all_data = {
                'users': [],
                'num_samples': [],
                'user_data': {}
            }
            for i in range(1, len(tar_paths)):
                with open(tar_paths[i], 'r') as f:
                    raw_data = ujson.load(f)
                    all_data['users'].extend(raw_data['users'])
                    all_data['num_samples'].extend(raw_data['num_samples'])
                    all_data['user_data'].update(raw_data['user_data'])
            with open(os.path.join(self.raw_folder, 'all_data.json'), 'w') as f:
                ujson.dump(all_data, f)
            os.remove(src_path)
            shutil.rmtree(tar_paths[0])
        else:
            with open(os.path.join(self.raw_folder, 'all_data.json'), 'r') as f:
                all_data = ujson.load(f)
        """Process Data"""
        Xs = []
        Ys = []
        sample_ids = []
        idx = 0
        for writer, v in all_data['user_data'].items():
            data, targets = v['x'], v['y']
            Xs.extend(data)
            Ys.extend(targets)
            sample_ids.extend([idx] * len(data))
            idx += 1
        Xs = torch.tensor(np.stack(Xs))
        Ys = torch.LongTensor(np.stack(Ys))
        sample_ids = torch.tensor(np.stack(sample_ids))
        num_samples = sample_ids.shape[0]
        s1 = int(num_samples * 0.9)
        s2 = num_samples - s1
        train_ids, test_ids = torch.utils.data.random_split(sample_ids, [s1, s2])
        train_indices = train_ids.indices
        test_indices = test_ids.indices
        train_data, train_targets, train_sample_id = Xs[train_indices], Ys[train_indices], sample_ids[train_indices]
        test_data, test_targets, test_sample_id = Xs[test_indices], Ys[test_indices], sample_ids[test_indices]
        torch.save((train_data, train_targets, train_sample_id), os.path.join(self.processed_folder, "training.pt"))
        torch.save((test_data, test_targets, test_sample_id), os.path.join(self.processed_folder, "test.pt"))

TaskCalculator = GeneralCalculator

class TaskGenerator(BuiltinClassGenerator):
    def __init__(self, rawdata_path=os.path.join(flgo.benchmark.path,'RAW_DATA', 'MNIST')):
        super(TaskGenerator, self).__init__('femnist_classification', rawdata_path, FEMNIST, torchvision.transforms.Compose([torchvision.transforms.ToTensor(),]))

class TaskPipe(BuiltinClassPipe):
    def __init__(self, task_name):
        super(TaskPipe, self).__init__(task_name, FEMNIST, torchvision.transforms.Compose([torchvision.transforms.ToTensor(),]))