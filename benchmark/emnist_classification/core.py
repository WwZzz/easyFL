from torchvision import datasets, transforms
from benchmark.toolkits import ClassificationCalculator, DefaultTaskGen, IDXTaskPipe
import numpy as np

class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients = 1, skewness = 0.5, local_hld_rate=0.2, seed=0):
        super(TaskGen, self).__init__(benchmark='emnist_classification',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/RAW_DATA/EMNIST',
                                      local_hld_rate=local_hld_rate,
                                      seed=seed
                                      )
        self.num_classes = 62
        self.save_task = IDXTaskPipe.save_task
        self.source_dict = {
            'class_path': 'torchvision.datasets',
            'class_name': 'EMNIST',
            'train_args': {
                'root': '"' + self.rawdata_path + '"',
                'split': "'byclass'",
                'download': 'True',
                'transform': 'transforms.Compose([transforms.ToTensor(),])',
                'train': 'True'
            },
            'test_args': {
                'root': '"' + self.rawdata_path + '"',
                'split': "'byclass'",
                'download': 'True',
                'transform': 'transforms.Compose([transforms.ToTensor(),])',
                'train': 'False'
            }
        }

    def load_data(self):
        self.train_data = datasets.EMNIST(self.rawdata_path, split='byclass', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        self.test_data = datasets.EMNIST(self.rawdata_path, split='byclass', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

    def partition(self):
        """
        for each client, s% data is i.i.d. sampled and 1-s% data is partitioned according to the sorted labels.
        """
        s = 1.0-self.skewness
        all_data = [_ for _ in range(len(self.train_data))]
        num_total_iid = int(s * len(self.train_data))
        iid_data = list(np.random.choice(all_data, num_total_iid, replace=False))
        all_data = list(set(all_data).difference(set(iid_data)))
        local_datas = np.array_split(iid_data, self.num_clients)
        local_datas = [data_idx.tolist() for data_idx in local_datas]
        if len(all_data)>0:
            dpairs = [[did, self.train_data[did][-1]] for did in all_data]
            z = zip([p[1] for p in dpairs], all_data)
            z = sorted(z)
            labels, all_data = zip(*z)
            local_niid_datas = np.array_split(all_data, self.num_clients)
            local_niid_datas = [data_idx.tolist() for data_idx in local_niid_datas]
            local_datas = [niid_k+iid_k for niid_k,iid_k in zip(local_niid_datas, local_datas)]
        return local_datas

class TaskPipe(IDXTaskPipe):
    def __init__(self):
        super(TaskPipe, self).__init__()

class TaskCalculator(ClassificationCalculator):
    def __init__(self, device):
        super(TaskCalculator, self).__init__(device)

