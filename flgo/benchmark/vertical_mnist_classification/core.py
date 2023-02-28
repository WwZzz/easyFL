from flgo.benchmark.mnist_classification.core import builtin_class, transforms
from flgo.benchmark.toolkits.base import BasicTaskPipe, BasicTaskCalculator
from flgo.benchmark.toolkits.cv.horizontal.image_classification import BuiltinClassGenerator
import torch
import os.path
from torch.utils.data import Dataset
import flgo
import random
import json

class TaskGenerator(BuiltinClassGenerator):
    def __init__(self, rawdata_path=os.path.join(flgo.benchmark.path,'RAW_DATA', 'MNIST')):
        super(TaskGenerator, self).__init__('vertical_mnist_classification', rawdata_path, builtin_class, transforms)

    def partition(self):
        self.local_datas = self.partitioner(self.train_data)
        self.num_objects = len(self.local_datas)
        if self.test_data is not None:
            self.test_local_datas = [{'sample_idxs':list(range(len(self.test_data))), 'pt_feature':pdata['pt_feature'], 'with_label':pdata['with_label']} for pdata in self.local_datas]

    def get_task_name(self):
        return '_'.join(['B-' + self.benchmark, 'P-' + str(self.partitioner), 'N-' + str(self.partitioner.num_parties)])

class PartialDataset(Dataset):
    def __init__(self, dataset, pt_feature=None, with_label=False, sample_idxs = []):
        self.dataset = dataset
        self.sample_idxs = sample_idxs
        self.pt_feature = pt_feature
        self.with_label = with_label

    def gen_id(self, *args, **kwargs) -> list:
        return list(range(len(self.dataset)))

    def partition(self, x):
        return (torch.split(x, self.pt_feature[1], dim=self.pt_feature[0]))[self.pt_feature[2]]

    def is_with_label(self):
        return self.with_label

    def __len__(self):
        return len(self.sample_idxs)

    def __getitem__(self, item):
        sidx = self.sample_idxs[item]
        x,y = self.dataset[sidx][0], self.dataset[sidx][1]
        sid = self.dataset.ids[sidx]
        if self.with_label:
            return self.partition(x), y, sid
        else:
            return self.partition(x), None, sid

    def get_batch_by_id(self, ids):
        try:
            xs = [self.partition(self.dataset[sid][0]) for sid in ids]
            ys = [self.dataset[sid][1] for sid in ids]
        except:
            raise ValueError("sample with id={} doesn't exists in the current dataset")
        return torch.stack(xs), torch.LongTensor(ys), ids

class TaskPipe(BasicTaskPipe):
    TaskDataset = PartialDataset
    def __init__(self, task_name):
        super().__init__(task_name)
        self.builtin_class = builtin_class
        self.transform = transforms

    def save_task(self, generator):
        party_names = self.gen_client_names(len(generator.local_datas))
        feddata = {'party_names': party_names,'rawdata_path': generator.rawdata_path, 'additional_option': generator.additional_option}
        for pid in range(len(party_names)):
            feddata[party_names[pid]] = {
                'data':{
                    'with_label': generator.local_datas[pid]['with_label'],
                    'pt_feature': generator.local_datas[pid]['pt_feature'],
                    'train':generator.local_datas[pid]['sample_idxs'],
                    'test': generator.test_local_datas[pid]['sample_idxs'],
                },
            }
        with open(os.path.join(self.task_path, 'data.json'), 'w') as outf:
            json.dump(feddata, outf)

    def load_data(self, running_time_option) -> dict:
        # load the datasets
        train_data = self.builtin_class(root=self.feddata['rawdata_path'], download=True, train=True, transform=self.transform, **self.feddata['additional_option'])
        test_data = self.builtin_class(root=self.feddata['rawdata_path'], download=True, train=False, transform=self.transform, **self.feddata['additional_option'])
        train_data.ids = list(range(len(train_data)))
        test_data.ids = list(range(len(test_data)))
        train_sidxs = list(range(len(train_data)))
        random.shuffle(train_sidxs)
        valid_sample_idxs = train_sidxs[:int(len(train_data)* running_time_option['train_holdout'])]
        train_sample_idxs = train_sidxs[int(len(train_data)* running_time_option['train_holdout']):]
        task_data = {}
        for pid, party_name in enumerate(self.feddata['party_names']):
            pdata = self.feddata[party_name]['data']
            with_label, pt_feature, train_idxs, test_idxs = pdata['with_label'], pdata['pt_feature'], pdata['train'], pdata['test']
            local_train_data = self.TaskDataset(train_data, pt_feature, with_label, train_sample_idxs)
            local_valid_data = self.TaskDataset(train_data, pt_feature, with_label, valid_sample_idxs)
            local_test_data = self.TaskDataset(test_data, pt_feature, with_label, test_idxs)
            task_data[party_name] = {'train':local_train_data, 'valid':local_valid_data, 'test':local_test_data}
        return task_data

class TaskCalculator(BasicTaskCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        super(TaskCalculator, self).__init__(device, optimizer_name)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.DataLoader = torch.utils.data.DataLoader

    def compute_loss(self, model, data):
        """
        :param model: the model to train
        :param data: the training dataset
        :return: dict of train-one-step's result, which should at least contains the key 'loss'
        """
        model.to(self.device)
        tdata = self.to_device(data)
        outputs = model(tdata[0])
        loss = self.criterion(outputs, tdata[-1])
        return {'loss': loss}

    @torch.no_grad()
    def test(self, model, dataset, batch_size=64, num_workers=0):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_dataloader(dataset, batch_size=batch_size, num_workers=num_workers)
        total_loss = 0.0
        num_correct = 0
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.to_device(batch_data)
            outputs = model(batch_data[0])
            batch_mean_loss = self.criterion(outputs, batch_data[-1]).item()
            y_pred = outputs.data.max(1, keepdim=True)[1]
            correct = y_pred.eq(batch_data[-1].data.view_as(y_pred)).long().cpu().sum()
            num_correct += correct.item()
            total_loss += batch_mean_loss * len(batch_data[-1])
        return {'accuracy': 1.0*num_correct/len(dataset), 'loss':total_loss/len(dataset)}

    def to_device(self, data):
        return data[0].to(self.device), data[1].to(self.device)

    def get_dataloader(self, dataset, batch_size=64, shuffle=True, num_workers=0):
        if self.DataLoader == None:
            raise NotImplementedError("DataLoader Not Found.")
        return self.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)