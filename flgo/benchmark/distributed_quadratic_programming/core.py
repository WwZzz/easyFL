from flgo.benchmark.base import BasicTaskGenerator, BasicTaskCalculator
from flgo.benchmark.base import XHorizontalTaskPipe as TaskPipe
import numpy as np
import torch
from torch.utils.data import DataLoader

class TaskGenerator(BasicTaskGenerator):
    def __init__(self, dimension=2, num_clients = 30, data_size = 400, std=1.0):
        super(TaskGenerator, self).__init__('distributed_quadratic_programming', '')
        self.dimension = dimension
        self.num_clients = num_clients
        self.data_size = data_size
        self.std = std

    def load_data(self):
        es = [[] for _ in range(self.num_clients)]
        emeans = []
        for i in range(self.num_clients):
            mean_ei = np.random.normal(0, self.std, self.dimension)
            emeans.append(mean_ei)
            es[i].extend(np.random.multivariate_normal(mean_ei, 0.05*np.eye(self.dimension), self.data_size).tolist())
        mean_e = np.mean(emeans, axis=0)
        self.test_data = {'x': [mean_e.tolist()]}
        self.local_datas = []
        for i in range(self.num_clients):
            self.local_datas.append({'x':es[i]})

    def get_task_name(self):
        return 'distqp_{}'.format(self.num_clients)

class TaskCalculator(BasicTaskCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        super(TaskCalculator, self).__init__(device, optimizer_name)
        self.DataLoader = DataLoader

    def compute_loss(self, model, data):
        tdata = self.to_device(data)
        outputs = model(tdata)
        loss = 0.5 * torch.mean(outputs)
        return {'loss': loss}

    @torch.no_grad()
    def test(self, model, data, batch_size=1, num_workers=0, pin_memory=False):
        tdata = self.to_device(data[0])
        outputs = model(tdata)
        loss = 0.5 * torch.mean(outputs)
        return {'loss': loss.item()}

    def to_device(self, data):
        return data[0].to(self.device)

    def get_dataloader(self, dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False, drop_last=False, *args, **kwargs):
        if self.DataLoader == None:
            raise NotImplementedError("DataLoader Not Found.")
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)