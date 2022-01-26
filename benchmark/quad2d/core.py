from benchmark.toolkits import BasicTaskGen, XTaskReader, BasicTaskCalculator
import numpy as np
import os.path
import ujson
import torch
from torch.utils.data import DataLoader

class TaskGen(BasicTaskGen):
    def __init__(self, dimension=2, dist_id = 5, num_clients = 30, skewness = 0.5, minvol=10, rawdata_path ='./benchmark/quad2d/data'):
        super(TaskGen, self).__init__(benchmark='quad2d',
                                      dist_id=dist_id,
                                      skewness=skewness,
                                      rawdata_path=rawdata_path)
        self.dimension = dimension
        self.num_clients = num_clients
        self.taskname = self.get_taskname()
        self.taskpath = os.path.join(self.rootpath, self.taskname)
        self.minvol = minvol

    def run(self):
        if not self._check_task_exist():
            self.create_task_directories()
        else:
            print("Task Already Exists.")
            return
        es = [[] for _ in range(self.num_clients)]
        emeans = []
        if self.dist_id==5:
            for i in range(self.num_clients):
                mean_ei = np.random.normal(0, self.skewness, self.dimension)
                emeans.append(mean_ei)
                if self.minvol == 1:
                    es[i].append(mean_ei.tolist())
                else:
                    es[i].extend(np.random.multivariate_normal(mean_ei, 0.05*np.eye(self.dimension), self.minvol).tolist())
        mean_e = np.mean(emeans, axis=0)
        self.cnames = self.get_client_names()
        self.test_data = {'x': [mean_e.tolist()]}
        feddata = {
            'store': 'X',
            'client_names': self.cnames,
            'dtest': self.test_data
        }
        for i in range(self.num_clients):
            feddata[self.cnames[i]] = {
                'dtrain': {
                    'x': es[i],
                },
                'dvalid': {
                    'x': [emeans[i].tolist()],
                }
            }
        with open(os.path.join(self.taskpath, 'data.json'), 'w') as outf:
            ujson.dump(feddata, outf)

class TaskReader(XTaskReader):
    def __init__(self, taskpath=''):
        super(TaskReader, self).__init__(taskpath)

class TaskCalculator(BasicTaskCalculator):
    def __init__(self, device):
        super(TaskCalculator, self).__init__(device)

    def get_loss(self, model, data):
        tdata = self.data_to_device(data)
        outputs = model(tdata)
        loss = 0.5 * torch.mean(outputs)
        return loss

    @torch.no_grad()
    def get_evaluation(self, model, data):
        tdata = self.data_to_device(data)
        outputs = model(tdata)
        loss = 0.5 * torch.mean(outputs)
        return loss.item()

    @torch.no_grad()
    def test(self, model, data):
        tdata = self.data_to_device(data)
        outputs = model(tdata)
        loss = 0.5 * torch.mean(outputs)
        return loss.item(), loss.item()

    def data_to_device(self, data):
        return data.to(self.device)

    def get_data_loader(self, dataset, batch_size=64, shuffle=True):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
