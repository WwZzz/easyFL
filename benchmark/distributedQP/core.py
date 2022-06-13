from benchmark.toolkits import BasicTaskGen, XTaskPipe, BasicTaskCalculator
import numpy as np
import os.path
import torch
from torch.utils.data import DataLoader

class TaskGen(BasicTaskGen):
    def __init__(self, dimension=2, dist_id = 5, num_clients = 30, skewness = 0.5, minvol=10, rawdata_path ='./benchmark/RAW_DATA/distributedQP', local_hld_rate=0.2, seed=0):
        super(TaskGen, self).__init__(benchmark='distributedQP',
                                      dist_id=dist_id,
                                      skewness=skewness,
                                      rawdata_path=rawdata_path,
                                      local_hld_rate=local_hld_rate,
                                      seed=seed)
        self.dimension = dimension
        self.num_clients = num_clients
        self.save_task = XTaskPipe.save_task
        self.taskname = self.get_taskname()
        self.taskpath = os.path.join(self.task_rootpath, self.taskname)
        self.minvol = minvol
        self.cnames = self.get_client_names()

    def run(self):
        """ Generate federated task"""
        # check if the task exists
        if self._check_task_exist():
            print("Task Already Exists.")
            return
        # read raw_data into self.train_data and self.test_data
        print('-----------------------------------------------------')
        print('Loading...')
        self.load_data()
        print('Done.')
        # save task infomation as .json file and the federated dataset
        print('-----------------------------------------------------')
        print('Saving data...')
        try:
            # create the directory of the task
            self.create_task_directories()
            self.save_task(self)
        except:
            self._remove_task()
            print("Failed to saving splited dataset.")
        print('Done.')
        return

    def load_data(self):
        es = [[] for _ in range(self.num_clients)]
        emeans = []
        data_size = self.minvol+int(self.minvol/(1-self.local_holdout_rate)*self.local_holdout_rate)
        for i in range(self.num_clients):
            mean_ei = np.random.normal(0, self.skewness, self.dimension)
            emeans.append(mean_ei)
            es[i].extend(np.random.multivariate_normal(mean_ei,0.05*np.eye(self.dimension),data_size).tolist())
        mean_e = np.mean(emeans, axis=0)
        self.test_data = {'x': [mean_e.tolist()]}
        self.train_data = []
        self.train_cidxs = [[] for _ in range(self.num_clients)]
        self.valid_cidxs = [[] for _ in range(self.num_clients)]
        did = 0
        for i in range(self.num_clients):
            self.train_data.extend(es[i])
            self.train_cidxs[i].extend([did + i for i in range(self.minvol)])
            self.valid_cidxs[i].extend([did + i for i in range(self.minvol, len(es[i]))])
            did = did + len(es[i])

class TaskPipe(XTaskPipe):
    def __init__(self):
        super(TaskPipe, self).__init__()

class TaskCalculator(BasicTaskCalculator):
    def __init__(self, device):
        super(TaskCalculator, self).__init__(device)

    def train_one_step(self, model, data):
        tdata = self.data_to_device(data)
        outputs = model(tdata)
        loss = 0.5 * torch.mean(outputs)
        return {'loss': loss}

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
        return {'loss': loss.item()}

    def data_to_device(self, data):
        return data.to(self.device)

    def get_data_loader(self, dataset, batch_size=64, shuffle=True):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
