"""
This is a non-official implementation of 'Subgraph Federated Learning with
Missing Neighbor Generation' (http://arxiv.org/abs/2106.13430). The FedSAGE+ will be added soon.
"""
from .fedbase import BasicClient,BasicServer
import torch
from utils import fmodule
import torch_geometric.utils

class TaskCalculator(fmodule.TaskCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        super(TaskCalculator, self).__init__(device, optimizer_name)
        self.criterion = torch.nn.NLLLoss()
        self.DataLoader1 = torch_geometric.loader.NeighborLoader
        self.DataLoader2 = torch_geometric.loader.DataLoader

    def train_one_step(self, model, data):
        tdata = self.data_to_device(data)
        outputs = model(tdata)
        loss = self.criterion(outputs[tdata.train_mask], tdata.y[tdata.train_mask])
        return {'loss': loss}

    @torch.no_grad()
    def test(self, model, dataset, batch_size=64, num_workers=0):
        loader = self.DataLoader2([dataset], batch_size=batch_size)
        total_loss = 0
        total_num_samples = 0
        for batch in loader:
            tdata = self.data_to_device(batch)
            outputs = model(tdata)
            loss = self.criterion(outputs[tdata.test_mask], tdata.y[tdata.test_mask])
            num_samples = len(tdata.x)
            total_loss += num_samples * loss
            total_num_samples += num_samples
        total_loss = total_loss.item()
        return {'loss': total_loss / total_num_samples}

    def data_to_device(self, data):
        return data.to(self.device)

    def get_data_loader(self, dataset, batch_size=64, shuffle=True, num_workers=0):
        return self.DataLoader1(dataset, batch_size=batch_size, shuffle=shuffle, num_neighbors=[3, 3], input_nodes=dataset.train_mask)

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.calculator = TaskCalculator(self.device, optimizer_name = option['optimizer'])


class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.calculator = TaskCalculator(self.device, optimizer_name = option['optimizer'])
