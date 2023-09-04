"""
This is a non-official implementation of personalized FL method FedProto (https://ojs.aaai.org/index.php/AAAI/article/view/20819).
The original implementation is at https://github.com/yuetan031/FedProto
"""
import collections
import copy
import torch
import torch.utils.data.dataset
import torch.nn as nn
import flgo.algorithm.fedbase
import flgo.utils.fmodule as fmodule
import numpy as np
import cvxpy as cvx

def pairwise(data):
    """ Simple generator of the pairs (x, y) in a tuple such that index x < index y.
    Args:
    data Indexable (including ability to query length) containing the elements
    Returns:
    Generator over the pairs of the elements of 'data'
    """
    n = len(data)
    for i in range(n):
        for j in range(i, n):
            yield (data[i], data[j])

class Server(flgo.algorithm.fedbase.BasicServer):
    def initialize(self):
        self.init_algo_para({'lmbd':0.1})
        self.num_classes = len(collections.Counter([d[-1] for d in self.test_data]))
        self.sample_option = 'full'
        with torch.no_grad():
            x = self.test_data[0]
            self.model.to('cpu')
            h = self.model.encoder(x[0].unsqueeze(0))
            self.dim = h.shape[-1]
        for c in self.clients:
            c.num_classes = self.num_classes
            c.dim = self.dim
        self.c = {}

    def pack(self, client_id, mtype=0):
        return {'c': copy.deepcopy(self.c),}

    def iterate(self):
        self.selected_clients = self.sample()
        res = self.communicate(self.selected_clients)
        cs, sizes_label = res['c'], res['sizes_label']
        self.c = self.aggregate(cs, sizes_label)
        return

    def aggregate(self, cs:list, sizes_label:list):
        if len(cs)==0: return self.c
        num_samples = np.sum(sizes_label, axis=0)
        num_j_clients = np.zeros(self.num_classes)
        new_c = {j:torch.zeros((1, self.dim)) for j in range(self.num_classes)}
        for j in range(self.num_classes):
            for ci, i, si in zip(cs, self.received_clients, sizes_label):
                if si[j]==0:continue
                num_j_clients += 1
                new_c[j] += (ci[j]*si[j]/num_samples[j])
            # new_c[j]/=num_j_clients
        return new_c

class Client(flgo.algorithm.fedbase.BasicClient):
    def initialize(self):
        self.model = copy.deepcopy(self.server.model).to('cpu')
        label_counter = collections.Counter([d[-1] for d in self.train_data])
        self.sizes_label = np.zeros(self.num_classes)
        for lb in range(self.num_classes):
            if lb in label_counter.keys():
                self.sizes_label[lb] = label_counter[lb]
        self.probs_label = self.sizes_label/self.sizes_label.sum()
        self.loss_mse = nn.MSELoss()

    def reply(self, svr_pkg):
        cg = self.unpack(svr_pkg)
        self.train(self.model, cg)
        return self.pack()

    def unpack(self, svr_pkg):
        return svr_pkg['c']

    def pack(self):
        c = {}
        self.model.to(self.device)
        with torch.no_grad():
            dataloader = self.calculator.get_dataloader(self.train_data, self.batch_size)
            for batch_id, batch_data in enumerate(dataloader):
                batch_data = self.calculator.to_device(batch_data)
                protos = self.model.encoder(batch_data[0]).detach()
                labels = batch_data[-1]
                for i in range(len(labels)):
                    yi = labels[i].item()
                    if yi not in c.keys():
                        c[yi] = protos[i]
                    else:
                        c[yi] += protos[i]
            for j in range(len(self.sizes_label)):
                if self.sizes_label[j]==0: continue
                c[j]/=self.sizes_label[j]
                c[j] = c[j].to('cpu')
        return {'c': c, 'sizes_label': self.sizes_label}

    @fmodule.with_multi_gpus
    def train(self, model, cg):
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        for iter in range(self.num_steps):
            model.zero_grad()
            batch_data = self.calculator.to_device(self.get_batch_data())
            protos = self.model.encoder(batch_data[0])
            labels = batch_data[-1]
            outputs = self.model.head(protos)
            loss_erm = self.calculator.criterion(outputs, labels)
            protos_new = copy.deepcopy(protos.data)
            for i,label in enumerate(labels):
                if label.item() in cg.keys():
                    protos_new[i, :] = cg[label.item()][0].data
            loss_reg = self.loss_mse(protos_new, protos)
            loss = loss_erm + self.lmbd*loss_reg
            loss.backward()
            optimizer.step()
        return