import os
import random

import torch
import ujson
from torch.utils.data import Subset
from torch_geometric.datasets import TUDataset
from benchmark.toolkits import BasicTaskCalculator, BasicTaskPipe
from benchmark.toolkits import DefaultTaskGen
import numpy as np
import collections
from torch_geometric.loader import DataLoader

class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients = 1, skewness = 0.5, local_hld_rate=0.2, seed=0):
        super(TaskGen, self).__init__(benchmark='enzymes_graph_classification',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/RAW_DATA/ENZYMES',
                                      local_hld_rate=local_hld_rate,
                                      seed=seed
                                      )
        self.num_classes = 6
        self.save_task = TaskPipe.save_task

    def load_data(self):
        self.all_data, self.perm = TUDataset(root=self.rawdata_path, name='ENZYMES').shuffle(return_perm=True)
        self.num_samples = len(self.all_data)
        k = int(0.9*self.num_samples)
        self.train_data = self.all_data[:k]
        self.test_data = list(range(self.num_samples))[k:]

    def partition(self):
        # Partition self.train_data according to the delimiter and return indexes of data owned by each client as [c1data_idxs, ...] where the type of each element is list(int)
        if self.dist_id == 0:
            """IID"""
            d_idxs = np.random.permutation(len(self.train_data))
            local_datas = np.array_split(d_idxs, self.num_clients)
            local_datas = [data_idx.tolist() for data_idx in local_datas]

        elif self.dist_id == 1:
            """label_skew_quantity"""
            dpairs = [[did, self.train_data[did].y] for did in range(len(self.train_data))]
            num = max(int((1 - self.skewness) * self.num_classes), 1)
            K = self.num_classes
            local_datas = [[] for _ in range(self.num_clients)]
            if num == K:
                for k in range(K):
                    idx_k = [p[0] for p in dpairs if p[1] == k]
                    np.random.shuffle(idx_k)
                    split = np.array_split(idx_k, self.num_clients)
                    for cid in range(self.num_clients):
                        local_datas[cid].extend(split[cid].tolist())
            else:
                times = [0 for _ in range(self.num_classes)]
                contain = []
                for i in range(self.num_clients):
                    current = []
                    j = 0
                    while (j < num):
                        mintime = np.min(times)
                        ind = np.random.choice(np.where(times == mintime)[0])
                        if (ind not in current):
                            j = j + 1
                            current.append(ind)
                            times[ind] += 1
                    contain.append(current)
                for k in range(K):
                    idx_k = [p[0] for p in dpairs if p[1] == k]
                    np.random.shuffle(idx_k)
                    split = np.array_split(idx_k, times[k])
                    ids = 0
                    for cid in range(self.num_clients):
                        if k in contain[cid]:
                            local_datas[cid].extend(split[ids].tolist())
                            ids += 1

        elif self.dist_id == 2:
            """label_skew_dirichlet"""
            """alpha = (-4log(skewness + epsilon))**4"""
            MIN_ALPHA = 0.01
            alpha = (-4 * np.log(self.skewness + 10e-8)) ** 4
            alpha = max(alpha, MIN_ALPHA)
            labels = [self.train_data[did].y for did in range(len(self.train_data))]
            lb_counter = collections.Counter(labels)
            p = np.array([1.0 * v / len(self.train_data) for v in lb_counter.values()])
            lb_dict = {}
            labels = np.array(labels)
            for lb in range(len(lb_counter.keys())):
                lb_dict[lb] = np.where(labels == lb)[0]
            proportions = [np.random.dirichlet(alpha * p) for _ in range(self.num_clients)]
            while np.any(np.isnan(proportions)):
                proportions = [np.random.dirichlet(alpha * p) for _ in range(self.num_clients)]
            while True:
                # generate dirichlet distribution till ||E(proportion) - P(D)||<=1e-5*self.num_classes
                mean_prop = np.mean(proportions, axis=0)
                error_norm = ((mean_prop - p) ** 2).sum()
                print("Error: {:.8f}".format(error_norm))
                if error_norm <= 1e-2 / self.num_classes:
                    break
                exclude_norms = []
                for cid in range(self.num_clients):
                    mean_excid = (mean_prop * self.num_clients - proportions[cid]) / (self.num_clients - 1)
                    error_excid = ((mean_excid - p) ** 2).sum()
                    exclude_norms.append(error_excid)
                excid = np.argmin(exclude_norms)
                sup_prop = [np.random.dirichlet(alpha * p) for _ in range(self.num_clients)]
                alter_norms = []
                for cid in range(self.num_clients):
                    if np.any(np.isnan(sup_prop[cid])):
                        continue
                    mean_alter_cid = mean_prop - proportions[excid] / self.num_clients + sup_prop[
                        cid] / self.num_clients
                    error_alter = ((mean_alter_cid - p) ** 2).sum()
                    alter_norms.append(error_alter)
                if len(alter_norms) > 0:
                    alcid = np.argmin(alter_norms)
                    proportions[excid] = sup_prop[alcid]
            local_datas = [[] for _ in range(self.num_clients)]
            self.dirichlet_dist = []  # for efficiently visualizing
            for lb in lb_counter.keys():
                lb_idxs = lb_dict[lb]
                lb_proportion = np.array([pi[lb] for pi in proportions])
                lb_proportion = lb_proportion / lb_proportion.sum()
                lb_proportion = (np.cumsum(lb_proportion) * len(lb_idxs)).astype(int)[:-1]
                lb_datas = np.split(lb_idxs, lb_proportion)
                self.dirichlet_dist.append([len(lb_data) for lb_data in lb_datas])
                local_datas = [local_data + lb_data.tolist() for local_data, lb_data in zip(local_datas, lb_datas)]
            self.dirichlet_dist = np.array(self.dirichlet_dist).T
            for i in range(self.num_clients):
                np.random.shuffle(local_datas[i])

        elif self.dist_id == 3:
            """label_skew_shard"""
            dpairs = [[did, self.train_data[did].y] for did in range(len(self.train_data))]
            self.skewness = min(max(0, self.skewness), 1.0)
            num_shards = max(int((1 - self.skewness) * self.num_classes * 2), 1)
            client_datasize = int(len(self.train_data) / self.num_clients)
            all_idxs = [i for i in range(len(self.train_data))]
            z = zip([p[1] for p in dpairs], all_idxs)
            z = sorted(z)
            labels, all_idxs = zip(*z)
            shardsize = int(client_datasize / num_shards)
            idxs_shard = range(int(self.num_clients * num_shards))
            local_datas = [[] for i in range(self.num_clients)]
            for i in range(self.num_clients):
                rand_set = set(np.random.choice(idxs_shard, num_shards, replace=False))
                idxs_shard = list(set(idxs_shard) - rand_set)
                for rand in rand_set:
                    local_datas[i].extend(all_idxs[rand * shardsize:(rand + 1) * shardsize])

        elif self.dist_id == 4:
            """label_skew_dirichlet with imbalance data size. The data"""
            # calculate alpha = (-4log(skewness + epsilon))**4
            MIN_ALPHA = 0.01
            alpha = (-4 * np.log(self.skewness + 10e-8)) ** 4
            alpha = max(alpha, MIN_ALPHA)
            # ensure imbalance data sizes
            total_data_size = len(self.train_data)
            mean_datasize = total_data_size / self.num_clients
            mu = np.log(mean_datasize) - 0.5
            sigma = 1
            samples_per_client = np.random.lognormal(mu, sigma, (self.num_clients)).astype(int)
            thresold = int(0.1 * total_data_size)
            delta = int(0.1 * thresold)
            crt_data_size = sum(samples_per_client)
            # force current data size to match the total data size
            while crt_data_size != total_data_size:
                if crt_data_size - total_data_size >= thresold:
                    maxid = np.argmax(samples_per_client)
                    samples = np.random.lognormal(mu, sigma, (self.num_clients))
                    new_size_id = np.argmin(
                        [np.abs(crt_data_size - samples_per_client[maxid] + s) for s in samples])
                    samples_per_client[maxid] = samples[new_size_id]
                elif crt_data_size - total_data_size >= delta:
                    maxid = np.argmax(samples_per_client)
                    samples_per_client[maxid] -= delta
                elif crt_data_size - total_data_size > 0:
                    maxid = np.argmax(samples_per_client)
                    samples_per_client[maxid] -= (crt_data_size - total_data_size)
                elif total_data_size - crt_data_size >= delta:
                    minid = np.argmin(samples_per_client)
                    samples_per_client[minid] += delta
                elif total_data_size - crt_data_size >= delta:
                    minid = np.argmin(samples_per_client)
                    samples = np.random.lognormal(mu, sigma, (self.num_clients))
                    new_size_id = np.argmin(
                        [np.abs(crt_data_size - samples_per_client[minid] + s) for s in samples])
                    samples_per_client[minid] = samples[new_size_id]
                else:
                    minid = np.argmin(samples_per_client)
                    samples_per_client[minid] += (total_data_size - crt_data_size)
                crt_data_size = sum(samples_per_client)
            # count the label distribution
            labels = [self.train_data[did].y for did in range(len(self.train_data))]
            lb_counter = collections.Counter(labels)
            p = np.array([1.0 * v / len(self.train_data) for v in lb_counter.values()])
            lb_dict = {}
            labels = np.array(labels)
            for lb in range(len(lb_counter.keys())):
                lb_dict[lb] = np.where(labels == lb)[0]
            proportions = [np.random.dirichlet(alpha * p) for _ in range(self.num_clients)]
            while np.any(np.isnan(proportions)):
                proportions = [np.random.dirichlet(alpha * p) for _ in range(self.num_clients)]
            sorted_cid_map = {k: i for k, i in
                              zip(np.argsort(samples_per_client), [_ for _ in range(self.num_clients)])}
            crt_id = 0
            while True:
                # generate dirichlet distribution till ||E(proportion) - P(D)||<=1e-5*self.num_classes
                mean_prop = np.sum([pi * di for pi, di in zip(proportions, samples_per_client)], axis=0)
                mean_prop = mean_prop / mean_prop.sum()
                error_norm = ((mean_prop - p) ** 2).sum()
                print("Error: {:.8f}".format(error_norm))
                if error_norm <= 1e-2 / self.num_classes:
                    break
                excid = sorted_cid_map[crt_id]
                crt_id = (crt_id + 1) % self.num_clients
                sup_prop = [np.random.dirichlet(alpha * p) for _ in range(self.num_clients)]
                del_prop = np.sum([pi * di for pi, di in zip(proportions, samples_per_client)], axis=0)
                del_prop -= samples_per_client[excid] * proportions[excid]
                alter_norms = []
                for cid in range(self.num_clients):
                    if np.any(np.isnan(sup_prop[cid])):
                        continue
                    alter_prop = del_prop + samples_per_client[excid] * sup_prop[cid]
                    alter_prop = alter_prop / alter_prop.sum()
                    error_alter = ((alter_prop - p) ** 2).sum()
                    alter_norms.append(error_alter)
                if len(alter_norms) > 0:
                    alcid = np.argmin(alter_norms)
                    proportions[excid] = sup_prop[alcid]
            local_datas = [[] for _ in range(self.num_clients)]
            self.dirichlet_dist = []  # for efficiently visualizing
            for lb in lb_counter.keys():
                lb_idxs = lb_dict[lb]
                lb_proportion = np.array([pi[lb] * si for pi, si in zip(proportions, samples_per_client)])
                lb_proportion = lb_proportion / lb_proportion.sum()
                lb_proportion = (np.cumsum(lb_proportion) * len(lb_idxs)).astype(int)[:-1]
                lb_datas = np.split(lb_idxs, lb_proportion)
                self.dirichlet_dist.append([len(lb_data) for lb_data in lb_datas])
                local_datas = [local_data + lb_data.tolist() for local_data, lb_data in zip(local_datas, lb_datas)]
            self.dirichlet_dist = np.array(self.dirichlet_dist).T
            for i in range(self.num_clients):
                np.random.shuffle(local_datas[i])
        return local_datas

class TaskPipe(BasicTaskPipe):
    TaskDataset = Subset
    @classmethod
    def save_task(cls, generator):
        feddata = {
            'store': 'IDX',
            'client_names': generator.cnames,
            'dtest': generator.test_data,
            'perm': generator.perm.tolist(),
        }
        for cid in range(len(generator.cnames)):
            feddata[generator.cnames[cid]] = {
                'dtrain': generator.train_cidxs[cid],
                'dvalid': generator.valid_cidxs[cid]
            }
        with open(os.path.join(generator.taskpath, 'data.json'), 'w') as outf:
            ujson.dump(feddata, outf)
        return
    @classmethod
    def load_task(cls, task_path):
        with open(os.path.join(task_path, 'data.json'), 'r') as inf:
            feddata = ujson.load(inf)
        dataset = TUDataset(root='./benchmark/RAW_DATA/ENZYMES', name='ENZYMES')
        dataset = dataset[feddata['perm']]
        test_data = dataset[feddata['dtest']]
        train_datas = []
        valid_datas = []
        for name in feddata['client_names']:
            train_data = feddata[name]['dtrain']
            valid_data = feddata[name]['dvalid']
            if cls._cross_validation:
                k = len(train_data)
                train_data.extend(valid_data)
                random.shuffle(train_data)
                all_data = train_data
                train_data = all_data[:k]
                valid_data = all_data[k:]
            if cls._train_on_all:
                train_data.extend(valid_data)
            train_datas.append(dataset[train_data])
            valid_datas.append(dataset[valid_data])
        return train_datas, valid_datas, test_data, feddata['client_names']

class TaskCalculator(BasicTaskCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        super(TaskCalculator, self).__init__(device, optimizer_name)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.DataLoader = DataLoader

    def train_one_step(self, model, data):
        """
        :param model: the model to train
        :param data: the training dataset
        :return: dict of train-one-step's result, which should at least contains the key 'loss'
        """
        tdata = self.data_to_device(data)
        outputs = model(tdata)
        loss = self.criterion(outputs, tdata.y)
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
        if batch_size == -1: batch_size = len(dataset)
        data_loader = self.get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers)
        total_loss = 0.0
        num_correct = 0
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.data_to_device(batch_data)
            outputs = model(batch_data)
            batch_mean_loss = self.criterion(outputs, batch_data.y).item()
            y_pred = outputs.argmax(dim=1)
            correct = int((y_pred == batch_data.y).sum())
            num_correct += correct
            total_loss += batch_mean_loss * len(batch_data.y)
        return {'accuracy': 1.0 * num_correct / len(dataset), 'loss': total_loss / len(dataset)}

    def data_to_device(self, data):
        return data.to(self.device)

    def get_data_loader(self, dataset, batch_size=64, shuffle=True, num_workers=0):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)




