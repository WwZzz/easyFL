import community.community_louvain
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from benchmark.toolkits import BasicTaskGen, BasicTaskCalculator, BasicTaskPipe
import torch_geometric.utils
import collections
import numpy as np
import os
import ujson


def node_index2mask(nodes, num_nodes):
    mask = torch.zeros(num_nodes)
    mask = mask.scatter(0, torch.tensor(nodes), 1).type(torch.bool)
    return mask

class TaskGen(BasicTaskGen):
    def __init__(self, dist_id, num_clients = 1, skewness = 0.5, local_hld_rate=0.2, seed=0):
        super(TaskGen, self).__init__(benchmark='pubmed_node_classification',
                                      dist_id=dist_id,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/RAW_DATA/PUBMED',
                                      local_hld_rate=local_hld_rate,
                                      seed=seed
                                      )
        self.num_clients = num_clients
        self.cnames = self.get_client_names()
        self.taskname = self.get_taskname()
        self.taskpath = os.path.join(self.task_rootpath, self.taskname)

    def holdout(self, local_datas):
        self.test_data = []
        self.train_datas = []
        self.valid_datas = []
        for i, local_data in enumerate(local_datas):
            num_nodes = len(local_data)
            if num_nodes<3: self.test_data.extend(local_data)
            k3 = num_nodes-3
            k1 = int(k3*0.1)
            k2 = int(k3*0.2)
            test_nodes = local_data[:k1]
            valid_nodes = local_data[k1:k2]
            train_nodes = local_data[k2:k3]
            train_nodes.append(local_data[num_nodes-3])
            valid_nodes.append(local_data[num_nodes-2])
            test_nodes.append(local_data[num_nodes-1])
            self.test_data.extend(test_nodes)
            self.train_datas.append(train_nodes)
            self.valid_datas.append(valid_nodes)

    def load_data(self):
        self.all_data = Planetoid(root=self.rawdata_path, name='PubMed')
        self.G = torch_geometric.utils.to_networkx(self.all_data[0], to_undirected=self.all_data[0].is_undirected())

    def partition(self):
        local_datas = [[] for _ in range(self.num_clients)]
        if self.dist_id==0:
            node_groups = community.community_louvain.best_partition(self.G)
            groups = collections.defaultdict(list)
            for ni, gi in node_groups.items():
                groups[gi].append(ni)
            groups = {k: groups[k] for k in list(range(len(groups)))}
            # ensure the number of groups is larger than the number of clients
            while len(groups)<self.num_clients:
                # find the group with the largest size
                groups_lens = [groups[k] for k in range(len(groups))]
                max_gi = np.argmax(groups_lens)
                # set the size of the new group
                min_glen = min(groups_lens)
                max_glen = max(groups_lens)
                if max_glen < 2 * min_glen: min_glen = max_glen // 2
                # split the group with the largest size into two groups
                nodes_in_gi = groups[max_gi]
                new_group_id = len(groups)
                groups[new_group_id] = nodes_in_gi[:min_glen]
                groups[max_gi] = nodes_in_gi[min_glen:]
            # allocate different groups to clients
            groups_lens = [groups[k] for k in range(len(groups))]
            group_ids = np.argsort(groups_lens)
            for gi in group_ids:
                cid = np.argmin([len(li) for li in local_datas])
                local_datas[cid].extend(groups[gi])
        elif self.dist_id==1:
            nodes = list(range(len(self.G.nodes)))
            marks = self.G.y




        return local_datas

    def run(self):
        if self._check_task_exist():
            print("Task Already Exists.")
            return
        self.load_data()
        local_datas = self.partition()
        self.holdout(local_datas)
        try:
            # create the directory of the task
            self.create_task_directories()
            TaskPipe.save_task(self)
        except Exception as e:
            print(e)
            self._remove_task()
            print("Failed to saving splited dataset.")
        print('Done.')
        return


class TaskPipe(BasicTaskPipe):
    @classmethod
    def save_task(cls, generator):
        feddata = {
            'store': 'node_id',
            'client_names': generator.cnames,
            'dtest': generator.test_data
        }
        for cid in range(len(generator.cnames)):
            feddata[generator.cnames[cid]] = {
                'dtrain': generator.train_datas[cid],
                'dvalid': generator.valid_datas[cid]
            }
        with open(os.path.join(generator.taskpath, 'data.json'), 'w') as outf:
            ujson.dump(feddata, outf)
        return

    @classmethod
    def load_task(cls, task_path):
        with open(os.path.join(task_path, 'data.json'), 'r') as inf:
            feddata = ujson.load(inf)
        all_data = Planetoid(root='./benchmark/RAW_DATA/PUBMED', name='PubMed')
        # reset node masks
        num_nodes = len(all_data.data.x)
        train_datas = [feddata[cname]['dtrain'] for cname in feddata['client_names']]
        valid_datas = [feddata[cname]['dvalid'] for cname in feddata['client_names']]
        test_data = feddata['dtest']
        all_train_nodes = []
        all_test_nodes = []
        for train_nodes in train_datas:all_train_nodes.extend(train_nodes)
        for test_nodes in valid_datas:all_test_nodes.extend(test_nodes)
        all_test_nodes.extend(test_data)
        train_mask = node_index2mask(all_train_nodes, num_nodes)
        test_mask = node_index2mask(all_test_nodes, num_nodes)
        all_data.data.train_mask = train_mask
        all_data.data.test_mask = test_mask
        for cid in range(len(feddata['client_names'])):
            local_node_indices = train_datas[cid]+valid_datas[cid]
            cdata = all_data.data.subgraph(torch.tensor(local_node_indices, dtype=torch.long))
            valid_datas[cid] = train_datas[cid] = cdata
        test_data = all_data.data.subgraph(torch.tensor(test_data, dtype=torch.long))
        return train_datas, valid_datas, test_data, feddata['client_names']

class TaskCalculator(BasicTaskCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        super(TaskCalculator, self).__init__(device, optimizer_name)
        self.criterion = torch.nn.NLLLoss()
        self.DataLoader = torch_geometric.loader.DataLoader

    def train_one_step(self, model, data):
        tdata = self.data_to_device(data)
        outputs = model(tdata)
        loss = self.criterion(outputs[tdata.train_mask], tdata.y[tdata.train_mask])
        return {'loss': loss}

    @torch.no_grad()
    def test(self, model, dataset, batch_size=64, num_workers=0):
        loader = self.DataLoader([dataset], batch_size=batch_size)
        total_loss = 0
        total_num_samples = 0
        for batch in loader:
            tdata = self.data_to_device(batch)
            outputs = model(tdata)
            loss = self.criterion(outputs[tdata.test_mask], tdata.y[tdata.test_mask])
            num_samples = len(tdata.x)
            total_loss+= num_samples*loss
            total_num_samples += num_samples
        total_loss=total_loss.item()
        return {'loss': total_loss/total_num_samples}

    def data_to_device(self, data):
        return data.to(self.device)

    def get_data_loader(self, dataset, batch_size=64, shuffle=True, num_workers=0):
        return self.DataLoader([dataset], batch_size=batch_size, shuffle=shuffle)