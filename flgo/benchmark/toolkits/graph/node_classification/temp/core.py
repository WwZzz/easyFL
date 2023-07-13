import os
import random
import torch
import torch_geometric.transforms as T
from flgo.benchmark.toolkits.graph.node_classification import GeneralCalculator, FromDatasetPipe, FromDatasetGenerator
from .config import train_data
try:
    import ujson as json
except:
    import json
try:
    from .config import test_data
except:
    test_data = None
try:
    from .config import val_data
except:
    val_data = None

class TaskGenerator(FromDatasetGenerator):
    def __init__(self, transductive=True):
        super(TaskGenerator, self).__init__(benchmark=os.path.split(os.path.dirname(__file__))[-1],
                                            train_data=train_data, val_data=val_data, test_data=test_data)
        self.transductive = transductive

class TaskPipe(FromDatasetPipe):
    def __init__(self, task_path):
        super(TaskPipe, self).__init__(task_path, train_data=train_data, val_data=val_data, test_data=test_data)

    def save_task(self, generator):
        client_names = self.gen_client_names(len(generator.local_datas))
        feddata = {'client_names': client_names, 'transductive': generator.transductive}
        for cid in range(len(client_names)):
            feddata[client_names[cid]] = {'data': generator.local_datas[cid], }
        with open(os.path.join(self.task_path, 'data.json'), 'w') as outf:
            json.dump(feddata, outf)
        return

    def load_data(self, running_time_option) -> dict:
        transductive = self.feddata['transductive']
        if self.test_data is not None:
            if self.val_data is None:
                if running_time_option['test_holdout']>0:
                    all_nodes = list(range(self.test_data.num_nodes))
                    random.shuffle(all_nodes)
                    k = int(self.test_data.num_nodes * running_time_option['test_holdout'])
                    val_nodes = all_nodes[:k]
                    test_nodes = all_nodes[k:]
                    if transductive:
                        test_mask = torch.BoolTensor([0 for _ in range(len(all_nodes))])
                        val_mask = torch.BoolTensor([0 for _ in range(len(all_nodes))])
                        val_mask[val_nodes] = True
                        test_mask[test_nodes] = True
                        self.test_data.val_mask = val_mask
                        self.test_data.test_mask = test_mask
                        server_test_data = self.TaskDataset(self.test_data, 'test')
                        server_val_data = self.TaskDataset(self.test_data, 'val')
                    else:
                        val_mask = torch.BoolTensor([0 for _ in range(len(all_nodes))])
                        val_mask[val_nodes] = True
                        val_data = self.test_data.subgraph(val_mask)
                        val_data.val_mask = torch.BoolTensor([1 for _ in range(val_data.num_nodes)])
                        test_mask = torch.BoolTensor([0 for _ in range(len(all_nodes))])
                        test_mask[test_nodes] = True
                        self.test_data.tesk_mask = test_mask
                        server_val_data = self.TaskDataset(val_data, 'val')
                        server_test_data = self.TaskDataset(self.test_data, 'test')
            else:
                self.test_data.test_mask = torch.BoolTensor([1 for _ in range(self.test_data.num_nodes)])
                self.val_data.val_mask = torch.BoolTensor([1 for _ in range(self.val_data.num_nodes)])
                server_test_data = self.TaskDataset(self.test_data, 'test')
                server_val_data = self.TaskDataset(self.val_data, 'val')
        elif self.val_data is not None:
            self.val_data.val_mask = torch.BoolTensor([1 for _ in range(self.val_data.num_nodes)])
            server_test_data = None
            server_val_data = self.TaskDataset(self.val_data, 'val')
        else:
            server_test_data = server_val_data = None
        task_data = {'server': {'test':server_test_data, 'val':server_val_data}}
        # rearrange data for clients
        num_val = running_time_option['train_holdout']
        if running_time_option['local_test']:
            num_val = 0.5 * num_val
            num_test = num_val
        else:
            num_test = 0.0
        for cid, cname in enumerate(self.feddata['client_names']):
            cnodes = self.feddata[cname]['data']
            cmask = torch.BoolTensor([0 for _ in range(self.train_data.num_nodes)])
            cmask[cnodes] = True
            cdata = self.train_data.subgraph(cmask)
            cdata = T.RandomNodeSplit(num_val=num_val, num_test=num_test)(cdata)
            if transductive:
                task_data[cname] = {'train': self.TaskDataset(cdata, 'train'), 'val': self.TaskDataset(cdata, 'val') if torch.any(cdata.val_mask) else None, 'test':self.TaskDataset(cdata, 'test') if torch.any(cdata.test_mask) else None}
            else:
                c_local_nodes = list(range(cdata.num_nodes))
                random.shuffle(c_local_nodes)
                ctrain_data = cdata.subgraph(cdata.train_mask)
                cval_data = cdata.subgraph(cdata.train_mask+cdata.val_mask)
                ctest_data = cdata
                task_data[cname] = {'train': self.TaskDataset(ctrain_data, 'train'), 'val': self.TaskDataset(cval_data, 'val') if num_val>0 else None, 'test': self.TaskDataset(ctest_data,'test') if num_test>0 else None}
        return task_data


TaskCalculator = GeneralCalculator