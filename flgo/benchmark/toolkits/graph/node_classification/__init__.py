import community.community_louvain
import torch_geometric.transforms as T
from torch_geometric.utils import mask_to_index, index_to_mask, from_networkx
import torch_geometric.utils
import collections
from flgo.benchmark.base import *
import networkx as nx

class BuiltinClassGenerator(BasicTaskGenerator):
    def __init__(self, benchmark, rawdata_path, builtin_class, transform=None, pre_transform=None, test_rate=0.2, transductive=True):
        super(BuiltinClassGenerator, self).__init__(benchmark, rawdata_path)
        self.builtin_class = builtin_class
        self.transform = transform
        self.pre_transform = pre_transform
        self.transductive = transductive
        self.test_rate = test_rate
        self.additional_option = {}
        self.train_additional_option = {}
        self.test_additional_option = {}
        self.download = True

    def load_data(self):
        default_init_para = {'root': self.rawdata_path, 'download':self.download, 'train':True, 'transform':self.transform, 'pre_transform':self.pre_transform}
        default_init_para.update(self.additional_option)
        if 'kwargs' not in self.builtin_class.__init__.__annotations__:
            pop_key = [k for k in default_init_para.keys() if k not in self.builtin_class.__init__.__annotations__]
            for k in pop_key: default_init_para.pop(k)
        self.dataset = T.RandomNodeSplit(split='train_rest', num_val=0.0, num_test=self.test_rate)(self.builtin_class(**default_init_para).data)
        self.G = torch_geometric.utils.to_networkx(self.dataset, to_undirected=self.dataset.is_undirected(), node_attrs=['x', 'y', 'train_mask', 'val_mask', 'test_mask'])
        self.test_nodes = mask_to_index(self.dataset.test_mask).tolist() if self.test_rate>0 else []
        self.train_nodes = mask_to_index(self.dataset.train_mask).tolist()
        self.train_data = nx.subgraph(self.G, self.train_nodes)

    def partition(self):
        self.local_datas = self.partitioner(self.train_data)
        self.num_clients = len(self.local_datas)

    def get_task_name(self):
        return '_'.join(['B-'+self.benchmark,  'P-None', 'N-'+str(self.num_clients)])

class BuiltinClassPipe(BasicTaskPipe):
    class TaskDataset(torch.utils.data.Dataset):
        def __init__(self, data, flag='test'):
            if data is None: return None
            self.flag = flag
            self.data = data
            if flag=='valid':
                self.mask = self.data.val_mask
            elif flag=='train':
                self.mask = self.data.train_mask
            else:
                self.mask = self.data.test_mask
            if len(self.mask)==0: return None
            self.test_mask = self.data.test_mask

        def change_mask_for_test(self):
            # train, test, val
            self.data.test_mask = self.mask

        def restore_mask(self):
            self.data.test_mask = self.test_mask

        def __getitem__(self, item):
            return self.data

        def __len__(self):
            return len(self.mask)

    def __init__(self, task_name, buildin_class, transform=None, pre_transform=None):
        super(BuiltinClassPipe, self).__init__(task_name)
        self.builtin_class = buildin_class
        self.pre_transform = pre_transform
        self.transform = transform
        # self.train_data = from_networkx(nx.subgraph(G, self.train_nodes))

    def save_task(self, generator):
        client_names = self.gen_client_names(len(generator.local_datas))
        feddata = {'client_names': client_names, 'server_data': generator.test_nodes,
                   'rawdata_path': generator.rawdata_path, 'additional_option': generator.additional_option,
                   'train_additional_option': generator.train_additional_option,
                   'test_additional_option': generator.test_additional_option,
                   'transductive': generator.transductive,
                   'test_rate': generator.test_rate,
                   }
        for cid in range(len(client_names)): feddata[client_names[cid]] = {'data': generator.local_datas[cid], }
        with open(os.path.join(self.task_path, 'data.json'), 'w') as outf:
            json.dump(feddata, outf)
        return

    def load_data(self, running_time_option) -> dict:
        default_init_para = {'root': self.feddata['rawdata_path'], 'download': True, 'train': True, 'transform': self.transform, 'pre_transform':self.pre_transform}
        default_init_para.update(self.feddata['additional_option'])
        if 'kwargs' not in self.builtin_class.__init__.__annotations__:
            pop_key = [k for k in default_init_para.keys() if k not in self.builtin_class.__init__.__annotations__]
            for k in pop_key: default_init_para.pop(k)
        self.dataset = T.RandomNodeSplit(split='train_rest', num_val=0.0, num_test=0.0)(
            self.builtin_class(**default_init_para).data)
        G = torch_geometric.utils.to_networkx(self.dataset, to_undirected=self.dataset.is_undirected(),
                                              node_attrs=['x', 'y', 'train_mask', 'val_mask', 'test_mask'])
        if self.feddata['test_rate']>0:
            test_nodes = self.feddata['server_data']
            all_nodes = test_nodes
            random.shuffle(test_nodes)
            k = int(len(test_nodes)*running_time_option['test_holdout'])
            test_nodes = test_nodes[k:]
            valid_nodes = test_nodes[:k]
            if self.feddata['transductive']:
                test_data = from_networkx(nx.subgraph(G, all_nodes))
                test_nodes = [all_nodes.index(i) for i in test_nodes]
                valid_nodes = [all_nodes.index(i) for i in valid_nodes]
                test_mask = torch.BoolTensor([0 for _ in range(len(all_nodes))])
                val_mask = torch.BoolTensor([0 for _ in range(len(all_nodes))])
                test_mask[test_nodes] = True
                val_mask[valid_nodes] = True
                test_data.test_mask = test_mask
                test_data.val_mask = val_mask
                valid_data = test_data
            else:
                test_data = from_networkx(nx.subgraph(G, test_nodes))
                valid_data = from_networkx(nx.subgraph(G, valid_nodes))
        else:
            test_data = None
            valid_data = None
        task_data = {'server': {'test':self.TaskDataset(test_data, 'test'), 'valid':self.TaskDataset(valid_data,'valid')}}
        # rearrange data for clients
        for cid, cname in enumerate(self.feddata['client_names']):
            all_local_nodes = self.feddata[cname]['data']
            k2 = int(running_time_option['train_holdout']*len(all_local_nodes))
            k1 = int(0.5*running_time_option['train_holdout']*len(all_local_nodes)) if running_time_option['local_test'] else 0
            local_test_nodes = all_local_nodes[:k1]
            local_valid_nodes = all_local_nodes[k1:k2]
            local_train_nodes = all_local_nodes[k2:]
            if self.feddata['transductive']:
                cdata = from_networkx(nx.subgraph(G, all_local_nodes))
                local_test_nodes = [all_local_nodes.index(i) for i in local_test_nodes]
                local_train_nodes = [all_local_nodes.index(i) for i in local_train_nodes]
                local_valid_nodes = [all_local_nodes.index(i) for i in local_valid_nodes]
                train_mask = torch.BoolTensor([0 for _ in range(len(all_local_nodes))])
                test_mask = torch.BoolTensor([0 for _ in range(len(all_local_nodes))])
                val_mask = torch.BoolTensor([0 for _ in range(len(all_local_nodes))])
                test_mask[local_test_nodes] = True
                val_mask[local_valid_nodes] = True
                train_mask[local_train_nodes] = True
                cdata.test_mask = test_mask
                cdata.train_mask = train_mask
                cdata.val_mask = val_mask
                task_data[cname] = {'train': self.TaskDataset(cdata, 'train'), 'valid': self.TaskDataset(cdata, 'valid'), 'test':self.TaskDataset(cdata, 'test')}
            else:
                ctest_data = from_networkx(nx.subgraph(G, local_test_nodes))
                cvalid_data = from_networkx(nx.subgraph(G, local_valid_nodes))
                ctrain_data = from_networkx(nx.subgraph(G, local_train_nodes))
                task_data[cname] = {'train': self.TaskDataset(ctrain_data, 'train'), 'valid': self.TaskDataset(cvalid_data, 'valid') if len(local_valid_nodes)>0 else None, 'test': self.TaskDataset(ctest_data,'test') if len(local_test_nodes)>0 else None}
        return task_data

"""
load_data -> return task_data = {
    'server': {'test': anything, 'xxx': anything},
    'Client01': {'train': anything, ...}
    ...
}

generate_objects -> [object1, object2, ...]
object1.name = task_data[0] = 'server'
object2.name = task_data[1] = 'Client01'
....

distribute:
    specify the object according to the name of the object
        object_x
        x_data = task_data[x_name] (i.e. {'xxx':anything, ...})
        for key in x_data:
            object_x.set_data(key, x_data[key])
            
set_data(data_name, data):
    setattr(self, data_name+'_data', data)
    
"""
class GeneralCalculator(BasicTaskCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        super(GeneralCalculator, self).__init__(device, optimizer_name)
        self.criterion = torch.nn.NLLLoss()
        self.DataLoader = torch_geometric.loader.DataLoader

    def compute_loss(self, model, data):
        tdata = self.data_to_device(data)
        outputs = model(tdata)
        loss = self.criterion(outputs[tdata.train_mask], tdata.y[tdata.train_mask])
        return {'loss': loss}

    @torch.no_grad()
    def test(self, model, dataset, batch_size=64, num_workers=0, pin_memory=False):
        dataset.change_mask_for_test()
        loader = self.DataLoader([dataset.data], batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        total_loss = 0
        total_correct = 0
        total_num_samples = 0
        for batch in loader:
            tdata = self.data_to_device(batch)
            outputs = model(tdata)
            loss = self.criterion(outputs[tdata.test_mask], tdata.y[tdata.test_mask])
            num_samples = len(tdata.x)
            total_loss += num_samples * loss
            total_correct += outputs[tdata.test_mask].max(1)[1].eq(tdata.y[tdata.test_mask]).sum().item()
            total_num_samples += num_samples
        total_loss = total_loss.item()
        dataset.restore_mask()
        return {'loss': total_loss / total_num_samples, 'accuracy':1.0*total_correct/total_num_samples}

    def data_to_device(self, data):
        return data.to(self.device)

    def get_dataloader(self, dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False):
        return self.DataLoader([dataset.data], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
