"""
This is a non-official implementation of the work 'Cross-Node Federated Graph
Neural Network for Spatio-Temporal Data Modeling' (https://arxiv.org/abs/2106.05223).

H: hidden_size
L: num_gru_layers
N: num_nodes: N
B: num_samples or batch_size:

The hidden states encoded by GRU has the shape of (L x H)
The total number of samples per batch is (N x B)
"""
import utils.fmodule
from .fedbase import BasicServer
import torch
from .fedavg import Client
import copy
from torch.utils.data import Dataset, TensorDataset
from torch_geometric.data import DataLoader, Data
import torch.nn as nn
import utils.fflow as flw

class AugSTNodeDataset(Dataset):
    """
    The dataset is augmented with additional graph encodings for
    preserving the ability of being called by DataLoader.
    """
    def __init__(self, tensordataset, graph_encodings=None):
        super().__init__()
        ts = [t for t in tensordataset.tensors]
        if graph_encodings is not None:
            ts.append(graph_encodings)
        self.tensors = tuple(ts)

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)

    def get_graph_encodings(self):
        return self.tensors[-1]

    @classmethod
    def concat_datasets(cls, datasets: list, align_dim=2):
        tensors = []
        for item in range(len(datasets[0].tensors)):
            ts = []
            for dataset in datasets:
                ts.append(dataset.tensors[item])
            tensors.append(torch.cat(ts, align_dim))
        return cls(TensorDataset(*tensors))

    def update_graph_encodings(self, graph_encodings):
        if len(graph_encodings) != len(self):
            raise RuntimeError("The length of graph_encodings doesn't match the length of the dataset")
        ts = [t for t in self.tensors]
        ts[-1] = graph_encodings
        self.tensors = tuple(ts)

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.lr_gcn = 0.001
        self.weight_decay_gcn = option['weight_decay']
        # self.server_batch_size = option['beta']
        # self.server_epoch = option['tau']
        self.server_epoch = 1
        self.server_batch_size = 48
        self.communicate_stage = 1
        self.client_encodings = [None for _ in self.clients]
        self.model_gru_num_layers = self.model.gru_num_layers
        self.model_hidden_size = self.model.hidden_size
        # init GCN owned by the server and its optimizer
        self.calculator = utils.fmodule.TaskCalculator(utils.fmodule.device)
        self.gcn = utils.fmodule.SvrModel(node_input_size=self.model_hidden_size, global_input_size=self.model_hidden_size).to(utils.fmodule.device)
        self.optimizer = self.calculator.get_optimizer(model=self.gcn, lr=self.lr_gcn, weight_decay = self.weight_decay_gcn)
        # read graph edge information from dataset
        self.adj = self.test_data.get_adj()
        self.edge = self.test_data.get_sparse_edge()
        self.mask = self.test_data.get_mask()
        self.masked_adj = self.test_data.get_masked_adj()
        self.masked_edge = self.test_data.get_masked_edge()
        # augmente the dataset with graph_encodings initialized as zero vector
        for cid, client in enumerate(self.clients):
            client.train_data = AugSTNodeDataset(client.train_data, torch.zeros(len(client.train_data), client.train_data[0][0].shape[2], self.model_gru_num_layers, self.model_hidden_size).float())
            client.valid_data = AugSTNodeDataset(client.valid_data, torch.zeros(len(client.valid_data), client.valid_data[0][0].shape[2], self.model_gru_num_layers, self.model_hidden_size).float())
        self.test_data = AugSTNodeDataset(self.test_data, torch.zeros(len(self.test_data), len(self.adj), self.model_gru_num_layers, self.model_hidden_size).float())
        self.total_train_data = AugSTNodeDataset.concat_datasets([c.train_data for c in self.clients], align_dim=2)
        self.total_valid_data = AugSTNodeDataset.concat_datasets([c.valid_data for c in self.clients], align_dim=2)

    def update_graph_encodings(self, data):
        data_loader = DataLoader(data, batch_size=self.server_batch_size, shuffle=False)
        updated_graph_encoding = []
        for batch_idx, batch_data in enumerate(data_loader):
            batch_data = self.calculator.data_to_device(batch_data)
            x, _, _, _, _, batch_num, node_num = self.model._format_input_data(batch_data)
            # L x (B * N) x H
            h_encode = self.model.forward_encoder(batch_data)
            # GCN input: Data(
            #   node_embedding: N x B x L x H
            #   edge_index: 2 x |E|, where 2 denotes the two nodes
            #   edge_attr:  (|E|,...,)
            graph_encoding = self.gcn(
                Data(
                    x=h_encode.view(h_encode.shape[0], batch_num, node_num, h_encode.shape[2]).permute(2, 1, 0, 3),
                    edge_index=self.masked_edge['edge_index'].to(utils.fmodule.device),
                    edge_attr=self.masked_edge['edge_attr'].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(utils.fmodule.device)
                )
            )
            updated_graph_encoding.append(graph_encoding.detach().clone().cpu())
        #N x B x L x H, cat along the dim of the batch_size
        updated_graph_encoding = torch.cat(updated_graph_encoding, dim=1)
        data.update_graph_encodings(updated_graph_encoding.permute(1, 0, 2, 3))

    def iterate(self, t):
        # (1) Federated learning of on-node models.
        self.communicate_stage = 1
        selected_clients = [cid for cid in range(self.num_clients)]
        res1 = self.communicate(selected_clients)
        base_models = res1['model']
        self.model = self.aggregate(base_models)

        # (2), (3), (4) update server.gcn.
        # For convenience, we omit the details of communication as the official one does
        # by preserving the server's accessibility to the whole training dataset and validation dataset.
        train_loader = DataLoader(self.total_train_data, batch_size=self.server_batch_size, shuffle=True)
        batches_seen = self.clients[0].calculator.batches_seen
        for epoch in range(self.server_epoch):
            self.model.train()
            self.gcn.train()
            for batch_idx, batch_data in enumerate(train_loader):
                batch_data = self.calculator.data_to_device(batch_data)
                x, x_attr, y, y_attr, graph_encodings, batch_num, node_num = self.model._format_input_data(batch_data)
                # L x (N*B) x H
                h_encode = self.model.forward_encoder(batch_data)
                # N x B x L x H
                graph_encoding = h_encode.view(h_encode.shape[0], batch_num, node_num, h_encode.shape[2]).permute(2, 1, 0, 3)
                # N x B x L x F
                graph_encoding = self.gcn(
                    Data(
                        x=graph_encoding,
                        edge_index=self.masked_edge['edge_index'].to(utils.fmodule.device),
                        edge_attr=self.masked_edge['edge_attr'].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(utils.fmodule.device)
                    )
                )
                y_pred = self.model.forward_decoder(batch_data, h_encode, batches_seen=batches_seen, return_encoding=False, training=True, server_graph_encoding=graph_encoding)
                loss = nn.MSELoss()(y_pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batches_seen += 1
        # update graph encodings
        self.update_graph_encodings(self.total_train_data)
        self.update_graph_encodings(self.total_valid_data)
        self.update_graph_encodings(self.test_data)
        train_encodings = self.total_train_data.get_graph_encodings()
        valid_encodings = self.total_valid_data.get_graph_encodings()
        for cid, client in enumerate(self.clients):
            client.calculator.batches_seen = batches_seen
            client.train_data.update_graph_encodings(train_encodings[:, cid:cid+1, :, :])
            client.valid_data.update_graph_encodings(valid_encodings[:, cid:cid+1, :, :])

class MyLogger(flw.Logger):
    def __init__(self):
        super(MyLogger, self).__init__()

    def log(self, server=None, current_round=-1):
        if len(self.output) == 0:
            self.output['meta'] = server.option
        test_metric = server.test()
        for met_name, met_val in test_metric.items():
            self.output['test_' + met_name].append(met_val)
        # output to stdout
        for key, val in self.output.items():
            if key == 'meta': continue
            print(self.temp.format(key, val[-1]))
        return

    def test(self, server, model):
        test_metric = server.test(model)
        return test_metric['loss']
