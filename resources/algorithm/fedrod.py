"""
This is a non-official implementation of personalized FL method Fed-ROD (https://openreview.net/forum?id=I1hQbx10Kxn).
To use this algorithm, the corresponding model should contain two submodules named `encoder` and `head`.
"""
import copy
from collections import OrderedDict
import torch
import torch.utils.data.dataset
import torch.nn as nn
import flgo.algorithm.fedbase
import flgo.utils.fmodule as fmodule
from flgo.utils.fmodule import FModule
import collections


def br_loss(y, logits, dist, lmbd=0.1):
    exp_logits = torch.exp(logits)
    weights = torch.pow(dist, lmbd)
    wexp = exp_logits * weights
    wsums = (1.0/wexp.sum(dim=1)).reshape(-1, 1)
    res = (wexp*wsums).gather(1, y.view(-1, 1))
    return res.mean()

class Server(flgo.algorithm.fedbase.BasicServer):
    def initialize(self):
        self.init_algo_para({'lmbd':0.1, 'num_hidden_layers':1, 'hidden_dim':100})
        self.num_classes = len(collections.Counter([d[-1] for d in self.test_data]))
        for c in self.clients: c.num_classes = self.num_classes
        self.hyper = self.num_hidden_layers>0
        self.hnet = self.init_hnet().to(self.device) if self.hyper else None

    def pack(self, client_id, mtype=0):
        return {'model':copy.deepcopy(self.model), 'hnet': self.hnet}

    def iterate(self):
        self.selected_clients = self.sample()
        # training
        res = self.communicate(self.selected_clients)
        models = res['model']
        hnets = res['hnet']
        self.model = self.aggregate(models)
        if len(hnets)>0 and hnets[0] is not None: self.hnet = self.aggregate(hnets)
        return len(models)>0

    def init_hnet(self):
        model = self.model
        embed_dim = self.num_classes
        hidden_dim = self.hidden_dim
        num_hidden_layers = self.num_hidden_layers
        class HyperNet(FModule):
            def __init__(self):
                super(HyperNet, self).__init__()
                # set mlp: embed -> hidden_feature
                layers = [
                    nn.Linear(embed_dim, hidden_dim),
                ]
                for _ in range(num_hidden_layers):
                    layers.append(nn.ReLU(inplace=True))
                    layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.dist_convertor = nn.Sequential(*layers)
                # set weight generator
                self.weights_shape = {}
                for n,p in model.head.named_parameters():
                    setattr(self, n, nn.Linear(hidden_dim, p.numel()))
                    self.weights_shape[n] = p.shape

            def forward(self, dist):
                features = self.dist_convertor(dist)
                weights = {k:getattr(self, k)(features).view(self.weights_shape[k]) for k in self.weights_shape.keys()}
                return weights
        return HyperNet()

class Client(flgo.algorithm.fedbase.BasicClient):
    def initialize(self):
        lb_counter = collections.Counter([d[-1] for d in self.train_data])
        self.dist = torch.zeros((1, self.num_classes))
        for k in lb_counter.keys():
            self.dist[0][k] = lb_counter[k]
        self.dist = self.dist / len(self.train_data)
        self.head = copy.deepcopy(self.server.model.head)
        self.hyper = self.num_hidden_layers>0

    def reply(self, svrpkg):
        model, hnet = self.unpack(svrpkg)
        self.train(model, hnet)
        return self.pack(model, hnet)

    def unpack(self, svrpkg):
        return svrpkg['model'], svrpkg['hnet']

    def pack(self, model, hnet):
        return {'model': model, 'hnet':hnet}

    @fmodule.with_multi_gpus
    def train(self, model, hnet=None):
        model.to(self.device)
        self.head.to(self.device)
        self.dist = self.dist.to(self.device)
        if self.hyper:
            hnet.to(self.device)
            head_weights = hnet(self.dist)
            self.head.load_state_dict(head_weights)
            hnet_optim = torch.optim.SGD(
                hnet.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum
            )
            inner_state = OrderedDict({k: tensor.data for k, tensor in head_weights.items()})
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        head_optimizer = self.calculator.get_optimizer(self.head, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for i in range(self.num_steps):
            model.train()
            optimizer.zero_grad()
            head_optimizer.zero_grad()
            if self.hyper:
                hnet_optim.zero_grad()
            batch_data = self.calculator.to_device(self.get_batch_data())
            emb = model.encoder(batch_data[0])
            ypred_global = model.head(emb)
            ypred_local = self.head(emb) + ypred_global
            loss_local = self.calculator.criterion(ypred_local, batch_data[1])
            loss_global = br_loss(batch_data[-1], ypred_global, self.dist*len(self.train_data), self.lmbd)
            loss = loss_global+loss_local
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
            optimizer.step()
            head_optimizer.step()
        if self.hyper:
            hnet_optim.zero_grad()
            final_state = model.head.state_dict()
            delta_theta = OrderedDict({k:inner_state[k]-final_state[k] for k in head_weights.keys()})
            # calculating phi gradient
            hnet_grads = torch.autograd.grad(
                list(head_weights.values()), hnet.parameters(), grad_outputs=list(delta_theta.values())
            )
            # update hnet weights
            for p, g in zip(hnet.parameters(), hnet_grads):
                p.grad = g
            torch.nn.utils.clip_grad_norm_(hnet.parameters(), 50)
            hnet_optim.step()
        # update self.model
        self.model = copy.deepcopy(model)
        with torch.no_grad():
            for ps, ph in zip(self.model.head.parameters(), self.head.parameters()):
                ps.data = ps.data+ph.data
        self.head.to('cpu')
        return