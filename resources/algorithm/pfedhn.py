"""
This is a non-official implementation of personalized FL method pFedHN (https://proceedings.mlr.press/v139/shamsian21a.html).
The original implementation is at https: //github.com/AvivSham/pFedHN
"""
import copy
from collections import OrderedDict
import torch
import torch.utils.data.dataset
import torch.nn as nn
import flgo.algorithm.fedbase
import flgo.utils.fmodule as fmodule
from flgo.utils.fmodule import FModule

class Server(flgo.algorithm.fedbase.BasicServer):
    def initialize(self):
        self.init_algo_para({'num_local_layers':0, "embed_dim":int(1+self.num_clients/4), 'hidden_dim':100, 'num_hidden_layers':1, })
        self.hnet = self.init_hnet().to(self.device)

    def pack(self, client_id, mtype=0):
        return {'model':copy.deepcopy(self.hnet),}

    def iterate(self):
        self.selected_clients = self.sample()
        # training
        hnets = self.communicate(self.selected_clients)['model']
        self.hnet = self.aggregate(hnets)
        return

    def init_hnet(self):
        model = self.model
        embed_dim = self.embed_dim
        hidden_dim = self.hidden_dim
        num_hidden_layers = self.num_hidden_layers
        num_local_layers = self.num_local_layers
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
                self.embed_convertor = nn.Sequential(*layers)
                # set weight generator
                weight_generators = {}
                self.weight_shapes = {}
                num_layers = len(list(model.parameters()))
                for i,(n, p) in enumerate(model.named_parameters()):
                    if i==num_layers-num_local_layers: break
                    weight_generators[n] = nn.Linear(hidden_dim, p.numel())
                    self.weight_shapes[n] = p.shape
                for k in weight_generators.keys():
                    setattr(self, k, weight_generators[k])

            def forward(self, embed):
                features = self.embed_convertor(embed)
                weights = {k:getattr(self, k)(features).view(self.weight_shapes[k]) for k in self.weight_shapes.keys()}
                return weights
        return HyperNet()

class Client(flgo.algorithm.fedbase.BasicClient):
    def initialize(self):
        self.embed = torch.randn((1, self.embed_dim)).to(self.device)
        self.model = copy.deepcopy(self.server.model).to('cpu')

    @fmodule.with_multi_gpus
    def train(self, hnet):
        hnet.to(self.device)
        self.model.to(self.device)
        self.embed = self.embed.detach().to(self.device)
        self.embed.requires_grad= True
        weights = hnet(self.embed)
        local_dict = self.model.state_dict()
        local_dict.update(weights)
        self.model.load_state_dict(local_dict)
        outer_optim = torch.optim.SGD(
            hnet.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum
        )
        embed_optim = torch.optim.SGD([self.embed], lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        inner_optim = self.calculator.get_optimizer(self.model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        inner_state = OrderedDict({k: tensor.data for k, tensor in weights.items()})
        for i in range(self.num_steps):
            self.model.train()
            inner_optim.zero_grad()
            outer_optim.zero_grad()
            embed_optim.zero_grad()
            batch_data = self.get_batch_data()
            loss = self.calculator.compute_loss(self.model, batch_data)['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 50)
            inner_optim.step()
        outer_optim.zero_grad()
        embed_optim.zero_grad()
        final_state = self.model.state_dict()
        delta_theta = OrderedDict({k:inner_state[k]-final_state[k] for k in weights.keys()})
        # calculating phi gradient
        hnet_grads = torch.autograd.grad(
            list(weights.values()), hnet.parameters(), grad_outputs=list(delta_theta.values())
        )
        # update hnet weights
        for p, g in zip(hnet.parameters(), hnet_grads):
            p.grad = g
        torch.nn.utils.clip_grad_norm_(hnet.parameters(), 50)
        outer_optim.step()
        embed_optim.step()
        self.model.to('cpu')
        return