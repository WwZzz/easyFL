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
        self.init_algo_para({'num_local_layers':2, "embed_dim":int(1+self.num_clients/4), 'hidden_dim':100, 'num_hidden_layers':1, })
        self.hnet = self.init_hnet().to(self.device)
        self.client_weights = {}
        self.outer_optim = torch.optim.SGD(
            self.hnet.parameters(), lr=self.option['learning_rate'], weight_decay=self.option['weight_decay'], momentum=self.option['momentum']
        )
    def pack(self, client_id, mtype=0):
        return {'w':self.client_weights[client_id]}

    def iterate(self):
        self.selected_clients = self.sample()
        client_weights = self.hnet(torch.LongTensor(self.selected_clients).to(self.device))
        self.client_weights = {ci:mi for ci, mi in zip(self.selected_clients, client_weights)}
        inner_states = {ci: OrderedDict({k: tensor.data for k, tensor in w.items()}) for ci,w in zip(self.selected_clients, client_weights)}
        # training
        models = self.communicate(self.selected_clients)['model']
        self.outer_optim.zero_grad()
        for ci,mi in zip(self.selected_clients, models):
            weights = self.client_weights[ci]
            inner_state = inner_states[ci]
            final_state = mi.state_dict()
            delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in inner_state.keys()})
            hnet_grads = torch.autograd.grad(
                list(weights.values()), self.hnet.parameters(), grad_outputs=list(delta_theta.values()), retain_graph=True
            )
            # update hnet weights
            for p, g in zip(self.hnet.parameters(), hnet_grads):
                if p.grad is None:
                    p.grad = g
                else: p.grad += g
        for p in self.hnet.parameters():
            p.grad/=len(models)
        clip_grad = self.option['clip_grad'] if self.option['clip_grad']>0 else 50
        torch.nn.utils.clip_grad_norm_(self.hnet.parameters(), clip_grad)
        self.outer_optim.step()
        self.client_weights = {}
        return

    def init_hnet(self):
        model = self.model
        embed_dim = self.embed_dim
        hidden_dim = self.hidden_dim
        num_hidden_layers = self.num_hidden_layers
        num_local_layers = self.num_local_layers
        num_clients = self.num_clients
        class HyperNet(FModule):
            def __init__(self):
                super(HyperNet, self).__init__()
                self.user_embeddings = nn.Embedding(num_embeddings=num_clients, embedding_dim=embed_dim)
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

            def forward(self, client_id):
                embed = self.user_embeddings(client_id)
                features = self.embed_convertor(embed)
                weights = {k: getattr(self, k)(features) for k in self.weight_shapes.keys()}
                res = [{k:v[i].view(self.weight_shapes[k]) for k,v in weights.items()} for i in range(len(client_id))]
                return res
        return HyperNet()

class Client(flgo.algorithm.fedbase.BasicClient):
    def initialize(self):
        self.model = copy.deepcopy(self.server.model).to(self.device)

    def unpack(self, received_pkg):
        model_dict = self.model.state_dict()
        model_dict.update(received_pkg['w'])
        self.model.load_state_dict(model_dict)
        return self.model