from utils import fmodule
from .fedbase import BaseServer, BaseClient
import copy
import torch
import math

class Server(BaseServer):
    def __init__(self, option, model, clients, dtest = None):
        super(Server, self).__init__(option, model, clients, dtest)
        # algorithm hyper-parameters
        self.learning_rate = option['learning_rate']
        self.alpha = option['alpha']
        self.tau = option['tau']
        self.client_last_sample_round = [-1 for i in range(self.num_clients)]
        self.client_grads_history = [0 for i in range(self.num_clients)]
        self.paras_name=['alpha','tau']

    def iterate(self, t):
        # sampling
        selected_clients = self.sample()
        # training locally
        ws, losses = self.communicate(selected_clients)
        grads = [fmodule.modeldict_sub(self.model.state_dict(), w) for w in ws]
        # update GH
        for cid, gi in zip(selected_clients, grads):
            self.client_grads_history[cid] = gi
            self.client_last_sample_round[cid] = t

        # project grads
        order_grads = copy.deepcopy(grads)
        order = [k for k in range(len(order_grads))]

        # sort client gradients according to their losses in ascending orders
        tmp = sorted(list(zip(losses, order)), key=lambda x: x[0])
        order = [x[1] for x in tmp]

        # keep the original direction for clients with the αm largest losses
        if self.alpha > 0:
            keep_original = order[math.ceil((len(order) - 1) * (1 - self.alpha)):]
        else:
            keep_original = []

        # mitigate internal conflicts by iteratively projecting gradients
        for i in range(len(order_grads)):
            if i in keep_original: continue
            for j in order:
                if (j == i):
                    continue
                else:
                    # calculate the dot of gi and gj
                    dot = fmodule.modeldict_dot(grads[j], order_grads[i])
                    if dot < 0:
                        order_grads[i] = fmodule.modeldict_sub(order_grads[i], fmodule.modeldict_scale(grads[j], 1.0 * dot / (
                                fmodule.modeldict_norm(grads[j]) ** 2)))

        # aggregate projected grads
        gt = self.aggregate(order_grads)
        # mitigate external conflicts
        if t >= self.tau:
            for k in range(0, self.tau):
                k = self.tau - k
                # create zero vector
                g_con = {}
                for layer in gt.keys():
                    g_con[layer] = torch.zeros_like(gt[layer])
                # calculate outside conflicts
                for cid in range(self.num_clients):
                    if self.client_last_sample_round[cid] == t - k:
                        if fmodule.modeldict_dot(self.client_grads_history[cid], gt) < 0:
                            g_con = fmodule.modeldict_add(g_con, self.client_grads_history[cid])
                dot = fmodule.modeldict_dot(gt, g_con)
                if dot < 0:
                    gt = fmodule.modeldict_sub(gt, fmodule.modeldict_scale(g_con, 1.0 * dot / (
                            fmodule.modeldict_norm(g_con) ** 2)))

        # ||gt||=||1/m*Σgi||
        gnorm = fmodule.modeldict_norm(self.aggregate(grads, p=[]))
        gt = fmodule.modeldict_scale(gt, 1.0 / fmodule.modeldict_norm(gt) * gnorm)

        w_new = fmodule.modeldict_sub(self.model.state_dict(), gt)
        self.model.load_state_dict(w_new)
        # output info
        loss_avg = sum(losses) / len(losses)
        return loss_avg

    def aggregate(self, ws, p=[]):
        return fmodule.modeldict_weighted_average(ws, p)

class Client(BaseClient):
    def __init__(self, option, name = '', data_train_dict = {'x':[],'y':[]}, data_val_dict={'x':[],'y':[]}, partition = 0.8, drop_rate = 0):
        super(Client, self).__init__(option, name, data_train_dict, data_val_dict, partition, drop_rate)