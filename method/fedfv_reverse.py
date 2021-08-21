from task import modelfuncs
from .fedbase import BaseServer, BaseClient
import copy
import torch
import math

class Server(BaseServer):
    def __init__(self, option, model, clients):
        super(Server, self).__init__(option, model, clients)
        # algorithm hyper-parameters
        self.learning_rate = option['learning_rate']
        self.alpha = option['alpha']
        self.tau = option['tau']
        self.client_last_sample_round = [-1 for i in range(self.num_clients)]
        self.client_grads_history = [0 for i in range(self.num_clients)]
        self.paras_name = ['alpha', 'tau']

    def communicate(self, cid):
        # setting client(cid)'s model with latest parameters
        self.trans_model.load_state_dict(self.model.state_dict())
        self.clients[cid].setModel(self.trans_model)
        # calculate the loss  Fk(w_(t-1)) of the global model on client[k]'s training dataset
        _, loss = self.clients[cid].test('train')
        # wait for replying of the update and loss
        return self.clients[cid].reply()[0], loss

    def iterate(self, t):
        ws, losses, grads = [], [], []
        selected_clients = self.sample()
        # training
        for cid in selected_clients:
            w, loss = self.communicate(cid)
            ws.append(w)
            losses.append(loss)
            gi = modelfuncs.modeldict_sub(self.model.state_dict(), w)
            grads.append(gi)
            # update GH
            self.client_grads_history[cid] = gi
            self.client_last_sample_round[cid] = t

        # project grads
        order_grads = copy.deepcopy(grads)
        order = [k for k in range(len(order_grads))]

        # reverse the order
        tmp = sorted(list(zip(losses, order)), reverse=True, key=lambda x: x[0])
        order = [x[1] for x in tmp]

        # keep the original direction for clients with the αm largest losses
        if self.alpha > 0:
            keep_original = order[math.ceil((len(order) - 1) * (1 - self.alpha)):]
        else:
            keep_original = []

        # # mitigate internal conflicts by iteratively projecting gradients
        for i in range(len(order_grads)):
            if i in keep_original: continue
            for j in order:
                if (j == i):
                    continue
                else:
                    # calculate the dot of gi and gj
                    dot = modelfuncs.modeldict_dot(grads[j], order_grads[i])
                    if dot < 0:
                        order_grads[i] = modelfuncs.modeldict_sub(order_grads[i], modelfuncs.modeldict_scale(grads[j], 1.0 * dot / (
                                    modelfuncs.modeldict_norm(grads[j]) ** 2)))

        # aggregate projected grads
        gt = self.aggregate(order_grads)
        # mitigate outside conflicts
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
                        if modelfuncs.modeldict_dot(self.client_grads_history[cid], gt) < 0:
                            g_con = modelfuncs.modeldict_add(g_con, self.client_grads_history[cid])
                dot = modelfuncs.modeldict_dot(gt, g_con)
                if dot < 0:
                    gt = modelfuncs.modeldict_sub(gt, modelfuncs.modeldict_scale(g_con, 1.0 * dot / (
                                modelfuncs.modeldict_norm(g_con) ** 2)))

        # ||gt||=||1/m*Σgi||
        gnorm = modelfuncs.modeldict_norm(self.aggregate(grads, p=[]))
        gt = modelfuncs.modeldict_scale(gt, 1.0 / modelfuncs.modeldict_norm(gt) * gnorm)

        w_new = modelfuncs.modeldict_sub(self.model.state_dict(), gt)
        self.model.load_state_dict(w_new)
        # output info
        loss_avg = sum(losses) / len(losses)
        return loss_avg

    def aggregate(self, ws, p=[]):
        return modelfuncs.modeldict_weighted_average(ws, p)

class Client(BaseClient):
    def __init__(self, option, name = '', data_train_dict = {'x':[],'y':[]}, data_test_dict={'x':[],'y':[]}, partition = True):
        super(Client, self).__init__(option, name, data_train_dict, data_test_dict, partition)