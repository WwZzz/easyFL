from utils import fmodule
from .fedbase import BasicServer, BasicClient
import copy
import math

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        # algorithm hyper-parameters
        self.learning_rate = option['learning_rate']
        self.alpha = option['alpha']
        self.tau = option['tau']
        self.client_last_sample_round = [-1 for i in range(self.num_clients)]
        self.client_grads_history = [0 for i in range(self.num_clients)]
        self.paras_name=['alpha','tau']

    def iterate(self, t):
        # sampling
        self.selected_clients = self.sample()
        # training locally
        ws, losses = self.communicate(self.selected_clients)
        if self.selected_clients == []: return
        grads = [self.model - w for w in ws]
        # update GH
        for cid, gi in zip(self.selected_clients, grads):
            self.client_grads_history[cid] = gi
            self.client_last_sample_round[cid] = t

        # project grads
        order_grads = copy.deepcopy(grads)
        order = [_ for _ in range(len(order_grads))]

        # sort client gradients according to their losses in ascending orders
        tmp = sorted(list(zip(losses, order)), key=lambda x: x[0])
        order = [x[1] for x in tmp]

        # keep the original direction for clients with the αm largest losses
        keep_original = []
        if self.alpha > 0:
            keep_original = order[math.ceil((len(order) - 1) * (1 - self.alpha)):]

        # mitigate internal conflicts by iteratively projecting gradients
        for i in range(len(order_grads)):
            if i in keep_original: continue
            for j in order:
                if (j == i):
                    continue
                else:
                    # calculate the dot of gi and gj
                    dot = grads[j].dot(order_grads[i])
                    if dot < 0:
                        order_grads[i] = order_grads[i] - grads[j] * dot / (grads[j].norm()**2)

        # aggregate projected grads
        gt = fmodule._model_average(order_grads)
        # mitigate external conflicts
        if t >= self.tau:
            for k in range(self.tau-1, -1, -1):
                # calculate outside conflicts
                gcs = [self.client_grads_history[cid] for cid in range(self.num_clients) if self.client_last_sample_round[cid] == t - k and gt.dot(self.client_grads_history[cid]) < 0]
                if gcs:
                    g_con = fmodule._model_sum(gcs)
                    dot = gt.dot(g_con)
                    if dot < 0:
                        gt = gt - g_con*dot/(g_con.norm()**2)

        # ||gt||=||1/m*Σgi||
        gnorm = fmodule._model_average(grads).norm()
        gt = gt/gt.norm()*gnorm

        self.model = self.model-gt
        return

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)