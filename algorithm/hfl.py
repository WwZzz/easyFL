"""This is a non-official implementation of 'Stragglers Are Not Disaster: A Hybrid Federated Learning Algorithm with Delayed Gradients' (http://arxiv.org/abs/2102.06329). """
import torch
from .fedasync import Server as AsyncServer
from .fedbase import BasicClient as Client
import utils.system_simulator as ss
import utils.fflow as flw
import copy
import numpy as np
from utils import fmodule

class Server(AsyncServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.init_algo_para({'period':2, 'agg_period':5, 'lambda0':0.5})
        self.tolerance_for_latency = 1000
        self.client_sampled_rounds = [0 for _ in self.clients]
        self.updated = True
        self.agg_candidate = []
        self.agg_cids = []
        self.model_history = {}

    def iterate(self):
        if self.current_round not in self.model_history.keys():
            device = torch.device('cpu')
            self.model_history[self.current_round] = copy.deepcopy(self.model).to(device)
        # Scheduler periodically triggers the idle clients to locally train the model
        self.selected_clients = self.sample() if (ss.clock.current_time % self.period) == 0 or ss.clock.current_time == 1 else []
        if len(self.selected_clients) > 0: flw.logger.info('Select clients {} at time {}'.format(self.selected_clients, ss.clock.current_time))
        # Record the timestamp of the selected clients
        for cid in self.selected_clients:
            self.client_sampled_rounds[cid] = self.current_round
        # Check the currently received models
        res = self.communicate(self.selected_clients, asynchronous=True)
        received_models = res['model']
        received_client_ids = res['__cid']
        self.agg_candidate.extend(received_models)
        self.agg_cids.extend(received_client_ids)
        if len(self.agg_candidate) > 0 and ss.clock.current_time % self.agg_period == 0:
            flw.logger.info('Receive new models from clients {} at time {}'.format(received_client_ids, ss.clock.current_time))
            models_sampled_round = [self.client_sampled_rounds[cid] for cid in self.agg_cids]
            # split the received into S1 and S2
            S1_cids = [cid for cid, r in zip(self.agg_cids, models_sampled_round) if r == self.current_round]
            S2_cids = [cid for cid, r in zip(self.agg_cids, models_sampled_round) if r < self.current_round]
            # dict to save S1,S2 model
            S1_models = {}
            S2_models = {}
            for model, cid in zip(self.agg_candidate, self.agg_cids):
                if cid in S1_cids:
                    S1_models[cid] = model
                elif cid in S2_cids:
                    S2_models[cid] = model
            # for clients in S1, calculate \hat{w^t}
            hat_model_t = self.aggregate([model for cid, model in S1_models.items()])
            # for clients in S2, calculate their gradient
            final_model_S2 = []
            # Synchronize and Asynchronous client
            if len(S2_cids) > 0:
                mean_lambda = 0.0
                p_cid = pow(len(S2_cids), -1)
                for cid in S2_cids:
                    grad_cid = S2_models[cid] - self.model_history[self.client_sampled_rounds[cid]]
                    # calculate their improved gradient according to (7)
                    grad_cid_horizontal = fmodule._model_to_tensor(grad_cid)
                    grad_cid_horizontal = grad_cid_horizontal.reshape(1, grad_cid_horizontal.shape[0])
                    # transpose
                    grad_cid_vertical = grad_cid_horizontal.t()
                    grad_cid_new = grad_cid + grad_cid_horizontal * grad_cid_vertical * (self.model_history[self.current_round] - self.model_history[self.client_sampled_rounds[cid]])
                    lambda_cid = self.lambda0 * np.exp(-self.current_round + self.client_sampled_rounds[cid])
                    final_model_cid = self.model_history[self.client_sampled_rounds[cid]] - self.lr * grad_cid_new
                    mean_lambda += lambda_cid
                    final_model_S2.append(final_model_cid * lambda_cid * p_cid)
                mean_lambda /= len(S2_models)
                self.model = (1.0 - mean_lambda) * hat_model_t + fmodule._model_average(final_model_S2)
            # Only Synchronize client
            else:
                self.model = hat_model_t
            # update aggregation round and the flag `updated`
            self.current_round += 1
            self.updated = True
        return

