"""
This is a non-official implementation of personalized FL method FedPAC (https://proceedings.mlr.press/v139/shamsian21a.html).
The original implementation is at https: //github.com/AvivSham/pFedHN
"""
import collections
import copy
import torch
import torch.utils.data.dataset
import torch.nn as nn
import flgo.algorithm.fedbase
import flgo.utils.fmodule as fmodule
import numpy as np
import cvxpy as cvx

def pairwise(data):
    """ Simple generator of the pairs (x, y) in a tuple such that index x < index y.
    Args:
    data Indexable (including ability to query length) containing the elements
    Returns:
    Generator over the pairs of the elements of 'data'
    """
    n = len(data)
    for i in range(n):
        for j in range(i, n):
            yield (data[i], data[j])

class Server(flgo.algorithm.fedbase.BasicServer):
    def initialize(self):
        self.init_algo_para({'lmbd':0.1})
        self.num_classes = len(collections.Counter([d[-1] for d in self.test_data]))
        for c in self.clients: c.num_classes = self.num_classes
        self.c = {}
        self.pheads = [self.model.head.state_dict() for _ in self.clients]

    def pack(self, client_id, mtype=0):
        return {
            'model': copy.deepcopy(self.model),
            'cg': copy.deepcopy(self.c),
            'r': self.current_round,
        }

    def iterate(self):
        self.selected_clients = self.sample()
        res = self.communicate(self.selected_clients)
        models, cs, Vars, Hs, sizes_label = res['model'], res['c'], res['v'], res['h'], res['sizes_label']
        # aggregate encoders directly
        self.model = self.aggregate(models)
        # update global protos
        self.c = self.protos_aggregation(cs, sizes_label)
        # Aggregate head for selected clients
        avg_weights = self.get_head_agg_weight(len(self.received_clients), Vars, Hs)
        for i,cid in enumerate(self.received_clients):
            if avg_weights[i] is not None:
                new_model = fmodule._model_average(models, avg_weights[i])
                self.communicate_with(cid, {'head': new_model.head, '__mtype__': 1})

    def get_head_agg_weight(self, num_users, Vars, Hs, *args, **kwargs):
        device = Hs[0][0].device
        num_cls = Hs[0].shape[0]  # number of classes
        d = Hs[0].shape[1]  # dimension of feature representation
        avg_weight = []
        for i in range(num_users):
            # ---------------------------------------------------------------------------
            # variance ter
            v = torch.tensor(Vars, device=device)
            # ---------------------------------------------------------------------------
            # bias term
            h_ref = Hs[i]
            dist = torch.zeros((num_users, num_users), device=device)
            for j1, j2 in pairwise(tuple(range(num_users))):
                h_j1 = Hs[j1]
                h_j2 = Hs[j2]
                h = torch.zeros((d, d), device=device)
                for k in range(num_cls):
                    h += torch.mm((h_ref[k] - h_j1[k]).reshape(d, 1), (h_ref[k] - h_j2[k]).reshape(1, d))
                dj12 = torch.trace(h)
                dist[j1][j2] = dj12
                dist[j2][j1] = dj12
            # QP solver
            p_matrix = torch.diag(v) + dist
            p_matrix = p_matrix.cpu().numpy()  # coefficient for QP problem
            evals, evecs = torch.linalg.eig(torch.tensor(p_matrix))
            # for numerical stablity
            p_matrix_new = 0
            p_matrix_new = 0
            for ii in range(num_users):
                if evals[ii].real >= 0.01:
                    p_matrix_new += evals[ii].real * torch.mm(evecs[:, ii].reshape(num_users, 1),
                                                              evecs[:, ii].reshape(1, num_users))
            p_matrix = p_matrix_new.numpy() if not np.all(np.linalg.eigvals(p_matrix) >= 0.0) else p_matrix
            # solve QP
            alpha = 0
            eps = 1e-3
            if np.all(np.linalg.eigvals(p_matrix) >= 0):
                alphav = cvx.Variable(num_users)
                obj = cvx.Minimize(cvx.quad_form(alphav, p_matrix))
                prob = cvx.Problem(obj, [cvx.sum(alphav) == 1.0, alphav >= 0])
                prob.solve()
                alpha = alphav.value
                alpha = [(i) * (i > eps) for i in alpha]  # zero-out small weights (<eps)
                if i == 0:
                    print('({}) Agg Weights of Classifier Head'.format(i + 1))
                    print(alpha, '\n')
            else:
                alpha = None  # if no solution for the optimization problem, use local classifier only

            avg_weight.append(alpha)

        return avg_weight

    def protos_aggregation(self, local_protos_list, local_sizes_list):
        agg_protos_label = {}
        agg_sizes_label = {}
        for idx in range(len(local_protos_list)):
            local_protos = local_protos_list[idx]
            local_sizes = local_sizes_list[idx]
            for label in local_protos.keys():
                if label in agg_protos_label:
                    agg_protos_label[label].append(local_protos[label])
                    agg_sizes_label[label].append(local_sizes[label])
                else:
                    agg_protos_label[label] = [local_protos[label]]
                    agg_sizes_label[label] = [local_sizes[label]]
        for [label, proto_list] in agg_protos_label.items():
            sizes_list = agg_sizes_label[label]
            proto = 0 * proto_list[0]
            for i in range(len(proto_list)):
                proto += sizes_list[i] * proto_list[i]
            agg_protos_label[label] = proto / sum(sizes_list)
        return agg_protos_label

class Client(flgo.algorithm.fedbase.BasicClient):
    def initialize(self):
        self.model = copy.deepcopy(self.server.model).to('cpu')
        label_counter = collections.Counter([d[-1] for d in self.train_data])
        self.sizes_label = torch.zeros(self.num_classes)
        for lb in range(self.num_classes):
            if lb in label_counter.keys():
                self.sizes_label[lb] = label_counter[lb]
        self.probs_label = self.sizes_label/self.sizes_label.sum()
        self.current_round = 1
        self.mse_loss = nn.MSELoss()
        self.actions = {0: self.reply, 1: self.set_local_head}

    def set_local_head(self, svr_pkg):
        with torch.no_grad():
            for phl, phs in zip(self.model.head.parameters(), svr_pkg['head'].parameters()):
                phl.data = phs.data.clone()
        return

    def reply(self, svr_pkg):
        self.unpack(svr_pkg)
        local_proto, v, h = self.train(self.model)
        return self.pack(local_proto, v, h)

    def unpack(self, svr_pkg):
        # load the parameters of encoder from the gloabl model to the local one
        global_model = svr_pkg['model']
        with torch.no_grad():
            for pg, pl in zip(global_model.encoder.parameters(), self.model.encoder.parameters()):
                pl.data = pg.data.clone()
        # update global protos
        self.global_protos = svr_pkg['cg']
        self.current_round = svr_pkg['r']
        return

    def pack(self, local_proto, v, h):
        return {
            'model': self.model,
            'v': v,
            'h': h.to('cpu'),
            'c': copy.deepcopy(local_proto),
            'sizes_label': self.sizes_label
        }

    @fmodule.with_multi_gpus
    def train(self, model):
        # compute v,h and local_ptotos1
        v,h,local_protos_list = self.statistics_extraction()
        local_protos1 = {lb:local_protos_list[lb].mean() for lb in local_protos_list.keys()}
        # set global proto
        global_protos = self.global_protos
        if global_protos=={}: global_protos=local_protos1
        self.last_model = copy.deepcopy(self.model)
        # fix encoder and optimize local head for one epoch
        for n,p in self.model.named_parameters():
            p.requires_grad = (n.split('.')[0]=='head')
        lr_g = 0.1
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr_g, momentum=self.momentum, weight_decay=self.weight_decay)
        dataloader = self.calculator.get_dataloader(self.train_data, self.batch_size)
        for batch_id, batch_data in enumerate(dataloader):
            optimizer.zero_grad()
            batch_data = self.calculator.to_device(batch_data)
            loss = self.calculator.compute_loss(self.model, batch_data)['loss']
            loss.backward()
            optimizer.step()
        for n,p in self.model.named_parameters():
            p.requires_grad = (n.split('.')[0]!='head')
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        for iter in range(self.num_steps):
            self.model.zero_grad()
            batch_data = self.calculator.to_device(self.get_batch_data())
            protos = self.model.encoder(batch_data[0])
            labels = batch_data[-1]
            outputs = self.model.head(protos)
            loss_erm = self.calculator.criterion(outputs, labels)
            loss_reg = 0.0
            if self.current_round>1:
                protos_new = protos.clone().detach()
                for i in range(len(labels)):
                    yi = labels[i].item()
                    if yi in global_protos:
                        protos_new[i] = global_protos[yi].detach()
                    else:
                        protos_new[i] = local_protos1[yi].detach()
                loss_reg = self.mse_loss(protos_new, protos)
            loss = loss_erm + self.lmbd*loss_reg
            loss.backward()
            optimizer.step()

        dataloader = self.calculator.get_dataloader(self.train_data, self.batch_size)
        local_protos_list = {}
        for batch_id, batch_data in enumerate(dataloader):
            batch_data = self.calculator.to_device(batch_data)
            features = self.model.encoder(batch_data[0])
            labels = batch_data[-1]
            feat_batch = features.clone().detach()
            for i in range(len(labels)):
                yi = labels[i].item()
                if yi in local_protos_list.keys():
                    local_protos_list[yi].append(feat_batch[i, :])
                else:
                    local_protos_list[yi] = [feat_batch[i, :]]
        local_protos2 = {lb: torch.stack(local_protos_list[lb]).mean() for lb in local_protos_list.keys()}
        return local_protos2, v, h

    def statistics_extraction(self):
        # extraction local statistics
        model = self.model
        g_params = model.head.state_dict()[list(model.head.state_dict().keys())[0]]
        d = g_params[0].shape[0]
        feature_dict = {}
        with torch.no_grad():
            dataloader = self.calculator.get_dataloader(self.train_data, batch_size=self.batch_size)
            for batch_id, batch_data in enumerate(dataloader):
                batch_data = self.calculator.to_device(batch_data)
                features = model.encoder(batch_data[0])
                labels = batch_data[-1]
                feat_batch = features.clone().detach()
                for i in range(len(labels)):
                    yi = labels[i].item()
                    if yi in feature_dict.keys():
                        feature_dict[yi].append(feat_batch[i, :])
                    else:
                        feature_dict[yi] = [feat_batch[i, :]]
        for k in feature_dict.keys():
            feature_dict[k] = torch.stack(feature_dict[k])
        py = self.probs_label
        py2 = py.mul(py)
        v = 0
        h_ref = torch.zeros((self.num_classes, d), device=self.device)
        for k in range(self.num_classes):
            if k in feature_dict.keys():
                feat_k = feature_dict[k]
                num_k = feat_k.shape[0]
                feat_k_mu = feat_k.mean(dim=0)
                h_ref[k] = py[k] * feat_k_mu
                v += (py[k] * torch.trace((torch.mm(torch.t(feat_k), feat_k) / num_k))).item()
                v -= (py2[k] * (torch.mul(feat_k_mu, feat_k_mu))).sum().item()
        v = v / len(self.train_data)
        return v, h_ref, feature_dict
