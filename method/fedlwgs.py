import torch
from .fedbase import BaseServer, BaseClient
import numpy as np
import copy
import cvxopt


class Server(BaseServer):
    def __init__(self, option, model, clients, dtest = None):
        super(Server, self).__init__(option, model, clients, dtest)
        # algorithm hyper-parameters
        self.lmd = np.ones(self.num_clients) * 1.0 / self.num_clients
        self.learning_rate = option['eta']
        self.epsilon = option['epsilon']
        self.paras_name = ['epsilon', 'eta']

    def iterate(self, t):
        # sample clients
        selected_clients = self.sample()
        # local training
        ws, losses = self.communicate(selected_clients)
        grads = [self.model-w for w in ws]
        gdicts = [g.state_dict() for g in grads]
        # initial aggregated gradient as zeros
        g_final = self.model.zeros_like()
        # initial lambda in MGDA
        nks = [self.client_vols[cid] for cid in selected_clients]
        nt = sum(nks)
        lmd0 = [1.0 * nk / nt for nk in nks]
        for layer in self.model.state_dict().keys():
            vecs = [gi[layer].view(-1) for gi in gdicts]
            # clip layer vec
            vecs_norm = [torch.div(vec, torch.norm(vec)) if not torch.all(vec == 0) else vec for vec in vecs]
            # 求解无矛盾的该层梯度gl
            # MGDA
            lmd = self.optim_lambda(vecs_norm, lmd0)
            lmd = [ele[0] for ele in lmd]
            gl = sum([v*l for v, l in zip(vecs_norm, lmd)])
            # 赋值该层梯度
            g_final.state_dict()[layer].data.copy_(gl.reshape(gdicts[0][layer].shape))
        self.model = self.model - self.learning_rate * g_final
        return

    def optim_lambda(self, grads, lmd0):
        n = len(grads)
        Jf = []
        for gi in grads:
            Jf.append(copy.deepcopy(gi).cpu().numpy())
        Jf = np.array(Jf)
        # target function
        # P = 2 * np.multiply(Jf.reshape(-1, 1).T, Jf.reshape(-1, 1))
        P = 2 * np.dot(Jf, Jf.T)
        q = np.array([[0] for i in range(n)])
        # equality constraint λ∈Δ
        A = np.ones(n).T
        b = np.array([1])
        # boundary
        lb = np.array([max(0, lmd0[i] - self.epsilon) for i in range(n)])
        ub = np.array([min(1, lmd0[i] + self.epsilon) for i in range(n)])
        G = np.zeros((2 * n, n))
        for i in range(n):
            G[i][i] = -1
            G[n + i][i] = 1
        h = np.zeros((2 * n, 1))
        for i in range(n):
            h[i] = -lb[i]
            h[n + i] = ub[i]
        res = self.quadprog(P, q, G, h, A, b)
        return res

    def quadprog(self, P, q, G, h, A, b):
        """
        Input: Numpy arrays, the format follows MATLAB quadprog function: https://www.mathworks.com/help/optim/ug/quadprog.html
        Output: Numpy array of the solution
        """
        P = cvxopt.matrix(P.tolist())
        q = cvxopt.matrix(q.tolist(), tc='d')
        G = cvxopt.matrix(G.tolist())
        h = cvxopt.matrix(h.tolist())
        A = cvxopt.matrix(A.tolist())
        b = cvxopt.matrix(b.tolist(), tc='d')
        sol = cvxopt.solvers.qp(P, q.T, G.T, h.T, A.T, b)
        return np.array(sol['x'])


class Client(BaseClient):
    def __init__(self, option, name='', data_train_dict={'x': [], 'y': []}, data_val_dict={'x': [], 'y': []}, train_rate=0.8, drop_rate=0):
        super(Client, self).__init__(option, name, data_train_dict, data_val_dict, train_rate, drop_rate)
