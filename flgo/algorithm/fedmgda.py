"""
This is a non-official implementation of 'FedMGDA+: Federated Learning meets Multi-objective Optimization' (http://arxiv.org/abs/2006.11489)
"""
import torch

from flgo.utils import fmodule
from flgo.algorithm.fedbase import BasicServer
from flgo.algorithm.fedavg import Client
import numpy as np
import copy
import cvxopt

class Server(BasicServer):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'eta':1, 'epsilon':0.1})

    def aggregate(self, models: list, *args, **kwargs):
        # calculate normalized gradients
        grads = [self.model - w for w in models]
        for i in range(len(grads)):
            gi_norm = 0.0
            for p in grads[i].parameters():
                gi_norm += (p**2).sum()
            grads[i] = grads[i]/(torch.sqrt(gi_norm) + 1e-8)
        # for gi in grads: gi.normalize()
        # calculate λ0
        nks = [len(self.clients[cid].train_data) for cid in self.received_clients]
        nt = sum(nks)
        lambda0 = [1.0 * nk / nt for nk in nks]
        # optimize lambdas to minimize ||λ'g||² s.t. λ∈Δ, ||λ - λ0||∞ <= ε
        op_lambda = self.optim_lambda(grads, lambda0)
        op_lambda = [ele[0] for ele in op_lambda]
        # aggregate grads
        dt = fmodule._model_average(grads, op_lambda)
        return self.model - dt * self.eta

    def optim_lambda(self, grads, lambda0):
        # create H_m*m = 2J'J where J=[grad_i]_n*m
        n = len(grads)
        Jt = []
        for gi in grads:
            Jt.append((copy.deepcopy(fmodule._modeldict_to_tensor1D(gi.state_dict())).cpu()).numpy())
        Jt = np.array(Jt)
        # target function
        P = 2 * np.dot(Jt, Jt.T)

        q = np.array([[0] for i in range(n)])
        # equality constraint λ∈Δ
        A = np.ones(n).T
        b = np.array([1])
        # boundary
        lb = np.array([max(0, lambda0[i] - self.epsilon) for i in range(n)])
        ub = np.array([min(1, lambda0[i] + self.epsilon) for i in range(n)])
        G = np.zeros((2*n,n))
        for i in range(n):
            G[i][i]=-1
            G[n+i][i]=1
        h = np.zeros((2*n,1))
        for i in range(n):
            h[i] = -lb[i]
            h[n+i] = ub[i]
        res=self.quadprog(P, q, G, h, A, b)
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