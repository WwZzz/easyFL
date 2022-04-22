from .fedbase import BasicServer, BasicClient
import numpy as np
from utils import fmodule
import copy
from itertools import product
from scipy.cluster.hierarchy import linkage, fcluster

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        # m = self.clients_per_round, M = self.data_vol = sum(n_i), n_i = self.client_vols[i]
        self.alg = option['alg']
        self.W = None
        self.paras_name=['alg']
        zero_model = fmodule._model_to_tensor(self.model-self.model).cpu().numpy()
        self.update_history = [copy.deepcopy(zero_model) for _ in range(self.num_clients)]
        self.distance_type = 'cos'

    def iterate(self, t):
        self.selected_clients = self.sample()
        # training
        models = self.communicate(self.selected_clients)['model']
        for model_k, cid in zip(models, self.selected_clients):
            self.update_history[cid]=fmodule._model_to_tensor(model_k-self.model).cpu().numpy()
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        self.model = self.aggregate(models, p = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
        return

    def update_w(self, m, M, ns, alg=1):
        if alg==1:
            # clustering based on data size
            if self.W: return self.W
            # q: current_data_amount
            current_data_amount = 0
            # k: current_cluster_idx
            current_cluster_idx = 0
            clients = [cid for cid in range(self.num_clients)]
            r = [[0 for _ in range(self.num_clients)] for _ in range(m)]
            # order clients by descending importance of n_i in ns
            tmp = zip(ns, clients)
            tmp = sorted(tmp, key=lambda x: x[0], reverse=True)
            b_prior = 0
            for n_cid, cid in tmp:
                current_data_amount = current_data_amount + m * n_cid
                ai = current_data_amount // M
                bi = current_data_amount % M
                if ai > current_cluster_idx:
                    r[current_cluster_idx][cid] = M - b_prior
                    for l in range(current_cluster_idx + 1, ai):
                        r[l][cid] = M
                    if bi!=0:
                        r[ai][cid] = bi
                else:
                    r[ai][cid] = bi - b_prior
                current_cluster_idx = ai
                # update b_i-1
                b_prior = bi
            return [[1.0 * rki / M for rki in rk] for rk in r]
        elif self.alg==2:
            # clustering based on client similarity
            epsilon = int(10 ** 10)
            sim_matrix = np.zeros((self.num_clients, self.num_clients))
            for i, j in product(range(self.num_clients), range(self.num_clients)):
                sim_matrix[i, j] = self.get_similarity(self.update_history[i], self.update_history[j])
            linkage_matrix = linkage(sim_matrix, "ward")

            # associate each client to a cluster
            weights = [1.0 * ni / M for ni in ns]
            augmented_weights = [1.0*ni/M for ni in ns]

            for i in range(len(linkage_matrix)):
                idx_1, idx_2 = int(linkage_matrix[i, 0]), int(linkage_matrix[i, 1])

                new_weight = np.array(
                    [augmented_weights[idx_1] + augmented_weights[idx_2]]
                )
                augmented_weights = np.concatenate((augmented_weights, new_weight))
                linkage_matrix[i, 2] = int(new_weight * epsilon)

            clusters = fcluster(
                linkage_matrix, int(epsilon / m), criterion="distance"
            )

            n_clients, n_clusters = len(clusters), len(set(clusters))

            # calculate the data volumn of each cluster
            pop_clusters = np.zeros((n_clusters, 2)).astype(int)
            for i in range(n_clusters):
                pop_clusters[i, 0] = i + 1
                for client in np.where(clusters == i + 1)[0]:
                    pop_clusters[i, 1] += int(weights[client] * epsilon * m)
            # sort the clusters according to the data volumn
            pop_clusters = pop_clusters[pop_clusters[:, 1].argsort()]

            distri_clusters = np.zeros((m, n_clients)).astype(int)

            # m biggest clusters that will remain unchanged
            kept_clusters = pop_clusters[n_clusters - m:, 0]
            # allocate weights for the clients in the kept clusters
            for idx, cluster in enumerate(kept_clusters):
                for client in np.where(clusters == cluster)[0]:
                    distri_clusters[idx, client] = int(weights[client] * m * epsilon)
            k = 0
            for j in pop_clusters[: n_clusters - m, 0]:
                clients_in_j = np.where(clusters == j)[0]
                np.random.shuffle(clients_in_j)
                for client in clients_in_j:
                    weight_client = int(weights[client] * epsilon * m)
                    while weight_client > 0:
                        sum_proba_in_k = np.sum(distri_clusters[k])
                        u_i = min(epsilon - sum_proba_in_k, weight_client)
                        distri_clusters[k, client] = u_i
                        weight_client += -u_i
                        sum_proba_in_k = np.sum(distri_clusters[k])
                        if sum_proba_in_k == 1 * epsilon:
                            k += 1
            distri_clusters = distri_clusters.astype(float)
            for l in range(m):
                distri_clusters[l] /= np.sum(distri_clusters[l])
            return distri_clusters.tolist()

    def sample(self):
        self.W = self.update_w(self.clients_per_round, self.data_vol, self.client_vols, self.alg)
        all_clients = [cid for cid in range(self.num_clients)]
        selected_clients = []
        for k in range(self.clients_per_round):
            cid = np.random.choice(all_clients, 1, p=self.W[k])[0]
            selected_clients.append(cid)
        return list(selected_clients)

    def get_similarity(self, g1, g2):
        if self.distance_type == "L1":
            return np.sum(np.abs(g1-g2))
        elif self.distance_type == "L2":
            return np.sum((g1-g2)**2)
        elif self.distance_type == "cos":
            ng1 = np.sum(g1**2)
            ng2 = np.sum(g2**2)
            if ng1==0.0 or ng2==0.0:
                return 0.0
            else:
                return np.arccos(np.sum(g1*g2)/(np.sqrt(ng1*ng2)))

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)


