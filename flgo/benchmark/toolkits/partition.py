from abc import abstractmethod, ABCMeta
import random

import numpy as np
import collections

class AbstractPartitioner(metaclass=ABCMeta):
    @ abstractmethod
    def __call__(self, *args, **kwargs):
        pass

class BasicPartitioner(AbstractPartitioner):
    def __call__(self, *args, **kwargs):
        return

    def register_generator(self, generator):
        self.generator = generator

    def data_imbalance_generator(self, num_clients, datasize, imbalance=0):
        if imbalance == 0:
            samples_per_client =  [int(datasize / num_clients) for _ in range(num_clients)]
            for _ in range(datasize%num_clients): samples_per_client[_] += 1
        else:
            imbalance = max(0.1, imbalance)
            sigma = imbalance
            mean_datasize = datasize / num_clients
            mu = np.log(mean_datasize) - sigma**2/2.0
            samples_per_client = np.random.lognormal(mu, sigma, (num_clients)).astype(int)
            thresold = int(imbalance**1.5 * (datasize - num_clients*10))
            delta = int(0.1 * thresold)
            crt_data_size = sum(samples_per_client)
            # force current data size to match the total data size
            while crt_data_size != datasize:
                if crt_data_size - datasize >= thresold:
                    maxid = np.argmax(samples_per_client)
                    maxvol = samples_per_client[maxid]
                    new_samples = np.random.lognormal(mu, sigma, (10*num_clients))
                    while min(new_samples)>maxvol:
                        new_samples = np.random.lognormal(mu, sigma, (10 * num_clients))
                    new_size_id = np.argmin([np.abs(crt_data_size - samples_per_client[maxid] + s - datasize) for s in new_samples])
                    samples_per_client[maxid] = new_samples[new_size_id]
                elif crt_data_size - datasize >= delta:
                    maxid = np.argmax(samples_per_client)
                    samples_per_client[maxid] -= delta
                elif crt_data_size - datasize > 0:
                    maxid = np.argmax(samples_per_client)
                    samples_per_client[maxid] -= (crt_data_size - datasize)
                elif datasize - crt_data_size >= thresold:
                    minid = np.argmin(samples_per_client)
                    minvol = samples_per_client[minid]
                    new_samples = np.random.lognormal(mu, sigma, (10*num_clients))
                    while max(new_samples)<minvol:
                        new_samples = np.random.lognormal(mu, sigma, (10 * num_clients))
                    new_size_id = np.argmin([np.abs(crt_data_size - samples_per_client[minid] + s - datasize) for s in new_samples])
                    samples_per_client[minid] = new_samples[new_size_id]
                elif datasize - crt_data_size >= delta:
                    minid = np.argmin(samples_per_client)
                    samples_per_client[minid] += delta
                else:
                    minid = np.argmin(samples_per_client)
                    samples_per_client[minid] += (datasize - crt_data_size)
                crt_data_size = sum(samples_per_client)
        return samples_per_client

class IIDPartitioner(BasicPartitioner):
    def __init__(self, num_clients=100, imbalance=0):
        self.num_clients = num_clients
        self.imbalance = imbalance

    def __str__(self):
        name = "iid"
        if self.imbalance>0: name += '_imb{:.1f}'.format(self.imbalance)
        return name

    def __call__(self, data):
        samples_per_client = self.data_imbalance_generator(self.num_clients, len(data), self.imbalance)
        d_idxs = np.random.permutation(len(data))
        local_datas = np.split(d_idxs, np.cumsum(samples_per_client))[:-1]
        local_datas = [di.tolist() for di in local_datas]
        return local_datas

class DirichletPartitioner(BasicPartitioner):
    def __init__(self, num_clients = 100, alpha=1.0, imbalance=0, flag_index=-1):
        self.num_clients = num_clients
        self.alpha = alpha
        self.imbalance = imbalance
        self.flag_index = flag_index

    def __str__(self):
        name = "dir{:.2f}".format(self.alpha)
        if self.imbalance>0: name += '_imb{:.1f}'.format(self.imbalance)
        return name

    def __call__(self, data):
        attrs = [d[self.flag_index] for d in data]
        num_attrs = len(set(attrs))
        samples_per_client = self.data_imbalance_generator(self.num_clients, len(data), self.imbalance)
        # count the label distribution
        lb_counter = collections.Counter(attrs)
        p = np.array([1.0 * v / len(data) for v in lb_counter.values()])
        lb_dict = {}
        attrs = np.array(attrs)
        for lb in range(len(lb_counter.keys())):
            lb_dict[lb] = np.where(attrs == lb)[0]
        proportions = [np.random.dirichlet(self.alpha * p) for _ in range(self.num_clients)]
        while np.any(np.isnan(proportions)):
            proportions = [np.random.dirichlet(self.alpha * p) for _ in range(self.num_clients)]
        sorted_cid_map = {k: i for k, i in zip(np.argsort(samples_per_client), [_ for _ in range(self.num_clients)])}
        error_increase_interval = 500
        max_error = 1e-6 / num_attrs
        loop_count = 0
        crt_id = 0
        crt_error = 100000
        while True:
            if loop_count >= error_increase_interval:
                loop_count = 0
                max_error = max_error * 10
            # generate dirichlet distribution till ||E(proportion) - P(D)||<=1e-5*self.num_classes
            mean_prop = np.sum([pi * di for pi, di in zip(proportions, samples_per_client)], axis=0)
            mean_prop = mean_prop / mean_prop.sum()
            error_norm = ((mean_prop - p) ** 2).sum()
            if crt_error - error_norm >= max_error:
                print("Error: {:.8f}".format(error_norm))
                crt_error = error_norm
            if error_norm <= max_error:
                break
            excid = sorted_cid_map[crt_id]
            crt_id = (crt_id + 1) % self.num_clients
            sup_prop = [np.random.dirichlet(self.alpha * p) for _ in range(self.num_clients)]
            del_prop = np.sum([pi * di for pi, di in zip(proportions, samples_per_client)], axis=0)
            del_prop -= samples_per_client[excid] * proportions[excid]
            for i in range(error_increase_interval-loop_count):
                alter_norms = []
                for cid in range(self.num_clients):
                    if np.any(np.isnan(sup_prop[cid])):
                        continue
                    alter_prop = del_prop + samples_per_client[excid] * sup_prop[cid]
                    alter_prop = alter_prop / alter_prop.sum()
                    error_alter = ((alter_prop - p) ** 2).sum()
                    alter_norms.append(error_alter)
                if min(alter_norms)<error_norm:
                    break
            if len(alter_norms) > 0 and min(alter_norms) < error_norm:
                alcid = np.argmin(alter_norms)
                proportions[excid] = sup_prop[alcid]
            loop_count += 1
        local_datas = [[] for _ in range(self.num_clients)]
        self.dirichlet_dist = []  # for efficiently visualizing
        for lb in lb_counter.keys():
            lb_idxs = lb_dict[lb]
            lb_proportion = np.array([pi[lb] * si for pi, si in zip(proportions, samples_per_client)])
            lb_proportion = lb_proportion / lb_proportion.sum()
            lb_proportion = (np.cumsum(lb_proportion) * len(lb_idxs)).astype(int)[:-1]
            lb_datas = np.split(lb_idxs, lb_proportion)
            self.dirichlet_dist.append([len(lb_data) for lb_data in lb_datas])
            local_datas = [local_data + lb_data.tolist() for local_data, lb_data in zip(local_datas, lb_datas)]
        self.dirichlet_dist = np.array(self.dirichlet_dist).T
        for i in range(self.num_clients): np.random.shuffle(local_datas[i])
        self.local_datas = local_datas
        return local_datas

class DiversityPartitioner(BasicPartitioner):
    def __init__(self, num_clients=100, diversity=1.0, flag_index=-1):
        self.num_clients = num_clients
        self.diversity = diversity
        self.flag_index = flag_index
        
    def __str__(self):
        name = "div{:.1f}".format(self.diversity)
        return name

    def __call__(self, data):
        labels = [d[self.flag_index] for d in data]
        num_classes = len(set(labels))
        dpairs = [[did, lb] for did,lb in zip(list(range(len(data))), labels)]
        num = max(int(self.diversity* num_classes), 1)
        K = num_classes
        local_datas = [[] for _ in range(self.num_clients)]
        if num == K:
            for k in range(K):
                idx_k = [p[0] for p in dpairs if p[1] == k]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k, self.num_clients)
                for cid in range(self.num_clients):
                    local_datas[cid].extend(split[cid].tolist())
        else:
            times = [0 for _ in range(num_classes)]
            contain = []
            for i in range(self.num_clients):
                current = []
                j = 0
                while (j < num):
                    mintime = np.min(times)
                    ind = np.random.choice(np.where(times == mintime)[0])
                    if (ind not in current):
                        j = j + 1
                        current.append(ind)
                        times[ind] += 1
                contain.append(current)
            for k in range(K):
                idx_k = [p[0] for p in dpairs if p[1] == k]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k, times[k])
                ids = 0
                for cid in range(self.num_clients):
                    if k in contain[cid]:
                        local_datas[cid].extend(split[ids].tolist())
                        ids += 1
        return local_datas

class GaussianPerturbationPartitioner(BasicPartitioner):
    def __init__(self, num_clients=100, imbalance=0.0, sigma=0.1, scale=0.1, feature_index=0):
        self.num_clients = num_clients
        self.imbalance = imbalance
        self.sigma = sigma
        self.scale = scale
        self.feature_index = feature_index

    def __str__(self):
        name = "perturb_gs{:.1f}_{:.1f}".format(self.sigma, self.scale)
        if self.imbalance>0: name += '_imb{:.1f}'.format(self.imbalance)
        return name

    def __call__(self, data):
        shape = tuple(np.array(data[0][self.feature_index].shape))
        samples_per_client = self.data_imbalance_generator(self.num_clients, len(data), self.imbalance)
        d_idxs = np.random.permutation(len(data))
        local_datas = np.split(d_idxs, np.cumsum(samples_per_client))[:-1]
        local_datas = [di.tolist() for di in local_datas]
        local_perturbation_means = [np.random.normal(0, self.sigma, shape) for _ in range(self.num_clients)]
        local_perturbation_stds = [0.1*np.ones(shape) for _ in range(self.num_clients)]
        local_perturbation = []
        for cid in range(self.num_clients):
            c_perturbation = [np.random.normal(local_perturbation_means[cid], local_perturbation_stds[cid]).tolist() for _ in range(len(local_datas[cid]))]
            local_perturbation.append(c_perturbation)
        self.local_perturbation = local_perturbation
        return local_datas

class IDPartitioner(BasicPartitioner):
    def __init__(self, num_clients=-1,  priority = 'max'):
        self.num_clients = int(num_clients)
        self.priorty = priority
        return

    def __str__(self):
        return 'id'

    def __call__(self, data):
        all_data = list(range(len(data)))
        data_owners = data.id
        local_datas = collections.defaultdict(list)
        for idx in range(len(all_data)):
            local_datas[data_owners[idx]].append(all_data[idx])
        local_datas = list(local_datas.values())
        if self.num_clients<0:
            self.num_clients = len(local_datas)
        elif self.priorty=='max':
            local_datas = sorted(local_datas, key=lambda x: len('x'), reverse=True)[:self.num_clients]
        elif self.priorty=='min':
            local_datas = sorted(local_datas, key=lambda x: len('x'))[:self.num_clients]
        elif self.priorty=='none':
            random.shuffle(local_datas)
            local_datas = local_datas[:self.num_clients]
        return local_datas
