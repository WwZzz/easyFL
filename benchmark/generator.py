"""
DISTRIBUTION OF DATASET
-----------------------------------------------------------------------------------
balance:
    iid:            0 : identical and independent distributions of the dataset among clients
    label skew:     1 Quantity:  each party owns data samples of a fixed number of labels.
                    2 Dirichlet: each party is allocated a proportion of the samples of each label according to Dirichlet distribution.
                    3 Shard: each party is allocated the same numbers of shards that is sorted by the labels of the data
-----------------------------------------------------------------------------------
depends on partitions:
    feature skew:   4 Noise: each party owns data samples of a fixed number of labels.
                    5 ID: For Shakespeare\FEMNIST, we divide and assign the writers (and their characters) into each party randomly and equally.
-----------------------------------------------------------------------------------
imbalance:
    iid:            6 Vol: only the vol of local dataset varies.
    niid:           7 Vol: for generating synthetic data
"""

import ujson
import numpy as np
import os.path
from torchvision import datasets, transforms
import gzip
import random
import urllib
import zipfile
import collections
import os
import re
import ssl
from scipy.special import softmax

ssl._create_default_https_context = ssl._create_unverified_context

data_dist = {
    0: 'iid',
    1: 'label_skew_quantity',
    2: 'label_skew_dirichlet',
    3: 'label_skew_shard',
    4: 'feature_skew_noise',
    5: 'feature_skew_id',
    6: 'iid_volumn_skew',
    7: 'niid_volumn_skew',
}

class TaskGenerator:
    def __init__(self, benchmark, num_classes, dist, num_clients = 1, beta = 0.5, noise = 0, minvol = 10, datapath ='', cnames = [], selected = None):
        self.benchmark = benchmark
        self.rootpath = './fedtask'

        self.datapath = datapath if datapath.endswith('/') else datapath+'/'
        if not os.path.exists(self.datapath):
            os.makedirs(self.datapath)
        self.datavol = -1
        self.num_classes = num_classes
        self.train_data = None
        self.test_data = None
        self.minvol = minvol

        self.num_clients = num_clients
        self.cnames = ['client '+str(i) for i in range(num_clients)] if cnames == [] else cnames
        self.noise = noise
        self.selected = selected

        self.dist = dist
        self.distname = data_dist[self.dist]
        self.supported_dists = []
        self.beta = beta

        self.taskname = self.benchmark+'_'+('client'+str(self.num_clients)+'_')+('dist'+str(self.dist)+'_')+('beta' + str(self.beta).replace(" ","") + '_')+('noise'+str(self.noise))
        self.savepath = self.rootpath + '/' + self.taskname
        self.output = None

    def load_data(self):
        """ load and pre-process the raw data, get the data volumn"""
        return

    def preprocess_data(self):
        """convert the data to {'x':[], 'y':[], 'p':[]} or other formats that can be divided by partition()"""
        return

    def add_noise(self, dtrains):
        """ add noise to parts of the training data"""
        return dtrains

    def generate(self):
        """ generate federated tasks"""
        # check if the task exists
        if os.path.exists(self.savepath):
            print('Task already exists!')
            return
        # read raw_data into self.train_data and self.test_data
        self.load_data()
        # convert self.train\test_data into dicts as {'x':[...], 'y':[...]} or {'x':[...], 'y':[...], 'p':[...]}(data['x'][k], d['y'][k], d['p'][k]) denotes the k_th data)
        self.preprocess_data()
        # partition self.train_data into a index-list, e.g. [[d66, d37, ...], [d9,...],...] where di denotes the index of the i_th training data
        udata_idxs = self.partition()
        # set dtrains = [cdata_1,... , cdata_m] where cdata_i = {'x':[...], 'y':[...]}
        dtrains = self.dsample(self.train_data, udata_idxs)
        # add noise to the clients' training data
        if self.noise > 0:
            dtrains = self.add_noise(dtrains)
        self.fill_output(dtrains)
        self.save_task()
        return

    def save_task(self):
        """save the task as task.json file"""
        # create task directory as ../fedtask/taskname/record
        if os.path.exists(self.savepath):
            print('Task already exists!')
            return
        else:
            os.makedirs(self.savepath+'/record')
        output_path = self.savepath + '/task.json'
        with open(output_path, 'w') as outfile:
            ujson.dump(self.output, outfile)

    def dsample(self, data, udata_idxs=[]):
        if udata_idxs == []:
            return data
        dtrains = [{} for i in range(self.num_clients)]
        for cidx in range(self.num_clients):
            d_idxs = udata_idxs[cidx]
            dtrains[cidx]['x'] = [data['x'][did] for did in d_idxs]
            dtrains[cidx]['y'] = [data['y'][did] for did in d_idxs]
        return dtrains

    def fill_output(self, dtrains):
        self.output = {
            'meta': {
                'benchmark': self.benchmark,
                'num_clients': self.num_clients,
                'dist': self.dist,
                'beta': self.beta
            },
            'clients': {},
            'dtest': self.test_data
        }

        for cidx in range(self.num_clients):
            self.output['clients'][self.cnames[cidx]] = {
                'dtrain': {'x': dtrains[cidx]['x'], 'y': dtrains[cidx]['y']},
                'dvalid': {'x': [], 'y': []},
                'dvol': len(dtrains[cidx]['y']),
            }
        return

    def partition(self):
        if self.dist == 0:
            """iid partition"""
            d_idxs = np.random.permutation(self.datavol)
            udata_idxs = np.array_split(d_idxs, self.num_clients)

        elif self.dist == 1:
            """label_skew_quantity"""
            num = self.beta
            K = self.num_classes
            if num == K:
                udata_idxs = [[] for i in range(self.num_clients)]
                for i in range(K):
                    idx_k = np.where(np.array(self.train_data['y']) == i)[0]
                    np.random.shuffle(idx_k)
                    split = np.array_split(idx_k, self.num_clients)
                    for j in range(self.num_clients):
                        udata_idxs[j].extend(split[j].tolist())
            else:
                times = [0 for i in range(self.num_classes)]
                contain = []
                for i in range(self.num_clients):
                    current = [i % K]
                    times[i % K] += 1
                    j = 1
                    while (j < num):
                        ind = random.randint(0, K - 1)
                        if (ind not in current):
                            j = j + 1
                            current.append(ind)
                            times[ind] += 1
                    contain.append(current)
                udata_idxs = [[] for i in range(self.num_clients)]
                for i in range(K):
                    idx_k = np.where(np.array(self.train_data['y']) == i)[0]
                    np.random.shuffle(idx_k)
                    split = np.array_split(idx_k, times[i])
                    ids = 0
                    for j in range(self.num_clients):
                        if i in contain[j]:
                            udata_idxs[j].extend(split[ids].tolist())
                            ids += 1

        elif self.dist == 2:
            """label_skew_dirichlet"""
            min_size = 0
            udata_idxs = [[] for i in range(self.num_clients)]
            while min_size < self.minvol:
                idx_batch = [[] for i in range(self.num_clients)]
                for k in range(self.num_classes):
                    idx_k = np.where(np.array(self.train_data['y']) == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(self.beta, self.num_clients))
                    ## Balance
                    proportions = np.array([p * (len(idx_j) < self.datavol / self.num_clients) for p, idx_j in zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])
            for j in range(self.num_clients):
                np.random.shuffle(idx_batch[j])
                udata_idxs[j].extend(idx_batch[j])

        elif self.dist == 3:
            """label_skew_shard"""
            # beta: the number of the shards (splitted by the sorted labels) owned by each client
            num_shards = int(self.beta)
            client_datasize = int(self.datavol / self.num_clients)
            all_idxs = [i for i in range(self.datavol)]
            labels = self.train_data['y']
            z = zip(labels, all_idxs)
            z = sorted(z)
            labels, all_idxs = zip(*z)
            shardsize = int(client_datasize / num_shards)
            idxs_shard = range(int(self.num_clients * num_shards))
            udata_idxs = [[] for i in range(self.num_clients)]
            for i in range(self.num_clients):
                rand_set = set(np.random.choice(idxs_shard, num_shards, replace=False))
                idxs_shard = list(set(idxs_shard) - rand_set)
                for rand in rand_set:
                    udata_idxs[i].extend(all_idxs[rand * shardsize:(rand + 1) * shardsize])

        elif self.dist == 4:
            self.noise = 1
            d_idxs = np.random.permutation(self.datavol)
            udata_idxs = np.array_split(d_idxs, self.num_clients)

        elif self.dist == 5:
            udata_idxs = []
            # self.num_clients = u_train.shape[0]
            # user = np.zeros(self.num_clients + 1, dtype=np.int32)
            # for i in range(1, self.num_clients + 1):
            #     user[i] = user[i - 1] + u_train[i - 1]
            # no = np.random.permutation(self.num_clients)
            # batch_idxs = np.array_split(no, self.num_clients)
            # net_dataidx_map = {i: np.zeros(0, dtype=np.int32) for i in range(self.num_clients)}
            # for i in range(self.num_clients):
            #     for j in batch_idxs[i]:
            #         net_dataidx_map[i] = np.append(net_dataidx_map[i], np.arange(user[j], user[j + 1]))

        elif self.dist == 6:
            minv = 0
            d_idxs = np.random.permutation(self.datavol)
            while minv < self.minvol:
                proportions = np.random.dirichlet(np.repeat(self.beta, self.num_clients))
                proportions = proportions / proportions.sum()
                minv = np.min(proportions * self.datavol)
            proportions = (np.cumsum(proportions) * len(d_idxs)).astype(int)[:-1]
            udata_idxs  = np.split(d_idxs, proportions)

        return udata_idxs

    def download_from_url(self, url= None, filepath = os.path.join('.','tmp')):
        if url: urllib.request.urlretrieve(url, filepath)
        return filepath

    def extract_from_zip(self, src_path, target_path):
        f = zipfile.ZipFile(src_path)
        f.extractall(target_path)
        targets = f.namelist()
        f.close()
        return [os.path.join(target_path, tar) for tar in targets]

class CIFAR100_TaskGenerator(TaskGenerator):
    def __init__(self, dist, num_clients = 1, beta = 0.5, noise = 0, minvol = 10, cnames = []):
        super(CIFAR100_TaskGenerator, self).__init__('cifar100', 100, dist, num_clients, beta, noise, minvol, './benchmark/cifar100/data', cnames)
        self.supported_dists = [0,1,2,3,4,6]

    def load_data(self):
        self.train_data = datasets.CIFAR100(self.datapath, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))]))
        self.test_data = datasets.CIFAR100(self.datapath, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))]))
        self.datavol = len(self.train_data)

    def preprocess_data(self):
        train_x = [self.train_data[did][0].tolist() for did in range(len(self.train_data))]
        train_y = [self.train_data[did][1] for did in range(len(self.train_data))]
        test_x = [self.test_data[did][0].tolist() for did in range(len(self.test_data))]
        test_y = [self.test_data[did][1] for did in range(len(self.test_data))]
        self.train_data = {'x':train_x, 'y':train_y}
        self.test_data = {'x': test_x, 'y': test_y}
        return

class CIFAR10_TaskGenerator(TaskGenerator):
    def __init__(self, dist, num_clients = 1, beta = 0.5, noise = 0, minvol = 10, cnames = []):
        super(CIFAR10_TaskGenerator, self).__init__('cifar10', 10, dist, num_clients, beta, noise, minvol, './benchmark/cifar10/data', cnames)
        self.supported_dists = [0, 1, 2, 3, 4, 6]

    def load_data(self):
        self.train_data = datasets.CIFAR10(self.datapath, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
        self.test_data = datasets.CIFAR10(self.datapath, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
        self.datavol = len(self.train_data)

    def preprocess_data(self):
        train_x = [self.train_data[did][0].tolist() for did in range(len(self.train_data))]
        train_y = [self.train_data[did][1] for did in range(len(self.train_data))]
        test_x = [self.test_data[did][0].tolist() for did in range(len(self.test_data))]
        test_y = [self.test_data[did][1] for did in range(len(self.test_data))]
        self.train_data = {'x':train_x, 'y':train_y}
        self.test_data = {'x': test_x, 'y': test_y}
        return

class MNIST_TaskGenerator(TaskGenerator):
    def __init__(self, dist, num_clients = 1, beta = 0.5, noise = 0, minvol = 10, cnames = []):
        super(MNIST_TaskGenerator, self).__init__('mnist', 10, dist, num_clients, beta, noise, minvol, './benchmark/mnist/data', cnames)
        self.supported_dists = [0, 1, 2, 3, 4, 6]

    def load_data(self):
        self.train_data = datasets.MNIST(self.datapath, train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        self.test_data = datasets.MNIST(self.datapath, train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        self.datavol = len(self.train_data)

    def preprocess_data(self):
        train_x = [self.train_data[did][0].tolist() for did in range(len(self.train_data))]
        train_y = [self.train_data[did][1] for did in range(len(self.train_data))]
        test_x = [self.test_data[did][0].tolist() for did in range(len(self.test_data))]
        test_y = [self.test_data[did][1] for did in range(len(self.test_data))]
        self.train_data = {'x':train_x, 'y':train_y}
        self.test_data = {'x': test_x, 'y': test_y}
        return

    def class_num(self):
        res = {}
        for i in range(self.num_classes):
            res[i] = 0
        for i in range(len(self.train_data['y'])):
            res[self.train_data['y'][i]]+=1
        return res

class FashionMNIST_TaskGenerator(TaskGenerator):
    def __init__(self, dist, num_clients = 1, beta = 0.5, noise = 0, minvol = 10, cnames = [], selected = [i for i in range(10)]):
        super(FashionMNIST_TaskGenerator, self).__init__('fashion_mnist', 10, dist, num_clients, beta, noise, minvol, './benchmark/fashion_mnist/data', cnames, selected)
        self.label_dict = {0: 'T-shirt', 1: 'Trouser', 2: 'pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Abkle boot'}
        self.selected = selected
        self.cnames = [self.label_dict[i] for i in self.selected]
        self.num_labels = len(selected)
        self.supported_dists = [0, 1, 2, 3, 4, 6]

    def load_data(self):
        self.train_data = datasets.FashionMNIST(self.datapath, train=True, download=True,
                                         transform=transforms.Compose([transforms.ToTensor()]))
        self.test_data = datasets.FashionMNIST(self.datapath, train=False, download=True,
                                        transform=transforms.Compose([transforms.ToTensor()]))
        X_train, y_train = self.load_mnist('fashion_mnist/data/FashionMNIST/raw/', kind='train')
        X_test, y_test = self.load_mnist('fashion_mnist/data/FashionMNIST/raw/', kind='t10k')
        mu = np.mean(X_train.astype(np.float32), 0)
        sigma = np.std(X_train.astype(np.float32), 0)
        self.X_train = ((X_train.astype(np.float32) - mu) / (sigma + 0.001)).tolist()
        self.X_test = ((X_test.astype(np.float32) - mu) / (sigma + 0.001)).tolist()
        self.y_train = y_train.tolist()
        self.y_test = y_test.tolist()


    def preprocess_data(self):
        X_trains = [[] for i in range(10)]
        y_trains = [[] for i in range(10)]
        for idx, item in enumerate(self.X_train):
            i = self.y_train[idx]
            if i in self.selected:
                X_trains[i].append(self.X_train[idx])
                y_trains[i].append(self.y_train[idx])
        X_tests = [[] for i in range(10)]
        y_tests = [[] for i in range(10)]
        for idx, item in enumerate(self.X_test):
            i = self.y_test[idx]
            if i in self.selected:
                X_tests[i].append(self.X_test[idx])
                y_tests[i].append(self.y_test[idx])
        xtrain = []
        ytrain = []
        xtest = []
        ytest = []
        for i in self.selected:
            xtrain.extend(X_trains[i])
            ytrain.extend(y_trains[i])
            xtest.extend(X_tests[i])
            ytest.extend(y_tests[i])
        cvt_labels = {}
        for i in range(len(self.selected)):
            cvt_labels[self.selected[i]] = i
        self.train_data = {'x':xtrain, 'y':[cvt_labels[i] for i in ytrain]}
        self.test_data = {'x': xtest, 'y': [cvt_labels[i] for i in ytest]}
        return

    def load_mnist(self, path, kind='train'):
        """Load MNIST data from `path`"""
        labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
        images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)
        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
        return images, labels

class Shakespeare_TaskGenerator(TaskGenerator):
    def __init__(self, dist, num_clients = 1, beta = 0.5, noise = 0, minvol = 10, cnames = [], selected = 0, val_frac=0.2):
        super(Shakespeare_TaskGenerator, self).__init__('shakespeare', 10, dist, num_clients, beta, noise, minvol, './benchmark/shakespeare/data', cnames, selected)
        self.RANDOM_SEED = 1234
        # Regular expression to capture an actors name, and line continuation
        self.CHARACTER_RE = re.compile(r'^  ([a-zA-Z][a-zA-Z ]*)\. (.*)')
        self.CONT_RE = re.compile(r'^    (.*)')
        # The Comedy of Errors has errors in its indentation so we need to use
        # different regular expressions.
        self.COE_CHARACTER_RE = re.compile(r'^([a-zA-Z][a-zA-Z ]*)\. (.*)')
        self.COE_CONT_RE = re.compile(r'^(.*)')
        self.ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
        self.NUM_LETTERS = len(self.ALL_LETTERS)
        self.SEQ_LENGTH = 80
        self.val_frac = val_frac
        self.supported_dists = [0, 5, 6]
        return

    def generate(self):
        self.load_data()
        self.preprocess_data()
        user_dict = self.partition()
        # fill output
        self.fill_output(user_dict)
        self.save_task()
        return

    def partition(self):
        if self.dist == 0:
            trainX = []
            trainY = []
            validX = []
            validY = []
            for user in self.user_dict.keys():
                trainX.extend(self.user_dict[user]['dtrain']['x'])
                trainY.extend(self.user_dict[user]['dtrain']['y'])
                validX.extend(self.user_dict[user]['dvalid']['x'])
                validY.extend(self.user_dict[user]['dvalid']['y'])
            user_dict = {}
            train_d_idxs = np.random.permutation(len(trainY))
            valid_d_idxs = np.random.permutation(len(validY))
            train_udata_idxs = np.array_split(train_d_idxs, self.num_clients)
            valid_udata_idxs = np.array_split(valid_d_idxs, self.num_clients)
            cnames = ['client '+str(i) for i in range(self.num_clients)]
            for cid in range(self.num_clients):
                user_dict[cnames[cid]] = {
                    'dtrain':{
                        'x':[trainX[did] for did in train_udata_idxs[cid]],
                        'y':[trainY[did] for did in train_udata_idxs[cid]],
                    },
                    'dvalid':{
                        'x': [validX[did] for did in valid_udata_idxs[cid]],
                        'y': [validY[did] for did in valid_udata_idxs[cid]],
                    },
                    'dvol':len(train_udata_idxs[cid])
                }
            return user_dict

        elif self.dist == 5:
            return self.user_dict

        elif self.dist == 6:
            minv = 0
            trainX = []
            trainY = []
            validX = []
            validY = []
            for user in self.user_dict.keys():
                trainX.extend(self.user_dict[user]['dtrain']['x'])
                trainY.extend(self.user_dict[user]['dtrain']['y'])
                validX.extend(self.user_dict[user]['dvalid']['x'])
                validY.extend(self.user_dict[user]['dvalid']['y'])
            user_dict = {}
            train_d_idxs = np.random.permutation(len(trainY))
            valid_d_idxs = np.random.permutation(len(validY))
            cnames = ['client ' + str(i) for i in range(self.num_clients)]
            while minv < self.minvol:
                proportions = np.random.dirichlet(np.repeat(self.beta, self.num_clients))
                proportions = proportions / proportions.sum()
                minv = np.min(proportions * self.datavol)
            train_proportions = (np.cumsum(proportions) * len(train_d_idxs)).astype(int)[:-1]
            train_udata_idxs = np.split(train_d_idxs, train_proportions)
            valid_proportions = (np.cumsum(proportions) * len(valid_d_idxs)).astype(int)[:-1]
            valid_udata_idxs = np.split(valid_d_idxs, valid_proportions)
            for cid in range(self.num_clients):
                user_dict[cnames[cid]] = {
                    'dtrain':{
                        'x':[trainX[did] for did in train_udata_idxs[cid]],
                        'y':[trainY[did] for did in train_udata_idxs[cid]],
                    },
                    'dvalid':{
                        'x': [validX[did] for did in valid_udata_idxs[cid]],
                        'y': [validY[did] for did in valid_udata_idxs[cid]],
                    },
                    'dvol':len(train_udata_idxs[cid])
                }
            return user_dict

    def fill_output(self, user_dict):
        self.output = {
            'meta': {
                'benchmark': self.benchmark,
                'num_clients': self.num_clients,
                'dist': self.dist,
                'beta': self.beta
            },
            'clients': user_dict,
            'dtest': self.test_data
        }
        return

    def load_data(self):
        # download and read raw dataset
        raw_path = os.path.join(self.datapath, 'raw_data')
        all_data_path = os.path.join(raw_path, 'all_data.json')
        if not os.path.exists(all_data_path):
            if not os.path.exists(raw_path):
                os.mkdir(raw_path)
            src_path = self.download_from_url("http://www.gutenberg.org/files/100/old/1994-01-100.zip", os.path.join(raw_path, 'tmp'))
            tar_paths = self.extract_from_zip(src_path, raw_path)
            os.remove(src_path)
            with open(tar_paths[0], 'r') as input_file:
                shakespeare_full = input_file.read()
            plays, discarded_lines = self._split_into_plays(shakespeare_full)
            users_and_plays, all_examples, num_skipped = self._get_examples_by_character(plays)
            all_data = {}
            for user in all_examples.keys():
                all_data[user] = {
                    'play': users_and_plays[user],
                    'sound_bites': all_examples[user],
                }
            with open(all_data_path,'w') as f:
                ujson.dump(all_data, f)
            os.remove(tar_paths[0])
        return all_data_path

    def example_to_text(self, examples):
        text = ' '.join(examples)
        text = re.sub(r"   *", r' ', text)
        X = []
        Y = []
        for i in range(0, len(text) - self.SEQ_LENGTH, 1):
            seq_in = text[i:i + self.SEQ_LENGTH]
            seq_out = text[i + self.SEQ_LENGTH]
            X.append(seq_in)
            Y.append(seq_out)
        return X,Y

    def X_text_to_vec(self, X):
        return [[self.ALL_LETTERS.find(c) for c in word] for word in X]

    def Y_text_to_vec(self, Y):
        return [self.ALL_LETTERS.find(c) for c in Y]

    def preprocess_data(self):
        with open(os.path.join(self.datapath, 'raw_data','all_data.json'),'r') as f:
            all_data = ujson.load(f)
        self.cnames = []
        user_dict = {}
        all_users = [user for user in all_data.keys()]
        train_users = []
        while(len(train_users)<self.num_clients):
            user = np.random.choice(all_users)
            if user in train_users: continue
            examples = all_data[user]['sound_bites']
            play = all_data[user]['play']
            # devide the sentence into training set and validating set
            num_val = max(int(len(examples) * self.val_frac), 1)
            train_examples = examples[:-num_val]
            val_examples = examples[-num_val:]
            assert len(val_examples) == num_val
            assert len(train_examples) >= len(val_examples)
            trainX, trainY = self.example_to_text(train_examples)
            valX, valY = self.example_to_text(val_examples)
            trainX = self.X_text_to_vec(trainX)
            trainY = self.Y_text_to_vec(trainY)
            valX = self.X_text_to_vec(valX)
            valY = self.Y_text_to_vec(valY)
            dvol = len(trainX)
            if dvol<self.minvol:
                continue
            if trainY and valY:
                train_users.append(user)
                self.cnames.append(user)
                user_dict[user] = {}
                user_dict[user]['dtrain'] = {'x': trainX, 'y': trainY}
                user_dict[user]['dvalid'] = {'x': valX, 'y': valY}
                user_dict[user]['dvol'] = len(trainY)
        test_users = []
        test_dict = {'x':[], 'y':[]}
        num_test = self.beta
        all_users = list(set(all_users)-set(train_users))
        while(len(test_users)<num_test and all_users):
            user = np.random.choice(all_users)
            if user in test_users: continue
            test_users.append(user)
            test_examples = all_data[user]['sound_bites']
            play = all_data[user]['play']
            testX, testY = self.example_to_text(test_examples)
            test_dict['x'].extend(self.X_text_to_vec(testX))
            test_dict['y'].extend(self.Y_text_to_vec(testY))
        self.test_data = test_dict
        self.user_dict = user_dict
        return

    def _split_into_plays(self, shakespeare_full):
        """Splits the full data by play."""
        # List of tuples (play_name, dict from character to list of lines)
        plays = []
        discarded_lines = []  # Track discarded lines.
        slines = shakespeare_full.splitlines(True)[1:]
        # skip contents, the sonnets, and all's well that ends well
        author_count = 0
        start_i = 0
        for i, l in enumerate(slines):
            if 'by William Shakespeare' in l:
                author_count += 1
            if author_count == 1:
                start_i = i - 5
                break
        slines = slines[start_i:]
        current_character = None
        comedy_of_errors = False
        for i, line in enumerate(slines):
            # This marks the end of the plays in the file.
            if i > 124195 - start_i:
                break
            # This is a pretty good heuristic for detecting the start of a new play:
            if 'by William Shakespeare' in line:
                current_character = None
                characters = collections.defaultdict(list)
                # The title will be 2, 3, 4, 5, 6, or 7 lines above "by William Shakespeare".
                for j in range(2,8):
                    if slines[i - j].strip():
                        title = slines[i-j]
                        break
                title = title.strip()

                assert title, (
                        'Parsing error on line %d. Expecting title 2 or 3 lines above.' %
                        i)
                comedy_of_errors = (title == 'THE COMEDY OF ERRORS')
                # Degenerate plays are removed at the end of the method.
                plays.append((title, characters))
                continue
            # match_character_regex
            match = (self.COE_CHARACTER_RE.match(line) if comedy_of_errors else self.CHARACTER_RE.match(line))
            if match:
                character, snippet = match.group(1), match.group(2)
                # Some character names are written with multiple casings, e.g., SIR_Toby
                # and SIR_TOBY. To normalize the character names, we uppercase each name.
                # Note that this was not done in the original preprocessing and is a
                # recent fix.
                character = character.upper()
                if not (comedy_of_errors and character.startswith('ACT ')):
                    characters[character].append(snippet)
                    current_character = character
                    continue
                else:
                    current_character = None
                    continue
            elif current_character:
                # _match_continuation_regex
                match = (self.COE_CONT_RE.match(line) if comedy_of_errors else self.CONT_RE.match(line))
                if match:
                    if comedy_of_errors and match.group(1).startswith('<'):
                        current_character = None
                        continue
                    else:
                        characters[current_character].append(match.group(1))
                        continue
            # Didn't consume the line.
            line = line.strip()
            if line and i > 2646:
                # Before 2646 are the sonnets, which we expect to discard.
                discarded_lines.append('%d:%s' % (i, line))
        # Remove degenerate "plays".
        return [play for play in plays if len(play[1]) > 1], discarded_lines

    def play_and_character(self, play, character):
        return re.sub('\\W+', '_',(play + '_' + character).replace(' ', '_'))

    def _get_examples_by_character(self, plays):
        skipped_characters = 0
        all_examples = collections.defaultdict(list)
        def add_examples(example_dict, example_tuple_list):
            for play, character, sound_bite in example_tuple_list:
                example_dict[self.play_and_character(
                    play, character)].append(sound_bite)
        users_and_plays = {}
        for play, characters in plays:
            curr_characters = list(characters.keys())
            for c in curr_characters:
                users_and_plays[self.play_and_character(play, c)] = play
            for character, sound_bites in characters.items():
                examples = [(play, character, sound_bite) for sound_bite in sound_bites]
                if len(examples) < 2:
                    skipped_characters += 1
                    # Skip characters with fewer than 2 lines since we need at least one
                    # train and one test line.
                    continue
                add_examples(all_examples, examples)
        return users_and_plays, all_examples, skipped_characters

class Synthetic_TaskGenerator(TaskGenerator):
    def __init__(self, num_classes=10, seed=931231, dimension=60 , dist = 0, num_clients = 30, beta = (0,0), noise = 0, minvol = 10, datapath ='./benchmark/synthetic/data', cnames = []):
        super(Synthetic_TaskGenerator, self).__init__('synthetic', num_classes, dist, num_clients, beta, noise, minvol, datapath, cnames)
        np.random.seed(seed)
        self.dimension = dimension
        self.num_classes = num_classes
        self.W_global = np.random.normal(0, 1, (self.dimension, self.num_classes))
        self.b_global = np.random.normal(0, 1, self.num_classes)
        self.supported_dists = [0, 5, 6, 7]

    def generate(self):
        xs, ys = self.gen_data(self.num_clients)
        x_tests = [di[int(0.8*len(di)):] for di in xs]
        x_trains = [di[:int(0.8*len(di))] for di in xs]
        y_tests = [di[int(0.8*len(di)):] for di in ys]
        y_trains = [di[:int(0.8*len(di))] for di in ys]
        # set dtrains = [cdata_1,... , cdata_m] where cdata_i = {'x':[...], 'y':[...]}
        dtrains = [{'x':x_trains[cid],'y':y_trains[cid]} for cid in range(self.num_clients)]
        X_test = []
        Y_test = []
        for i in range(len(y_tests)):
            X_test.extend(x_tests[i])
            Y_test.extend(y_tests[i])
        self.test_data = {'x':X_test, 'y':Y_test}
        # add noise to the clients' training data
        if self.noise > 0:
            dtrains = self.add_noise(dtrains)
        self.fill_output(dtrains)
        self.save_task()

    def softmax(self, x):
        ex = np.exp(x)
        sum_ex = np.sum(np.exp(x))
        return ex / sum_ex

    def gen_data(self, num_clients):
        self.dimension = 60
        if self.dist == 6 or self.dist ==7:
            samples_per_user = np.random.lognormal(4, 2, (num_clients)).astype(int) + self.minvol
        else:
            samples_per_user = [40*self.minvol for _ in range(self.num_clients)]
        X_split = [[] for _ in range(num_clients)]
        y_split = [[] for _ in range(num_clients)]
        #### define some eprior ####
        mean_W = np.random.normal(0, self.beta[0], num_clients)
        mean_b = mean_W
        B = np.random.normal(0, self.beta[1], num_clients)
        mean_x = np.zeros((num_clients, self.dimension))
        diagonal = np.zeros(self.dimension)
        for j in range(self.dimension):
            diagonal[j] = np.power((j + 1), -1.2)
        cov_x = np.diag(diagonal)
        for i in range(num_clients):
            mean_x[i] = np.ones(self.dimension) * B[i] if self.dist==0 else np.random.normal(B[i], 1, self.dimension)
        for i in range(num_clients):
            W = self.W_global if (self.dist==0 or self.dist==6) else np.random.normal(mean_W[i], 1, (self.dimension, self.num_classes))
            b = self.b_global if (self.dist==0 or self.dist==6) else np.random.normal(mean_b[i], 1, self.num_classes)
            xx = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
            yy = np.zeros(samples_per_user[i], dtype=int)
            for j in range(samples_per_user[i]):
                tmp = np.dot(xx[j], W) + b
                yy[j] = np.argmax(softmax(tmp))
            X_split[i] = xx.tolist()
            y_split[i] = yy.tolist()
        return X_split, y_split

