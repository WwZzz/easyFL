import torch.utils.data as tua
import os
import numpy as np
import json
import torch

root = os.path.dirname(__file__)

class Synthetic(tua.Dataset):
    def __init__(self, root:str, train=False, alpha=0.0, beta=0.0, datavol=400, dimension=60, num_classes=10,):
        self.alpha = alpha
        self.beta = beta
        self.datavol = datavol
        self.dimension = dimension
        self.num_classes = num_classes
        self.root = root
        self.train = train
        self.data_path = os.path.join(root, 'data.json')
        if not os.path.exists(self.data_path): self.gen_data()
        self.X, self.Y = self.read_data()

    def read_data(self):
        with open(self.data_path, 'r') as inf:
            data = json.load(inf)
        data = data['train'] if self.train else data['test']
        return data['x'], data['y']

    def gen_data(self):
        W_global = np.random.normal(0, 1, (self.dimension, self.num_classes))
        b_global = np.random.normal(0, 1, self.num_classes)
        v_global = np.zeros(self.dimension)
        # create Sigma = Diag([i^(-1.2) for i in range(60)])
        diagonal = np.zeros(self.dimension)
        for j in range(self.dimension):
            diagonal[j] = np.power((j + 1), -1.2)
        Sigma = np.diag(diagonal)
        # V
        datavol = self.datavol
        # Concept Skew
        if self.alpha>=0:
            Us = np.random.normal(0, self.alpha, 1)
            W = np.random.normal(Us[0], 1, (self.dimension, self.num_classes))
            b = np.random.normal(Us[0], 1, self.num_classes)
        else:
            W = W_global
            b = b_global
        # Feature Skew
        if self.beta>=0:
            B = np.random.normal(0, self.beta, 1)
            V = np.random.normal(B[0], 1, self.dimension)
        else:
            V = v_global
        # X_ki~N(v_k, Sigma)
        X_k = np.random.multivariate_normal(V, Sigma, datavol)
        Y_k = np.zeros(datavol, dtype=int)
        for i in range(datavol):
            tmp = np.dot(X_k[i], W) + b
            Y_k[i] = np.argmax(np.exp(tmp)/np.sum(np.exp(tmp)))
        X_split = X_k.tolist()
        y_split = Y_k.tolist()
        # split dataset to train and testing part
        k = int(datavol*0.85)
        x_train = X_split[:k]
        y_train = y_split[:k]
        x_test = X_split[k:]
        y_test = y_split[k:]
        test_data = {'x': x_test, 'y':y_test}
        train_data = {'x': x_train, 'y':y_train}
        data = {'train': train_data, 'test': test_data}
        with open(self.data_path, 'w') as outf:
            json.dump(data, outf)
        return

    def __getitem__(self, item):
        return torch.tensor(self.X[item]), torch.tensor(self.Y[item], dtype=torch.long)

    def __len__(self):
        return len(self.Y)

train_data = Synthetic(root, train=True)
test_data = Synthetic(root, train=False)
val_data = None

