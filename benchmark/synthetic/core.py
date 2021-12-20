from benchmark.toolkits import BasicTaskGen, XYTaskReader, ClassifyCalculator
from scipy.special import softmax
import numpy as np
import os.path
import ujson

class TaskGen(BasicTaskGen):
    def __init__(self, num_classes=10, dimension=60, dist_id = 0, num_clients = 30, skewness = (0, 0), minvol=10, rawdata_path ='./benchmark/synthetic/data'):
        super(TaskGen, self).__init__(benchmark='synthetic',
                                      dist_id=dist_id,
                                      skewness=skewness,
                                      rawdata_path=rawdata_path)
        self.dimension = dimension
        self.num_classes = num_classes
        self.num_clients = num_clients
        self.W_global = np.random.normal(0, 1, (self.dimension, self.num_classes))
        self.b_global = np.random.normal(0, 1, self.num_classes)
        self.minvol = minvol
        self.taskname = self.get_taskname()
        self.taskpath = os.path.join(self.rootpath, self.taskname)

    def run(self):
        if not self._check_task_exist():
            self.create_task_directories()
        else:
            print("Task Already Exists.")
            return

        xs, ys = self.gen_data(self.num_clients)
        x_trains = [di[:int(0.75 * len(di))] for di in xs]
        y_trains = [di[:int(0.75 * len(di))] for di in ys]
        x_valids = [di[int(0.75 * len(di)):int(0.90 * len(di))] for di in xs]
        y_valids = [di[int(0.75 * len(di)):int(0.90 * len(di))] for di in ys]
        x_tests = [di[int(0.90 * len(di)):] for di in xs]
        y_tests = [di[int(0.90 * len(di)):] for di in ys]
        self.cnames = self.get_client_names()
        X_test = []
        Y_test = []
        for i in range(len(y_tests)):
            X_test.extend(x_tests[i])
            Y_test.extend(y_tests[i])
        self.test_data = {'x': X_test, 'y': Y_test}
        feddata = {
            'store': 'XY',
            'client_names': self.cnames,
            'dtest': self.test_data
        }
        for cid in range(self.num_clients):
            feddata[self.cnames[cid]] = {
                'dtrain': {
                    'x': x_trains[cid],
                    'y': y_trains[cid]
                },
                'dvalid': {
                    'x': x_valids[cid],
                    'y': y_valids[cid]
                }
            }

        with open(os.path.join(self.taskpath, 'data.json'), 'w') as outf:
            ujson.dump(feddata, outf)

    def softmax(self, x):
        ex = np.exp(x)
        sum_ex = np.sum(np.exp(x))
        return ex / sum_ex

    def gen_data(self, num_clients):
        self.dimension = 60
        if self.dist_id == 6 or self.dist_id ==7:
            samples_per_user = np.random.lognormal(4, 2, (num_clients)).astype(int) + self.minvol
        else:
            samples_per_user = [40*self.minvol for _ in range(self.num_clients)]
        X_split = [[] for _ in range(num_clients)]
        y_split = [[] for _ in range(num_clients)]
        #### define some eprior ####
        mean_W = np.random.normal(0, self.skewness[0], num_clients)
        mean_b = mean_W
        B = np.random.normal(0, self.skewness[1], num_clients)
        mean_x = np.zeros((num_clients, self.dimension))
        diagonal = np.zeros(self.dimension)
        for j in range(self.dimension):
            diagonal[j] = np.power((j + 1), -1.2)
        cov_x = np.diag(diagonal)
        for i in range(num_clients):
            mean_x[i] = np.ones(self.dimension) * B[i] if self.dist_id == 0 else np.random.normal(B[i], 1, self.dimension)
        for i in range(num_clients):
            W = self.W_global if (self.dist_id == 0 or self.dist_id == 6) else np.random.normal(mean_W[i], 1, (self.dimension, self.num_classes))
            b = self.b_global if (self.dist_id == 0 or self.dist_id == 6) else np.random.normal(mean_b[i], 1, self.num_classes)
            xx = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
            yy = np.zeros(samples_per_user[i], dtype=int)
            for j in range(samples_per_user[i]):
                tmp = np.dot(xx[j], W) + b
                yy[j] = np.argmax(softmax(tmp))
            X_split[i] = xx.tolist()
            y_split[i] = yy.tolist()
        return X_split, y_split

class TaskReader(XYTaskReader):
    def __init__(self, taskpath=''):
        super(TaskReader, self).__init__(taskpath)

class TaskCalculator(ClassifyCalculator):
    def __init__(self, device):
        super(TaskCalculator, self).__init__(device)
