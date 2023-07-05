from flgo.benchmark.base import *
from flgo.benchmark.toolkits.cv.classification import GeneralCalculator
from flgo.benchmark.toolkits.partition import BasicPartitioner
import numpy as np

TaskPipe = XYHorizontalTaskPipe
TaskCalculator = GeneralCalculator

class TaskGenerator(BasicTaskGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__('fcube_classification', '')
        self.num_clients = 4
        self.num_classes = 2

    def load_data(self):
        X_train, y_train = [], []
        for loc in range(self.num_clients):
            for i in range(1000):
                p1 = random.random()
                p2 = random.random()
                p3 = random.random()
                if loc > 1:
                    p2 = -p2
                if loc % 2 == 1:
                    p3 = -p3
                if i % 2 == 0:
                    X_train.append([p1, p2, p3])
                    y_train.append(0)
                else:
                    X_train.append([-p1, -p2, -p3])
                    y_train.append(1)
        X_test, y_test = [], []
        for i in range(1000):
            p1 = random.random() * 2 - 1
            p2 = random.random() * 2 - 1
            p3 = random.random() * 2 - 1
            X_test.append([p1, p2, p3])
            if p1 > 0:
                y_test.append(0)
            else:
                y_test.append(1)
        idxs = np.linspace(0, 3999, 4000, dtype=np.int64)
        batch_idxs = np.array_split(idxs, self.num_clients)
        self.local_datas = [{'x': [X_train[did] for did in batch], 'y':[y_train[did] for did in batch]} for batch in batch_idxs]
        self.test_data = {'x':X_test, 'y':y_test}
        return

    def get_task_name(self):
        partitioner_name = 'feature_skew'
        return '_'.join(['B-'+self.benchmark, 'P-'+partitioner_name, 'N-'+str(self.num_clients)])
