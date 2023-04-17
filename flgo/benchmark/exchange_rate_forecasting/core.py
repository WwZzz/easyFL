from flgo.benchmark.toolkits.series.time_series_forecasting import BuiltinClassGenerator, BuiltinClassPipe, GeneralCalculator
import os
import flgo
from flgo.benchmark.toolkits.series.time_series_forecasting.datasets import BuiltinClassDataset
from flgo.benchmark.toolkits import download_from_url, extract_from_gz, normalized
import numpy as np
import ujson as json

path = os.path.join(flgo.benchmark.path,'RAW_DATA', 'EXCHANGE_RATE')

class ExchangeRate(BuiltinClassDataset):
    def __init__(self, root, train=True, window=24, horizon=12, normalize=2):
        self.normalize = normalize
        self.url = \
            "https://github.com/laiguokun/multivariate-time-series-data/raw/master/exchange_rate/exchange_rate.txt.gz"
        self.file = 'w_' + str(window) + '_h_' + str(horizon) + '_'
        self.window = window
        self.horizon = horizon
        super(ExchangeRate, self).__init__(root, train)

    def download(self):
        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)
        src_path = os.path.join(self.raw_folder, 'exchange_rate.txt.gz')
        if not os.path.exists(src_path):
            download_from_url(self.url, src_path)
        src_path = extract_from_gz(src_path, os.path.join(self.raw_folder, 'exchange_rate.txt'))
        raw_data = np.loadtxt(src_path, delimiter=',')
        data = normalized(raw_data, self.normalize)
        self.split(data)

    def split(self, data):
        n, m = data.shape
        s1 = int(n * 0.8)
        train_idx = range(self.window + self.horizon - 1, s1)
        test_idx = range(s1, n)
        train_data = self.batchify(train_idx, data)
        with open(os.path.join(self.processed_folder,
                               'w_' + str(self.window) + '_h_' + str(self.horizon) + '_train_data.json'), 'w') as f:
            json.dump(train_data, f)
        test_data = self.batchify(test_idx, data)
        with open(os.path.join(self.processed_folder,
                               'w_' + str(self.window) + '_h_' + str(self.horizon) + '_test_data.json'), 'w') as f:
            json.dump(test_data, f)

    def batchify(self, idx_set, data):
        n = len(idx_set)
        X = []
        Y = []
        for i in range(n):
            end = idx_set[i] - self.horizon + 1
            start = end - self.window
            X.append(data[start:end, :].tolist())
            Y.append(data[idx_set[i], :].tolist())
        return {'x': X, 'y': Y}

class TaskGenerator(BuiltinClassGenerator):
    def __init__(self, rawdata_path=path):
        super(TaskGenerator, self).__init__(os.path.split(os.path.dirname(__file__))[-1], rawdata_path, ExchangeRate)

class TaskPipe(BuiltinClassPipe):
    def __init__(self, task_path):
        super(TaskPipe, self).__init__(task_path, ExchangeRate)

TaskCalculator = GeneralCalculator