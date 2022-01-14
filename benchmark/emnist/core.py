from torchvision import datasets, transforms
from benchmark.toolkits import ClassifyCalculator, DefaultTaskGen, XYTaskReader

class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients = 1, skewness = 0.5):
        super(TaskGen, self).__init__(benchmark='emnist',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/emnist/data',
                                      )
        self.num_classes = 26
        self.save_data = self.XYData_to_json

    def load_data(self):
        self.train_data = datasets.EMNIST(self.rawdata_path, split='letters', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        self.test_data = datasets.EMNIST(self.rawdata_path, split='letters', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

    def convert_data_for_saving(self):
        train_x = [self.train_data[did][0].tolist() for did in range(len(self.train_data))]
        train_y = [self.train_data[did][1] for did in range(len(self.train_data))]
        test_x = [self.test_data[did][0].tolist() for did in range(len(self.test_data))]
        test_y = [self.test_data[did][1] for did in range(len(self.test_data))]
        self.train_data = {'x': train_x, 'y': train_y}
        self.test_data = {'x': test_x, 'y': test_y}
        return


class TaskReader(XYTaskReader):
    def __init__(self, taskpath=''):
        super(TaskReader, self).__init__(taskpath)


class TaskCalculator(ClassifyCalculator):
    def __init__(self, device):
        super(TaskCalculator, self).__init__(device)

