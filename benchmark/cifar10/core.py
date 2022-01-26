from torchvision import datasets, transforms
from benchmark.toolkits import ClassifyCalculator, DefaultTaskGen, IDXTaskReader

class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients = 1, skewness = 0.5):
        super(TaskGen, self).__init__(benchmark='cifar10',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/cifar10/data',
                                      )
        self.num_classes = 10
        self.save_data = self.IDXData_to_json
        self.datasrc = {
            'lib': 'torchvision.datasets',
            'class_name': 'CIFAR10',
            'args':["'"+self.rawdata_path+"'", 'download=True', 'transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])']
        }

    def load_data(self):
        self.train_data = datasets.CIFAR10(self.rawdata_path, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
        self.test_data = datasets.CIFAR10(self.rawdata_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))


class TaskReader(IDXTaskReader):
    def __init__(self, taskpath=''):
        super(TaskReader, self).__init__(taskpath)

class TaskCalculator(ClassifyCalculator):
    def __init__(self, device):
        super(TaskCalculator, self).__init__(device)
