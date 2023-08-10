import torchvision
from flgo.benchmark.toolkits.cv.classification import BuiltinClassGenerator, BuiltinClassPipe, GeneralCalculator
import flgo.benchmark
import os.path

builtin_class = torchvision.datasets.EMNIST
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
path = os.path.join(flgo.benchmark.data_root, 'EMNIST')

class TaskGenerator(BuiltinClassGenerator):
    def __init__(self, rawdata_path=path, split='byclass'):
        super(TaskGenerator, self).__init__(os.path.split(os.path.dirname(__file__))[-1], rawdata_path, builtin_class, transform)
        self.split = split
        self.additional_option = {'split': self.split}

    def load_data(self):
        self.train_data = torchvision.datasets.EMNIST(root=self.rawdata_path, split = self.split, download=True, train=True, transform=self.transform)
        self.test_data = torchvision.datasets.EMNIST(root=self.rawdata_path, split = self.split, download=True, train=False, transform=self.transform)

class TaskPipe(BuiltinClassPipe):
    def __init__(self, task_path):
        super(TaskPipe, self).__init__(task_path, builtin_class, transform)

TaskCalculator = GeneralCalculator