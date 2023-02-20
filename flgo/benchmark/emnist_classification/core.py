import torchvision
from flgo.benchmark.toolkits.cv.horizontal.image_classification import BuiltinClassGenerator, BuiltinClassPipe, GeneralCalculator
import flgo.benchmark
import os.path
TaskCalculator = GeneralCalculator

class TaskGenerator(BuiltinClassGenerator):
    def __init__(self, rawdata_path=os.path.join(flgo.benchmark.path,'RAW_DATA', 'EMNIST'), split='byclass'):
        super(TaskGenerator, self).__init__('emnist_classification', rawdata_path, torchvision.datasets.EMNIST, torchvision.transforms.Compose([torchvision.transforms.ToTensor(),]))
        self.split = split

    def load_data(self):
        self.train_data = torchvision.datasets.EMNIST(root=self.rawdata_path, split = self.split, download=True, train=True, transform=self.transform)
        self.test_data = torchvision.datasets.EMNIST(root=self.rawdata_path, split = self.split, download=True, train=False, transform=self.transform)

class TaskPipe(BuiltinClassPipe):
    def __init__(self, task_name):
        super(TaskPipe, self).__init__(task_name, torchvision.datasets.EMNIST, torchvision.transforms.Compose([torchvision.transforms.ToTensor(),]))