import torchvision
from benchmark.toolkits.cv.horizontal.image_classification import BuiltinClassGenerator, BuiltinClassPipe, GeneralCalculator
TaskCalculator = GeneralCalculator

class TaskGenerator(BuiltinClassGenerator):
    def __init__(self, split='byclass'):
        super(TaskGenerator, self).__init__('emnist_classification', './benchmark/RAW_DATA/EMNIST', torchvision.datasets.EMNIST, torchvision.transforms.Compose([torchvision.transforms.ToTensor(),]))
        self.split = split

    def load_data(self):
        self.train_data = torchvision.datasets.EMNIST(root=self.rawdata_path, split = self.split, download=True, train=True, transform=self.transform)
        self.test_data = torchvision.datasets.EMNIST(root=self.rawdata_path, split = self.split, download=True, train=False, transform=self.transform)

class TaskPipe(BuiltinClassPipe):
    def __init__(self, task_name):
        super(TaskPipe, self).__init__(task_name, torchvision.datasets.EMNIST, torchvision.transforms.Compose([torchvision.transforms.ToTensor(),]))