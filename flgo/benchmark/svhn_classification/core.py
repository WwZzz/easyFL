import torchvision
from flgo.benchmark.toolkits.cv.horizontal.image_classification import BuiltinClassGenerator, BuiltinClassPipe, GeneralCalculator
import flgo.benchmark
import os.path
import torch
transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

TaskCalculator = GeneralCalculator

class TaskGenerator(BuiltinClassGenerator):
    def __init__(self, rawdata_path=os.path.join(flgo.benchmark.path,'RAW_DATA', 'SVHN')):
        super(TaskGenerator, self).__init__('svhn_classification', rawdata_path, torchvision.datasets.SVHN, transforms)
        self.train_additional_option = {'split':'train'}
        self.test_additional_option = {'split':'test'}

    def load_data(self):
        self.train_data = self.builtin_class(root=self.rawdata_path, download=True, split='train', transform=self.transform)
        self.test_data = self.builtin_class(root=self.rawdata_path, download=True, split='test',transform=self.transform)

class TaskPipe(BuiltinClassPipe):
    def __init__(self, task_path):
        super(TaskPipe, self).__init__(task_path, torchvision.datasets.SVHN, transforms)