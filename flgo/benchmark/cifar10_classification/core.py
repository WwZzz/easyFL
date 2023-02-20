import torchvision
from flgo.benchmark.toolkits.cv.horizontal.image_classification import BuiltinClassGenerator, BuiltinClassPipe, GeneralCalculator
import flgo.benchmark
import os.path
transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

TaskCalculator = GeneralCalculator

class TaskGenerator(BuiltinClassGenerator):
    def __init__(self, rawdata_path=os.path.join(flgo.benchmark.path,'RAW_DATA', 'CIFAR10')):
        super(TaskGenerator, self).__init__('cifar10_classification', rawdata_path, torchvision.datasets.CIFAR10, transforms)

class TaskPipe(BuiltinClassPipe):
    def __init__(self, task_name):
        super(TaskPipe, self).__init__(task_name, torchvision.datasets.CIFAR10, transforms)