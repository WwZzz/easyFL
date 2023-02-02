import paddle.vision as pv
from benchmark.toolkits.cv.horizontal.image_classification import BuiltinClassGenerator, BuiltinClassPipe, GeneralCalculator

transforms = pv.transforms.Compose([pv.transforms.ToTensor(), pv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

TaskCalculator = GeneralCalculator

class TaskGenerator(BuiltinClassGenerator):
    def __init__(self):
        super(TaskGenerator, self).__init__('cifar10_classification', './benchmark/RAW_DATA/CIFAR10', pv.datasets.Cifar10, transforms)

class TaskPipe(BuiltinClassPipe):
    def __init__(self, task_name):
        super(TaskPipe, self).__init__(task_name, pv.datasets.Cifar10, transforms)