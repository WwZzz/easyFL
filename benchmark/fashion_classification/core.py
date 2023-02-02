import torchvision
from benchmark.toolkits.cv.horizontal.image_classification import BuiltinClassGenerator, BuiltinClassPipe, GeneralCalculator

builtin_class = torchvision.datasets.FashionMNIST
transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
TaskCalculator = GeneralCalculator

class TaskGenerator(BuiltinClassGenerator):
    def __init__(self):
        super(TaskGenerator, self).__init__('fashion_classification', './benchmark/RAW_DATA/FASHION', builtin_class, transforms)

class TaskPipe(BuiltinClassPipe):
    def __init__(self, task_name):
        super(TaskPipe, self).__init__(task_name, builtin_class, transforms)