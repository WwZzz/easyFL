import paddle.vision as pv
from benchmark.toolkits.cv.horizontal.image_classification import BuiltinClassGenerator, BuiltinClassPipe, GeneralCalculator

builtin_class = pv.datasets.MNIST
transforms = pv.transforms.Compose([pv.transforms.ToTensor(), pv.transforms.Normalize((0.1307,), (0.3081,))])
TaskCalculator = GeneralCalculator

class TaskGenerator(BuiltinClassGenerator):
    def __init__(self):
        super(TaskGenerator, self).__init__('mnist_classification', './benchmark/RAW_DATA/MNIST', builtin_class, transforms)

class TaskPipe(BuiltinClassPipe):
    def __init__(self, task_name):
        super(TaskPipe, self).__init__(task_name, builtin_class, transforms)