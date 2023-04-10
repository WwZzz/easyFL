import torchvision.datasets
import torchvision.transforms as transforms
from flgo.benchmark.toolkits.cv.detection import *
import torchvision.datasets
import os
import flgo

transform = transforms.ToTensor()
builtin_class = torchvision.datasets.VOCDetection
path = os.path.join(flgo.benchmark.path,'RAW_DATA', 'VOC')

class TaskGenerator(BuiltinClassGenerator):
    def __init__(self, rawdata_path=path, year='2007'):
        super(TaskGenerator, self).__init__(benchmark=__file__.split('/')[-2], rawdata_path=rawdata_path, builtin_class=builtin_class, transform=transform)
        self.additional_option = {'year':year}
        self.train_additional_option = {'image_set':'trainval'}
        self.test_additional_option = {'image_set':'test'} if year=='2007' else {}

class TaskPipe(BuiltinClassPipe):
    def __init__(self, task_path):
        super(TaskPipe, self).__init__(task_path, builtin_class, transform)

TaskCalculator = GeneralCalculator