from ..toolkits.rec.datasets import Ciao
from ..toolkits.rec.rating_prediction import BuiltinClassGenerator, BuiltinClassPipe, GeneralCalculator
import os
import flgo.benchmark

path = os.path.join(flgo.benchmark.data_root, 'CIAO')
builtin_class  = Ciao

class TaskGenerator(BuiltinClassGenerator):
    def __init__(self, rawdata_path=path, split='train', min_val=10, max_val=10e8):
        super(TaskGenerator, self).__init__(os.path.split(os.path.dirname(__file__))[-1], rawdata_path=rawdata_path, builtin_class=builtin_class)
        self.additional_option = {'split': split, 'download':True, 'min_val':min_val, 'max_val':max_val}

class TaskPipe(BuiltinClassPipe):
    def __init__(self, task_path):
        super(TaskPipe, self).__init__(task_path, builtin_class)

TaskCalculator = GeneralCalculator