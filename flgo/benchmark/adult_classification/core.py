from flgo.benchmark.toolkits.tabular.classification import BuiltinClassGenerator, BuiltinClassPipe, GeneralCalculator
from flgo.benchmark.toolkits.tabular.classification.datasets import Adult
import flgo.benchmark
import os.path

builtin_class = Adult
path = os.path.join(flgo.benchmark.path, 'RAW_DATA', 'ADULT')

TaskCalculator = GeneralCalculator

class TaskGenerator(BuiltinClassGenerator):
    def __init__(self, rawdata_path=path):
        super(TaskGenerator, self).__init__(os.path.split(os.path.dirname(__file__))[-1], rawdata_path, builtin_class)

class TaskPipe(BuiltinClassPipe):
    def __init__(self, task_path):
        super(TaskPipe, self).__init__(task_path, builtin_class)