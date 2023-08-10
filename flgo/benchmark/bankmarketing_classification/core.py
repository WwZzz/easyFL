from flgo.benchmark.toolkits.tabular.classification import BuiltinClassGenerator, BuiltinClassPipe, GeneralCalculator
from flgo.benchmark.toolkits.tabular.classification.datasets import BankMarketing
import flgo.benchmark
import os.path

builtin_class = BankMarketing
path = os.path.join(flgo.benchmark.data_root,  'BANKMARKETING')

TaskCalculator = GeneralCalculator

class TaskGenerator(BuiltinClassGenerator):
    def __init__(self, rawdata_path=path):
        super(TaskGenerator, self).__init__(os.path.split(os.path.dirname(__file__))[-1], rawdata_path, builtin_class)

class TaskPipe(BuiltinClassPipe):
    def __init__(self, task_path):
        super(TaskPipe, self).__init__(task_path, builtin_class)