from flgo.benchmark.toolkits.series.time_series_classification import BuiltinClassGenerator, BuiltinClassPipe, GeneralCalculator
import os
import flgo
from flgo.benchmark.toolkits.series.time_series_classification.datasets import UCRArchiveDataset

path = os.path.join(flgo.benchmark.data_root,  'UCRArchive')
builtin_class = UCRArchiveDataset

class TaskGenerator(BuiltinClassGenerator):
    def __init__(self, rawdata_path=path):
        super(TaskGenerator, self).__init__(os.path.split(os.path.dirname(__file__))[-1], rawdata_path, builtin_class)
        self.train_additional_option = {'dataset_name': 'ElectricDevices'}
        self.test_additional_option = {'dataset_name': 'ElectricDevices'}

class TaskPipe(BuiltinClassPipe):
    def __init__(self, task_path):
        super(TaskPipe, self).__init__(task_path, builtin_class)

TaskCalculator = GeneralCalculator