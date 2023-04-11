import torch_geometric
import flgo.benchmark
from flgo.benchmark.toolkits.graph.graph_classification import BuiltinClassPipe, GeneralCalculator, BuiltinClassGenerator
import os

path = os.path.join(flgo.benchmark.path,'RAW_DATA', 'ENZYMES')
builtin_class = torch_geometric.datasets.TUDataset
TaskCalculator = GeneralCalculator

class TaskGenerator(BuiltinClassGenerator):
    def __init__(self, rawdata_path=path):
        super(TaskGenerator, self).__init__(benchmark=os.path.split(os.path.dirname(__file__))[-1],rawdata_path=rawdata_path,
                                            builtin_class=builtin_class)
        self.additional_option = {'name': 'ENZYMES'}

class TaskPipe(BuiltinClassPipe):
    def __init__(self, task_name):
        super(TaskPipe, self).__init__(task_name, buildin_class=builtin_class, transform=None)