import  torch_geometric.datasets
import flgo.benchmark
from flgo.benchmark.toolkits.graph.node_classification import *

path = os.path.join(flgo.benchmark.path,'RAW_DATA', 'CORA')
TaskCalculator = GeneralCalculator
builtin_class = torch_geometric.datasets.Planetoid

class TaskGenerator(BuiltinClassGenerator):
    def __init__(self, rawdata_path=path):
        super(TaskGenerator, self).__init__(benchmark=os.path.split(os.path.dirname(__file__))[-1],
                                            rawdata_path=rawdata_path, builtin_class=builtin_class, transform=None, pre_transform=None)
        self.additional_option = {'name': 'Cora'}

class TaskPipe(BuiltinClassPipe):
    def __init__(self, task_path):
        super(TaskPipe, self).__init__(task_path, buildin_class=builtin_class, transform=None)