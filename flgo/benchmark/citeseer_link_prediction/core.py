import torch_geometric.datasets
import flgo.benchmark
from flgo.benchmark.toolkits.graph.link_prediction import *

path = os.path.join(flgo.benchmark.path,'RAW_DATA', 'CITESEER')
builtin_class = torch_geometric.datasets.Planetoid
TaskCalculator = GeneralCalculator

class TaskGenerator(BuiltinClassGenerator):
    def __init__(self, rawdata_path=path,  test_rate=0.2, test_node_split=True, disjoint_train_ratio=0.3, neg_sampling_ratio=1.0):
        super(TaskGenerator, self).__init__(benchmark=os.path.split(os.path.dirname(__file__))[-1],rawdata_path=rawdata_path,
                                            builtin_class=builtin_class, test_rate=test_rate, test_node_split=test_node_split, disjoint_train_ratio=disjoint_train_ratio, neg_sampling_ratio=neg_sampling_ratio)
        self.additional_option = {'name': 'Citeseer'}

class TaskPipe(BuiltinClassPipe):
    def __init__(self, task_name):
        super(TaskPipe, self).__init__(task_name, builtin_class=builtin_class)