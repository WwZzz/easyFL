from torch_geometric.datasets import Planetoid

from flgo.benchmark.toolkits.graph.horizontal.node_classification import *

TaskCalculator = NodeClassificationTaskCalculator
class TaskGenerator(NodeClassificationTaskGen):
    def __init__(self, rawdata_path='./RAW_DATA/CITESEER', *args, **kwargs):
        super(TaskGenerator, self).__init__(benchmark='citeseer_node_classification',
                                            rawdata_path=rawdata_path, builtin_class=Planetoid,
                                            dataset_name='CITESEER', num_clients=10)

class TaskPipe(NodeClassificationTaskPipe):
    def __init__(self, task_name):
        super(TaskPipe, self).__init__(task_name, Planetoid, transform=None)