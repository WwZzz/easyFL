from torch_geometric.datasets import TUDataset

from flgo.benchmark.toolkits.graph.horizontal.graph_classification import *

TaskCalculator = GraphClassificationTaskCalculator

class TaskGenerator(GraphClassificationTaskGen):
    def __init__(self, rawdata_path='./RAW_DATA/MUTAG',*args, **kwargs):
        super(TaskGenerator, self).__init__(benchmark='mutag_graph_classification',rawdata_path=rawdata_path,
                                            builtin_class=TUDataset, dataset_name='Mutag')

class TaskPipe(GraphClassificationTaskPipe):
    def __init__(self, task_name):
        super(TaskPipe, self).__init__(task_name, TUDataset, transform=None)