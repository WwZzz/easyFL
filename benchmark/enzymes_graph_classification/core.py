from torch_geometric.datasets import TUDataset

from benchmark.toolkits.graph.horizontal.graph_classification import *

TaskCalculator = GraphClassificationTaskCalculator

class TaskGenerator(GraphClassificationTaskGen):
    def __init__(self, *args, **kwargs):
        super(TaskGenerator, self).__init__(benchmark='enzymes_graph_classification',rawdata_path='./benchmark/RAW_DATA/ENZYMES',
                                            builtin_class=TUDataset, dataset_name='Enzymes')

class TaskPipe(GraphClassificationTaskPipe):
    def __init__(self, task_name):
        super(TaskPipe, self).__init__(task_name, TUDataset, transform=None)