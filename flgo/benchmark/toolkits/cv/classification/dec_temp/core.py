import os
from flgo.benchmark.toolkits.cv.classification import GeneralCalculator, FromDatasetGenerator, DecentralizedFromDatasetPipe
from .config import train_data
try:
    from .config import test_data
except:
    test_data = None
try:
    from .config import val_data
except:
    val_data = None
try:
    from .config import topology
except:
    topology = 'mesh'
try:
    from .config import adjacent
except:
    adjacent = None

class TaskGenerator(FromDatasetGenerator):
    def __init__(self):
        super(TaskGenerator, self).__init__(benchmark=os.path.split(os.path.dirname(__file__))[-1],
                                            train_data=train_data, val_data=val_data, test_data=test_data)

class TaskPipe(DecentralizedFromDatasetPipe):
    def __init__(self, task_path):
        super(TaskPipe, self).__init__(task_path, train_data=train_data, val_data=val_data, test_data=test_data)
        self.topology = topology
        self.adjacent = adjacent

TaskCalculator = GeneralCalculator