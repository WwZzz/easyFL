import os
from flgo.benchmark.toolkits.cv.classification import GeneralCalculator, FromDatasetPipe, FromDatasetGenerator
from .dataset import train_data, test_data

class TaskGenerator(FromDatasetGenerator):
    def __init__(self):
        super(TaskGenerator, self).__init__(benchmark=os.path.split(os.path.dirname(__file__))[-1],
                                            train_data=train_data, test_data=test_data)

class TaskPipe(FromDatasetPipe):
    def __init__(self, task_path):
        super(TaskPipe, self).__init__(task_path, train_data=train_data, test_data=test_data)

TaskCalculator = GeneralCalculator