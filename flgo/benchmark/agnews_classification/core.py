from ..toolkits.nlp.classification import DataPipeTaskPipe, DataPipeGenerator, GeneralCalculator
import torchtext.datasets
import os
import flgo.benchmark

path = os.path.join(flgo.benchmark.path, 'RAW_DATA','AG_NEWS')

def build_datapipes(root:str, split:str='train'):
    dp = torchtext.datasets.AG_NEWS(root=root, split=split)
    return dp

class TaskGenerator(DataPipeGenerator):
    def __init__(self, rawdata_path:str=path):
        super(TaskGenerator, self).__init__(benchmark=os.path.split(os.path.dirname(__file__))[-1], rawdata_path=rawdata_path, build_datapipes=build_datapipes)
        self.build_datapipes = build_datapipes
        self.additional_option = {'root':self.rawdata_path}
        self.train_additional_option = {'split': 'train'}
        self.test_additional_option = {'split':'test'}

class TaskPipe(DataPipeTaskPipe):
    def __init__(self, task_path):
        super(TaskPipe, self).__init__(task_path, build_datapipes)

TaskCalculator = GeneralCalculator