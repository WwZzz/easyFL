from torchvision import datasets, transforms
from benchmark.toolkits import ClassificationCalculator as TaskCalculator
from benchmark.toolkits import IDXTaskPipe as TaskPipe
from benchmark.toolkits import DefaultTaskGen
class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients = 1, skewness = 0.5, local_hld_rate=0.2, seed=0):
        super(TaskGen, self).__init__(benchmark='mnist_classification',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/RAW_DATA/MNIST',
                                      local_hld_rate=local_hld_rate,
                                      seed=seed
                                      )
        self.num_classes = 10
        self.save_task = TaskPipe.save_task
        self.visualize = self.visualize_by_class
        self.source_dict = {
            'class_path': 'torchvision.datasets',
            'class_name': 'MNIST',
            'train_args': {
                'root': '"'+self.rawdata_path+'"',
                'download': 'True',
                'transform': 'transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])',
                'train': 'True'
            },
            'test_args': {
                'root': '"'+self.rawdata_path+'"',
                'download': 'True',
                'transform': 'transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])',
                'train': 'False'
            }
        }

    def load_data(self):
        self.train_data = datasets.MNIST(self.rawdata_path, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        self.test_data = datasets.MNIST(self.rawdata_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))


# class TaskPipe(IDXTaskPipe):
#     def __init__(self):
#         super(TaskPipe, self).__init__()

# class TaskCalculator(ClassificationCalculator):
#     def __init__(self, device):
#         super(TaskCalculator, self).__init__(device)

